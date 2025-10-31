import argparse
import copy
import os
import warnings

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import scatter  # unwrap DataContainers to device via GPU id list
from torchpack.utils.config import configs

# ---- robust tqdm import ----
try:
    from torchpack.utils.tqdm import tqdm
except Exception:
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x, *a, **k): return x

# Silence Shapely deprecation spam (nuScenes maps)
try:
    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except Exception:
    pass

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            obj[k] = recursive_eval(obj[k], globals)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = recursive_eval(v, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)
    return obj


def _mkdir(path):
    if path:
        mmcv.mkdir_or_exist(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render camera/LiDAR visuals (single GPU)")
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"],
                        help="Render ground-truth or predictions")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Required when --mode pred")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None,
                        help="Keep only these class ids")
    parser.add_argument("--bbox-score", type=float, default=None,
                        help="Min score for predicted boxes")
    parser.add_argument("--map-score", type=float, default=0.5,
                        help="Threshold for BEV masks if used")
    parser.add_argument("--out-dir", type=str, default="viz")
    parser.add_argument("--no-map", action="store_true",
                        help="Disable BEV map/segmentation rendering (faster)")
    args, opts = parser.parse_known_args()

    # Load config (TorchPack-style) and allow CLI overrides
    configs.load(args.config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # Speed tweak
    torch.backends.cudnn.benchmark = bool(getattr(cfg, "cudnn_benchmark", False))

    # Single GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(0)

    # Data (non-distributed)
    dataset = build_dataset(cfg.data[args.split])
    workers = max(0, int(getattr(cfg.data, "workers_per_gpu", 2)))
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=min(workers, 2),  # small to reduce spawn overhead
        dist=False,
        shuffle=False,
    )

    # Model
    if args.mode == "pred":
        if not args.checkpoint:
            raise ValueError("--mode pred requires --checkpoint")
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")
        model = model.to(device)
        model.eval()
    else:
        model = None  # GT-only render

    # Output dirs
    _mkdir(args.out_dir)
    _mkdir(os.path.join(args.out_dir, "lidar"))
    if not args.no_map:
        _mkdir(os.path.join(args.out_dir, "map"))

    # Optional: limit frames via env var (e.g., VIZ_LIMIT=20)
    try:
        max_samples = int(os.environ.get("VIZ_LIMIT", "0"))
    except Exception:
        max_samples = 0

    for i, data in enumerate(tqdm(dataflow)):
        if max_samples and i >= max_samples:
            break

        # metas for filenames/transforms
        metas = data["metas"].data[0][0]
        name = "{}-{}".format(metas["timestamp"], metas["token"])

        # ---- Forward (prediction mode): unwrap DataContainers to CUDA:0 ----
        if args.mode == "pred":
            # IMPORTANT: pass integer GPU ids (e.g., [0]), not torch.device
            inputs = scatter(data, [0])[0]
            with torch.inference_mode():
                outputs = model(**inputs)
        else:
            outputs = None

        # -------------- Boxes (GT or Pred) --------------
        bboxes = None
        labels = None

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            b = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            l = data["gt_labels_3d"].data[0][0].numpy()
            if args.bbox_classes is not None:
                mask = np.isin(l, args.bbox_classes)
                b, l = b[mask], l[mask]
            # convert z-center to bottom-centered for LiDARInstance3DBoxes
            b[..., 2] -= b[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(b, box_dim=9)
            labels = l

        elif args.mode == "pred":
            out0 = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            if isinstance(out0, dict) and "boxes_3d" in out0:
                b = out0["boxes_3d"].tensor.detach().cpu().numpy()
                s = out0.get("scores_3d", None)
                l = out0.get("labels_3d", None)
                if l is None:
                    l = np.zeros((b.shape[0],), dtype=np.int64)

                if args.bbox_classes is not None:
                    mask = np.isin(l, args.bbox_classes)
                    b = b[mask]; l = l[mask]
                    if s is not None:
                        s = s[mask]
                if args.bbox_score is not None and s is not None:
                    mask = (s.detach().cpu().numpy() >= float(args.bbox_score))
                    b = b[mask]; l = l[mask]
                b[..., 2] -= b[..., 5] / 2
                bboxes = LiDARInstance3DBoxes(b, box_dim=9)
                labels = l

        # -------------- (Optional) BEV masks --------------
        masks = None
        if not args.no_map:
            if args.mode == "gt" and "gt_masks_bev" in data:
                m = data["gt_masks_bev"].data[0].numpy()
                masks = m.astype(bool)
            elif args.mode == "pred":
                out0 = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                if isinstance(out0, dict) and "masks_bev" in out0:
                    m = out0["masks_bev"].detach().cpu().numpy()
                    masks = m >= float(args.map_score)

        # -------------- Camera visualizations --------------
        if "img" in data and "filename" in metas:
            filenames = metas["filename"]
            lidar2image = metas["lidar2image"]
            for k, image_path in enumerate(filenames):
                cam_out = os.path.join(args.out_dir, f"camera-{k}")
                _mkdir(cam_out)
                image = mmcv.imread(image_path)
                visualize_camera(
                    os.path.join(cam_out, f"{name}.png"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=lidar2image[k],
                    classes=cfg.object_classes,
                )

        # -------------- LiDAR topdown --------------
        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in (0, 3)],
                ylim=[cfg.point_cloud_range[d] for d in (1, 4)],
                classes=cfg.object_classes,
            )

        # -------------- BEV map masks (only if not disabled) --------------
        if (masks is not None) and (not args.no_map):
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )


if __name__ == "__main__":
    main()

