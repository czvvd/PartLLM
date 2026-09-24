"""
Evaluation metrics and visualization utilities for 3D part segmentation.

Includes:
- compute_sample_miou: compute mIoU for one sample
- compute_part_count_error: compute the part-count prediction error
- save_classification_ply: save a color-coded PLY file
- EvalAccumulator: aggregate evaluation metrics across processes
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.distributed






PROMPT_MODE_TO_ID = {"grounding": 0, "open": 1, "promptable": 2}
PROMPT_MODE_NAMES = ["grounding", "open", "promptable"]
REPORT_PROMPT_MODE_NAMES = PROMPT_MODE_NAMES + ["refine"]

DATASET_TYPE_TO_ID = {
    "partnext": 0,
    "3dcompat200": 1,
    "unknown": 2,
}
DATASET_TYPE_NAMES = [
    "partnext",
    "3dcompat200",
    "unknown",
]
REPORT_DATASET_TYPE_NAMES = [
    "partnext",
    "3dcompat200",
]

SUBMODE_TO_ID = {"plain": 0, "refine": 1}
SUBMODE_NAMES = ["plain", "refine"]

MACRO_CATEGORY_TO_ID = {"character": 0, "weapon": 1, "vehicle": 2, "building": 3, "other": 4, "unknown": 5}
MACRO_CATEGORY_NAMES = ["character", "weapon", "vehicle", "building", "other", "unknown"]
REPORT_MACRO_CATEGORY_NAMES = ["character", "weapon", "vehicle", "building"]

PART_COUNT_BUCKET_TO_ID = {"5_12": 0, "13_20": 1, "21_30": 2, "31_plus": 3, "unknown": 4}
PART_COUNT_BUCKET_NAMES = ["5_12", "13_20", "21_30", "31_plus", "unknown"]
REPORT_PART_COUNT_BUCKET_NAMES = ["5_12", "13_20", "21_30", "31_plus"]

MODE20_COLORMAP_HEX = [
    "#BED1D4", "#C3E1B2", "#6E95B8", "#5DAE6E", "#E4B640",
    "#EAD4A4", "#DB6E75", "#E4AAA9", "#9C8DAC", "#CFCADA",
    "#94D5C2", "#C6F0D7", "#446B71", "#8AB6B8", "#E2A1CA",
    "#EBCBDC", "#D57E36", "#F5C39D", "#997862", "#D6C2B3",
]


def _hex_to_rgb(hex_color: str):
    """Convert a hexadecimal color string to an integer (R, G, B) tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


MODE20_COLORMAP_RGB = [_hex_to_rgb(h) for h in MODE20_COLORMAP_HEX]






def compute_sample_miou(logits: torch.Tensor, labels: torch.Tensor) -> Optional[float]:
    """
    Compute mean IoU for one sample, excluding the background class.

    Args:
        logits: [1, S+1, N] mask logits; the final channel (index S) is background
        labels: [1, N] ground-truth class per point (0..S-1 are parts, S is background)

    Returns:
        Mean IoU across part classes, or None if no valid IoU is available.
    """
    
    logits = logits.squeeze(0)
    labels = labels.squeeze(0)

    num_classes = logits.shape[0]
    num_parts = num_classes - 1         

    pred = logits.argmax(dim=0)

    ious = []
    for c in range(num_parts):          
        pred_mask = (pred == c)
        gt_mask = (labels == c)
        intersection = (pred_mask & gt_mask).sum().item()
        union = (pred_mask | gt_mask).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)

    if len(ious) == 0:
        return None
    return float(sum(ious) / len(ious))


def compute_part_count_error(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute the absolute error between predicted and ground-truth part counts.

    Args:
        logits: [1, S+1, N] mask logits; the final channel is background
        labels: [1, N] ground-truth class per point

    Returns:
        float: |pred_count - gt_count|
    """
    logits = logits.squeeze(0)
    labels = labels.squeeze(0)

    num_classes = logits.shape[0]
    bg_class = num_classes - 1   

    pred = logits.argmax(dim=0)

    
    pred_classes = set(pred.unique().tolist()) - {bg_class}
    gt_classes = set(labels.unique().tolist()) - {bg_class}

    pred_count = len(pred_classes)
    gt_count = len(gt_classes)

    return float(abs(pred_count - gt_count))






def save_classification_ply(
    points: np.ndarray,
    labels: np.ndarray,
    path: str,
    colormap=None,
    max_points: int = 50000,
):
    """
    Save a labeled point cloud as an ASCII PLY file with per-vertex RGB colors.

    Args:
        points: [N, 3] NumPy array of point coordinates
        labels: [N] integer NumPy array of per-point class indices
        path: output file path
        colormap: list of RGB colors; defaults to MODE20_COLORMAP_RGB
        max_points: maximum number of saved points; larger clouds are randomly downsampled
    """
    if colormap is None:
        colormap = MODE20_COLORMAP_RGB

    n = points.shape[0]
    if n > max_points:
        indices = np.random.choice(n, max_points, replace=False)
        points = points[indices]
        labels = labels[indices]
        n = max_points

    
    colors = np.full((n, 3), 180, dtype=np.uint8)  
    for i in range(n):
        label = int(labels[i])
        if 0 <= label < len(colormap):
            colors[i] = colormap[label]

    
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")






class EvalAccumulator:
    """
    Accumulate mIoU across validation samples and all-reduce over DP ranks.

    Tracked keys:
      - all
      - report mode: grounding / open / promptable / refine
      - dataset type: PartNeXt / 3DCoMPaT200
      - prompt mode x dataset type
      - part-count bucket: 5_12 / 13_20 / 21_30 / 31_plus
      - prompt mode x part-count bucket
    """

    _DATASET_KEYS = [f"dataset_{d}" for d in REPORT_DATASET_TYPE_NAMES]
    _MODE_DATASET_KEYS = [
        f"{pm}_dataset_{d}"
        for pm in REPORT_PROMPT_MODE_NAMES
        for d in REPORT_DATASET_TYPE_NAMES
    ]
    _BUCKET_KEYS = [f"bucket_{b}" for b in REPORT_PART_COUNT_BUCKET_NAMES]
    _MODE_BUCKET_KEYS = [f"{pm}_bucket_{b}" for pm in PROMPT_MODE_NAMES for b in REPORT_PART_COUNT_BUCKET_NAMES]
    _KEYS = ["all"] + REPORT_PROMPT_MODE_NAMES + _DATASET_KEYS + _MODE_DATASET_KEYS + _BUCKET_KEYS + _MODE_BUCKET_KEYS
    _STATS_PER_KEY = 2
    _TOTAL_STATS = len(_KEYS) * _STATS_PER_KEY

    def __init__(self):
        self.reset()

    def reset(self):
        self._stats = {k: [0.0, 0.0] for k in self._KEYS}

    def update(self, miou: Optional[float], prompt_mode: str, part_count_bucket: str = None, dataset_type: str = None):
        if miou is None:
            return

        keys = ["all"]
        if prompt_mode in REPORT_PROMPT_MODE_NAMES:
            keys.append(prompt_mode)
        if dataset_type in REPORT_DATASET_TYPE_NAMES:
            keys.append(f"dataset_{dataset_type}")
            if prompt_mode in REPORT_PROMPT_MODE_NAMES:
                keys.append(f"{prompt_mode}_dataset_{dataset_type}")
        if part_count_bucket in REPORT_PART_COUNT_BUCKET_NAMES:
            bucket_key = f"bucket_{part_count_bucket}"
            keys.append(bucket_key)
            if prompt_mode in PROMPT_MODE_NAMES:
                keys.append(f"{prompt_mode}_bucket_{part_count_bucket}")

        for key in keys:
            s = self._stats[key]
            s[0] += miou
            s[1] += 1.0

    def compute(self, dp_group, device: torch.device) -> dict:
        flat = []
        for k in self._KEYS:
            flat.extend(self._stats[k])
        tensor = torch.tensor(flat, dtype=torch.float64, device=device)

        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        flat_np = tensor.cpu().numpy()

        result = {}
        for i, k in enumerate(self._KEYS):
            base = i * self._STATS_PER_KEY
            miou_sum = float(flat_np[base + 0])
            miou_count = float(flat_np[base + 1])
            if miou_count > 0:
                if k == "all":
                    metric_key = "val/mIoU"
                elif k in REPORT_PROMPT_MODE_NAMES:
                    metric_key = f"val/mIoU_{k}"
                elif k.startswith("dataset_"):
                    dataset = k[len("dataset_"):]
                    metric_key = f"val_{dataset}/mIoU"
                elif k.startswith("bucket_"):
                    bucket = k[len("bucket_"):]
                    metric_key = f"val_{bucket}/mIoU"
                else:
                    metric_key = None
                    for mode in REPORT_PROMPT_MODE_NAMES:
                        dataset_prefix = f"{mode}_dataset_"
                        if k.startswith(dataset_prefix):
                            dataset = k[len(dataset_prefix):]
                            metric_key = f"val_{dataset}/mIoU_{mode}"
                            break
                        bucket_prefix = f"{mode}_bucket_"
                        if k.startswith(bucket_prefix):
                            bucket = k[len(bucket_prefix):]
                            metric_key = f"val_{bucket}/mIoU_{mode}"
                            break
                if metric_key is not None:
                    result[metric_key] = miou_sum / miou_count
        return result
