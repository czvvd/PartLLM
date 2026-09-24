"""
Point cloud data augmentation for PartNeXt 3D part segmentation.

Pure-function module. No dependency on Utonia transform registry.
Imported via _import_from_path() in partnext_dataset.py.

See spec: docs/superpowers/specs/2026-04-05-point-cloud-augmentation-design.md
"""

import math
import random
from collections.abc import Mapping
from typing import Callable

import numpy as np

DEFAULT_AUG_CONFIG = {
    "enabled": False,
    "tilt_xz": {
        "enabled": False,
        "angle": [-math.pi / 12, math.pi / 12],
        "p": 0.1,
    },
    "rotate_y": {
        "enabled": True,
        "angle": [-math.pi / 2, math.pi / 2],
        "p": 0.3,
    },
    "reorient_up_axis": {
        "enabled": False,
        "p": 0.15,
        "x_up_probability": 0.5,
    },
    "scale": {
        "enabled": True,
        "range": [0.9, 1.1],
        "p": 0.3,
    },
    "flip": {
        "enabled": False,
        "p_x": 0.5,
        "p_y": 0.5,
    },
    "jitter": {
        "enabled": False,
        "sigma": 0.005,
        "clip": 0.02,
    },
    "chromatic_jitter": {
        "enabled": True,
        "std": 0.01,
        "p": 0.15,
    },
    "chromatic_translation": {
        "enabled": True,
        "ratio": 0.02,
        "p": 0.2,
    },
    "color_drop": {
        "enabled": True,
        "p": 0.02,
        "fill": [128, 128, 128],
    },
}


def _tilt_xz(coord, normal, angle_range, p):
    """Tilt around a random horizontal axis while preserving handedness."""
    if random.random() >= p:
        return coord, normal, np.eye(3, dtype=np.float32)

    axis_angle = random.uniform(0.0, 2.0 * math.pi)
    axis = np.array(
        [math.cos(axis_angle), 0.0, math.sin(axis_angle)],
        dtype=np.float32,
    )
    angle = random.uniform(angle_range[0], angle_range[1])
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    one_minus_c = 1.0 - c
    R = np.array(
        [
            [
                c + x * x * one_minus_c,
                x * y * one_minus_c - z * s,
                x * z * one_minus_c + y * s,
            ],
            [
                y * x * one_minus_c + z * s,
                c + y * y * one_minus_c,
                y * z * one_minus_c - x * s,
            ],
            [
                z * x * one_minus_c - y * s,
                z * y * one_minus_c + x * s,
                c + z * z * one_minus_c,
            ],
        ],
        dtype=np.float32,
    )
    return coord @ R.T, normal @ R.T, R


def _get_sub_config(aug_config: Mapping, key: str) -> dict:
    default = DEFAULT_AUG_CONFIG.get(key, {})
    user = aug_config.get(key, {})
    if isinstance(user, Mapping):
        return {**default, **dict(user)}
    return default


def _rotate_y(coord, normal, angle_range, p):
    """Rotate around Y axis (up axis for this dataset)."""
    if random.random() >= p:
        return coord, normal, np.eye(3, dtype=np.float32)
    angle = random.uniform(angle_range[0], angle_range[1])
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    coord = coord @ R.T
    normal = normal @ R.T
    return coord, normal, R


def _reorient_up_axis(coord, normal, p, x_up_probability):
    """Map canonical +Y-up to +X-up or +Z-up with a proper rotation."""
    if random.random() >= p:
        return coord, normal, np.eye(3, dtype=np.float32)

    if random.random() < x_up_probability:

        R = np.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    else:

        R = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )

    return coord @ R.T, normal @ R.T, R


def _scale(coord, scale_range, p):
    if random.random() >= p:
        return coord, 1.0
    s = random.uniform(scale_range[0], scale_range[1])
    coord = coord * s
    return coord, s


def _flip(coord, normal, p_x, p_y):
    signs = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    if random.random() < p_x:
        coord[:, 0] = -coord[:, 0]
        normal[:, 0] = -normal[:, 0]
        signs[0] = -1.0
    if random.random() < p_y:
        coord[:, 1] = -coord[:, 1]
        normal[:, 1] = -normal[:, 1]
        signs[1] = -1.0
    return coord, normal, signs


def _jitter(coord, sigma, clip):
    noise = np.clip(sigma * np.random.randn(*coord.shape), -clip, clip).astype(np.float32)
    return coord + noise


def _chromatic_jitter(color, std, p):
    if random.random() >= p:
        return color
    noise = (np.random.randn(color.shape[0], 3) * std * 255).astype(np.float32)
    color_f = color.astype(np.float32) + noise
    return np.clip(color_f, 0, 255).astype(np.uint8)


def _chromatic_translation(color, ratio, p):
    if random.random() >= p:
        return color
    tr = ((np.random.rand(1, 3) - 0.5) * 255 * 2 * ratio).astype(np.float32)
    color_f = color.astype(np.float32) + tr
    return np.clip(color_f, 0, 255).astype(np.uint8)


def _color_drop(color, p, fill):
    if random.random() >= p:
        return color
    fill_arr = np.array(fill, dtype=np.uint8).reshape(1, 3)
    return np.broadcast_to(fill_arr, color.shape).copy()


def augment_point_cloud(
    coord: np.ndarray,
    color: np.ndarray,
    normal: np.ndarray,
    aug_config: Mapping,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Callable]:
    """
    Apply data augmentation to a point cloud.

    The caller must check aug_config["enabled"] before calling.
    Copies inputs internally; does not modify the original arrays.

    Args:
        coord:  [N, 3] float32 — point positions.
        color:  [N, 3] uint8   — RGB colors.
        normal: [N, 3] float32 — unit normals.
        aug_config: augmentation config dict.

    Returns:
        coord:        [N, 3] float32 — augmented coordinates.
        color:        [N, 3] uint8   — augmented colors.
        normal:       [N, 3] float32 — augmented normals.
        transform_fn: Callable[[np.ndarray], np.ndarray] for bbox linkage.
    """
    coord = coord.copy()
    color = color.copy()
    normal = normal.copy()

    M = np.eye(3, dtype=np.float32)
    applied_scale = 1.0

    cfg = _get_sub_config(aug_config, "tilt_xz")
    if cfg.get("enabled", False):
        coord, normal, R = _tilt_xz(coord, normal, cfg["angle"], cfg["p"])
        M = R @ M

    cfg = _get_sub_config(aug_config, "rotate_y")
    if cfg.get("enabled", True):
        coord, normal, R = _rotate_y(coord, normal, cfg["angle"], cfg["p"])
        M = R @ M



    cfg = _get_sub_config(aug_config, "reorient_up_axis")
    if cfg.get("enabled", False):
        coord, normal, R = _reorient_up_axis(
            coord,
            normal,
            cfg["p"],
            cfg.get("x_up_probability", 0.5),
        )
        M = R @ M

    cfg = _get_sub_config(aug_config, "scale")
    if cfg.get("enabled", True):
        coord, s = _scale(coord, cfg["range"], cfg["p"])
        applied_scale = float(s)
        M = np.diag(np.array([s, s, s], dtype=np.float32)) @ M

    cfg = _get_sub_config(aug_config, "flip")
    if cfg.get("enabled", True):
        coord, normal, signs = _flip(coord, normal, cfg["p_x"], cfg["p_y"])
        M = np.diag(signs) @ M

    cfg = _get_sub_config(aug_config, "jitter")
    if cfg.get("enabled", True):
        coord = _jitter(coord, cfg["sigma"], cfg["clip"])

    cfg = _get_sub_config(aug_config, "chromatic_jitter")
    if cfg.get("enabled", True):
        color = _chromatic_jitter(color, cfg["std"], cfg["p"])

    cfg = _get_sub_config(aug_config, "chromatic_translation")
    if cfg.get("enabled", True):
        color = _chromatic_translation(color, cfg["ratio"], cfg["p"])

    cfg = _get_sub_config(aug_config, "color_drop")
    if cfg.get("enabled", True):
        color = _color_drop(color, cfg["p"], cfg.get("fill", [128, 128, 128]))

    _M = M.copy()

    def transform_fn(points: np.ndarray) -> np.ndarray:
        return (points.astype(np.float32) @ _M.T).astype(np.float32)




    transform_fn.scale_factor = applied_scale

    return coord, color, normal, transform_fn


def transform_bbox_aabb(
    bbox: list[float],
    transform_fn: Callable,
) -> list[float]:
    """Transform AABB via 8-corner expansion."""
    x_min, y_min, z_min, x_max, y_max, z_max = bbox
    corners = np.array([
        [x_min, y_min, z_min], [x_min, y_min, z_max],
        [x_min, y_max, z_min], [x_min, y_max, z_max],
        [x_max, y_min, z_min], [x_max, y_min, z_max],
        [x_max, y_max, z_min], [x_max, y_max, z_max],
    ], dtype=np.float32)
    transformed = transform_fn(corners)
    new_min = transformed.min(axis=0)
    new_max = transformed.max(axis=0)
    return [float(new_min[0]), float(new_min[1]), float(new_min[2]),
            float(new_max[0]), float(new_max[1]), float(new_max[2])]


def tight_bboxes_from_point_masks(
    coord: np.ndarray,
    part_masks: np.ndarray,
    fallback_bboxes: list | None = None,
    min_points: int = 8,
) -> list:
    """Compute tight AABBs for the actual point-level segmentation targets.

    A transformed source AABB is generally looser than the AABB of the
    transformed part geometry. This function computes the latter directly
    from the sampled points used by the model. Parts with too few sampled
    points fall back to the transformed source box to avoid degenerate targets.
    """
    coord = np.asarray(coord, dtype=np.float32)
    masks = np.asarray(part_masks, dtype=bool)
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError(f"coord must have shape [N, 3], got {coord.shape}")
    if masks.ndim != 2 or masks.shape[1] != coord.shape[0]:
        raise ValueError(
            f"part_masks must have shape [S, N] with N={coord.shape[0]}, "
            f"got {masks.shape}"
        )

    result = []
    for part_idx, mask in enumerate(masks):
        point_count = int(mask.sum())
        if point_count >= min_points:
            part_coord = coord[mask]
            bbox_min = part_coord.min(axis=0)
            bbox_max = part_coord.max(axis=0)
            result.append(
                [
                    float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2]),
                    float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2]),
                ]
            )
        elif fallback_bboxes is not None and part_idx < len(fallback_bboxes):
            result.append(fallback_bboxes[part_idx])
        elif point_count > 0:
            part_coord = coord[mask]
            bbox_min = part_coord.min(axis=0)
            bbox_max = part_coord.max(axis=0)
            result.append(
                [
                    float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2]),
                    float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2]),
                ]
            )
        else:
            result.append(None)
    return result
