"""Utilities for annotation-free point-prompted mesh inference."""

from __future__ import annotations

import copy
import hashlib
from contextlib import contextmanager

import numpy as np
import torch

from partnext_dataset import (
    NOSEM_PROMPTABLE_TEMPLATE,
    SEMANTIC_PROMPTABLE_TEMPLATE,
    PartNeXtPoint3DDataset,
)
from refine_protocol import format_refine_prompt


def stable_seed(base_seed: int, *parts) -> int:
    payload = "\x1f".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


@contextmanager
def temporary_numpy_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(int(seed) % (2**32))
    try:
        yield
    finally:
        np.random.set_state(state)


def _clone_point_cloud_content(point_cloud_content, obj_surface):
    cloned = []
    for item in point_cloud_content:
        if isinstance(item, dict) and item.get("type") == "point_cloud":
            new_item = dict(item)
            new_item["point_cloud"] = obj_surface.to(torch.bfloat16)
            cloned.append(new_item)
        else:
            cloned.append(copy.deepcopy(item))
    return cloned


def build_colored_geo_cache(base_geo_cache, rgb_uint8, model, encoder_type, utonia_seed):
    """Rebuild point-cloud inputs with unchanged geometry and caller-provided RGB."""
    sampled_points = np.asarray(base_geo_cache["sampled_points"], dtype=np.float32)
    sampled_normals = np.asarray(base_geo_cache["sampled_normals"], dtype=np.float32)
    rgb_uint8 = np.asarray(rgb_uint8, dtype=np.uint8)
    if sampled_points.ndim != 2 or sampled_points.shape[1] != 3:
        raise ValueError("sampled_points must have shape [N, 3]")
    if sampled_normals.shape != sampled_points.shape or rgb_uint8.shape != sampled_points.shape:
        raise ValueError("normals and RGB must match sampled_points")

    obj_surface = PartNeXtPoint3DDataset._build_obj_surface(
        sampled_points, rgb_uint8.astype(np.float32) / 255.0
    )
    utonia_point_dict = None
    if encoder_type == "utonia":
        from transformers.models.qwen3_vl import utonia as _utonia

        transform = base_geo_cache.get("utonia_transform")
        if transform is None:
            raise ValueError("Utonia transform is unavailable")
        point_cloud = {
            "coord": sampled_points.copy(),
            "color": rgb_uint8.copy(),
            "normal": sampled_normals.copy(),
        }
        with temporary_numpy_seed(utonia_seed):
            transformed = transform(point_cloud)
        utonia_point_dict = _utonia.data.collate_fn([transformed])
        for key, value in list(utonia_point_dict.items()):
            if not isinstance(value, torch.Tensor):
                continue
            dtype = torch.float32 if value.is_floating_point() else value.dtype
            utonia_point_dict[key] = value.to(device=model.device, dtype=dtype)

    return {
        "diffusion_inputs": dict(base_geo_cache["diffusion_inputs"]),
        "utonia_point_dict": utonia_point_dict,
        "mesh_vis_info": base_geo_cache["mesh_vis_info"],
        "point_cloud_content": _clone_point_cloud_content(
            base_geo_cache["point_cloud_content"], obj_surface
        ),
    }


def build_promptable_prompt(template_mode: str, point_str: str) -> str:
    template = NOSEM_PROMPTABLE_TEMPLATE if template_mode == "nosem" else SEMANTIC_PROMPTABLE_TEMPLATE
    return template.format(point=point_str)


def build_refine_prompt(template_mode: str, click_history) -> str:
    return format_refine_prompt(click_history, semantic=template_mode)


def decode_promptable_mask(mask_logits, num_points: int):
    """Decode the fixed foreground query (class 0) without ground truth."""
    empty = np.zeros(int(num_points), dtype=bool)
    if mask_logits is None:
        return empty, {"num_classes": 0, "structure_valid": False}
    logits = mask_logits.detach().cpu().float()
    if logits.ndim != 3 or logits.shape[0] != 1:
        return empty, {"num_classes": 0, "structure_valid": False}
    if logits.shape[2] != int(num_points):
        raise ValueError(
            f"mask point count mismatch: logits={logits.shape[2]} expected={num_points}"
        )
    num_classes = int(logits.shape[1])
    if num_classes != 2:
        return empty, {"num_classes": num_classes, "structure_valid": False}
    return logits[0].argmax(dim=0).numpy() == 0, {
        "num_classes": num_classes,
        "structure_valid": True,
    }
