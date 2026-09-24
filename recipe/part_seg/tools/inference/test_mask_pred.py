#!/usr/bin/env python3
"""PartLLM full-shape 3D part segmentation inference.

The model generates part descriptions and segmentation tokens, then jointly
decodes all part classes and the background class into per-point logits.

Examples:
    # Run all built-in granularities.
    python test_mask_pred.py --model_path /path/to/ckpt --config_path /path/to/config --glb_dir /path/to/meshes

    # Run one granularity.
    python test_mask_pred.py ... --glb_dir /path/to/meshes --prompt_set fine

    # Explicitly request a target part count.
    python test_mask_pred.py ... --num_parts 5

    # Limit the number of meshes.
    python test_mask_pred.py ... --max_samples 10
"""
import argparse
import json
import os
import random
import shutil
import sys
import traceback
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import trimesh
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from partnext_dataset import (
    PartNeXtPoint3DDataset,
    SEMANTIC_OPEN_TEMPLATES,
    NOSEM_OPEN_TEMPLATES,
    LEGACY_SEMANTIC_TEMPLATES,
    LEGACY_NOSEM_TEMPLATES,
    _classify_nosem_bucket,
)
from mesh_utils import scene2meshes, normalize_meshes_diag
try:
    from loc_token_utils import parse_loc_tokens_line
except ImportError:
    parse_loc_tokens_line = None



def _ensure_custom_transformers():
    import transformers

    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, "..", "..", "..", ".."))
    src = os.path.join(project_root, "transformers-4-27", "src", "transformers", "models", "qwen3_vl")
    dst = os.path.join(os.path.dirname(transformers.__file__), "models", "qwen3_vl")

    if not os.path.isdir(src):
        print(f"[Warning] Custom qwen3_vl source not found: {src}")
        return

    src_file = os.path.join(src, "modeling_qwen3_vl.py")
    dst_file = os.path.join(dst, "modeling_qwen3_vl.py")
    if os.path.exists(dst_file) and os.path.getsize(src_file) == os.path.getsize(dst_file):
        return  

    print(f"[Info] Syncing custom qwen3_vl: {src} -> {dst}")
    for fname in os.listdir(src):
        s = os.path.join(src, fname)
        d = os.path.join(dst, fname)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

_ensure_custom_transformers()

from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLForConditionalGeneration
from verl.utils.model import get_hf_auto_model_class
from verl.models.transformers.qwen3_vl import qwen3_vl_base_forward, forward_with_normal_backend
from verl.utils import hf_tokenizer

from pca_features import FeatureCapture as _PCAFeatureCapture, is_enabled as _pca_enabled






def hex_to_rgb(hex_color: str):
    """Convert a hexadecimal color to an integer RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


MODE20_COLORMAP_HEX = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4",
    "#42D4F4", "#F032E6", "#BFEF45", "#FABED4", "#469990",
    "#DCBEFF", "#9A6324", "#FFFAC8", "#800000", "#AAFFC3",
    "#808000", "#FFD8B1", "#000075", "#A9A9A9", "#FF4500",
]

BG_COLOR_HEX = "#C0C0C0"  
BG_COLOR_RGB = hex_to_rgb(BG_COLOR_HEX)


MODE20_COLOR_NAMES = [
    "red", "green", "blue", "orange", "purple",
    "cyan", "magenta", "lime", "pink", "teal",
    "lavender", "brown", "light yellow", "maroon", "mint",
    "olive", "apricot", "navy", "gray", "orange red",
]


def bbox_aabb_to_wireframe_points(bbox, color=(0, 255, 0), num_samples=100):
    """Convert an axis-aligned bounding box to sampled wireframe points.

    Args:
        bbox: [x_min, y_min, z_min, x_max, y_max, z_max]
        color: integer RGB values in the range 0-255
        num_samples: number of sampled points per edge
    Returns:
        ``(vertices [M, 3], colors [M, 3])`` with uint8 colors.
    """
    x_min, y_min, z_min, x_max, y_max, z_max = bbox
    corners = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_max, y_max, z_max], [x_min, y_max, z_max],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    all_pts = []
    for i, j in edges:
        t = np.linspace(0, 1, num_samples, endpoint=True).reshape(-1, 1)
        all_pts.append(corners[i] * (1 - t) + corners[j] * t)
    verts = np.concatenate(all_pts, axis=0)
    colors = np.tile(np.array(color, dtype=np.uint8), (len(verts), 1))
    return verts, colors


def save_per_part_ply(points, pred_labels, part_idx, label, save_dir, bbox=None):
    """Save a single-part PLY with the selected part highlighted."""
    n = points.shape[0]
    mask = pred_labels == part_idx

    colors = np.full((n, 3), 180, dtype=np.uint8)  
    colors[mask] = [220, 60, 60]  

    safe_label = "".join(c if c.isalnum() or c in ("_", "-", " ") else "_" for c in label)[:50]
    filename = f"part_{part_idx}_{safe_label}.ply"
    path = os.path.join(save_dir, filename)

    if bbox is not None:
        bbox_verts, bbox_colors = bbox
        merged_points = np.concatenate([points, bbox_verts], axis=0)
        merged_colors = np.concatenate([colors, bbox_colors], axis=0)
    else:
        merged_points = points
        merged_colors = colors

    _write_ply_with_colors(merged_points, merged_colors, path)

    num_positive = mask.sum()
    print(f"  Part {part_idx} [{label}]: {num_positive}/{n} points")


def _write_ply_with_colors(points, colors, path):
    """Save a point cloud with RGB colors as an ASCII PLY file.

    Args:
        points: [N, 3] float
        colors: [N, 3] uint8 (0-255)
        path: output path
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = len(points)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            r, g, b = int(colors[i][0]), int(colors[i][1]), int(colors[i][2])
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def _save_legend(save_dir, filename, parts, colormap_hex):
    """Save a color legend for part classes and the background class."""
    path = os.path.join(save_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        for i, p in enumerate(parts):
            label = p["label"] if isinstance(p, dict) else str(p)
            if "<|BG|>" in label or "<|im_end|>" in label:
                continue
            color_name = MODE20_COLOR_NAMES[i % len(MODE20_COLOR_NAMES)]
            f.write(f"class {i}: [{color_name}]  {label}\n")
        f.write("class BG: [silver]  background\n")


def _format_legend_block(prefix, parts):
    """Format a color legend for a list of parts."""
    lines = []
    for i, p in enumerate(parts):
        label = p["label"] if isinstance(p, dict) else str(p)
        if "<|BG|>" in label or "<|im_end|>" in label:
            continue
        color_name = MODE20_COLOR_NAMES[i % len(MODE20_COLOR_NAMES)]
        bbox_str = ""
        if isinstance(p, dict) and p.get("bbox"):
            bbox_str = f"  bbox={p['bbox']}"
        lines.append(f"  class {i} [{color_name}]: {label}{bbox_str}")
    lines.append("  class BG [silver]: background")
    return "\n".join(lines)


def _save_info_txt(save_dir, user_prompt, response, gt_response, pred_parts, gt_parts):
    """Save the prompt, responses, and color legends in one text file."""
    path = os.path.join(save_dir, "info.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("PROMPT\n")
        f.write("=" * 60 + "\n")
        f.write((user_prompt or "(none)") + "\n\n")

        f.write("=" * 60 + "\n")
        f.write("PREDICTION\n")
        f.write("=" * 60 + "\n")
        f.write(_format_legend_block("Pred", pred_parts) + "\n\n")
        f.write("Raw response:\n")
        f.write(response + "\n\n")

        if gt_response is not None:
            f.write("=" * 60 + "\n")
            f.write("GROUND TRUTH (colors in gt.ply)\n")
            f.write("=" * 60 + "\n")
            f.write(_format_legend_block("GT", gt_parts) + "\n\n")
            f.write("Raw response:\n")
            f.write(gt_response + "\n")


def _get_part_colormap(parts, colormap_hex):
    """Build the class-index-to-RGB map with a fixed background color.

    Returns:
        list of (R, G, B) tuples, length = num_classes (parts + BG)
    """
    colors = []
    for i, p in enumerate(parts):
        label = p["label"] if isinstance(p, dict) else str(p)
        if "<|BG|>" in label or "<|im_end|>" in label:
            colors.append(BG_COLOR_RGB)
        else:
            colors.append(hex_to_rgb(colormap_hex[i % len(colormap_hex)]))
    return colors


def _fill_unsampled_faces(face_labels, combined_mesh):
    """Label unsampled faces by adjacency propagation and nearest neighbors.

    Args:
        face_labels: integer array; ``-1`` marks an unlabeled face
        combined_mesh: trimesh.Trimesh

    Returns:
        Updated array with all reachable faces labeled.
    """
    unlabeled = np.where(face_labels == -1)[0]
    if len(unlabeled) == 0:
        return face_labels

    
    adjacency = combined_mesh.face_adjacency
    for _ in range(10):  
        still_unlabeled = np.where(face_labels == -1)[0]
        if len(still_unlabeled) == 0:
            break
        unlabeled_set = set(still_unlabeled.tolist())
        changed = False
        for a, b in adjacency:
            if a in unlabeled_set and face_labels[b] != -1:
                face_labels[a] = face_labels[b]
                unlabeled_set.discard(a)
                changed = True
            elif b in unlabeled_set and face_labels[a] != -1:
                face_labels[b] = face_labels[a]
                unlabeled_set.discard(b)
                changed = True
        if not changed:
            break

    
    still_unlabeled = np.where(face_labels == -1)[0]
    if len(still_unlabeled) > 0:
        from scipy.spatial import cKDTree

        labeled_mask = face_labels != -1
        labeled_indices = np.where(labeled_mask)[0]
        if len(labeled_indices) > 0:
            centroids = combined_mesh.triangles_center
            tree = cKDTree(centroids[labeled_indices])
            _, nn_idx = tree.query(centroids[still_unlabeled])
            face_labels[still_unlabeled] = face_labels[labeled_indices[nn_idx]]

    return face_labels


def _fill_unsampled_faces_kdtree_fast(face_labels, combined_mesh):
    """Fill high-resolution meshes using one parallel nearest-face query."""
    labels = np.asarray(face_labels).copy()
    remaining = np.flatnonzero(labels < 0)
    if len(remaining) == 0:
        return labels
    labeled = np.flatnonzero(labels >= 0)
    if len(labeled) == 0:
        return labels

    from scipy.spatial import cKDTree

    centers = combined_mesh.triangles_center
    _, nearest = cKDTree(centers[labeled]).query(
        centers[remaining], workers=-1
    )
    labels[remaining] = labels[labeled[nearest]]
    return labels


def _apply_vertex_colors(combined_mesh, face_labels, parts, colormap_hex):
    """Convert per-face labels to per-vertex colors by majority vote.

    Args:
        combined_mesh: trimesh.Trimesh
        face_labels: [num_faces] numpy int — per-face class index
        parts: list of dict
        colormap_hex: list of hex color strings

    Returns:
        colored_mesh: trimesh.Trimesh with vertex_colors set
    """
    import trimesh as _trimesh
    num_verts = len(combined_mesh.vertices)
    faces = np.asarray(combined_mesh.faces, dtype=np.int64)
    labels = np.asarray(face_labels, dtype=np.int64)

    
    
    
    flat_vertices = faces.reshape(-1)
    flat_labels = np.repeat(labels, 3)
    valid = flat_labels >= 0
    if valid.any():
        num_classes = max(int(flat_labels[valid].max()) + 1, 1)
        votes = np.zeros((num_verts, num_classes), dtype=np.uint16)
        np.add.at(votes, (flat_vertices[valid], flat_labels[valid]), 1)
        vertex_labels = votes.argmax(axis=1)
        has_vote = np.bincount(
            flat_vertices[valid], minlength=num_verts
        ) > 0
    else:
        vertex_labels = np.zeros(num_verts, dtype=np.int64)
        has_vote = np.zeros(num_verts, dtype=bool)

    
    part_colors = _get_part_colormap(parts, colormap_hex)
    vertex_colors = np.full((num_verts, 4), [200, 200, 200, 255], dtype=np.uint8)
    palette_size = max(len(part_colors), int(vertex_labels.max()) + 1, 1)
    palette = np.full((palette_size, 4), [*BG_COLOR_RGB, 255], dtype=np.uint8)
    if part_colors:
        palette[:len(part_colors), :3] = np.asarray(part_colors, dtype=np.uint8)
    vertex_colors[has_vote] = palette[vertex_labels[has_vote]]

    colored_mesh = combined_mesh.copy()
    colored_mesh.visual = _trimesh.visual.ColorVisuals(mesh=colored_mesh, vertex_colors=vertex_colors)
    return colored_mesh


def _graph_cut_alpha_expansion(face_labels, face_logits_sum, face_point_count, combined_mesh, _lambda=1.0, iterations=1):
    """Refine face-label boundaries with alpha-expansion graph cut.

    Args:
        face_labels: current per-face labels
        face_logits_sum: accumulated class logits per face
        face_point_count: number of sampled points per face
        combined_mesh: trimesh.Trimesh
        _lambda: smoothness weight
        iterations: number of alpha-expansion iterations

    Returns:
        Refined per-face labels.
    """
    try:
        import igraph
    except ImportError:
        print("[Warning] igraph not installed, skipping graph cut. Install with: pip install python-igraph")
        return face_labels

    num_faces = len(face_labels)
    num_classes = face_logits_sum.shape[1]

    
    
    sampled_mask = face_point_count > 0
    face_probs = np.full((num_faces, num_classes), 1.0 / num_classes)
    if sampled_mask.any():
        avg_logits = face_logits_sum[sampled_mask] / face_point_count[sampled_mask, None]

        avg_logits -= avg_logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(avg_logits)
        face_probs[sampled_mask] = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    eps = 1e-10
    cost_data = -np.log(face_probs + eps)

    
    angles = combined_mesh.face_adjacency_angles
    cost_smoothness = -np.log(angles / np.pi + eps) * _lambda


    partition = face_labels.copy()
    labels = np.unique(partition)

    A = "alpha"
    B = "alpha_complement"

    face_adjacency = combined_mesh.face_adjacency

    for _iter in range(iterations):
        for label in labels:
            
            node2index = {A: 0, B: 1}
            for f in range(num_faces):
                node2index[f] = 2 + f

            aux_count = 0
            aux_nodes = {}
            for i, (f1, f2) in enumerate(face_adjacency):
                f1, f2 = int(f1), int(f2)
                if partition[f1] != partition[f2]:
                    key = (f1, f2)
                    if key not in aux_nodes:
                        aux_nodes[key] = num_faces + 2 + aux_count
                        node2index[key] = aux_nodes[key]
                        aux_count += 1

            total_nodes = num_faces + 2 + aux_count

            
            edges = []
            capacities = []


            for f in range(num_faces):
                
                edges.append((node2index[A], node2index[f]))
                capacities.append(float(cost_data[f, label]))
                
                if partition[f] == label:
                    edges.append((node2index[f], node2index[B]))
                    capacities.append(float("inf"))
                else:
                    edges.append((node2index[f], node2index[B]))
                    capacities.append(float(cost_data[f, partition[f]]))


            for i, (f1, f2) in enumerate(face_adjacency):
                f1, f2 = int(f1), int(f2)
                cs = float(cost_smoothness[i])
                if partition[f1] == partition[f2]:
                    if partition[f1] != label:
                        edges.append((node2index[f1], node2index[f2]))
                        capacities.append(cs)
                else:
                    key = (f1, f2)
                    a_idx = aux_nodes[key]
                    edges.append((a_idx, node2index[B]))
                    capacities.append(cs)
                    if partition[f1] != label:
                        edges.append((node2index[f1], a_idx))
                        capacities.append(cs)
                    if partition[f2] != label:
                        edges.append((a_idx, node2index[f2]))
                        capacities.append(cs)

            G = igraph.Graph(n=total_nodes, edges=edges, directed=False)
            G.es["capacity"] = capacities

            result = G.st_mincut(source=node2index[A], target=node2index[B], capacity="capacity")

            index2node = {v: k for k, v in node2index.items()}
            
            T_set = set(result.partition[1])
            T_faces = np.array([index2node[v] for v in T_set if isinstance(index2node.get(v), int)], dtype=np.int64)
            if len(T_faces) > 0:
                partition[T_faces] = label

    return partition


def _color_mesh_by_logits(
    combined_mesh,
    face_idx,
    logits,
    parts,
    colormap_hex,
    graph_cut_cfg=None,
    postprocess_cfg=None,
    precomputed_face_state=None,
    return_face_state=False,
):
    """Aggregate point logits into face labels and color the prediction mesh.

    Args:
        combined_mesh: trimesh.Trimesh
        face_idx: face index for each sampled point
        logits: per-point class logits with shape ``[S+1, N]``
        parts: list of dict with "label" key
        colormap_hex: list of hex color strings
        graph_cut_cfg: dict with keys {enabled, lambda, iterations} or None
        postprocess_cfg: dictionary returned by ``get_postprocess_cfg``
        precomputed_face_state: reusable aggregated face state
        return_face_state: attach reusable state to the returned mesh

    Returns:
        colored_mesh: trimesh.Trimesh with vertex_colors set
    """
    import time as _time

    _total_started = _time.perf_counter()
    num_faces = len(combined_mesh.faces)
    num_classes = logits.shape[0]

    instance_labels = None
    instance_to_class = None
    _reused_face_state = precomputed_face_state is not None
    _postprocess_mode = "reused"

    if _reused_face_state:
        face_logits_sum = precomputed_face_state["face_logits_sum"]
        face_point_count = precomputed_face_state["face_point_count"]
        face_labels = np.asarray(
            precomputed_face_state["face_labels"], dtype=np.int64
        ).copy()
        instance_labels = precomputed_face_state.get("instance_labels")
        instance_to_class = precomputed_face_state.get("instance_to_class")
        _aggregate_seconds = 0.0
        _postprocess_seconds = 0.0
    else:
        
        face_logits_sum = np.zeros((num_faces, num_classes), dtype=np.float64)
        face_point_count = np.zeros(num_faces, dtype=np.int64)
        fi = np.asarray(face_idx, dtype=np.int64)
        valid = (fi >= 0) & (fi < num_faces)
        np.add.at(face_logits_sum, fi[valid], logits[:, valid].T)
        np.add.at(face_point_count, fi[valid], 1)

        
        sampled_mask = face_point_count > 0
        face_labels = np.full(num_faces, -1, dtype=np.int64)
        face_labels[sampled_mask] = face_logits_sum[sampled_mask].argmax(axis=1)
        _aggregate_seconds = _time.perf_counter() - _total_started

        _postprocess_started = _time.perf_counter()
        use_fast_highpoly = bool(
            postprocess_cfg
            and postprocess_cfg.get("enabled", False)
            and postprocess_cfg.get("fast_highpoly", True)
            and num_faces >= postprocess_cfg.get("fast_highpoly_threshold", 500_000)
            and not postprocess_cfg.get("split_instances", False)
        )
        if use_fast_highpoly:
            _postprocess_mode = "fast_highpoly_kdtree"
            print(
                f"  [MeshPost] high-poly fast fill: {num_faces} faces; "
                "skip structured adjacency/voting/region merge",
                flush=True,
            )
            face_labels = _fill_unsampled_faces_kdtree_fast(
                face_labels, combined_mesh
            )
        elif postprocess_cfg and postprocess_cfg.get("enabled", False):
            _postprocess_mode = "structured"
            from postprocess import postprocess_face_labels

            result = postprocess_face_labels(
                face_labels, combined_mesh,
                stitch_shells=postprocess_cfg.get("stitch_shells", True),
                vote_iterations=postprocess_cfg.get("vote_iterations", 16),
                drop_small=postprocess_cfg.get("drop_small", True),
                rel_area_threshold=postprocess_cfg.get("rel_area_threshold", 0.001),
                merge_area_tail=postprocess_cfg.get("merge_area_tail", True),
                cumulative_threshold=postprocess_cfg.get("cumulative_threshold", 0.95),
                split=postprocess_cfg.get("split_instances", False),
                verbose=True,
            )
            face_labels = result["face_labels"]
            instance_labels = result.get("instance_labels")
            instance_to_class = result.get("instance_to_class")
        else:
            _postprocess_mode = "legacy_fill"
            face_labels = _fill_unsampled_faces(face_labels, combined_mesh)
        _postprocess_seconds = _time.perf_counter() - _postprocess_started

    
    _graph_cut_seconds = 0.0
    if graph_cut_cfg and graph_cut_cfg.get("enabled", False):
        _t0 = _time.perf_counter()
        if graph_cut_cfg.get("fast", True):
            
            
            from postprocess import graph_cut_fast
            band = graph_cut_cfg.get("band_rings", 3)
            converge = graph_cut_cfg.get("converge", True)
            adaptive_highpoly = bool(
                graph_cut_cfg.get("adaptive_highpoly", True)
                and num_faces >= graph_cut_cfg.get(
                    "highpoly_face_threshold", 500_000
                )
            )
            band_fallbacks = None
            if adaptive_highpoly:
                
                
                
                
                
                max_band_label_product = graph_cut_cfg.get(
                    "max_band_label_product", 500_000
                )
                max_band_faces = graph_cut_cfg.get("max_band_faces", 60_000)
                max_components = graph_cut_cfg.get("max_components", 128)
                band_fallbacks = graph_cut_cfg.get(
                    "band_fallbacks", (8, 4, 2, 1)
                )
                print(
                    "  [GraphCut] adaptive high-poly budget: "
                    f"band={band} fallbacks={tuple(band_fallbacks)}, "
                    f"max(band*labels)={max_band_label_product}, "
                    f"max_band={max_band_faces}, "
                    f"max_components={max_components}",
                    flush=True,
                )
            else:
                max_band_label_product = 0
                max_band_faces = 0
                max_components = 0
            print(f"  [GraphCut] fast mode (λ={graph_cut_cfg['lambda']}, "
                  f"band={band}, converge={converge})")
            face_labels = graph_cut_fast(
                face_labels, face_logits_sum, face_point_count, combined_mesh,
                _lambda=graph_cut_cfg["lambda"],
                iterations=graph_cut_cfg["iterations"],
                band_rings=band,
                converge=converge,
                smooth_mode=graph_cut_cfg.get("smooth_mode", "log"),
                n_jobs=graph_cut_cfg.get("n_jobs", None),
                max_band_label_product=max_band_label_product,
                max_band_faces=max_band_faces,
                max_components=max_components,
                max_sweeps=graph_cut_cfg.get("max_sweeps", 8),
                converge_min_change=graph_cut_cfg.get("converge_min_change", 0),
                band_fallbacks=band_fallbacks,
                verbose=True,
            )
        else:
            print(f"  [GraphCut] Running alpha-expansion (λ={graph_cut_cfg['lambda']}, iter={graph_cut_cfg['iterations']})")
            face_labels = _graph_cut_alpha_expansion(
                face_labels, face_logits_sum, face_point_count, combined_mesh,
                _lambda=graph_cut_cfg["lambda"],
                iterations=graph_cut_cfg["iterations"],
            )
        _graph_cut_seconds = _time.perf_counter() - _t0
        print(f"  [GraphCut] Done in {_graph_cut_seconds:.2f}s")

    
    _color_started = _time.perf_counter()
    colored_mesh = _apply_vertex_colors(combined_mesh, face_labels, parts, colormap_hex)
    _color_seconds = _time.perf_counter() - _color_started
    _timing = {
        "num_faces": int(num_faces),
        "num_classes": int(num_classes),
        "label_postprocess_mode": _postprocess_mode,
        "reused_face_state": bool(_reused_face_state),
        "face_aggregation_seconds": float(_aggregate_seconds),
        "label_postprocess_seconds": float(_postprocess_seconds),
        "structured_postprocess_seconds": float(
            _postprocess_seconds if _postprocess_mode == "structured" else 0.0
        ),
        "fast_fill_seconds": float(
            _postprocess_seconds
            if _postprocess_mode == "fast_highpoly_kdtree" else 0.0
        ),
        "graph_cut_seconds": float(_graph_cut_seconds),
        "vertex_color_seconds": float(_color_seconds),
        "total_seconds": float(_time.perf_counter() - _total_started),
    }
    print(
        "  [MeshPostTiming] "
        + " ".join(
            f"{key}={value:.4f}"
            for key, value in _timing.items()
            if key.endswith("_seconds")
        )
    )
    
    
    colored_mesh._face_labels = face_labels
    colored_mesh._instance_labels = instance_labels
    colored_mesh._instance_to_class = instance_to_class
    if return_face_state:
        colored_mesh._mesh_face_state = {
            "face_logits_sum": face_logits_sum,
            "face_point_count": face_point_count,
            "face_labels": np.asarray(face_labels).copy(),
            "instance_labels": instance_labels,
            "instance_to_class": instance_to_class,
        }
    return colored_mesh


def _apply_face_colors_mesh(combined_mesh, face_labels, parts, colormap_hex):
    """face_labels → unmerge-vertices flat mesh (V=3F, vertex_colors).

    Each face receives three private vertices with the same color, producing
    flat face colors in MeshLab and Blender.

    Returns: trimesh.Trimesh with .visual.vertex_colors set
    """
    import trimesh as _trimesh
    F = len(combined_mesh.faces)
    verts = np.asarray(combined_mesh.vertices, dtype=np.float32)
    faces = np.asarray(combined_mesh.faces, dtype=np.int64)
    new_verts = verts[faces.reshape(-1)]
    new_faces = np.arange(3 * F, dtype=np.int64).reshape(F, 3)

    part_colors = _get_part_colormap(parts, colormap_hex)
    palette = np.full((max(len(part_colors) + 1, 1), 3), BG_COLOR_RGB, dtype=np.uint8)
    for i, rgb in enumerate(part_colors):
        palette[i] = rgb
    labels = np.clip(face_labels, 0, len(palette) - 1).astype(np.int64)
    face_rgb = palette[labels]
    vert_rgb = np.repeat(face_rgb, 3, axis=0)
    vert_rgba = np.concatenate(
        [vert_rgb, np.full((vert_rgb.shape[0], 1), 255, dtype=np.uint8)], axis=1
    )
    flat_mesh = _trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    flat_mesh.visual = _trimesh.visual.ColorVisuals(mesh=flat_mesh, vertex_colors=vert_rgba)
    return flat_mesh


def _save_legend_image(save_dir, parts, colormap_hex, filename="legend.png"):
    """Generate a PNG color legend, or skip it when Pillow is unavailable."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[Warning] PIL not available, skipping legend image")
        return

    swatch_w, swatch_h = 30, 22
    padding = 8
    text_x_offset = swatch_w + 12
    row_height = swatch_h + padding
    font_size = 16

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    entries = []
    for i, p in enumerate(parts):
        label = p["label"] if isinstance(p, dict) else str(p)
        if "<|BG|>" in label or "<|im_end|>" in label:
            continue
        rgb = hex_to_rgb(colormap_hex[i % len(colormap_hex)])
        entries.append((rgb, f"[{i}] {label}"))
    entries.append((BG_COLOR_RGB, "[BG] background"))

    
    max_text_w = 0
    for _, text in entries:
        bbox = font.getbbox(text)
        max_text_w = max(max_text_w, bbox[2] - bbox[0])

    img_w = padding + text_x_offset + max_text_w + padding
    img_h = padding + len(entries) * row_height + padding

    img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = padding
    for rgb, text in entries:
        draw.rectangle([padding, y, padding + swatch_w, y + swatch_h], fill=rgb, outline=(0, 0, 0))
        draw.text((padding + text_x_offset, y + 2), text, fill=(0, 0, 0), font=font)
        y += row_height

    os.makedirs(save_dir, exist_ok=True)
    img.save(os.path.join(save_dir, filename))


def _int_bbox_to_float(bbox):
    """Convert integer [0, 1000] bbox coordinates back to float [-0.5, 0.5]."""
    if bbox is None:
        return None

    if all(isinstance(v, int) or (isinstance(v, float) and v == int(v)) for v in bbox):
        if all(0 <= v <= 1000 for v in bbox):
            return [v / 1000.0 - 0.5 for v in bbox]
    return bbox


def _strip_response_special_tokens(text: str) -> str:
    """Remove structural special tokens that are not semantic part names."""
    for tok in ("<|SEG|>", "<|BG|>", "<|im_end|>", "<|endoftext|>"):
        text = text.replace(tok, "")
    return text.strip()


class StopOnBgOrMaxSeg(StoppingCriteria):
    """Stop structured segmentation generation on BG, or on too many SEG queries.

    Greedy decoding can enter a local loop that repeats one part forever.  In
    joint decoding, <|BG|> semantically marks the end of the part list, and an
    excessive number of <|SEG|> tokens is almost certainly a degenerate loop.
    """

    def __init__(self, prompt_len: int, bg_token_id: int | None, seg_token_id: int | None, max_seg_tokens: int):
        self.prompt_len = int(prompt_len)
        self.bg_token_id = bg_token_id
        self.seg_token_id = seg_token_id
        self.max_seg_tokens = int(max_seg_tokens) if max_seg_tokens is not None else 0

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            return False
        gen_ids = input_ids[0, self.prompt_len:]
        if gen_ids.numel() <= 0:
            return False
        if self.bg_token_id is not None and int(gen_ids[-1].item()) == int(self.bg_token_id):
            return True
        if self.seg_token_id is not None and self.max_seg_tokens > 0:
            if int((gen_ids == int(self.seg_token_id)).sum().item()) >= self.max_seg_tokens:
                print(
                    f"[Warn] stopping generation after {self.max_seg_tokens} <|SEG|> tokens; "
                    "this usually indicates a greedy repetition loop."
                )
                return True
        return False


def ensure_bg_token_for_joint_mask(generated_ids, prompt_len, processor, model, args):
    """Inference-only structural guard for joint mask decoding.

    Joint training targets always include a trailing <|BG|> query token.
    Sampled generation can occasionally emit <|SEG|> and then stop before <|BG|>.
    Keep model/training forward strict, but synthesize the missing BG query before
    the final post-generation mask forward used only for visualization/inference.
    """
    if not getattr(args, "auto_add_bg", True):
        return generated_ids, False
    if generated_ids.ndim != 2 or generated_ids.shape[0] != 1:
        return generated_ids, False
    seg_token_id = getattr(model.config, "seg_token_id", None)
    bg_token_id = getattr(model.config, "bg_token_id", None)
    if seg_token_id is None or bg_token_id is None:
        return generated_ids, False

    seq = generated_ids[0]
    gen_seq = seq[prompt_len:]
    if not (gen_seq == seg_token_id).any():
        return generated_ids, False
    if (gen_seq == bg_token_id).any():
        return generated_ids, False

    end_token_ids = set()
    tokenizer = getattr(processor, "tokenizer", None)
    for value in [getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "pad_token_id", None)]:
        if isinstance(value, int) and value >= 0:
            end_token_ids.add(value)
        elif isinstance(value, (list, tuple)):
            end_token_ids.update(int(v) for v in value if isinstance(v, int) and v >= 0)
    if tokenizer is not None:
        for tok in ("<|im_end|>", "<|endoftext|>"):
            try:
                tok_id = tokenizer.convert_tokens_to_ids(tok)
            except Exception:
                tok_id = None
            if isinstance(tok_id, int) and tok_id >= 0:
                end_token_ids.add(tok_id)

    insert_pos = generated_ids.shape[1]
    if end_token_ids:
        for pos in range(prompt_len, generated_ids.shape[1]):
            if int(seq[pos].item()) in end_token_ids:
                insert_pos = pos
                break

    bg = torch.tensor([[bg_token_id]], dtype=generated_ids.dtype, device=generated_ids.device)
    fixed = torch.cat([generated_ids[:, :insert_pos], bg, generated_ids[:, insert_pos:]], dim=1)
    print(
        "[Warn] generated sequence has <|SEG|> but no <|BG|>; "
        "auto-inserted <|BG|> before final mask forward. "
        "Use --auto_add_bg false to disable this inference-only guard."
    )
    return fixed, True


def parse_response_parts(response: str):
    """Extract foreground part labels and bounding boxes from generated text.

    ``<|BG|>`` is the joint decoder's background query and is not a part.

    Returns:
        A list of ``{"line": str, "label": str, "bbox": list | None}`` dictionaries.
        bbox values are in float [-0.5, 0.5] range (auto-converted from integer [0, 1000] if needed).
    """
    lines = [line.strip() for line in response.split("\n") if line.strip()]
    parts = []
    for line in lines:
        line_wo_bg = line.replace("<|BG|>", "").replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        if not _strip_response_special_tokens(line_wo_bg):
            continue

        entry = {"line": line, "label": "", "bbox": None}

        
        if parse_loc_tokens_line is not None:
            parsed = parse_loc_tokens_line(line_wo_bg)
            if parsed is not None:
                bbox, label = parsed
                entry["label"] = _strip_response_special_tokens(label)
                entry["bbox"] = bbox
                if entry["label"] or entry["bbox"] is not None:
                    parts.append(entry)
                continue

        
        clean_line = _strip_response_special_tokens(line)
        if not clean_line:
            continue
        try:
            part_json = json.loads(clean_line)
            entry["label"] = _strip_response_special_tokens(str(part_json.get("label", "")))
            raw_bbox = part_json.get("bbox_aabb", None)
            entry["bbox"] = _int_bbox_to_float(raw_bbox)
        except (json.JSONDecodeError, KeyError, ValueError):
            
            clean = _strip_response_special_tokens(line)
            if clean:
                entry["label"] = clean

        if entry["label"] or entry["bbox"] is not None:
            parts.append(entry)
    return parts










def parse_bool(value):
    """Parse an explicit true/false command-line value."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def add_common_inference_args(parser):
    """Add arguments shared by all inference entry points."""
    parser.add_argument("--config", type=str, default="infer_config.yaml", help="YAML configuration file")
    parser.add_argument("--model_path", default=None, help="Model checkpoint path")
    parser.add_argument("--config_path", default=None, help="AutoConfig and AutoProcessor directory")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum samples; -1 means all")
    parser.add_argument("--seed", type=int, default=42, help="Generation and point-sampling seed")
    parser.add_argument("--sample_seed", type=int, default=42, help="Dataset-selection seed")
    parser.add_argument("--max_new_tokens", type=int, default=10240, help="Maximum generated tokens")
    parser.add_argument("--do_sample", type=parse_bool, default=False, metavar="{true,false}", help="Enable stochastic decoding")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p sampling value")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k value; non-positive disables it")
    parser.add_argument("--auto_add_bg", type=parse_bool, default=True, metavar="{true,false}", help="Append a missing background query before joint mask decoding")
    parser.add_argument("--max_generated_parts", type=int, default=64, help="Maximum generated part tokens; non-positive disables the limit")
    parser.add_argument("--save_per_part", type=parse_bool, default=False, metavar="{true,false}", help="Save one PLY per predicted part")
    parser.add_argument("--prompt_style", choices=["current", "legacy"], default="current", help="Prompt-template family")
    parser.add_argument("--sem_mode", choices=["all", "semantic", "nosem"], default="semantic", help="Run semantic prompts, non-semantic prompts, or both")
    parser.add_argument("--prompt_set", choices=["all", "coarse", "general", "fine"], default="all", help="Run all granularities or one selected granularity")
    parser.add_argument("--resume", type=parse_bool, default=True, metavar="{true,false}", help="Skip completed outputs")
    parser.add_argument("--save_mesh", type=parse_bool, default=True, metavar="{true,false}", help="Save colored prediction meshes")
    parser.add_argument("--postprocess", type=parse_bool, default=True, metavar="{true,false}", help="Enable mesh post-processing")
    parser.add_argument("--pp_drop_small", type=parse_bool, default=True, metavar="{true,false}", help="Remove and refill small connected regions")
    parser.add_argument("--pp_rel_area_threshold", type=float, default=0.001, help="Relative shell-area threshold for small regions")
    parser.add_argument("--graph_cut", type=parse_bool, default=True, metavar="{true,false}", help="Refine part boundaries with graph cut")


def apply_yaml_config(args, parser):
    """Merge YAML defaults with explicitly supplied command-line arguments."""
    explicitly_passed = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt in sys.argv:
                explicitly_passed.add(action.dest)
                break

    if getattr(args, "config", None):
        import yaml
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = os.path.join(script_dir, config_path)
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        for key, val in cfg.items():
            if key in explicitly_passed:
                continue
            if hasattr(args, key):
                setattr(args, key, val)
            else:
                setattr(args, key, val)


def load_eval_dataset(args, tokenizer, processor):
    """Build the evaluation dataframe and dataset."""
    dataset_config = DictConfig({
        "pad_mode": "no_padding",
        "truncation": "error",
        "max_length": 8192,
        "messages_key": "messages",
        "diffusion_messages_key": "diffusion_messages",
        "shuffle": False,
        "seed": args.seed,
        "utonia_apply_z_positive": bool(getattr(args, "apply_z_positive", False)),
    })
    dataset = PartNeXtPoint3DDataset(
        parquet_files=args.data_path,
        tokenizer=tokenizer,
        config=dataset_config,
        processor=processor,
        is_train=False,
    )
    df = dataset.dataframe.copy()
    if "_dataset_source" not in df.columns:
        raise RuntimeError("inference dataframe is missing required _dataset_source")
    return df, dataset


def get_graph_cut_cfg(args):
    """Build graph-cut configuration."""
    return {
        "enabled": getattr(args, "graph_cut", True),
        "lambda": 1.0,
        "iterations": 1,
        "fast": True,
        "band_rings": 3,
        "converge": True,
        "smooth_mode": "log",
        "n_jobs": None,
        "adaptive_highpoly": True,
        "band_fallbacks": (8, 4, 2, 1),
        "highpoly_face_threshold": 500_000,
        "max_band_label_product": 500_000,
        "max_band_faces": 60_000,
        "max_components": 128,
        "max_sweeps": 8,
        "converge_min_change": 0,
    }


def get_postprocess_cfg(args):
    """Build mesh post-processing configuration."""
    return {
        "enabled": getattr(args, "postprocess", True),
        "stitch_shells": True,
        "vote_iterations": 16,
        "drop_small": getattr(args, "pp_drop_small", True),
        "rel_area_threshold": getattr(args, "pp_rel_area_threshold", 0.001),
        "merge_area_tail": True,
        "cumulative_threshold": 0.95,
        "split_instances": False,
        "fast_highpoly": True,
        "fast_highpoly_threshold": 500_000,
    }


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_prompt_point_center_biased(sampled_points, part_mask, rng=None):
    """Sample a center-biased prompt point from a positive part mask.

    Args:
        sampled_points: sampled point coordinates
        part_mask: boolean or integer positive mask
        rng: optional NumPy generator for reproducibility

    Returns:
        Sampled prompt coordinate.
    """
    if rng is None:
        rng = np.random.default_rng()

    pos_indices = np.nonzero(part_mask)[0]
    num_pos = len(pos_indices)
    if num_pos == 0:
        raise ValueError("part_mask has no positive points")

    pos_points = sampled_points[pos_indices]
    centroid = pos_points.mean(axis=0)
    dists = np.linalg.norm(pos_points - centroid, axis=1)

    k = min(max(10, num_pos // 3), num_pos)
    if k >= num_pos:
        top_k_local = np.arange(num_pos)
    else:
        top_k_local = np.argpartition(dists, k - 1)[:k]
    chosen_local = top_k_local[rng.integers(0, len(top_k_local))]
    return pos_points[chosen_local]


def parse_args():
    parser = argparse.ArgumentParser(description="PartLLM full-shape segmentation")
    add_common_inference_args(parser)
    
    parser.add_argument("--num_parts", type=int, help="Requested number of parts")
    parser.add_argument("--glb_dir", type=str, default=None, help="Input mesh directory")
    parser.add_argument("--cache_dir", type=str, default=None,
        help="Optional cache created by preprocess_glb_cache.py")
    parser.add_argument(
        "--apply_z_positive",
        type=parse_bool,
        default=False,
        metavar="{true,false}",
        help=(
            "Apply legacy center shifting after Utonia coordinate normalization"
        ),
    )

    args = parser.parse_args()
    apply_yaml_config(args, parser)

    
    if not args.model_path:
        parser.error("--model_path is required")
    if not args.config_path:
        parser.error("--config_path is required")
    if not args.glb_dir:
        parser.error("--glb_dir is required")
    if args.num_parts is not None and args.num_parts <= 0:
        parser.error("--num_parts must be positive")

    return args


def matches_sem_mode(granularity, sem_mode: str) -> bool:
    if sem_mode == "all":
        return True
    is_nosem = (granularity == "nosem")
    if sem_mode == "nosem":
        return is_nosem
    return not is_nosem


def build_all_open_prompts(num_parts: int, granularity: str = None):
    """Build all templates matching one requested granularity."""
    prompts = []
    is_nosem = (granularity == "nosem")
    templates = NOSEM_OPEN_TEMPLATES if is_nosem else SEMANTIC_OPEN_TEMPLATES

    if granularity and granularity in templates:
        prompts.append((f"{granularity}_semantic", templates[granularity]))
        prompts.append((f"{granularity}_numeric", templates["numeric"].format(n=num_parts)))
    elif granularity == "medium":
        prompts.append(("medium_numeric", templates["numeric"].format(n=num_parts)))
        prompts.append(("medium_generic", templates["generic"]))
    else:
        
        for gran in ("coarse", "fine"):
            for tpl_set, prefix in [(SEMANTIC_OPEN_TEMPLATES, "sem"), (NOSEM_OPEN_TEMPLATES, "nosem")]:
                prompts.append((f"{gran}_{prefix}", tpl_set[gran]))
                prompts.append((f"{gran}_{prefix}_numeric", tpl_set["numeric"].format(n=num_parts)))
    for i, t in enumerate(LEGACY_SEMANTIC_TEMPLATES):
        prompts.append((f"legacy_sem_{i}", t))
    for i, t in enumerate(LEGACY_NOSEM_TEMPLATES):
        prompts.append((f"legacy_nosem_{i}", t))
    prompts.append(("generic", templates["generic"]))
    return prompts


def build_multi_granularity_prompts(num_parts: int):
    """Build representative prompts across the supported granularities."""
    prompts = []

    for gran in ("coarse", "fine"):
        prompts.append((gran, SEMANTIC_OPEN_TEMPLATES[gran]))
    
    nosem_bucket = _classify_nosem_bucket(num_parts)
    prompts.append(("nosem", NOSEM_OPEN_TEMPLATES[nosem_bucket]))

    prompts.append(("medium", SEMANTIC_OPEN_TEMPLATES["numeric"].format(n=num_parts)))

    prompts.append(("generic", SEMANTIC_OPEN_TEMPLATES["generic"]))
    return prompts


LEGACY_OPEN_PROMPTS = {
    "coarse": "Please roughly segment this 3D object in <point_cloud>.",
    "medium": "Please segment this 3D object into about {n} parts in <point_cloud>.",
    "fine": "Please segment all the detailed parts of this 3D object in <point_cloud>.",
}

LEGACY_GROUNDING_TEMPLATE = "Please segment all the {parts} in <point_cloud>."

def build_default_open_prompts(sem_mode: str, prompt_set: str = "all"):
    prompts = []
    selected = ("coarse", "general", "fine") if prompt_set == "all" else (prompt_set,)
    template_keys = {"coarse": "coarse", "general": "generic", "fine": "fine"}
    if sem_mode in {"all", "semantic"}:
        prompts.extend(
            (name, SEMANTIC_OPEN_TEMPLATES[template_keys[name]]) for name in selected
        )
    if sem_mode in {"all", "nosem"}:
        prompts.extend(
            (f"nosem_{name}", NOSEM_OPEN_TEMPLATES[template_keys[name]])
            for name in selected
        )
    return prompts


def build_glb_open_prompts(sem_mode: str, prompt_set: str = "all"):
    return build_default_open_prompts(sem_mode, prompt_set)


CURRENT_OPEN_PROMPTS = build_default_open_prompts("all")


def extract_unique_part_names(target_part_names):
    """Deduplicate part names while preserving first occurrence order."""
    if hasattr(target_part_names, "tolist"):
        target_part_names = target_part_names.tolist()
    return list(dict.fromkeys(target_part_names or []))


def build_legacy_open_prompt(granularity: str, num_parts: int) -> str:
    """Build a legacy full-shape prompt."""
    if granularity not in LEGACY_OPEN_PROMPTS:
        raise ValueError(f"Unsupported legacy open granularity: {granularity}")
    prompt = LEGACY_OPEN_PROMPTS[granularity]
    return prompt.format(n=num_parts) if "{n}" in prompt else prompt


def build_grounding_prompt(target_part_names, prompt_style: str = "current") -> str:
    """Build a text-guided segmentation prompt."""
    if prompt_style not in {"current", "legacy"}:
        raise ValueError(f"Unsupported grounding prompt style: {prompt_style}")
    unique_names = extract_unique_part_names(target_part_names)
    return LEGACY_GROUNDING_TEMPLATE.format(parts="; ".join(unique_names))


def _count_row_parts(row) -> int:
    return len(extract_unique_part_names(row.get("target_part_names", [])))


def _pick_open_row(open_rows, granularity: str, fallback_row):
    if len(open_rows) > 0:
        matched = open_rows[open_rows["granularity"] == granularity]
        if len(matched) > 0:
            return matched.iloc[0].copy()
        return open_rows.iloc[0].copy()
    return fallback_row.copy()


def is_completed_output(save_dir: str) -> bool:
    metadata_path = os.path.join(save_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(metadata, dict)
        and metadata.get("response") is not None
        and "num_parts_predicted" in metadata
        and os.path.exists(os.path.join(save_dir, "logits.npy"))
        and os.path.exists(os.path.join(save_dir, "face_idx.npy"))
    )


def build_open_prompt_specs(args, open_rows, fallback_row):
    """Return full-shape jobs as folder, row, and prompt tuples."""
    prompt_style = "current" if args.prompt_style == "parquet" else args.prompt_style

    if prompt_style == "legacy":
        if args.num_parts:
            raise ValueError("legacy prompt_style does not support --num_parts overrides")
        if len(open_rows) == 0:
            return []
        specs = []
        for granularity in ("coarse", "medium", "fine"):
            if not matches_sem_mode(granularity, args.sem_mode):
                continue
            matched = open_rows[open_rows["granularity"] == granularity]
            if len(matched) == 0:
                continue
            row = matched.iloc[0].copy()
            specs.append((granularity, row, build_legacy_open_prompt(granularity, _count_row_parts(row))))
        return specs

    if args.num_parts:
        specs = []
        row = _pick_open_row(open_rows, "medium", fallback_row)
        if args.sem_mode in {"all", "semantic"}:
            specs.append(("numeric", row, SEMANTIC_OPEN_TEMPLATES["numeric"].format(n=args.num_parts)))
        if args.sem_mode in {"all", "nosem"}:
            specs.append(("nosem_numeric", row, NOSEM_OPEN_TEMPLATES["numeric"].format(n=args.num_parts)))
        return specs

    specs = []
    for folder_name, prompt in build_default_open_prompts(args.sem_mode, args.prompt_set):
        granularity = "nosem" if folder_name.startswith("nosem") else folder_name
        row = _pick_open_row(open_rows, granularity, fallback_row)
        specs.append((folder_name, row, prompt))
    return specs






def load_model(args):
    """Load the model, processor, and tokenizer."""
    print("Loading model...")

    
    import os
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    hf_config = AutoConfig.from_pretrained(args.config_path, trust_remote_code=True, local_files_only=True)
    auto_class = get_hf_auto_model_class(hf_config=hf_config)

    model = auto_class.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        dtype=torch.bfloat16,
        config=hf_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        local_files_only=True,  
    )


    Qwen3VLModel.forward = qwen3_vl_base_forward
    Qwen3VLForConditionalGeneration.forward = forward_with_normal_backend

    processor = AutoProcessor.from_pretrained(args.config_path, trust_remote_code=True, local_files_only=True)
    tokenizer = hf_tokenizer(args.config_path, trust_remote_code=True, local_files_only=True)

    
    encoder_type = getattr(hf_config, "point_cloud_encoder_type", "utonia")
    if encoder_type != "utonia":
        raise ValueError("PartLLM supports only the Utonia point-cloud encoder")
    utonia_ckpt = getattr(hf_config, "utonia_ckpt_path", None)
    model.model.load_utonia_weights(ckpt_path=utonia_ckpt)

    from safetensors.torch import load_file
    import json as _json
    utonia_state = {}
    index_path = os.path.join(args.model_path, "model.safetensors.index.json")
    single_path = os.path.join(args.model_path, "model.safetensors")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = _json.load(f)
        utonia_shards = {
            shard for key, shard in index["weight_map"].items() if "utonia_model" in key
        }
        for shard in utonia_shards:
            shard_state = load_file(os.path.join(args.model_path, shard))
            for key, value in shard_state.items():
                if "utonia_model" in key:
                    utonia_state[key.replace("model.utonia_model.", "")] = value
    elif os.path.exists(single_path):
        shard_state = load_file(single_path)
        for key, value in shard_state.items():
            if "utonia_model" in key:
                utonia_state[key.replace("model.utonia_model.", "")] = value
    else:
        print("[Warning] No safetensors checkpoint found, using pretrained Utonia weights")
    if utonia_state:
        missing, unexpected = model.model.utonia_model.load_state_dict(utonia_state, strict=False)
        print(f"[Info] Utonia fine-tuned weights loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    
    model.to(torch.bfloat16).to("cuda")
    model.eval()

    
    if model.model.utonia_model is not None:
        model.model.utonia_model.to(torch.float32)
        print("[Info] Utonia model kept in float32 (spconv requirement)")

    utonia_num_tokens = getattr(hf_config, "utonia_num_tokens", 5000)
    processor.num_pc_tokens = utonia_num_tokens
    print(f"[Info] processor.num_pc_tokens = {utonia_num_tokens}")

    return model, processor, tokenizer, hf_config, encoder_type






def prepare_geometry(row, dataset, model, encoder_type):
    """Preprocess geometry once per model identifier.

    Returns:
        geo_cache: dict containing:
            - diffusion_inputs: optional model auxiliary tensors (on device)
            - utonia_point_dict: Utonia encoder inputs (on device) or None
            - mesh_vis_info: dict with face_idx and combined_mesh
            - point_cloud_content: cached point-cloud content items
    """
    dataset_source = str(row.get("_dataset_source", "")).strip()
    if dataset_source in {"", "unknown"}:
        dataset_df = getattr(dataset, "dataframe", None)
        inferred_sources = set()
        if dataset_df is not None and "_dataset_source" in dataset_df.columns:
            inferred_sources = {
                str(value).strip()
                for value in dataset_df["_dataset_source"].dropna().tolist()
                if str(value).strip()
            }
        if len(inferred_sources) == 1 and "unknown" not in inferred_sources:
            dataset_source = inferred_sources.pop()
            row = row.copy()
            row["_dataset_source"] = dataset_source
        else:
            raise ValueError(
                "parquet inference row has no recognized _dataset_source; "
                "use load_eval_dataset() and ensure the parquet path identifies "
                "its dataset source"
            )

    _messages, diffusion_model_inputs, cached_mesh_info = dataset._build_messages(row)

    
    diffusion_model_inputs.pop("classification_labels", None)

    device_diffusion = {}
    for k, v in diffusion_model_inputs.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.is_floating_point():
            v = v.to(model.dtype)
        device_diffusion[k] = v.to(model.device)


    utonia_point_dict = None
    if encoder_type == "utonia":
        from transformers.models.qwen3_vl import utonia as _utonia
        for key, val in cached_mesh_info.items():
            if key.endswith("__utonia_point"):
                utonia_point_dict = _utonia.data.collate_fn([val])
                for k2 in utonia_point_dict:
                    if isinstance(utonia_point_dict[k2], torch.Tensor):
                        if utonia_point_dict[k2].is_floating_point():
                            utonia_point_dict[k2] = utonia_point_dict[k2].to(
                                device=model.device, dtype=torch.float32
                            )
                        else:
                            utonia_point_dict[k2] = utonia_point_dict[k2].to(device=model.device)
                break

    mesh_vis_info = {
        "face_idx": cached_mesh_info.get("__face_idx"),
        "combined_mesh": cached_mesh_info.get("__combined_mesh"),
    }

    
    point_cloud_content = _messages[0]["content"] if _messages else None

    return {
        "diffusion_inputs": device_diffusion,
        "utonia_point_dict": utonia_point_dict,
        "mesh_vis_info": mesh_vis_info,
        "point_cloud_content": point_cloud_content,



        "sampled_points": cached_mesh_info.get("__sampled_points"),
        "sampled_colors": cached_mesh_info.get("__sampled_colors"),
        "sampled_normals": cached_mesh_info.get("__sampled_normals"),
        "utonia_transform": getattr(dataset, "utonia_transform", None),
    }


def prepare_prompt(row, geo_cache, dataset, processor, model, override_prompt):
    """Build prompt-specific model inputs from cached geometry.

    Returns:
        inputs, user_prompt_used, gt_response
    """
    user_prompt_used = override_prompt

    
    messages = row["messages"]
    gt_response = None
    if len(messages) >= 2:
        gt_msg = messages[1]
        gt_content = gt_msg.get("content", "") if isinstance(gt_msg, dict) else ""
        if isinstance(gt_content, list):
            gt_response = " ".join(
                item.get("text", "") for item in gt_content if isinstance(item, dict) and item.get("type") == "text"
            ).strip()
        else:
            gt_response = str(gt_content)

    
    if user_prompt_used is None:
        user_msg = messages[0] if messages else {}
        content = user_msg.get("content", "")
        if isinstance(content, list):
            user_prompt_used = " ".join(
                item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
            ).strip()
        else:
            user_prompt_used = str(content)

    
    user_text = override_prompt if override_prompt else (
        messages[0].get("content", "") if messages else ""
    )

    
    pc_content = geo_cache["point_cloud_content"]
    pc_item = None
    if pc_content:
        for item in pc_content:
            if isinstance(item, dict) and item.get("type") == "point_cloud":
                pc_item = item
                break

    if isinstance(user_text, str) and pc_item:
        
        parts = user_text.split("<point_cloud>")
        content_list = []
        for i, segment in enumerate(parts):
            if segment:
                content_list.append({"type": "text", "text": segment})
            if i < len(parts) - 1:
                content_list.append(pc_item)
        if not content_list:
            content_list = [pc_item]
        prompt_messages = [{"role": "user", "content": content_list}]
    elif isinstance(user_text, list):
        prompt_messages = [{"role": "user", "content": user_text}]
    else:
        prompt_messages = [messages[0]]

    text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    point_cloud_item = next(
        item for item in prompt_messages[0]["content"] if isinstance(item, dict) and item.get("type") == "point_cloud"
    )
    inputs = processor(
        text=text, point_clouds=point_cloud_item["point_cloud"], return_tensors="pt"
    ).to(model.device)

    
    for k, v in geo_cache["diffusion_inputs"].items():
        inputs[k] = v

    return inputs, user_prompt_used, gt_response


def prepare_inputs(row, dataset, processor, model, encoder_type, override_prompt):
    """Prepare model inputs for one sample."""
    geo_cache = prepare_geometry(row, dataset, model, encoder_type)
    inputs, user_prompt_used, gt_response = prepare_prompt(
        row, geo_cache, dataset, processor, model, override_prompt
    )
    return (
        inputs, geo_cache["utonia_point_dict"],
        user_prompt_used, gt_response, geo_cache["mesh_vis_info"],
    )



@torch.no_grad()
def run_inference(model, inputs, utonia_point_dict, processor, args):
    """Generate text and jointly decode all part and background masks."""
    
    
    if utonia_point_dict is not None:
        model.model._utonia_point_dict_cache = utonia_point_dict
    prompt_len = inputs["input_ids"].shape[1]
    bg_token_id = getattr(model.config, "bg_token_id", None)
    seg_token_id = getattr(model.config, "seg_token_id", None)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": bool(args.do_sample),
        "stopping_criteria": StoppingCriteriaList([
            StopOnBgOrMaxSeg(prompt_len, bg_token_id, seg_token_id, args.max_generated_parts)
        ]),
    }
    if bg_token_id is not None:
        eos_ids = set()
        tokenizer = getattr(processor, "tokenizer", None)
        for value in [
            getattr(tokenizer, "eos_token_id", None),
            getattr(model.generation_config, "eos_token_id", None),
        ]:
            if isinstance(value, int) and value >= 0:
                eos_ids.add(value)
            elif isinstance(value, (list, tuple)):
                eos_ids.update(int(v) for v in value if isinstance(v, int) and v >= 0)
        eos_ids.add(int(bg_token_id))
        generation_kwargs["eos_token_id"] = sorted(eos_ids)
    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
        if args.top_k is not None and args.top_k > 0:
            generation_kwargs["top_k"] = args.top_k

    generated_ids = model.generate(
        **inputs,
        **generation_kwargs,
    )

    
    response = processor.decode(generated_ids[0][prompt_len:], skip_special_tokens=False)

    
    
    
    generated_ids_for_mask, added_bg = ensure_bg_token_for_joint_mask(
        generated_ids, prompt_len, processor, model, args
    )
    if added_bg:
        response = processor.decode(generated_ids_for_mask[0][prompt_len:], skip_special_tokens=False)

    forward_inputs = {
        "input_ids": generated_ids_for_mask,
        "attention_mask": torch.ones_like(generated_ids_for_mask),
        "point_clouds": inputs["point_clouds"],
    }
    if utonia_point_dict is not None:
        forward_inputs["utonia_point_dict"] = utonia_point_dict

    pca_capture = _PCAFeatureCapture.attach(model) if _pca_enabled() else None
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            forward_outputs = model(**forward_inputs)
    finally:
        if pca_capture is not None:
            pca_capture.detach()
    mask_logits = forward_outputs.mask_decoder_outputs
    del forward_outputs, forward_inputs, generated_ids, generated_ids_for_mask

    return response, mask_logits, pca_capture






def process_and_save(
    response, mask_logits, inputs,
    row, save_dir, args, idx,
    user_prompt_used, gt_response,
    mesh_vis_info=None,
    pca_capture=None,
    include_metrics=True,
):
    """Parse one prediction and save its visualizations.

    Returns:
        metrics: dict with num_parts_predicted
    """
    point_cloud_xyz = inputs["point_clouds"][0][:, :3].cpu().float().numpy()

    if pca_capture is not None:
        try:
            pca_capture.dump(save_dir, point_cloud_xyz)
        except Exception as _e:
            print(f"[PCA] dump failed: {_e}")

    
    parts = parse_response_parts(response)
    num_parts_pred = len(parts)
    print(f"\n[Sample {idx}] {num_parts_pred} parts generated")
    for i, p in enumerate(parts):
        print(f"  [{i}] {p['label']}" + (f"  bbox={p['bbox']}" if p["bbox"] else ""))

    
    gt_parts = []
    if gt_response is not None:
        gt_parts = parse_response_parts(gt_response)
        print(f"  GT: {len(gt_parts)} parts")
        for i, p in enumerate(gt_parts):
            print(f"    [{i}] {p['label']}" + (f"  bbox={p['bbox']}" if p["bbox"] else ""))

    
    _save_info_txt(save_dir, user_prompt_used, response, gt_response, parts, gt_parts)

    
    _save_legend_image(save_dir, parts, MODE20_COLORMAP_HEX, "pred_legend.png")
    if gt_parts:
        _save_legend_image(save_dir, gt_parts, MODE20_COLORMAP_HEX, "gt_legend.png")

    metrics = {"num_parts_predicted": num_parts_pred}

    if mask_logits is None:
        print("[Warning] mask_decoder_outputs is None, skipping mask visualization")
        _save_metadata(
            save_dir,
            row,
            user_prompt_used,
            response,
            num_parts_pred,
            metrics if include_metrics else None,
        )
        return metrics


    
    logits = mask_logits.cpu().float()
    num_classes = logits.shape[1]
    num_points = logits.shape[2]

    print(f"[Info] mask decoder: {num_classes} classes (incl BG), {num_points} points")

    
    pred_labels = logits[0].argmax(dim=0).numpy()  

    
    np.save(os.path.join(save_dir, "logits.npy"), logits[0].numpy())
    if mesh_vis_info is not None and mesh_vis_info.get("face_idx") is not None:
        np.save(os.path.join(save_dir, "face_idx.npy"), np.array(mesh_vis_info["face_idx"]))

    
    save_mesh = (
        args.save_mesh
        and mesh_vis_info is not None
        and mesh_vis_info.get("combined_mesh") is not None
    )
    graph_cut_cfg = get_graph_cut_cfg(args) if save_mesh else {"enabled": False}

    if save_mesh:
        face_idx = mesh_vis_info["face_idx"]
        combined_mesh = mesh_vis_info["combined_mesh"]
        postprocess_cfg = get_postprocess_cfg(args)

        pred_colored = _color_mesh_by_logits(
            combined_mesh, face_idx, logits[0].numpy(), parts, MODE20_COLORMAP_HEX,
            graph_cut_cfg=graph_cut_cfg,
            postprocess_cfg=postprocess_cfg,
        )
        pred_mesh_path = os.path.join(save_dir, "pred_mesh.ply")
        pred_colored.export(pred_mesh_path)
        print(f"  Saved pred_mesh.ply ({len(combined_mesh.faces)} faces)")

        _face_labels = getattr(pred_colored, "_face_labels", None)
        if _face_labels is not None:
            np.save(os.path.join(save_dir, "face_labels.npy"), np.asarray(_face_labels))

        _inst = getattr(pred_colored, "_instance_labels", None)
        _inst2cls = getattr(pred_colored, "_instance_to_class", None)
        if _inst is not None and not graph_cut_cfg["enabled"]:
            _inst = np.asarray(_inst)
            np.save(os.path.join(save_dir, "instance_labels.npy"), _inst)
            if _inst2cls is not None:
                np.save(os.path.join(save_dir, "instance_to_class.npy"), np.asarray(_inst2cls))
            
            inst_parts = [{"label": f"inst_{i}"} for i in range(int(_inst.max()) + 1)]
            inst_mesh = _apply_face_colors_mesh(
                combined_mesh, _inst, inst_parts, MODE20_COLORMAP_HEX
            )
            inst_mesh.export(os.path.join(save_dir, "pred_mesh_instance.ply"))
            print(f"  Saved pred_mesh_instance.ply ({int(_inst.max()) + 1} instances)")

    stale_names = [
        "pred.ply", "pred_mesh_gc.ply", "face_labels_gc.npy",
        "mesh_postprocess_timing.json",
    ]
    if graph_cut_cfg["enabled"] or not save_mesh:
        stale_names.extend((
            "instance_labels.npy", "instance_to_class.npy", "pred_mesh_instance.ply",
        ))
    if not save_mesh:
        stale_names.extend(("pred_mesh.ply", "face_labels.npy"))
    for stale_name in stale_names:
        stale_path = os.path.join(save_dir, stale_name)
        if os.path.isfile(stale_path):
            os.remove(stale_path)

    
    if args.save_per_part:
        for part_idx, part_info in enumerate(parts):
            label = part_info["label"] or f"part_{part_idx}"
            bbox_wireframe = None
            if part_info["bbox"] is not None:
                color_rgb = hex_to_rgb(MODE20_COLORMAP_HEX[part_idx % len(MODE20_COLORMAP_HEX)])
                bbox_wireframe = bbox_aabb_to_wireframe_points(part_info["bbox"], color=color_rgb)
            save_per_part_ply(point_cloud_xyz, pred_labels, part_idx, label, save_dir, bbox=bbox_wireframe)

    
    _save_metadata(
        save_dir,
        row,
        user_prompt_used,
        response,
        num_parts_pred,
        metrics if include_metrics else None,
    )

    return metrics


def _save_metadata(save_dir, row, user_prompt_used, response, num_parts_pred, metrics=None):
    """Save sample metadata as JSON."""
    metadata = {
        "model_id": row.get("model_id", ""),
        "prompt_mode": row.get("prompt_mode", ""),
        "granularity": row.get("granularity", ""),
        "user_prompt": user_prompt_used,
        "num_parts_predicted": num_parts_pred,
        "response": response,
        "mesh_space": "normalized_model_space",
    }
    if metrics is not None:
        metadata["metrics"] = metrics
    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)






GLB_PROMPTS = build_glb_open_prompts("all")


def _load_mesh_file(mesh_path):
    """Load a supported mesh as a ``Trimesh`` or ``Scene`` object."""
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext != ".fbx":
        return trimesh.load(mesh_path)

    
    try:
        import bpy
        import tempfile

        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=mesh_path)

        
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            tmp_path = tmp.name
        bpy.ops.export_scene.gltf(filepath=tmp_path, export_format="GLB")
        loaded = trimesh.load(tmp_path)
        os.unlink(tmp_path)
        print(f"[Info] FBX loaded via bpy: {mesh_path}")
        return loaded
    except ImportError:
        pass
    except Exception as e:
        print(f"[Warning] bpy FBX load failed ({e}), trying pyassimp...")

    
    try:
        import pyassimp

        with pyassimp.load(mesh_path) as scene:
            mesh_list = []
            for mesh in scene.meshes:
                verts = np.array(mesh.vertices, dtype=np.float32)
                faces = np.array(mesh.faces, dtype=np.int64)
                if faces.shape[0] == 0:
                    continue
                tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                
                if hasattr(mesh, "colors") and len(mesh.colors) > 0:
                    colors = np.clip(np.array(mesh.colors[0])[:, :3] * 255, 0, 255).astype(np.uint8)
                    if len(colors) == len(verts):
                        tm.visual.vertex_colors = colors
                mesh_list.append(tm)
        if not mesh_list:
            raise ValueError("pyassimp returned no meshes")
        result = trimesh.util.concatenate(mesh_list) if len(mesh_list) > 1 else mesh_list[0]
        print(f"[Info] FBX loaded via pyassimp: {mesh_path}")
        return result
    except ImportError:
        pass
    except Exception as e:
        print(f"[Warning] pyassimp FBX load failed: {e}")

    raise RuntimeError(
        f"Cannot load FBX: {mesh_path}\n"
        "Install bpy (pip install bpy) or pyassimp (pip install pyassimp) for FBX support."
    )


def prepare_geometry_from_cache(
    model_id, cache_dir, model, hf_config, encoder_type,
    apply_z_positive=False,
):
    """Load geometry cached by ``preprocess_glb_cache.py``.

    The cache skips mesh loading,
      - merge_vertices + normalize_meshes_diag
      - trimesh.sample.sample_surface
      - face-normal calculation.

    Utonia collation and point-surface tensor construction still run normally.
    """
    import os as _os_c
    sub_dir = _os_c.path.join(cache_dir, model_id)
    geom_path = _os_c.path.join(sub_dir, "geom.npz")
    ply_path = _os_c.path.join(sub_dir, "combined_mesh.ply")
    if not _os_c.path.exists(geom_path) or not _os_c.path.exists(ply_path):
        raise FileNotFoundError(
            f"cache missing for {model_id}: {sub_dir}; run preprocess_glb_cache.py first"
        )

    data = np.load(geom_path)

    sampled_points = data["sampled_points"].astype(np.float32)
    face_idx = data["face_idx"].astype(np.int64)
    sampled_normals = data["sampled_normals"].astype(np.float32)
    sampled_rgb = data["sampled_rgb"].astype(np.uint8)
    pc_size = int(data["pc_size"])
    combined_mesh = trimesh.load(ply_path, process=False)
    if not isinstance(combined_mesh, trimesh.Trimesh):
        
        if isinstance(combined_mesh, trimesh.Scene):
            meshes_iter = list(combined_mesh.geometry.values())
            combined_mesh = (
                trimesh.util.concatenate(meshes_iter) if len(meshes_iter) > 1 else meshes_iter[0]
            )
        else:
            raise ValueError(f"unexpected mesh type from cache: {type(combined_mesh)}")



    sampled_rgb_f = sampled_rgb.astype(np.float32) / 255.0
    sharpedge_label = np.zeros((pc_size, 1), dtype=np.float32)
    obj_surface = torch.FloatTensor(
        np.concatenate([sampled_points, sampled_rgb_f, sharpedge_label], axis=-1)
    ).unsqueeze(0)

    device_diffusion = {}

    
    from transformers.models.qwen3_vl import utonia as _utonia
    utonia_scale = getattr(hf_config, "utonia_scale", 5.0)
    utonia_transform = _utonia.transform.default(
        utonia_scale,
        normalize_coord=True,
        apply_z_positive=apply_z_positive,
    )
    utonia_pc = dict(coord=sampled_points.copy(), color=sampled_rgb, normal=sampled_normals)
    utonia_point_dict = _utonia.data.collate_fn([utonia_transform(utonia_pc)])
    for key, value in utonia_point_dict.items():
        if isinstance(value, torch.Tensor):
            dtype = torch.float32 if value.is_floating_point() else value.dtype
            utonia_point_dict[key] = value.to(device=model.device, dtype=dtype)

    pc_item = {"type": "point_cloud", "point_cloud": obj_surface.to(torch.bfloat16)}
    return {
        "diffusion_inputs": device_diffusion,
        "utonia_point_dict": utonia_point_dict,
        "mesh_vis_info": {"face_idx": face_idx, "combined_mesh": combined_mesh},
        "point_cloud_content": [pc_item],
    }


def prepare_geometry_from_glb(
    mesh_path, model, hf_config, encoder_type,
    apply_z_positive=False,
    clean_mesh=False,
):
    """Build geometry inputs directly from a supported mesh file.

    Returns:
        Geometry cache compatible with ``prepare_geometry``.
    """
    
    loaded = _load_mesh_file(mesh_path)
    if isinstance(loaded, trimesh.Scene):
        mesh_list = scene2meshes(loaded)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh_list = [loaded]
    else:
        raise ValueError(f"Unsupported mesh type from {mesh_path}: {type(loaded)}")
    if not mesh_list:
        raise ValueError(f"No meshes found in {mesh_path}")
    model_id = os.path.splitext(os.path.basename(mesh_path))[0]

    source_vertices = np.vstack([
        np.asarray(mesh.vertices, dtype=np.float32) for mesh in mesh_list
    ])
    source_bbox_min = source_vertices.min(axis=0)
    source_bbox_max = source_vertices.max(axis=0)
    source_diag = float(np.linalg.norm(source_bbox_max - source_bbox_min))
    source_scale = 1.0 / source_diag if source_diag > 0 else 1.0
    source_shift = -((source_bbox_min + source_bbox_max) * source_scale / 2.0)
    mesh_list = normalize_meshes_diag(mesh_list, norm_diag_len=1.0)




    for i, m in enumerate(mesh_list):
        if m.visual.kind == "texture":
            try:
                mesh_list[i].visual = m.visual.to_color()
            except (IndexError, ValueError):
                nv = len(mesh_list[i].vertices)
                mesh_list[i].visual = trimesh.visual.ColorVisuals(
                    vertex_colors=np.full((nv, 4), [102, 102, 102, 255], dtype=np.uint8)
                )
        vc = mesh_list[i].visual.vertex_colors
        nv = len(mesh_list[i].vertices)
        if vc.ndim == 1 or (vc.ndim == 2 and len(vc) != nv):
            color = vc if vc.ndim == 1 else vc[0]
            mesh_list[i].visual.vertex_colors = np.tile(color, (nv, 1))

    combined_mesh = trimesh.util.concatenate(mesh_list) if len(mesh_list) > 1 else mesh_list[0]

    
    
    if clean_mesh:
        _f_before, _v_before = len(combined_mesh.faces), len(combined_mesh.vertices)
        combined_mesh.merge_vertices()
        combined_mesh.process(True)
        print(f"  [CleanMesh] faces {_f_before}→{len(combined_mesh.faces)}  "
              f"verts {_v_before}→{len(combined_mesh.vertices)}")

    
    pc_cfg = getattr(hf_config, "point_cloud_config", None)
    pc_size = 81920
    if pc_cfg is not None:
        pc_size = pc_cfg.get("pc_size", 81920) if isinstance(pc_cfg, dict) else getattr(pc_cfg, "pc_size", 81920)
    pc_size = getattr(hf_config, "pc_size", pc_size)

    sampled_points, face_idx, sampled_colors = trimesh.sample.sample_surface(
        combined_mesh, pc_size, seed=None, sample_color=True
    )


    _normal = combined_mesh.face_normals[face_idx]
    _normal = _normal / (np.linalg.norm(_normal, axis=-1, keepdims=True) + 1e-8)
    _normal = np.array(_normal, dtype=np.float32)

    sampled_points = np.asarray(sampled_points, dtype=np.float32)


    sampled_rgb = sampled_colors[:, :3].astype(np.float32) / 255.0
    sharpedge_label = np.zeros((pc_size, 1), dtype=np.float32)
    obj_surface = torch.FloatTensor(
        np.concatenate([sampled_points, sampled_rgb, sharpedge_label], axis=-1)
    ).unsqueeze(0)

    device_diffusion = {}


    from transformers.models.qwen3_vl import utonia as _utonia
    utonia_scale = getattr(hf_config, "utonia_scale", 5.0)
    utonia_transform = _utonia.transform.default(
        utonia_scale,
        normalize_coord=True,
        apply_z_positive=apply_z_positive,
    )
    utonia_pc = {
        "coord": np.asarray(sampled_points, dtype=np.float32),
        "color": sampled_colors[:, :3],
        "normal": _normal,
    }
    utonia_point_dict = _utonia.data.collate_fn([utonia_transform(utonia_pc)])
    for key, value in utonia_point_dict.items():
        if isinstance(value, torch.Tensor):
            dtype = torch.float32 if value.is_floating_point() else value.dtype
            utonia_point_dict[key] = value.to(device=model.device, dtype=dtype)

    
    pc_item = {"type": "point_cloud", "point_cloud": obj_surface.to(torch.bfloat16)}

    return {
        "diffusion_inputs": device_diffusion,
        "utonia_point_dict": utonia_point_dict,
        "mesh_vis_info": {"face_idx": face_idx, "combined_mesh": combined_mesh},
        "point_cloud_content": [pc_item],
        "sampled_points": sampled_points,
        "sampled_colors": sampled_colors[:, :3].astype(np.uint8),
        "sampled_normals": _normal,
        "utonia_transform": utonia_transform,
        "source_to_model": {
            "scale": source_scale,
            "shift": source_shift,
        },
    }


def prepare_prompt_from_template(prompt_text, geo_cache, processor, model):
    """Build model inputs from a prompt template and cached geometry."""
    if "<point_cloud>" not in prompt_text:
        raise ValueError("prompt_text must contain '<point_cloud>'")

    pc_item = None
    for item in geo_cache["point_cloud_content"]:
        if isinstance(item, dict) and item.get("type") == "point_cloud":
            pc_item = item
            break

    parts = prompt_text.split("<point_cloud>")
    content_list = []
    for i, segment in enumerate(parts):
        if segment:
            content_list.append({"type": "text", "text": segment})
        if i < len(parts) - 1:
            content_list.append(pc_item)
    if not content_list:
        content_list = [pc_item]

    prompt_messages = [{"role": "user", "content": content_list}]
    text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    point_cloud_item = next(
        item for item in prompt_messages[0]["content"] if isinstance(item, dict) and item.get("type") == "point_cloud"
    )
    inputs = processor(
        text=text, point_clouds=point_cloud_item["point_cloud"], return_tensors="pt"
    ).to(model.device)

    for k, v in geo_cache["diffusion_inputs"].items():
        inputs[k] = v

    return inputs, prompt_text


def main_glb_dir(args):
    """Run full-shape inference directly on a mesh folder."""
    os.makedirs(args.output_dir, exist_ok=True)

    
    model, processor, tokenizer, hf_config, encoder_type = load_model(args)

    
    SUPPORTED_EXT = (".glb", ".gltf", ".ply", ".obj", ".stl", ".off", ".fbx")
    use_cache = bool(getattr(args, "cache_dir", None))
    if use_cache:
        
        if not os.path.isdir(args.cache_dir):
            print(f"[Error] --cache_dir does not exist: {args.cache_dir}")
            return
        cache_ids = sorted([
            d for d in os.listdir(args.cache_dir)
            if os.path.isdir(os.path.join(args.cache_dir, d))
            and os.path.exists(os.path.join(args.cache_dir, d, "geom.npz"))
        ])
        if not cache_ids:
            print(f"[Error] No cache entries (geom.npz) under {args.cache_dir}")
            return
        
        mesh_files = [f"{mid}.cache" for mid in cache_ids]
        print(f"[Info] cache mode: {len(mesh_files)} cached meshes from {args.cache_dir}")
    else:
        mesh_files = sorted([
            f for f in os.listdir(args.glb_dir)
            if os.path.splitext(f.lower())[1] in SUPPORTED_EXT
        ])
        if not mesh_files:
            print(f"[Error] No mesh files found in {args.glb_dir}")
            return

    if args.max_samples > 0 and args.max_samples < len(mesh_files):
        random.seed(args.sample_seed)
        random.shuffle(mesh_files)
        mesh_files = mesh_files[:args.max_samples]
        mesh_files.sort()

    if args.prompt_style == "legacy":
        if args.num_parts is None:
            raise ValueError("legacy prompts require --num_parts")
        glb_prompts = [
            ("coarse", build_legacy_open_prompt("coarse", args.num_parts)),
            ("medium", build_legacy_open_prompt("medium", args.num_parts)),
            ("fine", build_legacy_open_prompt("fine", args.num_parts)),
        ]
    elif args.num_parts:
        glb_prompts = []
        if args.sem_mode in {"all", "semantic"}:
            glb_prompts.append((
                f"num_parts_{args.num_parts}",
                SEMANTIC_OPEN_TEMPLATES["numeric"].format(n=args.num_parts),
            ))
        if args.sem_mode in {"all", "nosem"}:
            glb_prompts.append((
                f"nosem_num_parts_{args.num_parts}",
                NOSEM_OPEN_TEMPLATES["numeric"].format(n=args.num_parts),
            ))
    else:
        glb_prompts = build_glb_open_prompts(args.sem_mode, args.prompt_set)
        print(f"[Info] mesh-folder mode: sem_mode={args.sem_mode}, prompt_set={args.prompt_set}")

    print(f"[Info] Processing {len(mesh_files)} GLB files × {len(glb_prompts)} prompts")

    
    num_success = 0
    num_failed = 0

    for mesh_name in tqdm(mesh_files, desc="Mesh Inference"):
        model_id = os.path.splitext(mesh_name)[0]
        if use_cache:
            mesh_path = os.path.join(args.cache_dir, model_id)  
        else:
            mesh_path = os.path.join(args.glb_dir, mesh_name)

        
        
        if args.resume and glb_prompts and all(
            is_completed_output(os.path.join(args.output_dir, model_id, gran_name))
            for gran_name, _ in glb_prompts
        ):
            print(f"[Skip] {model_id} all requested prompts already completed")
            continue

        
        try:
            if use_cache:
                geo_cache = prepare_geometry_from_cache(
                    model_id, args.cache_dir, model, hf_config, encoder_type,
                    apply_z_positive=args.apply_z_positive,
                )
            else:
                geo_cache = prepare_geometry_from_glb(
                    mesh_path, model, hf_config, encoder_type,
                    apply_z_positive=args.apply_z_positive,
                    clean_mesh=False,
                )
        except Exception as e:
            print(f"[Error] {model_id} geometry prep failed: {e}")
            traceback.print_exc()
            num_failed += len(glb_prompts)
            continue

        for gran_name, tmpl in glb_prompts:
            save_dir = os.path.join(args.output_dir, model_id, gran_name)
            if args.resume and is_completed_output(save_dir):
                print(f"[Skip] {model_id}/{gran_name} already completed -> {save_dir}")
                continue
            inputs = mask_logits = None

            
            dummy_row = {
                "model_id": model_id,
                "prompt_mode": f"open/{gran_name}",
                "granularity": gran_name,
            }

            try:
                inputs, user_prompt_used = prepare_prompt_from_template(
                    tmpl, geo_cache, processor, model
                )
                os.makedirs(save_dir, exist_ok=True)

                response, mask_logits, pca_capture = run_inference(
                    model, inputs, geo_cache["utonia_point_dict"], processor, args
                )

                process_and_save(
                    response, mask_logits, inputs,
                    dummy_row, save_dir, args, 0, user_prompt_used, None,
                    mesh_vis_info=geo_cache["mesh_vis_info"],
                    pca_capture=pca_capture,
                )

                num_success += 1
                print(f"[OK] {model_id}/{gran_name} -> {save_dir}")

            except Exception as e:
                num_failed += 1
                print(f"[Error] {model_id}/{gran_name}: {e}")
                traceback.print_exc()
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, "error.txt"), "w", encoding="utf-8") as f:
                    f.write(f"{type(e).__name__}: {e}\n")
                    traceback.print_exc(file=f)
            finally:
                del inputs, mask_logits

        del geo_cache
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"Done: {num_success} ok, {num_failed} failed ({len(mesh_files)} GLBs × {len(glb_prompts)} prompts)")






def main():
    args = parse_args()
    set_global_seed(args.seed)

    main_glb_dir(args)
    return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Info] Prompt style: {args.prompt_style}")
    print(f"[Info] Semantic mode: {args.sem_mode}")
    if args.prompt_style == "parquet":
        print("[Info] mapped the deprecated parquet prompt style to the current templates")

    
    model, processor, tokenizer, hf_config, encoder_type = load_model(args)

    
    print("Loading data...")
    df, dataset = load_eval_dataset(args, tokenizer, processor)

    
    
    model_ids = df["model_id"].unique().tolist()
    random.seed(args.sample_seed)
    random.shuffle(model_ids)
    if args.max_samples > 0:

        mid_to_cat = {}
        for _, row in df.drop_duplicates("model_id").iterrows():
            mid_to_cat[row["model_id"]] = row["type_id"]
        cat_to_mids = {}
        for mid in model_ids:
            cat = mid_to_cat.get(mid, "unknown")
            cat_to_mids.setdefault(cat, []).append(mid)

        selected = []
        cat_iters = {cat: iter(mids) for cat, mids in cat_to_mids.items()}
        cats = sorted(cat_iters.keys())
        while len(selected) < args.max_samples and cat_iters:
            empty = []
            for cat in cats:
                if cat not in cat_iters:
                    continue
                try:
                    selected.append(next(cat_iters[cat]))
                    if len(selected) >= args.max_samples:
                        break
                except StopIteration:
                    empty.append(cat)
            for cat in empty:
                del cat_iters[cat]
                cats.remove(cat)
        model_ids = selected

    
    model_id_to_idx = {}
    for i, row in df.iterrows():
        mid = row["model_id"]
        if mid not in model_id_to_idx:
            model_id_to_idx[mid] = i

    print(f"[Info] Processing {len(model_ids)} shapes")

    
    mid_to_open_rows = {}
    for mid in model_ids:
        model_rows = df[df["model_id"] == mid]
        mid_to_open_rows[mid] = model_rows[model_rows["prompt_mode"] == "open"]

    
    num_success = 0
    num_failed = 0
    num_skipped = 0
    total_runs = 0

    for mid in tqdm(model_ids, desc="Inference"):
        idx = model_id_to_idx[mid]
        row_orig = df.iloc[idx].copy()
        open_rows = mid_to_open_rows[mid]

        
        geo_row = open_rows.iloc[0].copy() if len(open_rows) > 0 else row_orig.copy()
        prompt_specs = build_open_prompt_specs(args, open_rows, row_orig)
        if not prompt_specs:
            print(f"[Warn] {mid} has no matching prompt specs for prompt_style={args.prompt_style}, skipping")
            continue
        total_runs += len(prompt_specs)
        try:
            geo_cache = prepare_geometry(geo_row, dataset, model, encoder_type)
        except Exception as e:
            print(f"[Error] {mid} geometry prep failed: {e}")
            traceback.print_exc()
            num_failed += len(prompt_specs)
            continue

        for gran_name, row, tmpl in prompt_specs:

            save_dir = os.path.join(args.output_dir, mid, gran_name)
            if args.resume and is_completed_output(save_dir):
                num_skipped += 1
                print(f"[Skip] {mid}/{gran_name} already completed -> {save_dir}")
                continue

            inputs = mask_logits = None

            try:
                inputs, user_prompt_used, gt_response = prepare_prompt(
                    row, geo_cache, dataset, processor, model, tmpl
                )
                os.makedirs(save_dir, exist_ok=True)

                response, mask_logits, pca_capture = run_inference(
                    model, inputs, geo_cache["utonia_point_dict"], processor, args
                )

                process_and_save(
                    response, mask_logits, inputs,
                    row, save_dir, args, idx, user_prompt_used, gt_response,
                    mesh_vis_info=geo_cache["mesh_vis_info"],
                    pca_capture=pca_capture,
                )

                num_success += 1
                print(f"[OK] {mid}/{gran_name} -> {save_dir}")

            except Exception as e:
                num_failed += 1
                print(f"[Error] {mid}/{gran_name}: {e}")
                traceback.print_exc()
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, "error.txt"), "w", encoding="utf-8") as f:
                    f.write(f"{type(e).__name__}: {e}\n")
                    traceback.print_exc(file=f)
            finally:
                del inputs, mask_logits

        
        del geo_cache
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"Done: {num_success} ok, {num_failed} failed, {num_skipped} skipped ({len(model_ids)} shapes, {total_runs} runs)")


if __name__ == "__main__":
    main()
