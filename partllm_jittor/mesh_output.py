"""Convert point logits into final per-face segmentation outputs."""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from .postprocess import graph_cut_fast, postprocess_face_labels


def palette(classes: int) -> np.ndarray:
    colors = np.random.RandomState(0).randint(48, 256, size=(classes, 4), dtype=np.uint8)
    colors[:, 3] = 255
    colors[-1] = (80, 80, 80, 255)
    return colors


def fill_unsampled_faces(labels: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    labels = labels.copy()
    sampled = labels >= 0
    if not sampled.any():
        raise ValueError("No mesh faces received point predictions")
    missing = np.flatnonzero(~sampled)
    if len(missing):
        known = np.flatnonzero(sampled)
        _, nearest = cKDTree(mesh.triangles_center[known]).query(mesh.triangles_center[missing])
        labels[missing] = labels[known[nearest]]
    return labels


def face_predictions(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
    logits: np.ndarray,
    postprocess: bool = True,
    graph_cut: bool = True,
    pp_drop_small: bool = True,
    pp_rel_area_threshold: float = 0.001,
) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    face_indices = np.asarray(face_indices, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[1] != len(face_indices) or not np.isfinite(logits).all():
        raise ValueError("Expected finite logits shaped [classes, sampled_points]")
    num_faces, num_classes = len(mesh.faces), logits.shape[0]
    if np.any(face_indices < 0) or np.any(face_indices >= num_faces):
        raise ValueError("Sampled face indices are outside the input mesh")
    sums = np.zeros((num_faces, num_classes), dtype=np.float64)
    counts = np.zeros(num_faces, dtype=np.int64)
    np.add.at(sums, face_indices, logits.T)
    np.add.at(counts, face_indices, 1)
    observed = counts > 0
    labels = np.full(num_faces, -1, dtype=np.int64)
    labels[observed] = sums[observed].argmax(axis=1)
    high_poly = num_faces >= 500_000
    if postprocess and not high_poly:
        labels = postprocess_face_labels(
            labels, mesh, drop_small=pp_drop_small,
            rel_area_threshold=pp_rel_area_threshold, verbose=True,
        )["face_labels"]
    else:
        labels = fill_unsampled_faces(labels, mesh)
    if graph_cut:
        labels = graph_cut_fast(
            labels, sums, counts, mesh,
            _lambda=1.0, iterations=1, band_rings=3, converge=True, max_sweeps=8,
            max_band_label_product=500_000 if high_poly else 0,
            max_band_faces=60_000 if high_poly else 0,
            max_components=128 if high_poly else 0,
            band_fallbacks=(8, 4, 2, 1) if high_poly else None,
            verbose=True,
        )
    return np.asarray(labels, dtype=np.int64)


def save_mesh(mesh: trimesh.Trimesh, labels: np.ndarray, classes: int, output_dir: Path,
              save_colored: bool = True, save_per_part: bool = False) -> None:
    colored = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    colored.visual.face_colors = palette(classes)[labels]
    if save_colored:
        colored.export(output_dir / "pred_mesh.ply")
    np.save(output_dir / "face_labels.npy", labels)
    if save_per_part:
        for label in range(classes - 1):
            selected = np.flatnonzero(labels == label)
            if len(selected):
                colored.submesh([selected], append=True).export(output_dir / f"part_{label:03d}.ply")
