"""Small utilities shared by the public mesh inference entry points."""

from __future__ import annotations

import os

import numpy as np

SUPPORTED_MESH_EXTENSIONS = (".glb", ".gltf", ".ply", ".obj", ".stl", ".off", ".fbx")


def collect_mesh_paths(path: str) -> list[str]:
    """Return one mesh or all supported meshes directly inside a directory."""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        if os.path.splitext(path)[1].lower() not in SUPPORTED_MESH_EXTENSIONS:
            raise ValueError(f"unsupported mesh format: {path}")
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"mesh path does not exist: {path}")
    meshes = sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.splitext(name)[1].lower() in SUPPORTED_MESH_EXTENSIONS
    )
    if not meshes:
        raise ValueError(f"no supported mesh files found in: {path}")
    return meshes


def source_point_to_model(point, transform: dict) -> np.ndarray:
    """Map a point from the original mesh frame to the model point-cloud frame."""
    point = np.asarray(point, dtype=np.float32)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"point must contain three finite coordinates, got {point!r}")
    point = point * float(transform["scale"]) + np.asarray(transform["shift"], dtype=np.float32)
    return np.asarray(point, dtype=np.float32)


def snap_to_sampled_point(point, sampled_points) -> tuple[np.ndarray, int, float]:
    """Snap a model-space coordinate to the sampled point used by the network."""
    sampled_points = np.asarray(sampled_points, dtype=np.float32)
    if sampled_points.ndim != 2 or sampled_points.shape[1] != 3 or len(sampled_points) == 0:
        raise ValueError("sampled_points must have shape [N, 3]")
    distances = np.linalg.norm(sampled_points - np.asarray(point, dtype=np.float32), axis=1)
    index = int(np.argmin(distances))
    return sampled_points[index].copy(), index, float(distances[index])
