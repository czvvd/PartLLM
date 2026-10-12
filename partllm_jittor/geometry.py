"""Mesh sampling and the non-neural Utonia default transform, without torch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(frozen=True)
class UtoniaPoints:
    """Equivalent arrays produced by Utonia's default transform before ToTensor."""

    coord: np.ndarray
    grid_coord: np.ndarray
    color: np.ndarray
    normal: np.ndarray
    feat: np.ndarray
    inverse: np.ndarray


def _sample_mesh(
    mesh: trimesh.Trimesh, point_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically sample surface points, RGB and face normals."""
    rng = np.random.RandomState(seed)
    areas = mesh.area_faces.astype(np.float64)
    if point_count <= 0 or not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("Mesh sampling requires positive point count and finite, nonzero surface area")
    face_indices = rng.choice(len(mesh.faces), size=point_count, p=areas / areas.sum())
    triangle = mesh.triangles[face_indices]
    barycentric = rng.random_sample((point_count, 2))
    flip = barycentric.sum(axis=1) > 1.0
    barycentric[flip] = 1.0 - barycentric[flip]
    weights = np.column_stack([1.0 - barycentric.sum(axis=1), barycentric])
    points = (triangle * weights[:, :, None]).sum(axis=1).astype(np.float32)
    normals = mesh.face_normals[face_indices].astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    colors = np.full((point_count, 3), 127, dtype=np.uint8)
    if hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors) == len(mesh.vertices):
        vertex_colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32)
        colors = np.clip((vertex_colors[mesh.faces[face_indices]] * weights[:, :, None]).sum(axis=1), 0, 255).astype(np.uint8)
    elif hasattr(mesh.visual, "face_colors") and len(mesh.visual.face_colors) == len(mesh.faces):
        colors = np.asarray(mesh.visual.face_colors[face_indices, :3], dtype=np.uint8)
    return points, colors, normals, face_indices.astype(np.int64)


def _load_triangle_mesh(mesh_path: str | Path) -> trimesh.Trimesh:
    """Load one scene or mesh as a single triangle mesh."""
    if Path(mesh_path).suffix.lower() == ".fbx":
        import assimp_py
        scene = assimp_py.import_file(str(mesh_path), assimp_py.Process_Triangulate | assimp_py.Process_PreTransformVertices)
        meshes = []
        for item in scene.meshes:
            vertices = np.asarray(item.vertices).reshape(-1, 3)
            faces = np.asarray(item.indices).reshape(-1, 3)
            meshes.append(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
        if not meshes:
            raise ValueError(f"No triangle mesh in {mesh_path}")
        loaded = trimesh.util.concatenate(meshes)
    else:
        loaded = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.dump() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle mesh in {mesh_path}")
        for item in meshes:
            if item.visual.kind == "texture":
                item.visual = item.visual.to_color()
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"No triangle mesh in {mesh_path}")
    return loaded


def mesh_frame(mesh: trimesh.Trimesh, points: np.ndarray):
    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    center = bounds.mean(axis=0)
    scale = 1.0 / max(float(np.linalg.norm(bounds[1] - bounds[0])), 1e-8)
    return (points - center) * scale, center, scale


def prepare_clicks(positive, negative, mesh, points):
    normalized, center, scale = mesh_frame(mesh, points)
    clicks = []
    for source, label in [(point, 1) for point in positive] + [(point, 0) for point in negative]:
        source = np.asarray(source, dtype=np.float32)
        if source.shape != (3,) or not np.isfinite(source).all():
            raise ValueError("Click coordinates must be three finite numbers")
        model_point = (source - center) * scale
        distances = np.linalg.norm(normalized - model_point, axis=1)
        index = int(distances.argmin())
        clicks.append({"point": normalized[index].tolist(), "point_index": index,
                       "label": label, "source_point": source.tolist(),
                       "snap_distance": float(distances[index])})
    return clicks


def load_initial_mask(path, num_faces):
    mask = np.load(path, allow_pickle=False)
    if mask.shape != (num_faces,) or mask.dtype.kind not in "biuf":
        raise ValueError(f"Initial mask must be a numeric or boolean array shaped [{num_faces}]")
    if not np.isfinite(mask).all() or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Initial mask must contain only 0/1 values")
    return mask.astype(bool)


def load_mesh_points(mesh_path: str | Path, point_count: int = 81920, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, colors, normals, _ = _sample_mesh(_load_triangle_mesh(mesh_path), point_count, seed)
    return points, colors, normals


def load_mesh_samples(
    mesh_path: str | Path, point_count: int = 81920, seed: int = 0
) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the mesh plus deterministic samples and their source face IDs."""
    mesh = _load_triangle_mesh(mesh_path)
    points, colors, normals, face_indices = _sample_mesh(mesh, point_count, seed)
    return mesh, points, colors, normals, face_indices


def load_points_npz(points_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an already-sampled point cloud from an NPZ archive.

    ``coords`` is required and must have shape ``[N, 3]``. ``colors`` and
    ``normals`` are optional; omitted colors default to neutral gray and
    omitted normals to zero.
    """
    with np.load(points_path, allow_pickle=False) as archive:
        if "coords" not in archive:
            raise ValueError(f"{points_path} must contain a 'coords' array")
        coords = np.asarray(archive["coords"], dtype=np.float32)
        colors = np.asarray(archive["colors"], dtype=np.uint8) if "colors" in archive else np.full(coords.shape, 127, dtype=np.uint8)
        normals = np.asarray(archive["normals"], dtype=np.float32) if "normals" in archive else np.zeros(coords.shape, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape [N, 3]")
    if colors.shape != coords.shape or normals.shape != coords.shape:
        raise ValueError("coords, colors and normals must all have shape [N, 3]")
    if len(coords) == 0:
        raise ValueError("point cloud must not be empty")
    if not np.isfinite(coords).all() or not np.isfinite(normals).all():
        raise ValueError("Point coordinates and normals must be finite")
    return coords, colors, normals


def _fnv_hash_vec(array: np.ndarray) -> np.ndarray:
    hashed = np.full(array.shape[0], np.uint64(14695981039346656037), dtype=np.uint64)
    for axis in range(array.shape[1]):
        hashed *= np.uint64(1099511628211)
        hashed = np.bitwise_xor(hashed, array[:, axis].astype(np.uint64, copy=False))
    return hashed


def utonia_default_transform(
    coords: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    scale: float = 5.0,
    grid_size: float = 0.01,
    seed: int = 0,
) -> UtoniaPoints:
    """Normalize coordinates, scale, sample voxels, and normalize colors."""
    coords = np.asarray(coords, dtype=np.float32).copy()
    colors = np.asarray(colors, dtype=np.uint8)
    normals = np.asarray(normals, dtype=np.float32)
    if not (len(coords) == len(colors) == len(normals)):
        raise ValueError("coords, colors and normals must have equal point counts")
    if len(coords) == 0 or not np.isfinite(coords).all() or not np.isfinite(normals).all():
        raise ValueError("Point coordinates and normals must be nonempty and finite")
    coords -= coords.mean(axis=0, keepdims=True)
    coords /= np.sqrt((coords * coords).sum(axis=1)).max() + 1e-8
    coords *= np.float32(scale)
    grid = np.floor(coords / grid_size).astype(np.int64)
    grid -= grid.min(axis=0, keepdims=True)
    keys = _fnv_hash_vec(grid)


    sorted_indices = np.argsort(keys, kind="stable")
    _, sorted_group, counts = np.unique(keys[sorted_indices], return_inverse=True, return_counts=True)
    rng = np.random.RandomState(seed)
    selected_offsets = np.cumsum(np.insert(counts, 0, 0)[:-1]) + rng.randint(0, counts.max(), counts.size) % counts
    selected = sorted_indices[selected_offsets]
    inverse = np.empty_like(sorted_group, dtype=np.int64)
    inverse[sorted_indices] = sorted_group
    selected_grid = grid[selected].astype(np.int32)
    selected_coord = coords[selected].astype(np.float32)
    selected_color = colors[selected, :3].astype(np.float32) / 255.0
    selected_normal = normals[selected].astype(np.float32)
    return UtoniaPoints(
        coord=selected_coord,
        grid_coord=selected_grid,
        color=selected_color,
        normal=selected_normal,
        feat=np.concatenate([selected_coord, selected_color, selected_normal], axis=1),
        inverse=inverse,
    )
