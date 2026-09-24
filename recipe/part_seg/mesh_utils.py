"""
mesh_utils.py — Shared mesh utility functions for 3D part segmentation.

Depends ONLY on trimesh and numpy. No torch, no transformers, no verl imports.
Used by:
  - recipe/part_seg/partnext_dataset.py          (file-path import via _import_from_path)
  - recipe/part_seg/tools/prepare_partnext_dataset.py  (sys.path import)
  - recipe/part_seg/tools/vis_parquet_data.py         (sys.path import)
  - recipe/part_seg/tools/prepare_compat200_dataset.py (future, sys.path import)
"""

import re

import numpy as np
import trimesh







_ACRONYMS = frozenset({
    "USB", "TV", "LED", "LCD", "PVC", "UV", "AC", "DC", "GPS", "VR", "AR",
    "HD", "PTT", "EVA", "PG", "RPG", "AK", "M4", "MP5", "ID", "IO", "PC",
    "BBQ", "DIY", "HDMI", "SSD", "CPU", "GPU",
})


_BLACKLIST_WORDS = frozenset({
    "component", "highlighted", "detached", "extracted", "detailed",
    "featuring", "representing", "designed", "showcasing", "resembling",
    "indicating", "suggesting", "appearing",
})


def normalize_part_name(name: str) -> str:
    """Normalize a part name to Title Case with acronym preservation and cleanup.

    - Replaces underscores with spaces, collapses whitespace
    - Strips leading/trailing punctuation
    - Removes blacklisted VLM artifact words
    - Applies Title Case while preserving known acronyms (USB, TV, LED, ...)

    Args:
        name: raw part name string

    Returns:
        Cleaned part name in Title Case (e.g. "Airplane Wing", "USB Port")
    """
    s = name.strip()
    if not s:
        return s
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" ,;.\"'")


    words = s.split()
    cleaned = [w for w in words if w.lower() not in _BLACKLIST_WORDS]
    if not cleaned:
        cleaned = words


    result = []
    for w in cleaned:
        upper = w.upper()
        if upper in _ACRONYMS:
            result.append(upper)
        elif "-" in w:

            parts = w.split("-")
            normalized_parts = []
            for p in parts:
                p_upper = p.upper()
                if p_upper in _ACRONYMS:
                    normalized_parts.append(p_upper)
                elif p.isupper() and len(p) <= 4:

                    normalized_parts.append(p.upper())
                else:
                    normalized_parts.append(p.capitalize())
            result.append("-".join(normalized_parts))
        else:
            result.append(w.capitalize())
    return " ".join(result)


def scene2meshes(scene):
    """Extract mesh list from a trimesh.Scene, applying transforms.

    Iterates over scene.geometry in trimesh's canonical order and returns
    one transformed trimesh.Trimesh per geometry node. The order here is
    the authoritative mesh_idx ordering used for face-level annotations.

    Args:
        scene: trimesh.Scene or trimesh.Trimesh

    Returns:
        list of trimesh.Trimesh (each with world-space transform applied)
    """
    if not isinstance(scene, trimesh.Scene):
        if isinstance(scene, trimesh.Trimesh):
            return [scene]
        raise ValueError(f"Input must be trimesh.Scene or trimesh.Trimesh, got {type(scene)}")
    meshes = []
    geometry_nodes = scene.graph.geometry_nodes
    for name, geometry in scene.geometry.items():
        if isinstance(geometry, trimesh.Trimesh):
            if name not in geometry_nodes:
                continue
            object_node_name = geometry_nodes[name]
            if len(object_node_name) != 1:

                object_node_name = [object_node_name[0]]
            object_node_name = object_node_name[0]
            if object_node_name not in scene.graph:
                continue
            transform, _ = scene.graph[object_node_name]
            geometry = geometry.copy()
            geometry.apply_transform(transform)
            meshes.append(geometry)
    return meshes


def scene2meshes_with_node_names(scene):
    """Same as scene2meshes() but also returns each mesh's graph node name.

    Needed for 3DCoMPaT200 part mapping, where part labels are stored per
    graph node rather than per mesh index.

    Args:
        scene: trimesh.Scene or trimesh.Trimesh

    Returns:
        list of (trimesh.Trimesh, node_name: str) pairs, in the same order
        as scene2meshes() so that mesh indices are consistent.
    """
    if not isinstance(scene, trimesh.Scene):
        if isinstance(scene, trimesh.Trimesh):
            return [(scene, "__root__")]
        raise ValueError(f"Input must be trimesh.Scene or trimesh.Trimesh, got {type(scene)}")
    result = []
    geometry_nodes = scene.graph.geometry_nodes
    for name, geometry in scene.geometry.items():
        if isinstance(geometry, trimesh.Trimesh):
            if name not in geometry_nodes:
                continue
            object_node_name = geometry_nodes[name]
            if len(object_node_name) != 1:

                object_node_name = [object_node_name[0]]
            object_node_name = object_node_name[0]
            if object_node_name not in scene.graph:
                continue
            transform, _ = scene.graph[object_node_name]
            geometry = geometry.copy()
            geometry.apply_transform(transform)
            result.append((geometry, object_node_name))
    return result


def normalize_meshes_diag(meshes, norm_diag_len=1.0):
    """Normalize meshes so the bounding-box diagonal == norm_diag_len, centered at origin.

    Args:
        meshes: list of trimesh.Trimesh (all treated as one object)
        norm_diag_len: target diagonal length (default 1.0)

    Returns:
        list of new trimesh.Trimesh with vertices rescaled and shifted
    """
    verts_all = np.vstack([np.asarray(m.vertices, dtype=np.float32) for m in meshes])
    bbox_min = np.min(verts_all, axis=0)
    bbox_max = np.max(verts_all, axis=0)
    diag_len = float(np.linalg.norm(bbox_max - bbox_min))
    scale = norm_diag_len / diag_len if diag_len > 0 else norm_diag_len
    verts_scaled = verts_all * scale
    center = (np.min(verts_scaled, axis=0) + np.max(verts_scaled, axis=0)) / 2.0
    shift = -center
    normalized = []
    for m in meshes:
        new_m = m.copy()
        new_m.vertices = np.asarray(new_m.vertices, dtype=np.float32) * scale + shift
        normalized.append(new_m)
    return normalized


def reorder_meshes_by_face_num(mesh_list, mesh_face_num):
    """Reorder mesh_list to match the annotation's mesh_face_num ordering.

    The annotation records {mesh_idx: num_faces} in a specific order that may
    differ from trimesh's scene.geometry iteration order. This function
    reorders mesh_list so that mesh_list[i].faces matches mesh_face_num[str(i)].

    Uses greedy matching by face count. If a match cannot be found, the mesh
    is left in its original position and a warning is printed.

    Args:
        mesh_list: list of trimesh.Trimesh
        mesh_face_num: dict {str(mesh_idx): int num_faces}

    Returns:
        reordered mesh_list (same length as input)
    """
    n_ann = len(mesh_face_num)
    if n_ann == 0 or len(mesh_list) == 0:
        return mesh_list

    target_counts = [mesh_face_num[str(i)] for i in range(n_ann)]
    actual_counts = [len(m.faces) for m in mesh_list]


    if actual_counts == target_counts:
        return mesh_list

    remaining = list(range(len(mesh_list)))
    reordered = [None] * n_ann
    for i, expected in enumerate(target_counts):
        found = False
        for j in remaining:
            if len(mesh_list[j].faces) == expected:
                reordered[i] = mesh_list[j]
                remaining.remove(j)
                found = True
                break
        if not found:

            if i < len(mesh_list):
                reordered[i] = mesh_list[i]
                if i in remaining:
                    remaining.remove(i)
            print(
                f"[WARN] reorder_meshes: no mesh matches face_num={expected} for index {i}, "
                f"available counts={[len(mesh_list[j].faces) for j in remaining]}"
            )


    for j in remaining:
        reordered.append(mesh_list[j])

    return [m for m in reordered if m is not None]
