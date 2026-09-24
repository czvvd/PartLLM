"""
Preprocess the 3DCoMPaT200 dataset into parquet format for training.

Converts 3DCoMPaT200 GLTF models (with composition-0 material textures) into
parquet files matching the PartNeXt 16-column schema so the existing
PartNeXtPoint3DDataset loader works without modification.

Each shape produces up to 2 depth levels (fine + coarse) x 3 prompt modes
(grounding, open, promptable) = up to 6 rows.

Usage:
    python recipe/part_seg/tools/prepare_compat200_dataset.py \
        --zip_path /path/to/Compat200.zip \
        --meta_dir /path/to/3DCoMPaT200/metadata \
        --loader_dir /path/to/3DCoMPaT200/loaders/3D \
        --output_dir /path/to/3DCoMPaT200/parquet \
        --glb_dir /path/to/3DCoMPaT200/glbs \
        --max_parts 64 \
        --seed 42 \
        --verify
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import datasets
import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mesh_utils import normalize_meshes_diag, normalize_part_name, scene2meshes, scene2meshes_with_node_names


def coord_to_int(coord: float) -> int:
    """Map coordinate from [-0.5, 0.5] to integer [0, 1000]."""
    return max(0, min(1000, int(round((coord + 0.5) * 1000))))


compat_gltf = None


def load_official_compat_loader(loader_dir=None):
    """Load the official 3DCoMPaT200 GLTF helper in this process."""
    global compat_gltf
    if compat_gltf is not None:
        return compat_gltf
    loader_dir = loader_dir or os.environ.get("PARTLLM_COMPAT200_LOADER_DIR")
    if not loader_dir or not os.path.isdir(os.path.join(loader_dir, "utils3D")):
        raise RuntimeError("3DCoMPaT200 loader not found; pass --loader_dir")
    if loader_dir not in sys.path:
        sys.path.insert(0, loader_dir)
    import utils3D.gltf as compat_gltf_module
    compat_gltf = compat_gltf_module
    return compat_gltf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)






def strip_instance_suffix(node_name: str) -> str:
    """'seat_01' → 'seat', 'leg_03' → 'leg', 'display' → 'display'."""
    parts = node_name.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return node_name


def classify_granularity(depth, first_valid_depth, last_valid_depth, num_parts):
    """Classify depth level into coarse/medium/fine granularity.

    Same logic as prepare_partnext_dataset.py.
    """
    is_first = depth == first_valid_depth
    is_last = depth == last_valid_depth

    if is_first and is_last:
        return "medium"
    if is_first:
        return "coarse"
    if is_last:
        return "fine" if num_parts >= 5 else "medium"
    if num_parts <= 4:
        return "coarse"
    elif num_parts >= 10:
        return "fine"
    else:
        return "medium"






def compute_bbox_from_meshes(mesh_vertices_by_idx, mesh_faces_by_idx, masks_dict, mask_ids):
    """Compute AABB for a set of mask_ids using pre-computed normalized vertices.

    Args:
        mesh_vertices_by_idx: dict {mesh_idx: np.ndarray of shape (V, 3)} (already normalized)
        mesh_faces_by_idx: dict {mesh_idx: np.ndarray of shape (F, 3)}
        masks_dict: {mask_id_str: {mesh_idx_str: [face_indices]}}
        mask_ids: list of int mask ids

    Returns:
        [x_min, y_min, z_min, x_max, y_max, z_max] rounded to 3 decimals
    """
    all_vertices = []
    for mid in mask_ids:
        mid_str = str(mid)
        if mid_str not in masks_dict:
            continue
        for mesh_idx_str, face_indices in masks_dict[mid_str].items():
            mesh_idx = int(mesh_idx_str)
            if mesh_idx not in mesh_vertices_by_idx:
                continue
            verts = mesh_vertices_by_idx[mesh_idx]
            faces = mesh_faces_by_idx[mesh_idx]
            face_indices_arr = np.array(face_indices, dtype=np.int64)
            valid = face_indices_arr[face_indices_arr < len(faces)]
            if len(valid) == 0:
                continue
            vert_indices = faces[valid].flatten()
            all_vertices.append(verts[vert_indices])

    if not all_vertices:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    all_vertices = np.vstack(all_vertices)
    mins = np.min(all_vertices, axis=0)
    maxs = np.max(all_vertices, axis=0)
    return [round(float(v), 3) for v in [mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2]]]






def export_glb(gltf_stream, zip_f, glb_path):
    """Load styled GLTF stream and export as GLB file."""
    scene = trimesh.load(
        gltf_stream,
        file_type=".gltf",
        force="scene",
        resolver=compat_gltf.ZipTextureResolver(zip_f=zip_f),
    )
    os.makedirs(os.path.dirname(glb_path), exist_ok=True)
    scene.export(glb_path, file_type="glb")
    return scene


def verify_glb_roundtrip(original_scene, glb_path):
    """Verify GLB round-trip by checking vertex count hash for meshes with same face count."""
    reloaded = trimesh.load(glb_path, force="scene")
    orig_meshes = scene2meshes(original_scene)
    reload_meshes = scene2meshes(reloaded)

    if len(orig_meshes) != len(reload_meshes):
        return False, f"mesh count mismatch: {len(orig_meshes)} vs {len(reload_meshes)}"

    for i, (om, rm) in enumerate(zip(orig_meshes, reload_meshes)):
        if len(om.faces) != len(rm.faces):
            return False, f"mesh {i} face count: {len(om.faces)} vs {len(rm.faces)}"

        oh = hashlib.md5(np.asarray(om.vertices, dtype=np.float32).tobytes()).hexdigest()
        rh = hashlib.md5(np.asarray(rm.vertices, dtype=np.float32).tobytes()).hexdigest()
        if oh != rh:
            return False, f"mesh {i} vertex hash mismatch"

    return True, "ok"






def build_masks_and_face_num(scene, parts_fine_set):
    """Build masks_dict and mesh_face_num from a scene using node names.

    Each GLTF node = one mesh. The node name (with instance suffix stripped)
    is the fine part name. Nodes not in parts_fine are skipped (logged).

    Meshes sharing the same part name share a mask_id.

    Uses lightweight enumeration (no mesh copy/transform) for speed.
    The mesh_idx ordering follows scene.geometry.items() — same as scene2meshes().

    Args:
        scene: trimesh.Scene (loaded from GLTF/GLB)
        parts_fine_set: set of valid fine part names

    Returns:
        masks_dict: {mask_id_str: {mesh_idx_str: [face_indices]}}
        mesh_face_num: {mesh_idx_str: int}
        part_name_to_mask_id: {part_name: mask_id}
        skipped_nodes: list of node names that were not in parts_fine
    """

    geometry_nodes = scene.graph.geometry_nodes
    mesh_entries = []
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
            mesh_entries.append((object_node_name, len(geometry.faces)))

    masks_dict = {}
    mesh_face_num = {}
    part_name_to_mask_id = {}
    skipped_nodes = []
    next_mask_id = 0

    for mesh_idx, (node_name, n_faces) in enumerate(mesh_entries):
        mesh_idx_str = str(mesh_idx)
        mesh_face_num[mesh_idx_str] = n_faces

        part_name = strip_instance_suffix(node_name)
        if part_name not in parts_fine_set:
            skipped_nodes.append(node_name)
            continue


        if part_name not in part_name_to_mask_id:
            part_name_to_mask_id[part_name] = next_mask_id
            next_mask_id += 1

        mask_id = part_name_to_mask_id[part_name]
        mask_id_str = str(mask_id)

        if mask_id_str not in masks_dict:
            masks_dict[mask_id_str] = {}


        masks_dict[mask_id_str][mesh_idx_str] = list(range(n_faces))

    return masks_dict, mesh_face_num, part_name_to_mask_id, skipped_nodes






def process_single_shape(
    shape_id,
    split,
    zip_path,
    meta_dir,
    glb_dir,
    parts_fine_set,
    hier_coarse,
    classes,
    max_parts,
    do_verify,
    zip_f=None,
    textures_map=None,
    styles_cache=None,
):
    """Process one 3DCoMPaT200 shape into training samples.

    Returns list of sample dicts (up to 6 per shape: 2 depths x 3 prompt modes).
    If zip_f/textures_map/styles_cache are provided, they are reused (serial mode).
    Otherwise opened per-call (parallel mode).
    """
    load_official_compat_loader()

    cat_hex = shape_id[:2]
    owns_zip = False

    try:
        if zip_f is None:
            zip_f = zipfile.ZipFile(zip_path, "r")
            owns_zip = True
        if textures_map is None:
            textures_map = json.load(zip_f.open("textures_map.json", "r"))


        gltf_f = compat_gltf.load_gltf(shape_id, zip_file=zip_f, models_dir="models/")


        if styles_cache is not None and shape_id in styles_cache:
            style_entry = styles_cache[shape_id]
        else:
            style_entry = compat_gltf.load_styles(
                shape_id=shape_id,
                style_id="0",
                split=split,
                comp_k=0,
                sem_level="fine",
                zip_file=zip_f,
                styles_dir="styles/",
            )


        styled_stream = compat_gltf.apply_style(
            gltf_f, style_entry, textures_file_map=textures_map
        )


        glb_path = os.path.join(glb_dir, f"{shape_id}.glb")
        if os.path.exists(glb_path):
            scene = trimesh.load(glb_path, force="scene")
        else:
            scene = export_glb(styled_stream, zip_f, glb_path)


        masks_dict, mesh_face_num, part_name_to_mask_id, skipped_nodes = build_masks_and_face_num(
            scene, parts_fine_set
        )

        if skipped_nodes:
            logger.warning(f"{shape_id}: skipped {len(skipped_nodes)} unknown nodes: {skipped_nodes}")

        if not part_name_to_mask_id:
            if owns_zip:
                zip_f.close()
            return []


        if do_verify:
            ok, msg = verify_glb_roundtrip(scene, glb_path)
            if not ok:
                logger.warning(f"GLB verification failed for {shape_id}: {msg}")

        if owns_zip:
            zip_f.close()

    except Exception as e:
        logger.warning(f"Error processing {shape_id}: {e}")
        return []


    try:
        geometry_nodes = scene.graph.geometry_nodes

        all_verts_list = []
        mesh_vertices_by_idx = {}
        mesh_faces_by_idx = {}
        idx = 0
        for name, geometry in scene.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                if name not in geometry_nodes:
                    continue
                node = geometry_nodes[name]
                if len(node) < 1 or node[0] not in scene.graph:
                    continue
                transform, _ = scene.graph[node[0]]
                verts = np.asarray(geometry.vertices, dtype=np.float32)

                verts_h = np.hstack([verts, np.ones((len(verts), 1), dtype=np.float32)])
                verts_transformed = (verts_h @ transform.T)[:, :3]
                all_verts_list.append(verts_transformed)
                mesh_vertices_by_idx[idx] = verts_transformed
                mesh_faces_by_idx[idx] = np.asarray(geometry.faces, dtype=np.int64)
                idx += 1

        if not all_verts_list:
            return []


        all_verts = np.vstack(all_verts_list)
        bbox_min = np.min(all_verts, axis=0)
        bbox_max = np.max(all_verts, axis=0)
        diag_len = float(np.linalg.norm(bbox_max - bbox_min))
        scale = 1.0 / diag_len if diag_len > 0 else 1.0
        all_verts_scaled = all_verts * scale
        center = (np.min(all_verts_scaled, axis=0) + np.max(all_verts_scaled, axis=0)) / 2.0
        shift = -center

        for idx in mesh_vertices_by_idx:
            mesh_vertices_by_idx[idx] = mesh_vertices_by_idx[idx] * scale + shift

    except Exception as e:
        logger.warning(f"Failed to compute bboxes for {shape_id}: {e}")
        return []



    type_id = cat_hex
    cat_idx = int(cat_hex, 16)
    object_category = normalize_part_name(classes[cat_idx]) if cat_idx < len(classes) else "unknown"



    fine_parts = []
    for part_name, mask_id in part_name_to_mask_id.items():
        fine_parts.append({
            "name": part_name,
            "mask_ids": [mask_id],
            "display_name": normalize_part_name(part_name),
        })


    coarse_map = hier_coarse.get(cat_hex, {})
    coarse_groups = defaultdict(list)
    for part_name, mask_id in part_name_to_mask_id.items():
        coarse_name = coarse_map.get(part_name)
        if coarse_name is None:

            coarse_name = part_name
        coarse_groups[coarse_name].append(mask_id)

    coarse_parts = []
    for coarse_name, mask_ids in coarse_groups.items():
        coarse_parts.append({
            "name": coarse_name,
            "mask_ids": sorted(mask_ids),
            "display_name": normalize_part_name(coarse_name),
        })



    depths_info = []


    if len(fine_parts) > 1 and (max_parts is None or len(fine_parts) <= max_parts):
        depths_info.append((1, "fine", fine_parts))


    if len(coarse_parts) > 1 and (max_parts is None or len(coarse_parts) <= max_parts):

        fine_partition = frozenset(frozenset(p["mask_ids"]) for p in fine_parts)
        coarse_partition = frozenset(frozenset(p["mask_ids"]) for p in coarse_parts)
        if coarse_partition != fine_partition:
            depths_info.append((0, "coarse", coarse_parts))

    if not depths_info:
        return []


    valid_depths = sorted(d for d, _, _ in depths_info)
    first_valid_depth = valid_depths[0]
    last_valid_depth = valid_depths[-1]

    masks_json = json.dumps(masks_dict)
    mesh_face_num_json = json.dumps(mesh_face_num)

    samples = []

    for depth, depth_level, parts in depths_info:

        parts_with_bbox = []
        for p in parts:
            bbox = compute_bbox_from_meshes(mesh_vertices_by_idx, mesh_faces_by_idx, masks_dict, p["mask_ids"])
            cx = (bbox[0] + bbox[3]) / 2
            cy = (bbox[1] + bbox[4]) / 2
            cz = (bbox[2] + bbox[5]) / 2
            parts_with_bbox.append((p, bbox, (cx, cy, cz)))
        parts_with_bbox.sort(key=lambda x: x[2])


        target_part_names = []
        target_mask_ids = []
        target_bboxes = []
        assistant_lines = []
        for p, bbox, _ in parts_with_bbox:
            display_name = p["display_name"]
            bbox_list = [coord_to_int(float(v)) for v in bbox]
            bbox_json = json.dumps({"bbox_aabb": bbox_list, "label": display_name})
            line = f"{bbox_json}<|SEG|>"
            assistant_lines.append(line)
            target_part_names.append(display_name)
            target_mask_ids.append(p["mask_ids"])
            target_bboxes.append(bbox)

        assistant_text = "\n".join(assistant_lines) + "\n<|BG|>"

        unique_labels = list(dict.fromkeys(target_part_names))
        num_parts = len(parts)
        granularity = classify_granularity(depth, first_valid_depth, last_valid_depth, num_parts)

        base_sample = {
            "model_id": shape_id,
            "type_id": type_id,
            "object_category": object_category,
            "depth": depth,
            "depth_level": depth_level,
            "glb_path": glb_path,
            "masks_json": masks_json,
            "mesh_face_num_json": mesh_face_num_json,
            "target_part_names": target_part_names,
            "target_mask_ids": target_mask_ids,
            "target_bboxes": target_bboxes,
            "point_cloud": [glb_path],
            "vertex": [glb_path],
            "diffusion_messages": {
                "x": "<vertex>",
                "contexts": {"main": "<messages>"},
                "targets": "<vertex_label>",
                "targets_type": "classification",
            },
        }


        grounding_sample = dict(base_sample)
        grounding_sample["prompt_mode"] = "grounding"
        grounding_sample["messages"] = [
            {"role": "user", "content": f"Please segment all the {'; '.join(unique_labels)} in <point_cloud>."},
            {"role": "assistant", "content": assistant_text},
        ]
        grounding_sample["granularity"] = None
        samples.append(grounding_sample)


        open_sample = dict(base_sample)
        open_sample["prompt_mode"] = "open"
        open_sample["messages"] = [
            {"role": "user", "content": "Please segment all parts of this 3D object in <point_cloud>."},
            {"role": "assistant", "content": assistant_text},
        ]
        open_sample["granularity"] = granularity
        samples.append(open_sample)


        promptable_sample = dict(base_sample)
        promptable_sample["prompt_mode"] = "promptable"
        promptable_sample["messages"] = [
            {"role": "user", "content": "<point_cloud>"},
            {"role": "assistant", "content": ""},
        ]
        promptable_sample["granularity"] = None
        samples.append(promptable_sample)

    return samples






def main():
    parser = argparse.ArgumentParser(description="Preprocess 3DCoMPaT200 dataset to parquet")
    parser.add_argument(
        "--zip_path",
        required=True,
        help="Path to Compat200.zip",
    )
    parser.add_argument(
        "--meta_dir",
        required=True,
        help="Path to metadata directory",
    )
    parser.add_argument(
        "--loader_dir",
        required=True,
        help="Path to the official 3DCoMPaT200 loaders/3D directory",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--glb_dir",
        required=True,
        help="Output directory for GLB files",
    )
    parser.add_argument("--max_parts", type=int, default=64, help="Max parts per sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_proc", type=int, default=32, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N shapes (for testing)")
    parser.add_argument("--verify", action="store_true", help="Verify GLB round-trip")
    parser.add_argument("--splits", nargs="+", default=None, help="Only process these output splits (e.g. test_official)")
    args = parser.parse_args()

    loader_dir = os.path.abspath(os.path.expanduser(args.loader_dir))
    if not os.path.isdir(os.path.join(loader_dir, "utils3D")):
        parser.error("--loader_dir must contain the official utils3D package")
    os.environ["PARTLLM_COMPAT200_LOADER_DIR"] = loader_dir
    load_official_compat_loader(loader_dir)

    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.glb_dir, exist_ok=True)


    with open(os.path.join(args.meta_dir, "parts_fine.json")) as f:
        parts_fine = json.load(f)
    parts_fine_set = set(parts_fine)
    logger.info(f"Loaded {len(parts_fine)} fine parts")

    with open(os.path.join(args.meta_dir, "classes.json")) as f:
        classes = json.load(f)
    logger.info(f"Loaded {len(classes)} classes")

    with open(os.path.join(args.meta_dir, "split.json")) as f:
        split_info = json.load(f)
    logger.info(
        f"Split info: train={len(split_info['train'])}, valid={len(split_info['valid'])}, test={len(split_info['test'])}"
    )

    with open(os.path.join(args.meta_dir, "hier_coarse.json")) as f:
        hier_coarse = json.load(f)
    logger.info(f"Loaded hier_coarse for {len(hier_coarse)} categories")


    zip_f = zipfile.ZipFile(args.zip_path, "r")
    styled_sets = {}
    for compat_split in ["train", "valid"]:
        styles = json.load(zip_f.open(f"styles/{compat_split}/comp_fine_0.json"))
        styled_sets[compat_split] = set(key.split("__")[0] for key in styles)
    zip_f.close()



    splits_to_process = [
        ("train", split_info["train"], styled_sets["train"], "train"),
        ("test", split_info["valid"], styled_sets["valid"], "valid"),
    ]

    for out_name, shape_ids, styled_shapes, compat_split in splits_to_process:
        if args.splits and out_name not in args.splits:
            logger.info(f"Skipping {out_name} (not in --splits {args.splits})")
            continue

        shape_ids = [sid for sid in shape_ids if sid in styled_shapes]
        shape_ids.sort()

        if args.limit is not None:
            shape_ids = shape_ids[: args.limit]

        logger.info(f"Processing {out_name}: {len(shape_ids)} shapes (compat split={compat_split})")

        all_samples = []
        skipped = 0

        if args.num_proc <= 1:

            zip_f = zipfile.ZipFile(args.zip_path, "r")
            textures_map = json.load(zip_f.open("textures_map.json", "r"))

            all_styles = compat_gltf.load_style_json(
                split=compat_split, comp_k=0, sem_level="fine",
                zip_file=zip_f, styles_dir="styles/",
            )

            styles_cache = {}
            for key, entry in all_styles.items():
                sid = key.split("__")[0]
                styles_cache[sid] = entry

            for i, shape_id in enumerate(shape_ids):
                try:
                    samples = process_single_shape(
                        shape_id, compat_split, args.zip_path, args.meta_dir,
                        args.glb_dir, parts_fine_set, hier_coarse, classes, args.max_parts, args.verify,
                        zip_f=zip_f, textures_map=textures_map, styles_cache=styles_cache,
                    )
                    if not samples:
                        skipped += 1
                    all_samples.extend(samples)
                except Exception as e:
                    skipped += 1
                    logger.warning(f"Error processing {shape_id}: {e}")
                if (i + 1) % 500 == 0 or (i + 1) % 50 == 0 and (i + 1) <= 500:
                    logger.info(f"  {out_name}: {i + 1}/{len(shape_ids)} shapes, {len(all_samples)} samples")
            zip_f.close()
        else:
            with ProcessPoolExecutor(max_workers=args.num_proc) as executor:
                futures = {
                    executor.submit(
                        process_single_shape,
                        shape_id, compat_split, args.zip_path, args.meta_dir,
                        args.glb_dir, parts_fine_set, hier_coarse, classes, args.max_parts, args.verify,
                    ): shape_id
                    for shape_id in shape_ids
                }
                done_count = 0
                for future in as_completed(futures, timeout=14400):
                    done_count += 1
                    shape_id = futures[future]
                    try:
                        samples = future.result(timeout=300)
                        if not samples:
                            skipped += 1
                        all_samples.extend(samples)
                    except TimeoutError:
                        skipped += 1
                        logger.warning(f"Timeout processing {shape_id}, skipping")
                    except Exception as e:
                        skipped += 1
                        logger.warning(f"Error processing {shape_id}: {e}")
                    if done_count % 500 == 0:
                        logger.info(f"  {out_name}: {done_count}/{len(shape_ids)} shapes, {len(all_samples)} samples")

        logger.info(f"{out_name}: {len(all_samples)} samples from {len(shape_ids)} shapes, {skipped} skipped")

        if not all_samples:
            logger.warning(f"No samples for {out_name}, skipping")
            continue


        out_path = os.path.join(args.output_dir, f"{out_name}.parquet")
        ds = datasets.Dataset.from_list(all_samples)
        ds.to_parquet(out_path)
        logger.info(f"Saved {out_name} ({len(ds)} samples) to {out_path}")


        sub_groups = defaultdict(list)
        for s in all_samples:
            key = (s.get("depth_level", "unknown"), s.get("prompt_mode", "grounding"))
            sub_groups[key].append(s)

        for (dlevel, pmode), group_samples in sorted(sub_groups.items()):
            sub_name = f"{out_name}_{dlevel}_{pmode}"
            sub_path = os.path.join(args.output_dir, f"{sub_name}.parquet")
            sub_ds = datasets.Dataset.from_list(group_samples)
            sub_ds.to_parquet(sub_path)
            logger.info(f"  Saved {sub_name}: {len(sub_ds)} samples -> {sub_path}")


        depth_counts = Counter(s["depth"] for s in all_samples)
        level_counts = Counter(s.get("depth_level", "unknown") for s in all_samples)
        prompt_counts = Counter(s.get("prompt_mode", "grounding") for s in all_samples)
        logger.info(f"  Depth distribution: {dict(sorted(depth_counts.items()))}")
        logger.info(f"  Level distribution: {dict(level_counts)}")
        logger.info(f"  Prompt distribution: {dict(prompt_counts)}")
        granularity_counts = Counter(
            s.get("granularity", "N/A") for s in all_samples if s.get("prompt_mode") == "open"
        )
        if granularity_counts:
            logger.info(f"  Granularity distribution (open only): {dict(granularity_counts)}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
