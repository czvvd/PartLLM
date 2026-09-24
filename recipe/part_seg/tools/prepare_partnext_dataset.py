"""
Preprocess the PartNeXt dataset into parquet format for training.

Converts PartNeXt annotations (HF datasets) + GLB files into a parquet file
with conversation-format training samples. Each 3D model is decomposed at
multiple hierarchy depth levels to produce coarse-to-fine segmentation samples.

Usage:
    python recipe/part_seg/tools/prepare_partnext_dataset.py \
        --annotation_dir /path/to/PartNeXt \
        --glb_dir /path/to/PartNeXt/glbs \
        --output_dir /path/to/PartNeXt/parquet \
        --max_parts 64 \
        --depth_mode leaf \
        --num_proc 32
"""

import argparse
import ast
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mesh_utils import scene2meshes, normalize_meshes_diag, reorder_meshes_by_face_num, normalize_part_name


def coord_to_int(coord: float) -> int:
    """Map coordinate from [-0.5, 0.5] to integer [0, 1000]."""
    return max(0, min(1000, int(round((coord + 0.5) * 1000))))
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import datasets
import numpy as np
import trimesh

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)






def collect_mask_ids(node):
    """Collect all maskIds in a subtree (node + all descendants)."""
    ids = []
    if "maskId" in node:
        ids.append(node["maskId"])
    if "children" in node:
        for child in node["children"]:
            ids.extend(collect_mask_ids(child))
    return ids


def collect_nodes_at_depth(node, current_depth, target_depth):
    """
    Collect *groups* at a given target depth.

    At the target depth, each node (which may be a leaf or an internal node)
    becomes one "part". Its mask = union of all descendant maskIds.

    Returns list of dicts: [{"name": str, "nodeId": int, "mask_ids": [int, ...]}]
    """


    if "children" not in node and "maskId" in node:
        return [{"name": node["name"], "nodeId": node["nodeId"], "mask_ids": [node["maskId"]]}]

    if current_depth == target_depth:

        mask_ids = collect_mask_ids(node)
        if mask_ids:
            return [{"name": node["name"], "nodeId": node["nodeId"], "mask_ids": mask_ids}]
        return []


    if "children" not in node:
        return []
    results = []
    for child in node["children"]:
        results.extend(collect_nodes_at_depth(child, current_depth + 1, target_depth))
    return results


def get_max_depth(node, current_depth=0):
    """Get the maximum depth of the hierarchy tree."""
    if "children" not in node:
        return current_depth
    return max(get_max_depth(child, current_depth + 1) for child in node["children"])


def classify_granularity(depth, first_valid_depth, last_valid_depth, num_parts):
    """Classify depth level into coarse/medium/fine granularity.

    Args:
        depth: Current depth.
        first_valid_depth: min of valid depth set (after >1 part, dedup, max_parts filtering).
        last_valid_depth: max of valid depth set.
        num_parts: Number of parts at this depth.
    """
    is_first = (depth == first_valid_depth)
    is_last = (depth == last_valid_depth)


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


def compute_bbox_from_faces(merged_mesh, mesh_face_num, masks_dict, mask_ids):
    """
    Compute AABB for a set of maskIds using the concatenated (merged) mesh.

    Uses annotation's mesh_face_num as face offsets to compute global face indices,
    consistent with the official PartNeXt code.

    Args:
        merged_mesh: trimesh.Trimesh, concatenated from reordered mesh_list
        mesh_face_num: dict {mesh_idx_str: num_faces}
        masks_dict: {part_id_str: {mesh_idx_str: [face_indices]}}
        mask_ids: list of int maskIds

    Returns:
        [x_min, y_min, z_min, x_max, y_max, z_max] rounded to 3 decimals
    """

    n_meshes = len(mesh_face_num)
    face_offsets = {}
    offset = 0
    for i in range(n_meshes):
        face_offsets[str(i)] = offset
        offset += mesh_face_num[str(i)]

    all_vertices = []
    for mid in mask_ids:
        mid_str = str(mid)
        if mid_str not in masks_dict:
            continue
        for mesh_idx_str, face_indices in masks_dict[mid_str].items():
            if mesh_idx_str not in face_offsets:
                continue
            global_indices = np.array(face_indices) + face_offsets[mesh_idx_str]
            valid = global_indices[global_indices < len(merged_mesh.faces)]
            if len(valid) == 0:
                continue
            face_verts = merged_mesh.vertices[merged_mesh.faces[valid].flatten()]
            all_vertices.append(face_verts)

    if not all_vertices:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    all_vertices = np.vstack(all_vertices)
    mins = np.min(all_vertices, axis=0)
    maxs = np.max(all_vertices, axis=0)
    return [round(float(v), 3) for v in [mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2]]]


def process_single_row(row, glb_dir, max_parts, depth_mode="all", prompt_mode="grounding"):
    """
    Process a single PartNeXt annotation row into training samples.

    Args:
        depth_mode: Controls which depth levels to generate samples for.
            - "all": all valid depth levels (default)
            - "leaf": only the maximum depth (leaf-level parts)
            - "coarse": only the first depth with >1 parts
            - "coarse_and_fine": coarse (first >1 parts) + fine (leaf)
            - comma-separated ints, e.g. "1,3": only those specific depths
        prompt_mode: Controls the user prompt style.
            - "grounding": provide part names in user prompt (default)
            - "open": open-ended segmentation, no part names given
            - "promptable": placeholder for point-prompt segmentation
            - "both": alias for "all"
            - "all": generate grounding, open, and promptable samples per depth

    Returns a list of sample dicts, one per valid depth level.
    """
    model_id = row["model_id"]
    type_id = row["type_id"]
    glb_path = os.path.join(glb_dir, type_id, model_id + ".glb")

    if not os.path.exists(glb_path):
        return []

    masks_dict = ast.literal_eval(row["masks"])
    mesh_face_num = ast.literal_eval(row["mesh_face_num"])
    hierarchy_list = ast.literal_eval(row["hierarchyList"])

    if not hierarchy_list:
        return []

    root = hierarchy_list[0]
    max_d = get_max_depth(root)



    depth_to_level = {}
    if depth_mode == "all":
        target_depths = list(range(1, max_d + 1))
        for d in target_depths:
            depth_to_level[d] = str(d)
    elif depth_mode == "leaf":
        target_depths = [max_d]
        depth_to_level[max_d] = "fine"
    elif depth_mode == "coarse":
        target_depths = []
        for d in range(1, max_d + 1):
            parts = collect_nodes_at_depth(root, 0, d)
            if len(parts) > 1:
                target_depths = [d]
                depth_to_level[d] = "coarse"
                break
    elif depth_mode == "coarse_and_fine":
        coarse_depth = None
        for d in range(1, max_d + 1):
            parts = collect_nodes_at_depth(root, 0, d)
            if len(parts) > 1:
                coarse_depth = d
                break
        if coarse_depth is not None and coarse_depth != max_d:
            target_depths = [coarse_depth, max_d]
            depth_to_level[coarse_depth] = "coarse"
            depth_to_level[max_d] = "fine"
        elif coarse_depth is not None:

            target_depths = [coarse_depth]
            depth_to_level[coarse_depth] = "fine"
        else:
            target_depths = [max_d]
            depth_to_level[max_d] = "fine"
    else:

        target_depths = [int(d.strip()) for d in depth_mode.split(",")]
        target_depths = [d for d in target_depths if 1 <= d <= max_d]
        for d in target_depths:
            depth_to_level[d] = str(d)

    if not target_depths:
        return []


    try:
        scene = trimesh.load(glb_path, force="scene")
        mesh_list = scene2meshes(scene)
        if not mesh_list:
            return []
        mesh_list = normalize_meshes_diag(mesh_list, norm_diag_len=1.0)
    except Exception as e:
        logger.warning(f"Failed to load GLB {glb_path}: {e}")
        return []


    mesh_list = reorder_meshes_by_face_num(mesh_list, mesh_face_num)


    merged_mesh = trimesh.util.concatenate(mesh_list) if len(mesh_list) > 1 else mesh_list[0]


    valid_depths_info = []
    prev_mask_id_sets = None

    for depth in target_depths:
        parts = collect_nodes_at_depth(root, 0, depth)

        if len(parts) <= 1:
            continue
        if max_parts is not None and len(parts) > max_parts:
            continue

        current_mask_id_sets = frozenset(frozenset(p["mask_ids"]) for p in parts)
        if current_mask_id_sets == prev_mask_id_sets:
            continue
        prev_mask_id_sets = current_mask_id_sets

        parts_with_bbox = []
        for p in parts:
            bbox = compute_bbox_from_faces(merged_mesh, mesh_face_num, masks_dict, p["mask_ids"])
            cx = (bbox[0] + bbox[3]) / 2
            cy = (bbox[1] + bbox[4]) / 2
            cz = (bbox[2] + bbox[5]) / 2
            parts_with_bbox.append((p, bbox, (cx, cy, cz)))
        parts_with_bbox.sort(key=lambda x: x[2])

        valid_depths_info.append((depth, parts, parts_with_bbox))

    if not valid_depths_info:
        return []

    first_valid_depth = valid_depths_info[0][0]
    last_valid_depth = valid_depths_info[-1][0]


    samples = []

    for depth, parts, parts_with_bbox in valid_depths_info:
        unique_labels = list(dict.fromkeys(p["name"] for p, _, _ in parts_with_bbox))

        assistant_lines = []
        target_part_names = []
        target_mask_ids = []
        target_bboxes = []
        for p, bbox, _ in parts_with_bbox:
            bbox_list = [coord_to_int(float(v)) for v in bbox]
            bbox_json = json.dumps({"bbox_aabb": bbox_list, "label": p["name"]})
            line = f"{bbox_json}<|SEG|>"
            assistant_lines.append(line)
            target_part_names.append(p["name"])
            target_mask_ids.append(p["mask_ids"])
            target_bboxes.append(bbox)

        assistant_text = "\n".join(assistant_lines)
        assistant_text += "\n<|BG|>"

        num_parts = len(parts)
        granularity = classify_granularity(depth, first_valid_depth, last_valid_depth, num_parts)

        effective_mode = "all" if prompt_mode == "both" else prompt_mode
        prompts_to_generate = []
        if effective_mode in ("grounding", "all"):
            prompts_to_generate.append(
                ("grounding", f"Please segment all the {'; '.join(unique_labels)} in <point_cloud>.")
            )
        if effective_mode in ("open", "all"):
            prompts_to_generate.append(
                ("open", "Please segment all parts of this 3D object in <point_cloud>.")
            )

        for pmode, user_content in prompts_to_generate:
            sample = {
                "model_id": model_id,
                "type_id": type_id,
                "depth": depth,
                "depth_level": depth_to_level.get(depth, str(depth)),
                "prompt_mode": pmode,
                "glb_path": glb_path,
                "masks_json": row["masks"],
                "mesh_face_num_json": row["mesh_face_num"],
                "target_part_names": target_part_names,
                "target_mask_ids": target_mask_ids,
                "target_bboxes": target_bboxes,
                "object_category": normalize_part_name(root["name"]) if hierarchy_list else None,
                "point_cloud": [glb_path],
                "vertex": [glb_path],
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_text},
                ],
                "diffusion_messages": {
                    "x": "<vertex>",
                    "contexts": {"main": "<messages>"},
                    "targets": "<vertex_label>",
                    "targets_type": "classification",
                },
            }
            sample["granularity"] = granularity if pmode == "open" else None
            sample["object_category"] = normalize_part_name(root["name"]) if hierarchy_list else None
            samples.append(sample)

        if effective_mode in ("promptable", "all"):
            sample = {
                "model_id": model_id,
                "type_id": type_id,
                "depth": depth,
                "depth_level": depth_to_level.get(depth, str(depth)),
                "prompt_mode": "promptable",
                "glb_path": glb_path,
                "masks_json": row["masks"],
                "mesh_face_num_json": row["mesh_face_num"],
                "target_part_names": target_part_names,
                "target_mask_ids": target_mask_ids,
                "target_bboxes": target_bboxes,
                "object_category": normalize_part_name(root["name"]) if hierarchy_list else None,
                "point_cloud": [glb_path],
                "vertex": [glb_path],
                "messages": [
                    {"role": "user", "content": "<point_cloud>"},
                    {"role": "assistant", "content": ""},
                ],
                "diffusion_messages": {
                    "x": "<vertex>",
                    "contexts": {"main": "<messages>"},
                    "targets": "<vertex_label>",
                    "targets_type": "classification",
                },
                "granularity": None,
                "object_category": normalize_part_name(root["name"]) if hierarchy_list else None,
            }
            samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Preprocess PartNeXt dataset to parquet")
    parser.add_argument(
        "--annotation_dir",
        required=True,
        help="Path to PartNeXt HF dataset directory",
    )
    parser.add_argument(
        "--glb_dir",
        required=True,
        help="Path to GLB files directory",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for parquet files",
    )
    parser.add_argument("--max_parts", type=int, default=64, help="Max parts per sample")
    parser.add_argument(
        "--depth_mode",
        default="all",
        help='Depth selection: "all", "leaf", "coarse", "coarse_and_fine", or comma-separated ints',
    )
    parser.add_argument(
        "--prompt_mode",
        default="grounding",
        choices=["grounding", "open", "both", "promptable", "all"],
        help='Prompt style: "grounding" (part names in prompt), "open" (no names), "both"/"all" (all types), "promptable" (point prompt)',
    )
    parser.add_argument("--num_proc", type=int, default=32, help="Number of parallel workers")
    parser.add_argument("--test_size", type=float, default=0.05, help="Test split ratio (stratified by type_id)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)


    logger.info(f"Loading annotations from {args.annotation_dir}")
    annotation = datasets.load_from_disk(args.annotation_dir)
    logger.info(f"Loaded {len(annotation)} annotations")



    from collections import defaultdict

    split_file = os.path.join(args.output_dir, "split.json")
    if os.path.exists(split_file):
        with open(split_file) as f:
            split_info = json.load(f)
        test_model_ids = set(split_info["test_model_ids"])
        train_model_ids = set(split_info["train_model_ids"])
        logger.info(
            f"Loaded existing split from {split_file}: "
            f"{len(train_model_ids)} train, {len(test_model_ids)} test"
        )
    else:

        type_to_models = defaultdict(list)
        for i in range(len(annotation)):
            type_to_models[annotation[i]["type_id"]].append(annotation[i]["model_id"])

        type_to_models = {tid: sorted(set(mids)) for tid, mids in type_to_models.items()}

        test_model_ids = set()
        train_model_ids = set()
        rng = random.Random(args.seed)
        for tid in sorted(type_to_models.keys()):
            mids = list(type_to_models[tid])
            rng.shuffle(mids)
            n_test = max(1, int(len(mids) * args.test_size))
            test_model_ids.update(mids[:n_test])
            train_model_ids.update(mids[n_test:])


        os.makedirs(args.output_dir, exist_ok=True)
        split_info = {
            "test_size": args.test_size,
            "seed": args.seed,
            "num_type_ids": len(type_to_models),
            "train_model_ids": sorted(train_model_ids),
            "test_model_ids": sorted(test_model_ids),
        }
        with open(split_file, "w") as f:
            json.dump(split_info, f, indent=2)
        logger.info(
            f"Generated stratified split by {len(type_to_models)} type_ids: "
            f"{len(train_model_ids)} train, {len(test_model_ids)} test. "
            f"Saved to {split_file}"
        )

    rows = [annotation[i] for i in range(len(annotation))]
    all_model_ids = set(r["model_id"] for r in rows)
    known_model_ids = train_model_ids | test_model_ids
    new_model_ids = all_model_ids - known_model_ids
    if new_model_ids:
        logger.warning(
            f"Found {len(new_model_ids)} model_ids in annotation not present in split.json "
            f"(they will be skipped). Delete {split_file} and re-run to regenerate split."
        )
    train_rows = [r for r in rows if r["model_id"] in train_model_ids]
    test_rows = [r for r in rows if r["model_id"] in test_model_ids]


    for split_name, split_rows in [("train", train_rows), ("test", test_rows)]:
        logger.info(
            f"Processing {split_name}: {len(split_rows)} rows, "
            f"depth_mode={args.depth_mode}, prompt_mode={args.prompt_mode}"
        )

        all_samples = []
        skipped = 0
        with ProcessPoolExecutor(max_workers=args.num_proc) as executor:
            futures = {
                executor.submit(
                    process_single_row, row, args.glb_dir, args.max_parts, args.depth_mode, args.prompt_mode
                ): i
                for i, row in enumerate(split_rows)
            }
            done_count = 0
            for future in as_completed(futures, timeout=7200):
                done_count += 1
                try:
                    samples = future.result(timeout=120)
                    if not samples:
                        skipped += 1
                    all_samples.extend(samples)
                except TimeoutError:
                    skipped += 1
                    logger.warning(f"Timeout processing {split_name} row {futures[future]}, skipping")
                except Exception as e:
                    skipped += 1
                    logger.warning(f"Error processing {split_name} row {futures[future]}: {e}")
                if done_count % 1000 == 0:
                    logger.info(f"  {split_name}: {done_count}/{len(split_rows)} rows, {len(all_samples)} samples")

        logger.info(f"{split_name}: {len(all_samples)} samples, {skipped} skipped")

        if not all_samples:
            logger.warning(f"No samples for {split_name}, skipping")
            continue


        out_path = os.path.join(args.output_dir, f"{split_name}.parquet")
        ds = datasets.Dataset.from_list(all_samples)
        ds.to_parquet(out_path)
        logger.info(f"Saved {split_name} ({len(ds)} samples) to {out_path}")


        from collections import Counter, defaultdict

        sub_groups = defaultdict(list)
        for s in all_samples:
            key = (s.get("depth_level", "unknown"), s.get("prompt_mode", "grounding"))
            sub_groups[key].append(s)

        for (dlevel, pmode), group_samples in sorted(sub_groups.items()):
            sub_name = f"{split_name}_{dlevel}_{pmode}"
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
        granularity_counts = Counter(s.get("granularity", "N/A") for s in all_samples if s.get("prompt_mode") == "open")
        if granularity_counts:
            logger.info(f"  Granularity distribution (open only): {dict(granularity_counts)}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
