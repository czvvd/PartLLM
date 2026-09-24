"""
Preprocess the PartVerse dataset into parquet format for training.

Converts PartVerse objects (12K objects, 88K parts from Objaverse) into
parquet files matching the PartNeXt 16-column schema so the existing
PartNeXtPoint3DDataset loader works without modification.

Geometry & color source: textured_part_glbs/{uid}/{label}.glb assembled
into a single scene GLB.  Each part GLB = one label = one mesh (or mesh
group).  masks_json / mesh_face_num_json / bboxes are all derived from
this assembled geometry, so color and supervision share the same mesh.

face2label.json + info.json are used only for cross-validation (label
consistency, face count plausibility), not as the primary supervision.

Each object produces 3 samples (grounding, open, promptable).

Usage:
    python recipe/part_seg/tools/prepare_partverse_dataset.py \
        --data_root /path/to/PartVerse \
        --output_dir /path/to/PartVerse/parquet \
        --glb_dir /path/to/PartVerse/glbs \
        --overwrite_glb
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict

import datasets
import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mesh_utils import normalize_meshes_diag, normalize_part_name, scene2meshes

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")






def coord_to_int(coord: float) -> int:
    """Map coordinate from [-0.5, 0.5] to integer [0, 1000]."""
    return max(0, min(1000, int(round((coord + 0.5) * 1000))))






def extract_part_name(caption: str) -> str:
    """Extract a concise part name from a verbose text caption.

    Args:
        caption: raw caption string from text_captions.json

    Returns:
        Clean part name in Title Case (e.g. "Metal Handrail")
    """
    s = caption.strip().rstrip(".")
    s = re.sub(r"^(A|An|The)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^close-up(\s+view)?\s+of\s+(the\s+|a\s+|an\s+)?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^detailed\s+(view|3D\s+model)\s+of\s+(the\s+|a\s+|an\s+)?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^section\s+of\s+(the\s+|a\s+|an\s+)?", "", s, flags=re.IGNORECASE)
    s = re.sub(r",\s*(extracted|detached|highlighted|likely|specifically|which|showcasing|designed|opened)\b.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+(extracted|detached|attached)\s+from\b.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+isolated\s+from\b.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+worn\s+by\b.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+resembling\b.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+(from|of|for)\s+(the|a|an)\b.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+with\s+(a|an|the)\b.*$", "", s, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+component$", "", s, flags=re.IGNORECASE).strip()
    if cleaned:
        s = cleaned
    s = s.strip().title()
    s = re.sub(r"'S\b", "'s", s)
    return s






def classify_granularity_flat(num_parts: int) -> str:
    """Classify number of parts into coarse/medium/fine."""
    if num_parts <= 4:
        return "coarse"
    elif num_parts <= 9:
        return "medium"
    else:
        return "fine"






def compute_bbox_from_vertices(verts: np.ndarray) -> list:
    """Compute AABB [x_min, y_min, z_min, x_max, y_max, z_max] from vertex array."""
    if len(verts) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mins = np.min(verts, axis=0)
    maxs = np.max(verts, axis=0)
    return [round(float(v), 3) for v in [mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2]]]






def process_single_object(
    uid: str,
    data_root: str,
    glb_dir: str,
    text_captions: dict,
    max_parts: int,
    overwrite_glb: bool = False,
    caption_to_name: dict = None,
) -> list:
    """Process one PartVerse object into training samples.

    Geometry & color source: textured_part_glbs/{uid}/{label}.glb.
    Each part GLB is loaded with force="mesh" (merges sub-geometries),
    then assembled into a scene where each mesh = one label.  masks_json,
    mesh_face_num_json, and bboxes are all derived from this assembled
    geometry so that color and supervision share the same mesh.

    face2label.json is used only for label-set cross-validation.

    Args:
        uid: object UID
        data_root: path to PartVerse root
        glb_dir: directory to write output GLB files
        text_captions: {uid: {label_str: [short, long]}}
        max_parts: skip objects with more than this many parts (None = no limit)
        overwrite_glb: if True, re-export GLB even if it already exists

    Returns:
        List of sample dicts (3 per object: grounding, open, promptable).
        Empty list if the object should be skipped.
    """
    try:
        anno_dir = os.path.join(data_root, "anno_infos", uid)
        part_glb_dir = os.path.join(data_root, "textured_part_glbs", uid)

        if not os.path.isdir(part_glb_dir):
            return []


        f2l_path = os.path.join(anno_dir, f"{uid}_face2label.json")
        with open(f2l_path) as f:
            face2label = json.load(f)
        f2l_labels = sorted(set(face2label.values()))


        glb_files = [fn for fn in os.listdir(part_glb_dir) if fn.endswith(".glb")]
        glb_labels = sorted(int(fn.replace(".glb", "")) for fn in glb_files)


        valid_labels = sorted(set(f2l_labels) & set(glb_labels))
        num_parts = len(valid_labels)

        if num_parts < 2:
            return []
        if max_parts is not None and num_parts > max_parts:
            return []


        uid_captions = text_captions.get(uid, {})
        caption_labels = set(int(k) for k in uid_captions.keys())

        valid_labels = sorted(set(valid_labels) & caption_labels)
        if len(valid_labels) < 2:
            return []


        f2l_total = len(face2label)


        part_meshes = []
        for label in valid_labels:
            glb_path = os.path.join(part_glb_dir, f"{label}.glb")
            mesh = trimesh.load(glb_path, force="mesh")
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                logger.warning(f"{uid}: empty mesh for label {label}, skipping object")
                return []
            part_meshes.append((label, mesh))




        textured_face_total = sum(len(m.faces) for _, m in part_meshes)
        if textured_face_total != f2l_total:
            logger.info(
                f"{uid}: textured faces ({textured_face_total}) != "
                f"face2label faces ({f2l_total}); using textured_part_glbs as authoritative geometry"
            )



        assembled_scene = trimesh.Scene()
        for label, mesh in part_meshes:
            assembled_scene.add_geometry(mesh, geom_name=f"part_{label:04d}")


        assembled_glb_path = os.path.join(glb_dir, f"{uid}.glb")
        os.makedirs(glb_dir, exist_ok=True)
        if overwrite_glb or not os.path.exists(assembled_glb_path):
            assembled_scene.export(assembled_glb_path, file_type="glb")





        reloaded = trimesh.load(assembled_glb_path, force="scene")
        if not isinstance(reloaded, trimesh.Scene):
            logger.warning(f"{uid}: reloaded as {type(reloaded).__name__}, expected Scene, skipping")
            return []

        raw_meshes = scene2meshes(reloaded)




        geom_items = [(n, g) for n, g in reloaded.geometry.items() if isinstance(g, trimesh.Trimesh)]
        if len(geom_items) != len(raw_meshes):
            logger.warning(
                f"{uid}: Trimesh geometry count {len(geom_items)} != scene2meshes count {len(raw_meshes)}, skipping"
            )
            return []

        label_order = []
        for geom_name, _ in geom_items:
            match = re.match(r"part_(\d+)", geom_name)
            if not match:
                logger.warning(f"{uid}: unparseable geometry name '{geom_name}' after reload, skipping")
                return []
            label_order.append(int(match.group(1)))

        if set(label_order) != set(valid_labels):
            logger.warning(
                f"{uid}: label set mismatch after reload: "
                f"expected {sorted(valid_labels)}, got {sorted(label_order)}, skipping"
            )
            return []

        mesh_idx_for_label = {label: idx for idx, label in enumerate(label_order)}


        pre_export_counts = {label: len(mesh.faces) for label, mesh in part_meshes}
        for label in valid_labels:
            midx = mesh_idx_for_label[label]
            if pre_export_counts[label] != len(raw_meshes[midx].faces):
                logger.warning(
                    f"{uid}: face count changed for label {label}: "
                    f"{pre_export_counts[label]}→{len(raw_meshes[midx].faces)}, skipping"
                )
                return []


        mesh_face_num = {}
        for idx, mesh in enumerate(raw_meshes):
            mesh_face_num[str(idx)] = len(mesh.faces)


        norm_meshes = normalize_meshes_diag(raw_meshes, norm_diag_len=1.0)


        masks_dict = {}
        for label in valid_labels:
            midx = mesh_idx_for_label[label]
            n_faces = len(norm_meshes[midx].faces)
            masks_dict[str(label)] = {str(midx): list(range(n_faces))}

        masks_json = json.dumps(masks_dict)
        mesh_face_num_json = json.dumps(mesh_face_num)



        _JUNK_NAMES = {"component", "highlighted", "detached", "extracted", "detailed",
                       "featuring", "representing", "designed", "showcasing", "object", "part"}
        label_to_name = {}
        fallback_count = 0
        for label in valid_labels:
            short_cap = uid_captions[str(label)][0]
            if caption_to_name and short_cap in caption_to_name:
                name = normalize_part_name(caption_to_name[short_cap])
            else:
                name = normalize_part_name(extract_part_name(short_cap))
                fallback_count += 1

            if name.lower().strip() in _JUNK_NAMES:
                logger.debug(f"{uid}: skipping part {label} with junk name '{name}'")
                continue
            label_to_name[label] = name
        if fallback_count > 0:
            logger.debug(f"{uid}: {fallback_count}/{len(valid_labels)} parts used regex fallback")


        valid_labels = [l for l in valid_labels if l in label_to_name]
        if len(valid_labels) < 2:
            logger.debug(f"{uid}: <2 valid parts after name filtering, skipping")
            return []


        parts_with_bbox = []
        for label in valid_labels:
            midx = mesh_idx_for_label[label]
            verts = np.asarray(norm_meshes[midx].vertices, dtype=np.float32)
            bbox = compute_bbox_from_vertices(verts)
            cx = (bbox[0] + bbox[3]) / 2
            cy = (bbox[1] + bbox[4]) / 2
            cz = (bbox[2] + bbox[5]) / 2
            parts_with_bbox.append({
                "label": label,
                "name": label_to_name[label],
                "bbox": bbox,
                "center": (cx, cy, cz),
            })


        parts_with_bbox.sort(key=lambda p: p["center"])

        target_part_names = [p["name"] for p in parts_with_bbox]
        target_mask_ids = [[p["label"]] for p in parts_with_bbox]
        target_bboxes = [p["bbox"] for p in parts_with_bbox]


        assistant_lines = []
        for p in parts_with_bbox:
            bbox_ints = [coord_to_int(float(v)) for v in p["bbox"]]
            bbox_json = json.dumps({"bbox_aabb": bbox_ints, "label": p["name"]})
            assistant_lines.append(f"{bbox_json}<|SEG|>")
        assistant_text = "\n".join(assistant_lines) + "\n<|BG|>"

        unique_names = list(dict.fromkeys(target_part_names))
        granularity = classify_granularity_flat(len(valid_labels))

        base_sample = {
            "model_id": uid,
            "type_id": "partverse",
            "depth": 0,
            "depth_level": "fine",
            "object_category": None,
            "glb_path": assembled_glb_path,
            "masks_json": masks_json,
            "mesh_face_num_json": mesh_face_num_json,
            "target_part_names": target_part_names,
            "target_mask_ids": target_mask_ids,
            "target_bboxes": target_bboxes,
            "point_cloud": [assembled_glb_path],
            "vertex": [assembled_glb_path],
            "diffusion_messages": {
                "x": "<vertex>",
                "contexts": {"main": "<messages>"},
                "targets": "<vertex_label>",
                "targets_type": "classification",
            },
        }

        samples = []


        g = dict(base_sample)
        g["prompt_mode"] = "grounding"
        g["messages"] = [
            {"role": "user", "content": f"Please segment all the {'; '.join(unique_names)} in <point_cloud>."},
            {"role": "assistant", "content": assistant_text},
        ]
        g["granularity"] = None
        samples.append(g)


        o = dict(base_sample)
        o["prompt_mode"] = "open"
        o["messages"] = [
            {"role": "user", "content": "Please segment all parts of this 3D object in <point_cloud>."},
            {"role": "assistant", "content": assistant_text},
        ]
        o["granularity"] = granularity
        samples.append(o)


        p = dict(base_sample)
        p["prompt_mode"] = "promptable"
        p["messages"] = [
            {"role": "user", "content": "<point_cloud>"},
            {"role": "assistant", "content": ""},
        ]
        p["granularity"] = None
        samples.append(p)

        return samples

    except Exception as e:
        logger.warning(f"Error processing {uid}: {e}", exc_info=True)
        return []






def main():
    parser = argparse.ArgumentParser(description="Preprocess PartVerse dataset to parquet")
    parser.add_argument("--data_root", required=True, help="Extracted PartVerse directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for Parquet files")
    parser.add_argument("--glb_dir", required=True, help="Output directory for assembled GLB files")
    parser.add_argument("--max_parts", type=int, default=64, help="Max parts per sample (0 = no limit)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N objects")
    parser.add_argument("--num_proc", type=int, default=0, help="Parallel workers (0 = serial)")
    parser.add_argument("--test_size", type=float, default=0.05, help="Held-out object ratio")
    parser.add_argument("--overwrite_glb", action="store_true", help="Re-export GLBs even if they exist")
    parser.add_argument(
        "--caption_to_name_json",
        default=None,
        help="Path to caption_to_name.json (LLM-standardized part names). "
        "If provided, captions found in this mapping use the standardized name; "
        "otherwise falls back to regex extract_part_name().",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    max_parts = args.max_parts if args.max_parts > 0 else None

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.glb_dir, exist_ok=True)


    caption_to_name = None
    if args.caption_to_name_json and os.path.isfile(args.caption_to_name_json):
        with open(args.caption_to_name_json, encoding="utf-8") as f:
            caption_to_name = json.load(f)
        logger.info(f"Loaded {len(caption_to_name)} caption→name mappings from {args.caption_to_name_json}")


    captions_path = os.path.join(args.data_root, "text_captions.json")
    logger.info(f"Loading text_captions from {captions_path}")
    with open(captions_path) as f:
        text_captions = json.load(f)
    logger.info(f"Loaded captions for {len(text_captions)} objects")


    anno_dir = os.path.join(args.data_root, "anno_infos")
    uids = sorted(d for d in os.listdir(anno_dir) if os.path.isdir(os.path.join(anno_dir, d)))
    logger.info(f"Found {len(uids)} objects in anno_infos/")

    if args.limit is not None:
        uids = uids[: args.limit]
        logger.info(f"Limiting to first {args.limit} objects")

    all_samples = []
    skipped = 0

    if args.num_proc > 0:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.num_proc) as executor:
            futures = {
                executor.submit(
                    process_single_object, uid, args.data_root, args.glb_dir,
                    text_captions, max_parts,
                    overwrite_glb=args.overwrite_glb,
                    caption_to_name=caption_to_name,
                ): uid
                for uid in uids
            }
            done = 0
            for future in as_completed(futures):
                uid = futures[future]
                try:
                    samples = future.result()
                except Exception as e:
                    logger.warning(f"Worker error for {uid}: {e}")
                    samples = []
                if not samples:
                    skipped += 1
                all_samples.extend(samples)
                done += 1
                if done % 500 == 0 or (done % 50 == 0 and done <= 500):
                    logger.info(f"  {done}/{len(uids)} objects, {len(all_samples)} samples")
    else:
        for i, uid in enumerate(uids):
            samples = process_single_object(
                uid, args.data_root, args.glb_dir, text_captions, max_parts,
                overwrite_glb=args.overwrite_glb,
                caption_to_name=caption_to_name,
            )
            if not samples:
                skipped += 1
            all_samples.extend(samples)
            if (i + 1) % 500 == 0 or ((i + 1) % 50 == 0 and (i + 1) <= 500):
                logger.info(f"  {i + 1}/{len(uids)} objects, {len(all_samples)} samples")

    logger.info(f"Processed {len(uids)} objects: {len(all_samples)} samples, {skipped} skipped")

    if not all_samples:
        logger.warning("No samples produced, exiting")
        return


    object_ids = sorted({sample["model_id"] for sample in all_samples})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(object_ids)
    n_test = max(1, int(len(object_ids) * args.test_size))
    test_ids = set(object_ids[:n_test])
    split_samples = {
        "train": [sample for sample in all_samples if sample["model_id"] not in test_ids],
        "test": [sample for sample in all_samples if sample["model_id"] in test_ids],
    }
    with open(os.path.join(args.output_dir, "split.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": args.seed,
                "test_size": args.test_size,
                "train_model_ids": sorted(set(object_ids) - test_ids),
                "test_model_ids": sorted(test_ids),
            },
            handle,
            indent=2,
        )

    for split_name, samples in split_samples.items():
        out_path = os.path.join(args.output_dir, f"{split_name}.parquet")
        dataset = datasets.Dataset.from_list(samples)
        dataset.to_parquet(out_path)
        logger.info(f"Saved {len(dataset)} samples to {out_path}")
        if split_name != "train":
            continue
        sub_groups = defaultdict(list)
        for sample in samples:
            sub_groups[sample.get("prompt_mode", "grounding")].append(sample)
        for prompt_mode, group in sorted(sub_groups.items()):
            sub_path = os.path.join(args.output_dir, f"train_{prompt_mode}.parquet")
            datasets.Dataset.from_list(group).to_parquet(sub_path)
            logger.info(f"  Saved train_{prompt_mode}: {len(group)} samples -> {sub_path}")


    prompt_counts = Counter(s.get("prompt_mode") for s in all_samples)
    gran_counts = Counter(s.get("granularity") for s in all_samples if s.get("prompt_mode") == "open")
    part_counts = [len(s["target_part_names"]) for s in all_samples if s.get("prompt_mode") == "open"]
    logger.info(f"Prompt distribution: {dict(prompt_counts)}")
    logger.info(f"Granularity distribution (open only): {dict(gran_counts)}")
    if part_counts:
        logger.info(f"Part count stats: min={min(part_counts)}, max={max(part_counts)}, mean={sum(part_counts)/len(part_counts):.1f}")


    if caption_to_name is not None:
        all_names = []
        for s in all_samples:
            all_names.extend(s.get("target_part_names", []))
        unique_names = set(all_names)
        name_lengths = [len(n.split()) for n in all_names]
        logger.info(f"Part name stats: {len(all_names)} total, {len(unique_names)} unique")
        if name_lengths:
            logger.info(
                f"  Word count: mean={sum(name_lengths)/len(name_lengths):.1f}, "
                f"max={max(name_lengths)}"
            )
            long_names = [n for n in unique_names if len(n.split()) > 6]
            if long_names:
                logger.info(f"  ⚠ {len(long_names)} names > 6 words: {sorted(long_names)[:5]}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
