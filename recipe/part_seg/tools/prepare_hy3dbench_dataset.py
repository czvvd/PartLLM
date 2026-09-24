"""
Preprocess the HY3D-Bench dataset into parquet format for training.

HY3D-Bench (240K objects) provides watertight per-part meshes in NPZ archives.
This script:
  1. Loads per-part PLY meshes from mesh NPZ files
  2. Normalizes and computes per-part bboxes
  3. Builds KNN face-label mapping from full.ply faces to per-part labels (watertight)
  4. Generates two prompt modes: open (nosem) and promptable
  5. Outputs 16+4 column parquet (standard 16 + watertight_* fields)
  6. Dataset loader reads NPZ directly at training time (no GLB export)

Usage:
    python recipe/part_seg/tools/prepare_hy3dbench_dataset.py \\
        --data_root /path/to/HY3D-Bench/part \\
        --output_dir /path/to/HY3D-Bench/parquet \\
        --num_proc 16 --skip_watertight
"""

import argparse
import io
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import datasets
import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from mesh_utils import normalize_meshes_diag, scene2meshes
from prepare_partnext_dataset import coord_to_int

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")






def compute_bbox_from_vertices(verts: np.ndarray) -> list:
    """Compute AABB [x_min, y_min, z_min, x_max, y_max, z_max] from vertex array."""
    if len(verts) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mins = np.min(verts, axis=0)
    maxs = np.max(verts, axis=0)
    return [round(float(v), 3) for v in [mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2]]]






def build_watertight_face_labels(full_mesh, part_meshes, max_distance=0.01):
    """Map each face of full_mesh to a part label using KNN on face centroids.

    Args:
        full_mesh: trimesh.Trimesh (watertight full mesh, already normalized)
        part_meshes: list of trimesh.Trimesh (per-part meshes, already normalized)
        max_distance: reject mapping if nearest centroid is further than this

    Returns:
        face_labels: np.ndarray [N_faces] of int (part index), or None if mapping fails
        max_dist: float, maximum distance across all mappings
    """
    from scipy.spatial import cKDTree

    full_centroids = full_mesh.triangles_center

    all_part_centroids = []
    all_part_labels = []
    for i, m in enumerate(part_meshes):
        c = m.triangles_center
        all_part_centroids.append(c)
        all_part_labels.extend([i] * len(c))

    all_part_centroids = np.vstack(all_part_centroids)
    all_part_labels = np.array(all_part_labels)

    tree = cKDTree(all_part_centroids)
    dists, indices = tree.query(full_centroids, k=1)

    max_dist = float(dists.max())
    if max_dist > max_distance:
        return None, max_dist

    face_labels = all_part_labels[indices]
    return face_labels, max_dist


def build_watertight_masks_json(face_labels, num_parts):
    """Build masks_json for a single watertight mesh from face labels.

    Args:
        face_labels: [N_faces] int array, each value is a part index
        num_parts: total number of parts

    Returns:
        masks_json: str, e.g. {"0": {"0": [face_indices]}, "1": {"0": [...]}, ...}
        mesh_face_num_json: str, e.g. {"0": N_faces}
    """
    masks = {}
    for part_id in range(num_parts):
        face_indices = np.where(face_labels == part_id)[0].tolist()
        if face_indices:
            masks[str(part_id)] = {"0": face_indices}

    masks_json = json.dumps(masks)
    mesh_face_num_json = json.dumps({"0": len(face_labels)})
    return masks_json, mesh_face_num_json


def build_watertight_bboxes(full_mesh, face_labels, num_parts):
    """Compute per-part bboxes from watertight mesh using face labels.

    Args:
        full_mesh: normalized trimesh.Trimesh
        face_labels: [N_faces] int array
        num_parts: total number of parts

    Returns:
        list of [x_min, y_min, z_min, x_max, y_max, z_max] per part
    """
    bboxes = []
    for part_id in range(num_parts):
        part_face_mask = (face_labels == part_id)
        if not np.any(part_face_mask):
            bboxes.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        part_faces = full_mesh.faces[part_face_mask]
        part_vert_indices = np.unique(part_faces.flatten())
        part_verts = np.asarray(full_mesh.vertices[part_vert_indices], dtype=np.float32)
        bboxes.append(compute_bbox_from_vertices(part_verts))
    return bboxes






def process_single_object(
    uid: str,
    mesh_npz_path: str,
    max_parts: int,
    knn_max_distance: float = 0.01,
    skip_watertight: bool = False,
) -> list:
    """Process one HY3D-Bench object into training samples.

    Args:
        uid: object UID (filename without .npz)
        mesh_npz_path: path to mesh NPZ file
        max_parts: skip objects with more than this many parts
        knn_max_distance: max KNN distance for watertight face mapping

    Returns:
        List of sample dicts (2 per object: open nosem + promptable).
        Empty list if the object should be skipped.
    """
    try:
        data = np.load(mesh_npz_path, allow_pickle=True)
        keys = sorted(data.keys())


        part_keys = [k for k in keys if k.startswith("part_") and k.endswith(".ply")]
        num_parts = len(part_keys)

        if num_parts < 2:
            return []
        if max_parts is not None and num_parts > max_parts:
            return []


        sorted_part_keys = sorted(
            part_keys, key=lambda x: int(re.search(r"(\d+)", x).group())
        )


        part_meshes = []
        for pk in sorted_part_keys:
            m = trimesh.load(io.BytesIO(data[pk].tobytes()), file_type="ply", process=False)
            if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
                return []
            part_meshes.append(m)


        mesh_face_num = {str(i): len(m.faces) for i, m in enumerate(part_meshes)}


        norm_part_meshes = normalize_meshes_diag(part_meshes, norm_diag_len=1.0)






        masks_json = ""
        mesh_face_num_json = json.dumps(mesh_face_num)


        parts_with_info = []
        for i, m in enumerate(norm_part_meshes):
            verts = np.asarray(m.vertices, dtype=np.float32)
            bbox = compute_bbox_from_vertices(verts)
            cx = (bbox[0] + bbox[3]) / 2
            cy = (bbox[1] + bbox[4]) / 2
            cz = (bbox[2] + bbox[5]) / 2
            parts_with_info.append({
                "orig_idx": i,
                "name": "Part",
                "bbox": bbox,
                "center": (cx, cy, cz),
            })


        assembled_glb_path = mesh_npz_path


        parts_with_info.sort(key=lambda p: p["center"])

        target_part_names = [p["name"] for p in parts_with_info]
        target_mask_ids = [[p["orig_idx"]] for p in parts_with_info]
        target_bboxes = [p["bbox"] for p in parts_with_info]


        wt_glb_path = None
        wt_masks_json = None
        wt_mesh_face_num_json = None
        wt_target_bboxes = None

        if not skip_watertight and "full.ply" in keys:

            full_mesh_raw = trimesh.load(
                io.BytesIO(data["full.ply"].tobytes()), file_type="ply", process=False
            )
            if not isinstance(full_mesh_raw, trimesh.Trimesh) or len(full_mesh_raw.faces) == 0:
                full_mesh_raw = None

            if full_mesh_raw is not None:

                face_labels_raw, max_dist = build_watertight_face_labels(
                    full_mesh_raw, part_meshes, max_distance=knn_max_distance
                )


                if face_labels_raw is not None:
                    covered_parts = set(np.unique(face_labels_raw))
                    expected_parts = set(range(num_parts))
                    if covered_parts != expected_parts:
                        missing = expected_parts - covered_parts
                        logger.info(
                            f"{uid}: watertight KNN missing parts {missing}, skipping watertight"
                        )
                        face_labels_raw = None

                if face_labels_raw is not None:
                    norm_full_list = normalize_meshes_diag([full_mesh_raw], norm_diag_len=1.0)
                    norm_full = norm_full_list[0]

                    wt_glb_path = mesh_npz_path + "::full"

                    wt_masks_json, wt_mesh_face_num_json = build_watertight_masks_json(
                        face_labels_raw, num_parts
                    )

                    raw_wt_bboxes = build_watertight_bboxes(norm_full, face_labels_raw, num_parts)
                    wt_target_bboxes = [raw_wt_bboxes[p["orig_idx"]] for p in parts_with_info]
                else:
                    if max_dist > knn_max_distance:
                        logger.info(f"{uid}: KNN max_dist={max_dist:.4f} > {knn_max_distance}, skipping watertight")


        assistant_lines = []
        for p in parts_with_info:
            bbox_ints = [coord_to_int(float(v)) for v in p["bbox"]]
            bbox_json = json.dumps({"bbox_aabb": bbox_ints, "label": p["name"]})
            assistant_lines.append(f"{bbox_json}<|SEG|>")
        assistant_text = "\n".join(assistant_lines) + "\n<|BG|>"


        base_sample = {
            "model_id": uid,
            "type_id": "hy3dbench",
            "depth": 0,
            "depth_level": "nosem",
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

            "object_category": None,

            "watertight_mesh_path": wt_glb_path,
            "watertight_masks_json": wt_masks_json,
            "watertight_mesh_face_num_json": wt_mesh_face_num_json,
            "watertight_target_bboxes": wt_target_bboxes,
        }

        samples = []


        o = dict(base_sample)
        o["prompt_mode"] = "open"
        o["messages"] = [
            {"role": "user", "content": "Please segment this 3D object into distinct parts in <point_cloud>."},
            {"role": "assistant", "content": assistant_text},
        ]
        o["granularity"] = "nosem"
        samples.append(o)


        p = dict(base_sample)
        p["prompt_mode"] = "promptable"
        p["messages"] = [
            {"role": "user", "content": "<point_cloud>"},
            {"role": "assistant", "content": ""},
        ]
        p["granularity"] = "nosem"
        samples.append(p)

        return samples

    except Exception as e:
        logger.warning(f"Error processing {uid}: {e}", exc_info=True)
        return []


def process_chunk_to_shard(
    chunk: list,
    shard_path: str,
    max_parts,
    knn_max_distance: float,
    skip_watertight: bool,
) -> dict:
    """Process a chunk of NPZ files and write results directly to a shard parquet.

    This runs inside a worker process. The worker writes the parquet file itself
    so the main process never needs to hold all samples in memory.

    If the shard already exists (from a previous run), skip processing entirely —
    just read row count and return stats. This enables resumable runs.

    Args:
        chunk: list of (uid, npz_path) tuples
        shard_path: output parquet path for this chunk
        max_parts: skip objects with > max_parts parts
        knn_max_distance: KNN threshold for watertight mapping
        skip_watertight: skip watertight KNN mapping

    Returns:
        dict with stats: {'shard_path', 'num_samples', 'num_objects_done',
                          'skipped', 'wt_count', 'prompt_counts', 'resumed'}
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from collections import Counter


    if os.path.isfile(shard_path):
        try:
            md = pq.read_metadata(shard_path)
            if md.num_rows > 0:

                t = pq.read_table(shard_path, columns=["prompt_mode", "watertight_mesh_path"])
                pmodes = t["prompt_mode"].to_pylist()
                wts = t["watertight_mesh_path"].to_pylist()
                pc = Counter(pmodes)

                wt_cnt = sum(1 for pm, wt in zip(pmodes, wts) if pm == "open" and wt)
                return {
                    "shard_path": shard_path,
                    "num_samples": md.num_rows,
                    "num_objects_done": len(chunk),
                    "skipped": 0,
                    "wt_count": wt_cnt,
                    "prompt_counts": dict(pc),
                    "resumed": True,
                }
        except Exception:

            pass

    samples = []
    skipped = 0
    wt_count = 0
    done = 0
    prompt_counts = Counter()

    for uid, path in chunk:
        objs = process_single_object(
            uid, path, max_parts, knn_max_distance, skip_watertight,
        )
        done += 1
        if not objs:
            skipped += 1
        else:
            if objs[0].get("watertight_mesh_path"):
                wt_count += 1
            for s in objs:
                prompt_counts[s.get("prompt_mode")] += 1
            samples.extend(objs)


    num_samples = 0
    if samples:
        try:
            table = pa.Table.from_pylist(samples)
            pq.write_table(table, shard_path)
            num_samples = table.num_rows
        except Exception as e:

            return {
                "shard_path": None,
                "num_samples": 0,
                "num_objects_done": done,
                "skipped": skipped,
                "wt_count": wt_count,
                "prompt_counts": dict(prompt_counts),
                "error": f"{type(e).__name__}: {e}",
            }

        del table
        del samples

    return {
        "shard_path": shard_path if num_samples > 0 else None,
        "num_samples": num_samples,
        "num_objects_done": done,
        "skipped": skipped,
        "wt_count": wt_count,
        "prompt_counts": dict(prompt_counts),
    }


def _process_chunk_star(args_tuple):
    """Helper: unpack tuple args for multiprocessing.Pool.imap_unordered."""
    return process_chunk_to_shard(*args_tuple)






def discover_mesh_npz_files(data_root: str) -> list:
    """Find all mesh NPZ files under data_root/meshes/."""
    meshes_dir = os.path.join(data_root, "meshes")
    if not os.path.isdir(meshes_dir):
        logger.error(f"Meshes directory not found: {meshes_dir}")
        return []

    npz_files = []
    for shard in sorted(os.listdir(meshes_dir)):
        shard_dir = os.path.join(meshes_dir, shard)
        if not os.path.isdir(shard_dir) or shard.endswith(".tar.gz"):
            continue
        for fname in os.listdir(shard_dir):
            if fname.endswith(".npz"):
                uid = fname[:-4]
                npz_files.append((uid, os.path.join(shard_dir, fname)))

    return npz_files






def main():
    parser = argparse.ArgumentParser(description="Preprocess HY3D-Bench dataset to parquet")
    parser.add_argument("--data_root", required=True, help="Extracted HY3D-Bench part directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for Parquet files")
    parser.add_argument("--max_parts", type=int, default=64, help="Max parts per sample (0 = no limit)")
    parser.add_argument("--knn_max_distance", type=float, default=0.01, help="Max KNN distance for watertight mapping")
    parser.add_argument("--skip_watertight", action="store_true", help="Skip watertight KNN mapping (much faster, no watertight augmentation)")
    parser.add_argument("--checkpoint_interval", type=int, default=10000, help="Save incremental parquet every N objects (0 = disable, serial mode only)")
    parser.add_argument("--chunk_size", type=int, default=50, help="Objects per worker chunk (parallel mode). Smaller = lower per-worker memory peak.")
    parser.add_argument("--skip_pass2", action="store_true", help="Skip Pass 2 (train/test split write). Useful when you want to inspect shards before committing to a split.")
    parser.add_argument("--only_pass2", action="store_true", help="Skip Pass 0 (NPZ discovery + chunk processing) and go straight to Pass 1+Pass 2. Requires shards/ directory already populated.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N objects")
    parser.add_argument("--num_proc", type=int, default=0, help="Parallel workers (0 = serial)")
    parser.add_argument("--test_size", type=float, default=0.05, help="Test split ratio")
    args = parser.parse_args()

    np.random.seed(args.seed)
    max_parts = args.max_parts if args.max_parts > 0 else None

    os.makedirs(args.output_dir, exist_ok=True)


    if args.only_pass2:
        import pyarrow as pa
        import pyarrow.parquet as pq
        shard_dir = os.path.join(args.output_dir, "shards")
        if not os.path.isdir(shard_dir):
            logger.error(f"--only_pass2 requires {shard_dir} to exist")
            return
        logger.info(f"--only_pass2 mode: reading shards from {shard_dir}, skipping NPZ discovery and chunk processing")
        _run_pass1_and_pass2(args, shard_dir)
        return


    npz_files = discover_mesh_npz_files(args.data_root)
    logger.info(f"Found {len(npz_files)} mesh NPZ files")

    if args.limit is not None:
        npz_files = npz_files[: args.limit]
        logger.info(f"Limiting to first {args.limit} objects")

    all_samples = []
    skipped = 0
    wt_count = 0
    checkpoint_interval = args.checkpoint_interval



    shard_dir = os.path.join(args.output_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    shard_counter = [0]

    def flush_shard(samples):
        """Write current samples to a new shard file, then clear the buffer.

        Returns the shard path (for tracking) or None if no samples.
        """
        if not samples:
            return None
        import pyarrow as pa
        import pyarrow.parquet as pq
        import traceback
        try:
            idx = shard_counter[0]
            shard_path = os.path.join(shard_dir, f"shard_{idx:05d}.parquet")
            logger.info(f"  Flushing {len(samples)} samples → {shard_path}")
            table = pa.Table.from_pylist(samples)
            pq.write_table(table, shard_path)
            logger.info(f"  Flushed shard_{idx:05d} ({table.num_rows} rows, {table.num_columns} cols)")
            shard_counter[0] += 1
            return shard_path
        except Exception as e:
            logger.error(f"  FLUSH FAILED: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            return None

    def save_checkpoint(samples, output_dir, label="checkpoint"):
        """Checkpoint: flush current buffer samples to a shard file and clear the buffer in-place."""
        if not samples:
            return
        path = flush_shard(samples)
        if path is not None:
            samples.clear()
            logger.info(f"  [{label}] Buffer cleared, shard persisted")

    if args.num_proc > 0:







        chunk_size = args.chunk_size
        chunks = [
            npz_files[i : i + chunk_size] for i in range(0, len(npz_files), chunk_size)
        ]

        chunk_args = []
        for ci, chunk in enumerate(chunks):
            shard_path = os.path.join(shard_dir, f"shard_{ci:05d}.parquet")
            chunk_args.append((chunk, shard_path))

        logger.info(f"Split {len(npz_files)} objects into {len(chunks)} chunks of ~{chunk_size}")

        done = 0





        def _worker_args_gen():
            for chunk, shard_path in chunk_args:
                yield (chunk, shard_path, max_parts, args.knn_max_distance, args.skip_watertight)

        with mp.Pool(processes=args.num_proc, maxtasksperchild=5) as pool:
            for stats in pool.imap_unordered(_process_chunk_star, _worker_args_gen()):
                shard_path = stats.get("shard_path") or "(none)"
                if stats.get("error"):
                    logger.error(f"  Chunk error: {stats['error']}")

                done += stats.get("num_objects_done", 0)
                skipped += stats.get("skipped", 0)
                wt_count += stats.get("wt_count", 0)

                written = stats.get("shard_path")
                if written:
                    resumed_flag = " [RESUMED]" if stats.get("resumed") else ""
                    logger.info(
                        f"  done {done}/{len(npz_files)} | shard written: {os.path.basename(written)}{resumed_flag} "
                        f"({stats['num_samples']} samples) | total_skipped={skipped}, wt={wt_count}"
                    )
                else:
                    logger.info(
                        f"  done {done}/{len(npz_files)} | chunk had no samples | "
                        f"total_skipped={skipped}, wt={wt_count}"
                    )
    else:

        for i, (uid, path) in enumerate(npz_files):
            samples = process_single_object(
                uid, path, max_parts, args.knn_max_distance,
                args.skip_watertight,
            )
            if not samples:
                skipped += 1
            else:
                if samples[0].get("watertight_mesh_path"):
                    wt_count += 1
                all_samples.extend(samples)
            if (i + 1) % 500 == 0 or (i + 1) == len(npz_files):
                logger.info(f"  {i + 1}/{len(npz_files)} objects, {len(all_samples)} samples, {skipped} skipped, {wt_count} watertight")
            if checkpoint_interval > 0 and (i + 1) % checkpoint_interval == 0:
                save_checkpoint(all_samples, args.output_dir, label=f"{i+1}/{len(npz_files)}")

    logger.info(
        f"Processed {len(npz_files)} objects: {skipped} skipped, {wt_count} with watertight"
    )


    if all_samples:
        flush_shard(all_samples)
        all_samples.clear()

    import pyarrow as pa
    import pyarrow.parquet as pq


    shard_files = sorted(
        os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith(".parquet")
    )
    logger.info(f"Total shards written: {len(shard_files)}")

    if not shard_files:
        logger.warning("No shards produced, exiting")
        return


    logger.info("Pass 1: collecting model_ids from shards...")
    all_uids = set()
    total_rows = 0
    for sp in shard_files:
        t = pq.read_table(sp, columns=["model_id"])
        all_uids.update(t["model_id"].to_pylist())
        total_rows += t.num_rows
    logger.info(f"  Total: {total_rows} rows, {len(all_uids)} unique model_ids")

    if args.skip_pass2:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Pass 2 SKIPPED (--skip_pass2 set).")
        logger.info(f"All {len(shard_files)} shards are ready at: {shard_dir}")
        logger.info(f"Total {total_rows} samples across {len(all_uids)} unique model_ids.")
        logger.info("Re-run without --skip_pass2 to produce train/test split.")
        logger.info("=" * 60)
        return


    uids_list = sorted(all_uids)
    np.random.shuffle(uids_list)
    n_test = max(1, int(len(uids_list) * args.test_size))
    test_uids = set(uids_list[:n_test])
    train_uids = set(uids_list[n_test:])
    logger.info(f"  train uids: {len(train_uids)}, test uids: {len(test_uids)}")


    logger.info("Pass 2: streaming shards into train/test/per-mode parquet files...")
    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    per_mode_paths = {
        "open": os.path.join(args.output_dir, "train_open.parquet"),
        "promptable": os.path.join(args.output_dir, "train_promptable.parquet"),
    }

    train_writer = None
    test_writer = None
    mode_writers = {}

    prompt_counts = Counter()
    wt_available = 0
    part_counts = []
    train_rows = 0
    test_rows = 0
    mode_rows = Counter()

    def _cast_large_strings(table):
        """Cast all string columns to large_string to avoid 2GB offset overflow
        when `take()` accumulates rows with very long string values (e.g. masks_json).
        """
        import pyarrow.types as pat
        new_fields = []
        need_cast = False
        for f in table.schema:
            if pat.is_string(f.type):
                new_fields.append(f.with_type(pa.large_string()))
                need_cast = True
            else:
                new_fields.append(f)
        if not need_cast:
            return table
        new_schema = pa.schema(new_fields)
        return table.cast(new_schema)

    for sp in shard_files:
        t = pq.read_table(sp)

        t = _cast_large_strings(t)
        mid_list = t["model_id"].to_pylist()
        pmode_list = t["prompt_mode"].to_pylist()
        wt_list = t["watertight_mesh_path"].to_pylist()
        part_names_list = t["target_part_names"].to_pylist()

        train_mask = [m in train_uids for m in mid_list]

        for pm, wt, pnames in zip(pmode_list, wt_list, part_names_list):
            prompt_counts[pm] += 1
            if pm == "open":
                part_counts.append(len(pnames) if pnames else 0)
                if wt:
                    wt_available += 1

        train_idx = [i for i, m in enumerate(train_mask) if m]
        test_idx = [i for i, m in enumerate(train_mask) if not m]

        if train_idx:
            train_sub = t.take(train_idx)
            if train_writer is None:
                train_writer = pq.ParquetWriter(train_path, train_sub.schema)
            train_writer.write_table(train_sub)
            train_rows += train_sub.num_rows

            sub_pmodes = train_sub["prompt_mode"].to_pylist()
            for pmode, mode_path in per_mode_paths.items():
                mode_idx = [i for i, p in enumerate(sub_pmodes) if p == pmode]
                if mode_idx:
                    mode_sub = train_sub.take(mode_idx)
                    if pmode not in mode_writers:
                        mode_writers[pmode] = pq.ParquetWriter(mode_path, mode_sub.schema)
                    mode_writers[pmode].write_table(mode_sub)
                    mode_rows[pmode] += mode_sub.num_rows

        if test_idx:
            test_sub = t.take(test_idx)
            if test_writer is None:
                test_writer = pq.ParquetWriter(test_path, test_sub.schema)
            test_writer.write_table(test_sub)
            test_rows += test_sub.num_rows

        del t

    if train_writer is not None:
        train_writer.close()
        logger.info(f"Saved train ({train_rows} samples) to {train_path}")
    if test_writer is not None:
        test_writer.close()
        logger.info(f"Saved test ({test_rows} samples) to {test_path}")
    for pmode, w in mode_writers.items():
        w.close()
        logger.info(f"  Saved train_{pmode}: {mode_rows[pmode]} samples -> {per_mode_paths[pmode]}")

    logger.info(f"Prompt distribution: {dict(prompt_counts)}")
    logger.info(f"Watertight available (open samples): {wt_available}/{prompt_counts.get('open', 0)}")
    if part_counts:
        logger.info(
            f"Part count stats: min={min(part_counts)}, max={max(part_counts)}, "
            f"mean={sum(part_counts) / len(part_counts):.1f}"
        )
    logger.info(f"Done! Shards kept at {shard_dir} (can be deleted to save space).")


if __name__ == "__main__":
    main()
