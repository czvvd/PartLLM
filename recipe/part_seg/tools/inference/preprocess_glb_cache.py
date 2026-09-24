#!/usr/bin/env python3
"""Pre-compute mesh -> sampled point cloud cache for high-poly meshes.

For high-face-count FBX, GLB, and OBJ meshes:
- load FBX with assimp_py and other formats with trimesh
- merge duplicate vertices produced by assimp
- apply normalize_meshes_diag(1.0) to match training
- sample_surface(pc_size) + face_idx + face_normals[face_idx]

Outputs one cache per mesh:
    <cache_dir>/<model_id>/
        geom.npz: sampled_points, face_idx, sampled_normals,
                  sampled_rgb (uint8 RGB), sampled_sharpedge (zeros)
        combined_mesh.ply: normalized combined mesh in binary trimesh PLY format,
                            reused by _color_mesh_by_logits during inference

When reading the cache, inference only needs to:
    1) load geom.npz and assemble the obj_surface tensor
    2) run utonia.transform and collate_fn on the GPU

Parallelism uses multiprocessing.Pool with nprocs=min(cpu_count, 32) by default.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np
import trimesh

SUPPORTED_EXT = (".glb", ".gltf", ".ply", ".obj", ".stl", ".off", ".fbx")


def _load_fbx_assimp(mesh_path):
    """Load an FBX file with assimp_py and return a trimesh.Trimesh.

    assimp_py returns flattened vertices and indices, which are reshaped to (-1, 3).
    Process_PreTransformVertices bakes skeletal and hierarchical transforms into the
    vertices to avoid misaligned multi-mesh transforms. JoinIdenticalVertices is not
    used because trimesh.merge_vertices(digits_vertex=5) provides better control and
    preserves the one-to-one face-index mapping.
    """
    import assimp_py
    flags = assimp_py.Process_Triangulate | assimp_py.Process_PreTransformVertices
    scene = assimp_py.import_file(mesh_path, flags)
    if scene is None or not scene.meshes:
        raise RuntimeError(f"assimp returned empty scene: {mesh_path}")
    ml = []
    for mesh in scene.meshes:
        v = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
        idx = np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3)
        if idx.shape[0] == 0:
            continue
        
        vc = None
        if hasattr(mesh, "colors") and mesh.colors:
            try:
                c0 = np.asarray(mesh.colors[0], dtype=np.float32)
                if c0.size == v.shape[0] * 4:
                    vc = np.clip(c0.reshape(-1, 4)[:, :3] * 255, 0, 255).astype(np.uint8)
                elif c0.size == v.shape[0] * 3:
                    vc = np.clip(c0.reshape(-1, 3) * 255, 0, 255).astype(np.uint8)
            except Exception:
                vc = None
        tm = trimesh.Trimesh(vertices=v, faces=idx, process=False)
        if vc is not None and len(vc) == len(v):
            tm.visual.vertex_colors = np.concatenate(
                [vc, np.full((len(vc), 1), 255, dtype=np.uint8)], axis=-1
            )
        ml.append(tm)
    if not ml:
        raise RuntimeError(f"no non-empty mesh: {mesh_path}")
    return trimesh.util.concatenate(ml) if len(ml) > 1 else ml[0]


def _load_mesh_any(mesh_path):
    """Load FBX with assimp_py and all other supported formats with trimesh."""
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".fbx":
        return _load_fbx_assimp(mesh_path)
    return trimesh.load(mesh_path)


def _to_mesh_list(loaded):
    """Convert a trimesh.Scene or Trimesh to a mesh list without flattening transforms."""
    if isinstance(loaded, trimesh.Scene):
        
        sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
        from mesh_utils import scene2meshes
        return scene2meshes(loaded)
    if isinstance(loaded, trimesh.Trimesh):
        return [loaded]
    raise ValueError(f"Unsupported mesh type: {type(loaded)}")


def _normalize_and_combine(mesh_list):
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from mesh_utils import normalize_meshes_diag
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
    combined = trimesh.util.concatenate(mesh_list) if len(mesh_list) > 1 else mesh_list[0]
    return combined


def preprocess_one(args_tuple):
    """Cache one mesh and return (model_id, ok, message, timing_dict)."""
    mesh_path, cache_dir, pc_size, merge_digits, sample_seed = args_tuple
    model_id = os.path.splitext(os.path.basename(mesh_path))[0]
    sub_dir = os.path.join(cache_dir, model_id)
    os.makedirs(sub_dir, exist_ok=True)
    geom_path = os.path.join(sub_dir, "geom.npz")
    ply_path = os.path.join(sub_dir, "combined_mesh.ply")
    done_marker = os.path.join(sub_dir, ".done")
    if os.path.exists(done_marker) and os.path.exists(geom_path) and os.path.exists(ply_path):
        return (model_id, True, "skip (already cached)", {})

    t = {}
    try:
        t0 = time.time()
        loaded = _load_mesh_any(mesh_path)
        mesh_list = _to_mesh_list(loaded)
        if not mesh_list:
            raise RuntimeError("empty mesh_list")
        t["load"] = time.time() - t0

        t0 = time.time()
        combined = _normalize_and_combine(mesh_list)
        
        if merge_digits is not None and merge_digits > 0:
            combined.merge_vertices(digits_vertex=merge_digits)
        t["normalize_merge"] = time.time() - t0

        
        t0 = time.time()
        
        
        sampled_points, face_idx = trimesh.sample.sample_surface(
            combined, pc_size, seed=sample_seed
        )
        sampled_points = np.asarray(sampled_points, dtype=np.float32)
        face_idx = np.asarray(face_idx, dtype=np.int64)
        
        try:
            vc = np.asarray(combined.visual.vertex_colors, dtype=np.uint8)
            sampled_rgb = vc[combined.faces[face_idx][:, 0]][:, :3]
        except Exception:
            sampled_rgb = np.full((pc_size, 3), 102, dtype=np.uint8)
        t["sample"] = time.time() - t0


        t0 = time.time()
        _normal = np.asarray(combined.face_normals[face_idx], dtype=np.float32)
        _normal = _normal / (np.linalg.norm(_normal, axis=-1, keepdims=True) + 1e-8)
        t["normals"] = time.time() - t0

        
        t0 = time.time()
        np.savez_compressed(
            geom_path,
            sampled_points=sampled_points,
            face_idx=face_idx,
            sampled_normals=_normal,
            sampled_rgb=sampled_rgb,
            pc_size=np.int64(pc_size),
        )
        
        
        combined.export(ply_path)
        with open(done_marker, "w") as f:
            f.write("ok\n")
        t["save"] = time.time() - t0

        total = sum(t.values())
        return (
            model_id, True,
            f"V={len(combined.vertices)} F={len(combined.faces)} total={total:.1f}s",
            t,
        )
    except Exception as e:
        tb = traceback.format_exc()
        return (model_id, False, f"{type(e).__name__}: {e}\n{tb}", t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", required=True, help="Directory containing FBX, GLB, PLY, or OBJ files")
    parser.add_argument("--cache_dir", required=True, help="Output cache directory")
    parser.add_argument("--pc_size", type=int, default=81920)
    parser.add_argument(
        "--merge_digits", type=int, default=5,
        help="digits_vertex for trimesh.merge_vertices; assimp FBX requires at least 4 to recover topology",
    )
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--nprocs", type=int, default=0, help="0 = auto")
    parser.add_argument("--max_samples", type=int, default=0, help="Maximum number of meshes; 0 processes all meshes")
    parser.add_argument("--shuffle_seed", type=int, default=666)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(args.mesh_dir)
        if os.path.splitext(f.lower())[1] in SUPPORTED_EXT
    )
    if not files:
        print(f"[Error] no mesh files in {args.mesh_dir}")
        sys.exit(1)
    if args.max_samples > 0 and args.max_samples < len(files):
        rng = np.random.default_rng(args.shuffle_seed)
        idxs = rng.permutation(len(files))[: args.max_samples]
        files = sorted(files[i] for i in idxs)

    nprocs = args.nprocs or min(mp.cpu_count(), 32)
    nprocs = min(nprocs, len(files))
    tasks = [
        (
            os.path.join(args.mesh_dir, f),
            args.cache_dir,
            args.pc_size,
            args.merge_digits,
            args.sample_seed,
        )
        for f in files
    ]

    print(
        f"[preprocess] mesh_dir={args.mesh_dir}  files={len(files)}  "
        f"cache_dir={args.cache_dir}  pc_size={args.pc_size}  nprocs={nprocs}"
    )

    t0 = time.time()
    results = []
    with mp.Pool(nprocs) as pool:
        for i, res in enumerate(pool.imap_unordered(preprocess_one, tasks)):
            results.append(res)
            model_id, ok, msg, _ = res
            tag = "OK" if ok else "FAIL"
            print(f"[{i+1:3d}/{len(tasks)}] {tag} {model_id}: {msg}")
    elapsed = time.time() - t0

    n_ok = sum(1 for _, ok, *_ in results if ok)
    n_fail = len(results) - n_ok
    print(
        f"\n[preprocess] done: ok={n_ok} fail={n_fail} elapsed={elapsed:.1f}s "
        f"({elapsed/max(1,len(results)):.2f}s/mesh avg wall-clock)"
    )

    summary_path = os.path.join(args.cache_dir, "_preprocess_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "mesh_dir": args.mesh_dir,
                "total": len(results),
                "ok": n_ok,
                "fail": n_fail,
                "pc_size": args.pc_size,
                "merge_digits": args.merge_digits,
                "sample_seed": args.sample_seed,
                "nprocs": nprocs,
                "elapsed_sec": elapsed,
                "failures": [
                    {"model_id": mid, "msg": msg.splitlines()[0] if msg else ""}
                    for mid, ok, msg, _ in results
                    if not ok
                ],
            },
            f,
            indent=2,
        )
    print(f"[preprocess] summary -> {summary_path}")
    if n_fail > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
