"""PartLLM mesh inference with Jittor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re

from .checkpoint import load_config, validate_checkpoint
from .prompts import select_prompts

MESH_EXTENSIONS = {".glb", ".gltf", ".ply", ".obj", ".stl", ".off", ".fbx"}


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Use true or false")


def collect_inputs(mesh, points_npz=None):
    if points_npz is not None:
        if not points_npz.is_file():
            raise ValueError(f"Point-cloud file not found: {points_npz}")
        return [(points_npz.stem, points_npz, False)]
    if mesh.is_file():
        if mesh.suffix.lower() not in MESH_EXTENSIONS:
            raise ValueError(f"Unsupported mesh format: {mesh.suffix}")
        return [(mesh.stem, mesh, True)]
    if not mesh.is_dir():
        raise ValueError(f"Mesh input not found: {mesh}")
    result = [(path.stem, path, True) for path in sorted(mesh.iterdir())
              if path.is_file() and path.suffix.lower() in MESH_EXTENSIONS]
    if not result:
        raise ValueError(f"No supported meshes found in {mesh}")
    names = [row[0] for row in result]
    if len(names) != len(set(names)):
        raise ValueError("Mesh filenames in a batch must have unique stems")
    return result


def part_labels(text, count):
    labels = []
    for index, segment in enumerate(text.split("<|SEG|>")[:count]):
        match = re.search(r'"label"\s*:\s*("(?:[^"\\]|\\.)*")', segment)
        labels.append(json.loads(match.group(1)) if match else f"Part {index + 1}")
    return labels


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    def option(name, *aliases, **kwargs):
        flags = ["--" + name]
        if "_" in name:
            flags.append("--" + name.replace("_", "-"))
        flags.extend(aliases)
        parser.add_argument(*dict.fromkeys(flags), dest=name, **kwargs)

    option("model_path", "--model-dir", type=Path, required=True)
    option("config_path", type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--mesh_path", "--mesh-path", "--mesh", dest="mesh_path", type=Path)
    inputs.add_argument("--points_npz", "--points-npz", type=Path)
    option("task", choices=["full_shape", "text_guided", "interactive", "refine"], default="full_shape")
    option("part_names", nargs="+")
    option("sem_mode", choices=["semantic", "nosem", "all"], default="semantic")
    option("prompt_set", choices=["all", "coarse", "general", "fine"], default="all")
    option("num_parts", type=int)
    option("point", nargs=3, type=float, action="append", default=[])
    option("negative_point", nargs=3, type=float, action="append", default=[])
    option("initial_mask", type=Path)
    option("semantic", type=parse_bool, default=True)
    option("output_dir", type=Path, required=True)
    option("cache_dir", type=Path)
    option("num_points", type=int, default=81920)
    option("max_new_tokens", type=int, default=10240)
    option("max_samples", type=int, default=-1)
    option("seed", type=int, default=42)
    option("sample_seed", type=int, default=42)
    option("do_sample", type=parse_bool, default=False)
    option("temperature", type=float, default=0.7)
    option("top_p", type=float, default=0.8)
    option("top_k", type=int, default=20)
    option("max_generated_parts", type=int, default=64)
    for name, default in (("auto_add_bg", True), ("resume", True), ("save_mesh", True),
                          ("save_per_part", False), ("postprocess", True),
                          ("graph_cut", True), ("pp_drop_small", True)):
        option(name, type=parse_bool, default=default, metavar="{true,false}")
    option("pp_rel_area_threshold", type=float, default=0.001)
    return parser


def validate_args(args):
    if args.num_points <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--num_points and --max_new_tokens must be positive")
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max_samples must be positive or -1")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("Require temperature > 0 and 0 < top_p <= 1")
    if not 0 <= args.pp_rel_area_threshold <= 1:
        raise ValueError("--pp_rel_area_threshold must be in [0, 1]")
    if args.task in {"interactive", "refine"}:
        if args.points_npz or args.cache_dir:
            raise ValueError("Interactive/refine requires original mesh geometry")
        if args.task == "refine" and not args.initial_mask:
            raise ValueError("Refine requires --initial_mask")
        if args.initial_mask and not args.mesh_path.is_file():
            raise ValueError("--initial_mask requires exactly one mesh file")
        if not args.point and (not args.initial_mask or not args.negative_point):
            raise ValueError("Provide a positive --point, or an initial mask and corrective points")
    elif args.point or args.negative_point or args.initial_mask:
        raise ValueError("Point and initial-mask options require interactive/refine")
    if args.cache_dir and args.task != "full_shape":
        raise ValueError("--cache_dir is supported for full_shape only")


def completed(output, args, is_mesh, interactive=False):
    required = ["metadata.json", "response.txt", "face_labels.npy" if is_mesh else "point_labels.npy"]
    if is_mesh and args.save_mesh:
        required.append("pred_mesh.ply")
    if interactive:
        required.append("binary_mask.npy")
    if not all((output / name).is_file() for name in required):
        return False
    try:
        metadata = json.loads((output / "metadata.json").read_text())
    except (ValueError, OSError):
        return False
    return not args.save_per_part or metadata.get("save_per_part", False)


class Inference:
    def __init__(self, args):
        import jittor as jt
        from .hf_weights import LazySafeTensorStore
        from .mask_decoder import JittorMaskDecoder, MaskDecoderConfig, WeightStore
        from .pipeline import PartSegPrompt
        from .qwen3_text import JittorQwen3Text, Qwen3TextConfig
        self.args, self.jt = args, jt
        jt.flags.use_cuda = 1
        jt.set_global_seed(args.seed)
        self.config_dir = args.config_path or args.model_path
        self.config = load_config(self.config_dir)
        self.point_tokens = int(self.config.get("utonia_num_tokens", 4096))
        self.prompt_builder = PartSegPrompt(self.config_dir, point_token_count=self.point_tokens)
        self.decoder = JittorMaskDecoder(
            MaskDecoderConfig.from_hf_config(self.config_dir / "config.json"),
            WeightStore.from_checkpoint(args.model_path),
        )
        self.encoder_store = LazySafeTensorStore(args.model_path, dtype="float32")
        self.qwen_store = LazySafeTensorStore(args.model_path, dtype="bfloat16")
        self.qwen = JittorQwen3Text(Qwen3TextConfig.from_hf_config(self.config_dir / "config.json"), self.qwen_store)

    def release(self):
        self.encoder_store.release_all()
        self.qwen_store.release_all()
        self.decoder.weights.release_all()

    def encode(self, coords, colors, normals):
        from .geometry import utonia_default_transform
        from .utonia_encoder import JittorUtoniaEncoder
        print(f"[Encode] {len(coords)} surface points", flush=True)
        points = utonia_default_transform(coords, colors, normals,
                                         scale=float(self.config.get("utonia_scale", 5.0)), seed=self.args.seed)
        encoded = JittorUtoniaEncoder(self.encoder_store).encode(points, num_tokens=self.point_tokens)
        arrays = tuple(value.numpy() for value in (encoded.point_embeddings, encoded.centers, encoded.coords, encoded.feat_raw))
        del encoded
        self.encoder_store.release_all()
        print(f"[Encode] {self.point_tokens} point tokens ready", flush=True)
        return arrays

    def predict(self, encoded, prompt):
        import numpy as np
        args, jt = self.args, self.jt
        embedding, centers, coords, features = encoded
        generated = self.prompt_builder.generate(
            self.qwen, prompt, embedding, args.max_new_tokens,
            auto_add_bg=args.auto_add_bg, do_sample=args.do_sample, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, seed=args.seed, max_generated_parts=args.max_generated_parts,
        )
        self.qwen_store.release_all()
        logits = self.decoder(
            jt.array(embedding.astype(np.float32)[None]), jt.array(generated.token_hidden_states.astype(np.float32)),
            jt.array(centers.astype(np.float32)[None]), jt.array(coords.astype(np.float32)[None]),
            feat_raw=jt.array(features.astype(np.float32)[None]),
        ).numpy()[0]
        self.decoder.weights.release_all()
        if not np.isfinite(logits).all():
            raise ValueError("Mask decoder produced non-finite logits")
        return generated.text, logits

    def save(self, output, input_path, prompt, text, logits, mesh, face_indices, clicks=None):
        import numpy as np
        from .mesh_output import face_predictions, save_mesh
        args = self.args
        output.mkdir(parents=True, exist_ok=True)
        if mesh is not None:
            labels = face_predictions(mesh, face_indices, logits, args.postprocess, args.graph_cut,
                                      args.pp_drop_small, args.pp_rel_area_threshold)
            save_mesh(mesh, labels, logits.shape[0], output, args.save_mesh, args.save_per_part)
            if clicks is not None:
                np.save(output / "binary_mask.npy", (labels == 0).astype(np.uint8))
        else:
            np.save(output / "point_labels.npy", logits.argmax(axis=0).astype(np.int64))
        metadata = {"input": str(input_path), "prompt": prompt, "seed": args.seed,
                    "part_labels": part_labels(text, logits.shape[0] - 1), "background_label": logits.shape[0] - 1,
                    "point_tokens": self.point_tokens, "sampled_points": logits.shape[1],
                    "postprocess": bool(mesh is not None and args.postprocess),
                    "graph_cut": bool(mesh is not None and args.graph_cut), "save_per_part": args.save_per_part}
        if clicks is not None:
            metadata["clicks"] = clicks
        (output / "response.txt").write_text(text, encoding="utf-8")
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    def run_shape(self, shape_id, input_path, is_mesh, prompts):
        import numpy as np
        import trimesh
        from .geometry import load_mesh_samples, load_points_npz, prepare_clicks, load_initial_mask
        from .refine_protocol import encode_refine_mask_colors, format_refine_prompt
        args = self.args
        interactive = args.task in {"interactive", "refine"}
        shape_dir = args.output_dir / shape_id
        if interactive:
            final = shape_dir / f"click_{len(args.point) + len(args.negative_point)}"
            if args.resume and completed(final, args, True, True):
                print(f"[Skip] {shape_id}", flush=True)
                return
        elif args.resume and all(completed(shape_dir if args.task == "text_guided" else shape_dir / name,
                                           args, is_mesh) for name, _ in prompts):
            print(f"[Skip] {shape_id}", flush=True)
            return
        mesh, face_indices = None, None
        if args.cache_dir:
            entry = args.cache_dir / shape_id
            with np.load(entry / "geom.npz", allow_pickle=False) as geom:
                raw_coords, colors, normals, face_indices = (geom[key].copy() for key in
                    ("sampled_points", "sampled_rgb", "sampled_normals", "face_idx"))
            mesh = trimesh.load(entry / "combined_mesh.ply", process=False)
        elif is_mesh:
            mesh, raw_coords, colors, normals, face_indices = load_mesh_samples(input_path, args.num_points, seed=args.seed)
        else:
            raw_coords, colors, normals = load_points_npz(input_path)
        if interactive:
            clicks = prepare_clicks(args.point, args.negative_point, mesh, raw_coords)
            current_mask = load_initial_mask(args.initial_mask, len(mesh.faces))[face_indices] if args.initial_mask else None
            for index, click in enumerate(clicks):
                history = clicks[:index + 1]
                if current_mask is None:
                    x, y, z = click["point"]
                    template = "Please segment and name the part at" if args.semantic else "Please segment the geometric part at"
                    prompt = f"{template} ({x:.3f}, {y:.3f}, {z:.3f}) in <point_cloud>."
                    active_colors = colors
                else:
                    active_colors = encode_refine_mask_colors(colors, current_mask)
                    prompt = format_refine_prompt(history, semantic=args.semantic)
                print(f"[Infer] {shape_id}/click_{index + 1}", flush=True)
                encoded = self.encode(raw_coords, active_colors, normals)
                text, logits = self.predict(encoded, prompt)
                if logits.shape[0] != 2:
                    raise ValueError(f"Interactive inference requires foreground plus background, got {logits.shape[0]} queries")
                current_mask = logits.argmax(axis=0) == 0
                output = shape_dir / f"click_{index + 1}"
                self.save(output, input_path, prompt, text, logits, mesh, face_indices, history)
                print(f"[OK] {output}", flush=True)
        else:
            encoded = self.encode(raw_coords, colors, normals)
            for name, prompt in prompts:
                output = shape_dir if args.task == "text_guided" else shape_dir / name
                if args.resume and completed(output, args, is_mesh):
                    print(f"[Skip] {output}", flush=True)
                    continue
                print(f"[Infer] {shape_id}/{name}", flush=True)
                text, logits = self.predict(encoded, prompt)
                self.save(output, input_path, prompt, text, logits, mesh, face_indices)
                print(f"[OK] {output}", flush=True)


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        prompts = [] if args.task in {"interactive", "refine"} else select_prompts(
            args.task, args.sem_mode, args.prompt_set, args.part_names, args.num_parts)
        if args.cache_dir:
            inputs = [(path.name, path / "combined_mesh.ply", True) for path in sorted(args.cache_dir.iterdir())
                      if path.is_dir() and (path / "geom.npz").is_file()]
            if not inputs:
                raise ValueError("No geom.npz entries found in --cache_dir")
        else:
            inputs = collect_inputs(args.mesh_path, args.points_npz)
        validate_checkpoint(args.model_path, args.config_path)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    if args.max_samples > 0 and args.max_samples < len(inputs):
        if args.task == "full_shape":
            inputs = sorted(random.Random(args.sample_seed).sample(inputs, args.max_samples))
        else:
            inputs = inputs[:args.max_samples]
    runtime = Inference(args)
    failed = 0
    with runtime.jt.no_grad():
        for shape_id, path, is_mesh in inputs:
            try:
                runtime.run_shape(shape_id, path, is_mesh, prompts)
            except Exception as exc:
                failed += 1
                print(f"[Failed] {shape_id}: {type(exc).__name__}: {exc}", flush=True)
                import traceback
                traceback.print_exc()
            finally:
                runtime.release()
    print(f"Done: {len(inputs) - failed} successful, {failed} failed shapes", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
