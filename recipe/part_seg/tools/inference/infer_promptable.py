#!/usr/bin/env python3
"""Interactive part segmentation for a mesh or directory of meshes."""

import argparse
import os
import sys
import traceback

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mesh_inference_utils import collect_mesh_paths, snap_to_sampled_point, source_point_to_model
from refine_protocol import encode_refine_mask_colors
from test_mask_pred import (
    add_common_inference_args,
    apply_yaml_config,
    load_model,
    parse_bool,
    prepare_geometry_from_glb,
    prepare_prompt_from_template,
    process_and_save,
    run_inference,
    set_global_seed,
)
from promptable_inference_utils import (
    build_colored_geo_cache,
    build_promptable_prompt,
    build_refine_prompt,
    decode_promptable_mask,
    stable_seed,
)
import test_mask_pred as _inference_core

_inference_core.MODE20_COLORMAP_HEX[:] = ["#E6194B"] * 20
_inference_core.BG_COLOR_HEX = "#C0C0C0"
_inference_core.BG_COLOR_RGB = (192, 192, 192)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Segment a part from coordinates in the original mesh coordinate system. "
            "Repeat --point or --negative_point to add corrective clicks."
        )
    )
    add_common_inference_args(parser)
    parser.add_argument("--mesh_path", required=True, help="Mesh file or directory of meshes.")
    parser.add_argument(
        "--point",
        nargs=3,
        type=float,
        action="append",
        default=[],
        metavar=("X", "Y", "Z"),
        help="Positive point in the original mesh coordinates. May be repeated.",
    )
    parser.add_argument(
        "--negative_point",
        nargs=3,
        type=float,
        action="append",
        default=[],
        metavar=("X", "Y", "Z"),
        help="Negative corrective point in the original mesh coordinates. May be repeated.",
    )
    parser.add_argument(
        "--initial_mask",
        type=str,
        default=None,
        help=(
            "Path to a NumPy .npy binary face mask with shape [num_faces]; "
            "1=True is foreground and 0=False is background."
        ),
    )
    parser.add_argument(
        "--require_initial_mask",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--semantic",
        type=parse_bool,
        default=True,
        metavar="{true,false}",
        help="Also predict the part name (default: enabled).",
    )
    parser.add_argument(
        "--apply_z_positive",
        type=parse_bool,
        default=False,
        metavar="{true,false}",
        help="Enable the legacy Utonia ground shift (default: disabled).",
    )
    args = parser.parse_args()
    apply_yaml_config(args, parser)
    if not args.model_path or not args.config_path:
        parser.error("--model_path and --config_path are required")
    if args.require_initial_mask and not args.initial_mask:
        parser.error("refine mode requires --initial_mask")
    if args.initial_mask:
        if not args.point and not args.negative_point:
            parser.error("refine mode requires at least one --point or --negative_point")
    elif not args.point:
        parser.error("interactive mode requires at least one positive --point")
    return args


def _load_initial_face_mask(path, geometry):
    """Load a binary per-face mask and map it to the sampled model points."""
    if not str(path).lower().endswith(".npy"):
        raise ValueError("--initial_mask must be a NumPy .npy file")
    face_mask = np.load(path, allow_pickle=False)
    combined_mesh = geometry["mesh_vis_info"]["combined_mesh"]
    num_faces = len(combined_mesh.faces)
    if face_mask.shape != (num_faces,):
        raise ValueError(
            f"initial mask must have shape ({num_faces},), got {face_mask.shape}"
        )
    if face_mask.dtype != np.bool_:
        if not np.issubdtype(face_mask.dtype, np.number):
            raise TypeError("initial mask must contain boolean or numeric 0/1 values")
        if not np.isfinite(face_mask).all() or not np.all((face_mask == 0) | (face_mask == 1)):
            raise ValueError("initial mask must contain only 0/1 values")
        face_mask = face_mask.astype(bool)
    face_idx = np.asarray(geometry["mesh_vis_info"]["face_idx"], dtype=np.int64)
    if np.any(face_idx < 0) or np.any(face_idx >= num_faces):
        raise ValueError("sampled face indices are incompatible with the initial mask")
    return face_mask[face_idx], face_mask


def _save_binary_face_mask(round_dir, current_mask, geometry):
    """Save the final foreground as a reusable 0/1 per-face mask."""
    for filename in ("face_labels_gc.npy", "face_labels.npy"):
        path = os.path.join(round_dir, filename)
        if os.path.exists(path):
            face_labels = np.load(path, allow_pickle=False)
            binary = (np.asarray(face_labels) == 0).astype(np.uint8)
            np.save(os.path.join(round_dir, "binary_mask.npy"), binary)
            return

    num_faces = len(geometry["mesh_vis_info"]["combined_mesh"].faces)
    face_idx = np.asarray(geometry["mesh_vis_info"]["face_idx"], dtype=np.int64)
    current_mask = np.asarray(current_mask, dtype=bool)
    positive = np.bincount(face_idx, weights=current_mask.astype(np.int64), minlength=num_faces)
    total = np.bincount(face_idx, minlength=num_faces)
    binary = (positive * 2 >= total) & (total > 0)
    np.save(os.path.join(round_dir, "binary_mask.npy"), binary.astype(np.uint8))


def _prepare_clicks(args, geometry):
    clicks = []
    source_clicks = [(point, 1) for point in args.point]
    source_clicks.extend((point, 0) for point in args.negative_point)
    for source_point, label in source_clicks:
        model_point = source_point_to_model(source_point, geometry["source_to_model"])
        snapped, point_index, distance = snap_to_sampled_point(
            model_point, geometry["sampled_points"]
        )
        clicks.append(
            {
                "point": snapped,
                "point_index": point_index,
                "label": label,
                "prompt_type": "include" if label else "exclude",
                "source_point": [float(value) for value in source_point],
                "model_point_before_snap": model_point.tolist(),
                "snap_distance": distance,
            }
        )
    return clicks


def infer_mesh(args, mesh_path, model, processor, hf_config, encoder_type):
    model_id = os.path.splitext(os.path.basename(mesh_path))[0]
    shape_dir = os.path.join(args.output_dir, model_id)
    final_round = os.path.join(
        shape_dir, f"click_{len(args.point) + len(args.negative_point)}"
    )
    if (
        args.resume
        and os.path.exists(os.path.join(final_round, "metadata.json"))
        and os.path.exists(os.path.join(final_round, "binary_mask.npy"))
    ):
        print(f"[Skip] {model_id} already completed")
        return "skipped"

    geometry = prepare_geometry_from_glb(
        mesh_path,
        model,
        hf_config,
        encoder_type,
        apply_z_positive=args.apply_z_positive,
        clean_mesh=False,
    )
    clicks = _prepare_clicks(args, geometry)
    template_mode = "semantic" if args.semantic else "nosem"
    initial_face_mask = None
    if args.initial_mask:
        current_mask, initial_face_mask = _load_initial_face_mask(args.initial_mask, geometry)
    else:
        current_mask = np.zeros(len(geometry["sampled_points"]), dtype=bool)
    original_rgb = np.asarray(geometry["sampled_colors"], dtype=np.uint8)
    utonia_seed = stable_seed(args.seed, model_id, "utonia")
    for round_index, click in enumerate(clicks):
        history = clicks[: round_index + 1]
        if round_index == 0 and initial_face_mask is None:
            rgb = original_rgb
            x, y, z = click["point"]
            prompt = build_promptable_prompt(template_mode, f"({x:.3f}, {y:.3f}, {z:.3f})")
        else:
            rgb = encode_refine_mask_colors(original_rgb, current_mask)
            prompt = build_refine_prompt(template_mode, history)

        active_geometry = build_colored_geo_cache(
            geometry, rgb, model, encoder_type, utonia_seed
        )
        round_dir = os.path.join(shape_dir, f"click_{round_index + 1}")
        inputs = mask_logits = pca_capture = None
        try:
            inputs, user_prompt = prepare_prompt_from_template(
                prompt, active_geometry, processor, model
            )
            set_global_seed(stable_seed(args.seed, model_id, round_index + 1, "generation"))
            response, mask_logits, pca_capture = run_inference(
                model,
                inputs,
                active_geometry["utonia_point_dict"],
                processor,
                args,
            )
            current_mask, structure = decode_promptable_mask(
                mask_logits, len(geometry["sampled_points"])
            )
            if not structure["structure_valid"]:
                raise RuntimeError(
                    "interactive inference must produce one foreground query plus background; "
                    f"got {structure['num_classes']} classes"
                )
            os.makedirs(round_dir, exist_ok=True)
            row = {
                "model_id": model_id,
                "prompt_mode": "interactive",
                "granularity": template_mode,
            }
            process_and_save(
                response,
                mask_logits,
                inputs,
                row,
                round_dir,
                args,
                round_index,
                user_prompt,
                None,
                mesh_vis_info=active_geometry["mesh_vis_info"],
                pca_capture=pca_capture,
                include_metrics=False,
            )
            _save_binary_face_mask(round_dir, current_mask, geometry)
        finally:
            if pca_capture is not None:
                pca_capture.detach()
            del inputs, mask_logits, pca_capture
            torch.cuda.empty_cache()

    print(f"[OK] {model_id} -> {shape_dir}")
    return "successful"


def main():
    args = parse_args()
    mesh_paths = collect_mesh_paths(args.mesh_path)
    if args.initial_mask and len(mesh_paths) != 1:
        raise ValueError("refine mode accepts exactly one mesh for each --initial_mask")
    if args.max_samples > 0:
        mesh_paths = mesh_paths[: args.max_samples]
    os.makedirs(args.output_dir, exist_ok=True)
    model, processor, _tokenizer, hf_config, encoder_type = load_model(args)
    counts = {"successful": 0, "failed": 0, "skipped": 0}

    for mesh_path in tqdm(mesh_paths, desc="Interactive segmentation"):
        try:
            status = infer_mesh(
                args, mesh_path, model, processor, hf_config, encoder_type
            )
            counts[status] += 1
        except Exception as exc:
            counts["failed"] += 1
            model_id = os.path.splitext(os.path.basename(mesh_path))[0]
            error_dir = os.path.join(args.output_dir, model_id)
            os.makedirs(error_dir, exist_ok=True)
            with open(os.path.join(error_dir, "error.txt"), "w", encoding="utf-8") as handle:
                handle.write(f"{type(exc).__name__}: {exc}\n")
            print(f"[Error] {model_id}: {exc}")
            traceback.print_exc()

    print(
        f"[Done] {counts['successful']} successful, "
        f"{counts['failed']} failed, {counts['skipped']} skipped"
    )


if __name__ == "__main__":
    main()
