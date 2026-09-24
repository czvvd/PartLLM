#!/usr/bin/env python3
"""Text-guided part segmentation for a mesh or a directory of meshes."""

import argparse
import os
import sys
import traceback

import torch
from tqdm import tqdm

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mesh_inference_utils import collect_mesh_paths
from test_mask_pred import (
    add_common_inference_args,
    apply_yaml_config,
    build_grounding_prompt,
    load_model,
    parse_bool,
    prepare_geometry_from_glb,
    prepare_prompt_from_template,
    process_and_save,
    run_inference,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment named parts in a mesh without annotations or Parquet input."
    )
    add_common_inference_args(parser)
    parser.add_argument("--mesh_path", required=True, help="Mesh file or directory of meshes.")
    parser.add_argument(
        "--part_names",
        nargs="+",
        required=True,
        help='One or more part names, for example: --part_names "seat" "chair back" leg',
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
    args.part_names = list(dict.fromkeys(name.strip() for name in args.part_names if name.strip()))
    if not args.part_names:
        parser.error("--part_names must contain at least one non-empty name")
    return args


def main():
    args = parse_args()
    mesh_paths = collect_mesh_paths(args.mesh_path)
    if args.max_samples > 0:
        mesh_paths = mesh_paths[: args.max_samples]
    os.makedirs(args.output_dir, exist_ok=True)

    model, processor, _tokenizer, hf_config, encoder_type = load_model(args)
    prompt = build_grounding_prompt(args.part_names, prompt_style="current")
    successful = 0
    failed = 0

    for mesh_path in tqdm(mesh_paths, desc="Text-guided segmentation"):
        model_id = os.path.splitext(os.path.basename(mesh_path))[0]
        save_dir = os.path.join(args.output_dir, model_id)
        if args.resume and os.path.exists(os.path.join(save_dir, "metadata.json")):
            print(f"[Skip] {model_id} already completed")
            continue
        inputs = mask_logits = pca_capture = None
        try:
            geometry = prepare_geometry_from_glb(
                mesh_path,
                model,
                hf_config,
                encoder_type,
                apply_z_positive=args.apply_z_positive,
                clean_mesh=False,
            )
            inputs, user_prompt = prepare_prompt_from_template(prompt, geometry, processor, model)
            response, mask_logits, pca_capture = run_inference(
                model, inputs, geometry["utonia_point_dict"], processor, args
            )
            os.makedirs(save_dir, exist_ok=True)
            row = {"model_id": model_id, "prompt_mode": "text_guided", "granularity": "semantic"}
            process_and_save(
                response,
                mask_logits,
                inputs,
                row,
                save_dir,
                args,
                0,
                user_prompt,
                None,
                mesh_vis_info=geometry["mesh_vis_info"],
                pca_capture=pca_capture,
            )
            successful += 1
            print(f"[OK] {model_id} -> {save_dir}")
        except Exception as exc:
            failed += 1
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "error.txt"), "w", encoding="utf-8") as handle:
                handle.write(f"{type(exc).__name__}: {exc}\n")
            print(f"[Error] {model_id}: {exc}")
            traceback.print_exc()
        finally:
            if pca_capture is not None:
                pca_capture.detach()
            del inputs, mask_logits, pca_capture
            torch.cuda.empty_cache()

    print(f"[Done] {successful} successful, {failed} failed")


if __name__ == "__main__":
    main()
