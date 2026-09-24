#!/usr/bin/env python3
"""Create a vLLM model view without copying or modifying checkpoint weights."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from partllm_vllm.constants import ARCHITECTURE_NAME

SMALL_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def copy_first_existing(filename: str, roots: tuple[Path, ...], output: Path) -> None:
    for root in roots:
        source = root / filename
        if source.is_file():
            shutil.copy2(source, output / filename)
            return


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    config_root = args.config.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty; refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)

    for filename in SMALL_MODEL_FILES:
        copy_first_existing(filename, (config_root, checkpoint), output)

    config_path = output / "config.json"
    if not config_path.is_file():
        raise SystemExit("config.json was not found")
    config = json.loads(config_path.read_text())
    config["architectures"] = [ARCHITECTURE_NAME]
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    weight_files = sorted(checkpoint.glob("model*.safetensors"))
    index_file = checkpoint / "model.safetensors.index.json"
    if index_file.is_file():
        weight_files.append(index_file)
    if not any(path.suffix == ".safetensors" for path in weight_files):
        raise SystemExit(f"checkpoint contains no safetensors: {checkpoint}")
    for source in weight_files:
        destination = output / source.name
        if not destination.exists():
            os.symlink(source, destination)

    manifest = {
        "architecture": ARCHITECTURE_NAME,
        "checkpoint": str(checkpoint),
        "config": str(config_root),
        "weights": [path.name for path in weight_files],
        "weight_mode": "absolute_symlink_read_only",
    }
    (output / "bridge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
