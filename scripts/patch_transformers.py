#!/usr/bin/env python3
"""Install or verify PartLLM's custom Qwen3-VL implementation."""

import argparse
import filecmp
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify without copying")
    args = parser.parse_args()

    import transformers

    root = Path(__file__).resolve().parents[1]
    source = root / "transformers-4-27" / "src" / "transformers" / "models" / "qwen3_vl"
    target = Path(transformers.__file__).resolve().parent / "models" / "qwen3_vl"
    required = ["modeling_qwen3_vl.py", "configuration_qwen3_vl.py", "mask_decoder.py"]

    if args.check:
        missing = [name for name in required if not (target / name).is_file()]
        changed = [
            name
            for name in required
            if (target / name).is_file() and not filecmp.cmp(source / name, target / name, shallow=False)
        ]
        if missing or changed:
            raise SystemExit(f"custom Qwen3-VL is not installed; missing={missing}, changed={changed}")
        print(f"custom Qwen3-VL verified at {target}")
        return

    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)
    print(f"installed custom Qwen3-VL into {target}")


if __name__ == "__main__":
    main()

