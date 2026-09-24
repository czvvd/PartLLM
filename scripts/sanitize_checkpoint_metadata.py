#!/usr/bin/env python3
"""Audit and optionally sanitize text metadata in a Hugging Face checkpoint."""

import argparse
import json
import re
from pathlib import Path


TEXT_SUFFIXES = {".json", ".jinja", ".txt", ".md"}
FORBIDDEN = re.compile(
    r"/vinowan-cfs|/data/work/|zhezhu|gameai|cos_user|cos_password|s3://|9\.135\.140\.83",
    re.IGNORECASE,
)


def findings(checkpoint: Path) -> list[str]:
    hits = []
    for path in sorted(checkpoint.iterdir()):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN.search(line):
                hits.append(f"{path.name}:{line_no}: {line.strip()}")
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--write", action="store_true", help="set local-only config paths to null")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        parser.error(f"missing {config_path}")

    if args.write:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["utonia_ckpt_path"] = None
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    hits = findings(checkpoint)
    if hits:
        print("\n".join(hits))
        raise SystemExit("checkpoint metadata audit failed")
    print(f"checkpoint metadata audit passed: {checkpoint}")


if __name__ == "__main__":
    main()

