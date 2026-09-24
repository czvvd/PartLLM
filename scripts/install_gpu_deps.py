#!/usr/bin/env python3
"""Install PartLLM CUDA extensions for the active PyTorch environment."""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys


PYG_WHEELS = {
    (2, 7): {"cu118", "cu126", "cu128"},
    (2, 8): {"cu126", "cu128", "cu129"},
}


def parse_version(value: str, name: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", value)
    if not match:
        raise SystemExit(f"Could not parse {name} version: {value!r}")
    return int(match.group(1)), int(match.group(2))


def run(command: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install torch-scatter, torch-cluster, spconv, and flash-attn."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument("--skip-flash-attn", action="store_true")
    parser.add_argument("--skip-spconv", action="store_true")
    parser.add_argument("--torch-version", help=argparse.SUPPRESS)
    parser.add_argument("--cuda-version", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if platform.system() != "Linux":
        raise SystemExit("PartLLM CUDA extensions currently require Linux.")

    if args.torch_version and args.cuda_version:
        torch_version = args.torch_version
        cuda_version = args.cuda_version
    else:
        try:
            import torch
        except ImportError as exc:
            raise SystemExit("PyTorch must be installed before GPU extensions.") from exc
        torch_version = torch.__version__
        cuda_version = torch.version.cuda or ""

    if not cuda_version:
        raise SystemExit(
            "The active PyTorch build has no CUDA support. Install a CUDA-enabled "
            "PyTorch build, then rerun this script."
        )

    torch_major_minor = parse_version(torch_version, "PyTorch")
    cuda_major_minor = parse_version(cuda_version, "CUDA")
    cuda_tag = f"cu{cuda_major_minor[0]}{cuda_major_minor[1]}"

    supported_cuda = PYG_WHEELS.get(torch_major_minor)
    if supported_cuda is None or cuda_tag not in supported_cuda:
        supported = ", ".join(sorted(supported_cuda or [])) or "none"
        raise SystemExit(
            f"No tested PyG wheel combination for PyTorch {torch_version} and "
            f"CUDA {cuda_version}. Supported CUDA builds for PyTorch "
            f"{torch_major_minor[0]}.{torch_major_minor[1]} are: {supported}."
        )

    torch_wheel_version = f"{torch_major_minor[0]}.{torch_major_minor[1]}.0"
    pyg_url = f"https://data.pyg.org/whl/torch-{torch_wheel_version}+{cuda_tag}.html"
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--only-binary=:all:",
            "--no-index",
            "--find-links",
            pyg_url,
            "torch-scatter",
            "torch-cluster",
        ],
        dry_run=args.dry_run,
    )

    if not args.skip_spconv:
        if cuda_major_minor[0] == 12:
            spconv_package = "spconv-cu120"
        elif cuda_major_minor == (11, 8):
            spconv_package = "spconv-cu118"
        else:
            raise SystemExit(
                f"No automated spconv package mapping for CUDA {cuda_version}."
            )
        run(
            [sys.executable, "-m", "pip", "install", spconv_package],
            dry_run=args.dry_run,
        )

    if not args.skip_flash_attn:
        run(
            [sys.executable, "-m", "pip", "install", "ninja", "packaging"],
            dry_run=args.dry_run,
        )
        build_env = os.environ.copy()
        build_env.setdefault("MAX_JOBS", str(min(4, os.cpu_count() or 1)))
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "flash-attn>=2.4.3",
                "--no-build-isolation",
            ],
            dry_run=args.dry_run,
            env=build_env,
        )

    if not args.dry_run:
        modules = ["torch_cluster", "torch_scatter"]
        if not args.skip_spconv:
            modules.append("spconv")
        if not args.skip_flash_attn:
            modules.append("flash_attn")
        subprocess.run(
            [
                sys.executable,
                "-c",
                "; ".join(f"import {module}" for module in modules)
                + "; print('PartLLM GPU dependencies are ready.')",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
