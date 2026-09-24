#!/usr/bin/env python3
"""Install a tested CUDA-enabled PyTorch build for the active NVIDIA driver."""

from __future__ import annotations

import os
import re
import subprocess
import sys


BUILDS = {
    "cu118": ("2.7.1", "0.22.1"),
    "cu126": ("2.8.0", "0.23.0"),
    "cu128": ("2.8.0", "0.23.0"),
}


def detect_cuda_tag() -> str:
    override = os.environ.get("PARTLLM_TORCH_CUDA")
    if override:
        if override not in BUILDS:
            choices = ", ".join(BUILDS)
            raise SystemExit(f"PARTLLM_TORCH_CUDA must be one of: {choices}")
        return override

    try:
        output = subprocess.check_output(
            ["nvidia-smi"], text=True, stderr=subprocess.STDOUT
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "An NVIDIA GPU and driver are required. Set PARTLLM_TORCH_CUDA "
            "to cu118, cu126, or cu128 to select a build explicitly."
        ) from exc

    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
    if not match:
        raise SystemExit("Could not detect the CUDA version reported by nvidia-smi.")
    version = tuple(map(int, match.groups()))
    if version >= (12, 8):
        return "cu128"
    if version >= (12, 6):
        return "cu126"
    if version >= (11, 8):
        return "cu118"
    raise SystemExit(f"CUDA {version[0]}.{version[1]} is not supported.")


def current_build_matches(tag: str, torch_version: str) -> bool:
    try:
        import torch
    except ImportError:
        return False
    cuda = torch.version.cuda or ""
    expected_cuda = f"{tag[2:4]}.{tag[4:]}"
    return torch.__version__.split("+")[0] == torch_version and cuda.startswith(expected_cuda)


def main() -> None:
    tag = detect_cuda_tag()
    torch_version, torchvision_version = BUILDS[tag]
    if current_build_matches(tag, torch_version):
        print(f"Compatible PyTorch {torch_version} ({tag}) is already installed.")
        return

    index_url = f"https://download.pytorch.org/whl/{tag}"
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"torch=={torch_version}",
        f"torchvision=={torchvision_version}",
        "--index-url",
        index_url,
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
