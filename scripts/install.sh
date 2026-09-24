#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python -m pip install --upgrade pip setuptools wheel
python scripts/install_torch.py
python -m pip install -r requirements.txt
python scripts/install_gpu_deps.py "$@"
python -m pip install -e .
python scripts/patch_transformers.py

echo "PartLLM installation completed."
