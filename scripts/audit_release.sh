#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

forbidden='(/vinowan-cfs|zhezhu|cos_user|cos_password|gameai|s3://|INTERNAL_MESH|internal[_-](train|val|data)|Paper_Open_Source_Watertight|9\.135\.140\.83)'
if grep -RInE "$forbidden" \
  --exclude-dir=.git --exclude='vocab.json' --exclude='merges.txt' \
  --exclude='audit_release.sh' --exclude='sanitize_checkpoint_metadata.py' .; then
  echo "Release audit failed: forbidden internal reference found." >&2
  exit 1
fi

credentials='(-----BEGIN [A-Z ]*PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|hf_[A-Za-z0-9]{20,})'
if grep -RInE "$credentials" \
  --exclude-dir=.git --exclude='vocab.json' --exclude='merges.txt' \
  --exclude='audit_release.sh' .; then
  echo "Release audit failed: credential-like value found." >&2
  exit 1
fi

if find . -path './.git' -prune -o -type f \
  \( -name '.env' -o -name '*.pem' -o -name '*.key' \
     -o -name '*.parquet' -o -name '*.pkl' -o -name '*.pt' -o -name '*.pth' \
     -o -name '*.ckpt' -o -name '*.safetensors' -o -name '*.glb' \) \
  -print -quit | grep -q .; then
  echo "Release audit failed: data or checkpoint artifact found." >&2
  exit 1
fi

echo "Release audit passed."
