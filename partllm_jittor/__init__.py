"""PartLLM inference with Jittor."""

import os
from pathlib import Path
import sys

os.environ.setdefault("conv_opt", "1")

if "python_config_path" not in os.environ:
    candidate = Path(sys.base_prefix) / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}-config"
    if candidate.is_file():
        os.environ["python_config_path"] = str(candidate)
