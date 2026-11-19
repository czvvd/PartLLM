"""Run a PartLLM inference entry point using the Jittor Torch frontend."""

import os
from pathlib import Path
import runpy
import sys


def configure_checkpoint_optimizations(jt, torch):
    prepared = os.environ.get("PARTLLM_PREPARED_NORMAL_DIR")
    parallel = os.environ.get("PARTLLM_PARALLEL_CONSTRUCTOR_NORMAL") == "1"
    if not (prepared or parallel):
        return

    import transformers

    if transformers.__version__ != "4.57.3":
        raise RuntimeError("Checkpoint optimizations require Transformers 4.57.3")
    if os.environ.get("PARTLLM_JTORCH_CPU_LOAD") != "2":
        raise RuntimeError("Checkpoint optimizations require PARTLLM_JTORCH_CPU_LOAD=2")

    counts = {}
    if prepared:
        from jittor import compiler
        from prepared_normal_candidate import install

        os.environ.setdefault("PARTLLM_PREPARED_NORMAL_MODE", "reuse")
        identity = os.environ.get("PARTLLM_PREPARED_NORMAL_ID", "partllm-prepared-normal-v1")
        os.environ["PARTLLM_PREPARED_NORMAL_ID"] = f"{identity}:{compiler.cache_path}"
        install(jt, torch, counts)

    if parallel:
        from functools import wraps

        from cpu_normal_candidate import install_constructor_draw
        from jittor.compat.torch import checkpoint_linear

        if os.environ.get("PARTLLM_JTORCH_DEFER_LINEAR_INIT") != "1":
            raise RuntimeError("Parallel constructor initialization requires deferred Linear initialization")
        scope = install_constructor_draw(jt, counts)
        load = checkpoint_linear.load_with_deferred_linear_initialization

        @wraps(load)
        def load_with_parallel_normal(*args, **kwargs):
            with scope():
                return load(*args, **kwargs)

        checkpoint_linear.load_with_deferred_linear_initialization = load_with_parallel_normal


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_partllm.py PARTLLM_SCRIPT [arguments...]")

    import jittor as jt
    import torch

    if not hasattr(torch, "_torch_compat_install_context"):
        raise RuntimeError("Set JITTOR_TORCH_SHIM=1 before running this launcher")

    if os.environ.get("PARTLLM_SINGLE_TOKEN_OPT") == "1":
        from partllm_runtime_optimizations import install

        install()
    elif os.environ.get("PARTLLM_FAST_DECODER_KNN") == "1":
        from partllm_mask_decoder_knn import install_decoder_knn

        install_decoder_knn()

    configure_checkpoint_optimizations(jt, torch)
    entry = Path(sys.argv[1]).resolve()
    sys.argv = [str(entry), *sys.argv[2:]]
    sys.path.insert(0, str(entry.parent))
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
