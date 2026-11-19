"""Parallel CPU normal draws for the PartLLM checkpoint constructor."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

HEADER = Path(__file__).with_name("cpu_normal_parallel.h").read_text(encoding="utf-8")


def parallel_normal(jt, shape):
    return jt.code(shape, "float32", [], cpu_header=HEADER,
                   cpu_src="jt_parallel_normal(out0_p, out0->num, *jittor::get_random_engine());")


def install_constructor_draw(jt, counts):
    """Leave the draw lazy at the checkpoint's original construction boundary."""
    from jittor._core.dtypes import dtype_name

    original = jt.init.gauss
    active = ContextVar("partllm_large_embedding_normal", default=False)

    @wraps(original)
    def gauss(shape, dtype="float32", mean=0.0, std=1.0, generator=None):
        if (not active.get() or jt.flags.use_cuda or jt.flags.amp_reg
                or generator is not None or tuple(shape) != (151936, 2560)
                or dtype_name(dtype) != "bfloat16" or mean != 0.0 or std != 1.0):
            return original(shape, dtype, mean, std, generator)
        counts["parallel_embedding_draws"] = counts.get("parallel_embedding_draws", 0) + 1
        draw = parallel_normal(jt, tuple(shape)).cast("bfloat16")
        return draw * std + mean

    @contextmanager
    def scoped():
        token = active.set(True)
        try:
            yield
        finally:
            active.reset(token)

    jt.init.gauss = gauss
    return scoped
