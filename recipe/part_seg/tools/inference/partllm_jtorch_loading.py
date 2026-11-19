"""Optional CPU checkpoint staging for the standalone PartLLM inference CLI."""

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import mmap
import os


@dataclass(frozen=True, slots=True)
class _TensorMetadata:
    shape: list[int]
    dtype: object


def _replace(stack, owner, name, replacement):
    original = getattr(owner, name)
    stack.callback(setattr, owner, name, original)
    setattr(owner, name, replacement)


def _metadata_reads(stack):
    from transformers import modeling_utils

    original = modeling_utils.load_state_dict

    def load_state_dict(checkpoint_file, is_quantized=False, map_location="cpu", weights_only=True):
        if is_quantized or map_location != "meta" or not checkpoint_file.endswith(".safetensors"):
            return original(checkpoint_file, is_quantized, map_location, weights_only)
        with modeling_utils.safe_open(checkpoint_file, framework="pt") as shard:
            metadata = shard.metadata()
            if metadata is not None and metadata.get("format") not in ("pt", "tf", "flax", "mlx"):
                raise OSError(f"Invalid safetensors format metadata in {checkpoint_file}")
            dtypes = modeling_utils.str_to_torch_dtype
            result = {}
            for key in shard.keys():
                entry = shard.get_slice(key)
                dtype = entry.get_dtype()
                if dtype not in dtypes:
                    raise ValueError(f"Cannot load safetensors of unknown dtype {dtype}")
                result[key] = _TensorMetadata(entry.get_shape(), dtypes[dtype])
            return result

    _replace(stack, modeling_utils, "load_state_dict", load_state_dict)


def _borrow_cpu_payloads(stack, jt, torch, checkpoint_dir):
    from jittor.compat.torch.frontend import tensor_frontend
    from jittor.compat.torch.serialization import safetensors as codec

    if not callable(getattr(jt, "reinterpret_view", None)) or not callable(getattr(jt, "from_dlpack", None)):
        return

    root = os.path.realpath(checkpoint_dir)
    reader_type = codec._PySafeOpen
    original_init = reader_type.__init__
    original_entry = reader_type._entry
    original_tensor = codec._to_tensor

    def init(reader, filename, *args, **kwargs):
        original_init(reader, filename, *args, **kwargs)
        if os.path.realpath(os.path.dirname(filename)) == root:
            reader._partllm_mapping = mmap.mmap(reader._file.fileno(), 0, access=mmap.ACCESS_COPY)

    def entry(reader, key):
        mapping = getattr(reader, "_partllm_mapping", None)
        if mapping is None:
            return original_entry(reader, key)
        info = reader._header[key]
        start, end = info["data_offsets"]
        base = reader._data_offset
        return info["dtype"], info["shape"], memoryview(mapping)[base + start:base + end]

    def to_tensor(array, dtype, device):
        if (
            device not in (None, "cpu") or dtype not in ("BF16", "F32", "F16")
            or array.nbytes < 65536 or not array.flags.c_contiguous or not array.flags.writeable
        ):
            return original_tensor(array, dtype, device)
        with tensor_frontend(torch.Tensor, device="cpu"):
            value = jt.from_dlpack(array, copy=False)
        if dtype == "BF16":
            value = value.view(torch.bfloat16)
        value.requires_grad_(False)
        return value

    _replace(stack, reader_type, "__init__", init)
    _replace(stack, reader_type, "_entry", entry)
    _replace(stack, codec, "_to_tensor", to_tensor)


@contextmanager
def checkpoint_cpu_load(checkpoint_dir):
    import torch

    option = os.environ.get("PARTLLM_JTORCH_CPU_LOAD", "0")
    if option not in ("1", "2") or not hasattr(torch, "_torch_compat_install_context"):
        yield False
        return

    import jittor as jt
    import transformers

    fast = option == "2" and transformers.__version__ == "4.57.3"
    previous = jt.flags.use_cuda_host_allocator
    jt.flags.use_cuda_host_allocator = 0
    try:
        with jt.flag_scope(use_cuda=0), ExitStack() as stack:
            if fast:
                _metadata_reads(stack)
                _borrow_cpu_payloads(stack, jt, torch, checkpoint_dir)
            yield fast
            jt.sync_all(True)
    finally:
        jt.flags.use_cuda_host_allocator = previous


def load_finetuned_utonia(model, checkpoint, state):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import utonia

    if (
        state and not (checkpoint and os.path.isfile(checkpoint))
        and os.environ.get("PARTLLM_UTONIA_ALLOW_DOWNLOAD", "0") != "1"
    ):
        class CheckpointUtonia(utonia.model.PointTransformerV3):
            @staticmethod
            def _init_weights(module):
                pass

        encoder = CheckpointUtonia(
            in_channels=int(getattr(model.config, "utonia_in_channels", 9)),
            enc_mode=True,
            traceable=True,
            enc_channels=tuple(model.utonia_enc_channels),
        )
        if all(name in state and state[name].shape == value.shape
               for name, value in encoder.state_dict().items()):
            model.utonia_model = encoder.to(
                device=model.utonia_proj.weight.device,
                dtype=model.utonia_proj.weight.dtype,
            )
            return
    model.load_utonia_weights(ckpt_path=checkpoint)
