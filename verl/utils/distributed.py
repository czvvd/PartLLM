# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for distributed training."""

import ctypes
import os
import socket
from datetime import timedelta

import ray
import torch
import torch.distributed

from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device, is_npu_available


def set_numa_affinity():
    if is_npu_available:

        return

    initialized = False
    try:
        libnuma = ctypes.CDLL("libnuma.so")
        if libnuma.numa_available() < 0:
            return

        import pynvml

        pynvml.nvmlInit()
        initialized = True
        device_name = "NPU" if is_npu_available else "GPU"
        local_rank = int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        pynvml.nvmlDeviceSetCpuAffinity(handle)
    except ImportError:
        print("Warning: pynvml not available, skipping NUMA affinity setup")
    except Exception as e:
        print(f"Warning: Failed to set NUMA affinity: {e}")
    finally:
        if initialized:
            pynvml.nvmlShutdown()


def _safe_cuda_probe(probe):
    try:
        return probe()
    except Exception as exc:
        return f"<error:{exc}>"


def _parse_visible_devices(visible_devices: str | None) -> list[str] | None:
    if visible_devices is None:
        return None

    visible_devices = visible_devices.strip()
    if not visible_devices:
        return []

    return [device.strip() for device in visible_devices.split(",") if device.strip()]


def _build_cuda_init_debug_info(local_rank: int, rank: int, world_size: int) -> str:
    return (
        f"[verl.distributed] rank={rank} local_rank={local_rank} world_size={world_size} "
        f"local_world_size={os.environ.get('LOCAL_WORLD_SIZE', '<unset>')} "
        f"host={socket.gethostname()} pid={os.getpid()} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"cuda_available={_safe_cuda_probe(torch.cuda.is_available)} "
        f"device_count={_safe_cuda_probe(torch.cuda.device_count)} "
        f"DIST_INIT_METHOD={os.environ.get('DIST_INIT_METHOD', '<unset>')}"
    )


def _validate_cuda_local_rank(local_rank: int, rank: int, world_size: int) -> None:
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if local_rank < 0 or local_rank >= local_world_size:
        raise RuntimeError(
            "LOCAL_RANK is outside LOCAL_WORLD_SIZE before CUDA initialization. "
            f"{_build_cuda_init_debug_info(local_rank=local_rank, rank=rank, world_size=world_size)}"
        )

    visible_devices = _parse_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if visible_devices is not None and local_rank >= len(visible_devices):
        raise RuntimeError(
            "LOCAL_RANK is outside the visible CUDA device range before CUDA initialization. "
            f"visible_device_count={len(visible_devices)} "
            f"{_build_cuda_init_debug_info(local_rank=local_rank, rank=rank, world_size=world_size)}"
        )

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            raise RuntimeError(
                "LOCAL_RANK is outside torch.cuda.device_count() before CUDA initialization. "
                f"device_count={device_count} "
                f"{_build_cuda_init_debug_info(local_rank=local_rank, rank=rank, world_size=world_size)}"
            )


def get_device_mesh_timeout_second() -> int | None:
    timeout_second = os.environ.get("VERL_DEVICE_MESH_TIMEOUT_SECONDS")
    if timeout_second is None:
        timeout_second = os.environ.get("VERL_GLOBAL_PG_TIMEOUT_SECONDS")

    if timeout_second in (None, ""):
        return None

    return int(timeout_second)


def set_process_group_timeout(timeout_second: int | None, group=None) -> None:
    if timeout_second is None:
        return

    from torch.distributed.distributed_c10d import _set_pg_timeout

    _set_pg_timeout(timedelta(seconds=timeout_second), group=group)


def set_device_mesh_timeout(device_mesh, timeout_second: int | None = None) -> None:
    if timeout_second is None:
        timeout_second = get_device_mesh_timeout_second()

    if timeout_second is None:
        return

    for group in device_mesh.get_all_groups():
        set_process_group_timeout(timeout_second=timeout_second, group=group)


def initialize_global_process_group(timeout_second=36000):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device_name = get_device_name()
    backend = get_nccl_backend()

    if device_name == "cuda":
        print(_build_cuda_init_debug_info(local_rank=local_rank, rank=rank, world_size=world_size), flush=True)
        _validate_cuda_local_rank(local_rank=local_rank, rank=rank, world_size=world_size)

    accelerator_device = None
    if device_name != "cpu":
        get_torch_device().set_device(local_rank)
        print(
            f"[verl.distributed] rank={rank} bound {device_name}:{local_rank} before init_process_group",
            flush=True,
        )
        if device_name == "cuda":
            accelerator_device = torch.device(device_name, local_rank)

    init_process_group_kwargs = {
        "backend": backend,
        "timeout": timedelta(seconds=timeout_second),
        "init_method": os.environ.get("DIST_INIT_METHOD", None),
    }




    use_pg_device_id = accelerator_device is not None and os.environ.get("VERL_ENABLE_PG_DEVICE_ID", "0") == "1"
    if use_pg_device_id:
        init_process_group_kwargs["device_id"] = accelerator_device
        print(
            f"[verl.distributed] rank={rank} init_process_group will bind device_id={accelerator_device}",
            flush=True,
        )
    elif accelerator_device is not None:
        print(
            f"[verl.distributed] rank={rank} init_process_group without device_id to avoid DeviceMesh eager subgroup connect",
            flush=True,
        )

    try:
        torch.distributed.init_process_group(**init_process_group_kwargs)
    except TypeError:
        init_process_group_kwargs.pop("device_id", None)
        torch.distributed.init_process_group(**init_process_group_kwargs)

    return local_rank, rank, world_size


def destroy_global_process_group():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def initialize_global_process_group_ray(timeout_second=None):


    import torch.distributed

    timeout = timedelta(seconds=timeout_second) if timeout_second is not None else None

    if not torch.distributed.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.distributed.init_process_group(
            backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
            rank=rank,
            world_size=world_size,
            timeout=timeout,
            init_method=os.environ.get("DIST_INIT_METHOD", None),
        )
