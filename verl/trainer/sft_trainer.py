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


import atexit
import os
import threading
from functools import partial

from tensordict.tensorclass import NonTensorData

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging

import hydra
import torch
if os.getenv("PYTORCH_DISABLE_CUDNN_SDP", "1") == "1" and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)
import torch.distributed
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint import CheckpointHandler
from verl.utils.dataset.dataset_utils import SFTTensorCollator
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import auto_set_device, get_device_name
from verl.utils.distributed import destroy_global_process_group
from verl.utils.logger import log_with_rank
from verl.utils.tracking import Tracking
from verl.workers.engine_workers import TrainingWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))

PART_SEG_REPORT_CATEGORIES = ["character", "weapon", "vehicle", "building"]
PART_SEG_REPORT_PROMPT_MODES = ["grounding", "open", "promptable"]
PART_SEG_REPORT_MODES = PART_SEG_REPORT_PROMPT_MODES + ["refine"]
PART_SEG_REPORT_DATASETS = [
    "partnext",
    "3dcompat200",
]


def _mean_scalar_metric(value):
    """Convert per-microbatch metric lists/tensors into one scalar for logging."""
    if isinstance(value, torch.Tensor):
        return value.detach().float().mean().item()
    if isinstance(value, (list, tuple)):
        scalars = []
        for item in value:
            if isinstance(item, torch.Tensor):
                scalars.extend(item.detach().float().reshape(-1).tolist())
            elif isinstance(item, (list, tuple)):
                scalars.append(_mean_scalar_metric(item))
            else:
                scalars.append(float(item))
        return torch.tensor(scalars, dtype=torch.float32).mean().item() if scalars else float("nan")
    return value


def _normalize_scalar_metrics(metrics: dict) -> dict:
    normalized = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            normalized[key] = {sub_key: _mean_scalar_metric(sub_value) for sub_key, sub_value in value.items()}
        else:
            normalized[key] = _mean_scalar_metric(value)
    return normalized


def _filter_part_seg_metrics_for_current_eval(metric_dict: dict) -> dict:
    """Keep the compact scalar validation report."""
    ordered_keys = [
        "val/ce_loss",
        "val/seg_loss",
        "val/seg_focal_loss",
        "val/seg_dice_loss",
        "val/seg_loss_grounding",
        "val/seg_loss_open",
        "val/seg_loss_promptable",
        "val/seg_loss_refine",
        "val/mIoU",
        "val/mIoU_grounding",
        "val/mIoU_open",
        "val/mIoU_promptable",
        "val/mIoU_refine",
        "val_partnext/seg_loss",
        "val_3dcompat200/seg_loss",
    ]
    filtered = {k: metric_dict.get(k, float("nan")) for k in ordered_keys}
    for k, v in metric_dict.items():
        if k.startswith("val/vis_sample_"):
            filtered[k] = v
    return filtered


class _ValLossAccumulator:
    """Accumulate per-sample CE / seg losses across the validation set.

    Aggregation keys are kept minimal: ``"all"`` for global stats and the
    four report modes (grounding/open/promptable/refine) for per-mode seg loss.
    Part-count bucket aggregation is omitted from the compact report.

    Per-key stats layout:
        ``[ce_sum, ce_count, seg_sum, seg_count,
           focal_sum, focal_count, dice_sum, dice_count]``

    Focal/Dice components are only populated for the ``"all"`` key, since
    we deliberately keep wandb panels uncluttered: per-mode breakdowns
    only report the combined ``seg_loss``.
    """

    _DATASET_KEYS = [f"dataset_{d}" for d in PART_SEG_REPORT_DATASETS]
    _MODE_DATASET_KEYS = [f"{pm}_dataset_{d}" for pm in PART_SEG_REPORT_PROMPT_MODES for d in PART_SEG_REPORT_DATASETS]
    _KEYS = ["all"] + PART_SEG_REPORT_MODES + _DATASET_KEYS + _MODE_DATASET_KEYS
    _STATS_PER_KEY = 8

    def __init__(self):
        self._stats = {key: [0.0] * self._STATS_PER_KEY for key in self._KEYS}

    @staticmethod
    def _prompt_mode_key(prompt_mode: str | None) -> str | None:
        if prompt_mode in PART_SEG_REPORT_MODES:
            return prompt_mode
        return None

    def update_ce(self, ce_loss: float, weight: float, dataset_type: str | None = None):
        if weight <= 0:
            return
        keys = ["all"]
        if dataset_type in PART_SEG_REPORT_DATASETS:
            keys.append(f"dataset_{dataset_type}")
        for key in keys:
            self._stats[key][0] += float(ce_loss) * float(weight)
            self._stats[key][1] += float(weight)

    def update_seg(
        self,
        seg_loss: float,
        prompt_mode: str | None = None,
        focal_loss: float | None = None,
        dice_loss: float | None = None,
        dataset_type: str | None = None,
    ):
        keys = ["all"]
        mode_key = self._prompt_mode_key(prompt_mode)
        if mode_key is not None:
            keys.append(mode_key)
        if dataset_type in PART_SEG_REPORT_DATASETS:
            keys.append(f"dataset_{dataset_type}")
            if mode_key in PART_SEG_REPORT_PROMPT_MODES:
                keys.append(f"{mode_key}_dataset_{dataset_type}")
        for key in keys:
            self._stats[key][2] += float(seg_loss)
            self._stats[key][3] += 1.0

        if focal_loss is not None:
            self._stats["all"][4] += float(focal_loss)
            self._stats["all"][5] += 1.0
        if dice_loss is not None:
            self._stats["all"][6] += float(dice_loss)
            self._stats["all"][7] += 1.0

    def compute(self, dp_group, device: torch.device) -> dict:
        flat = []
        for key in self._KEYS:
            flat.extend(self._stats[key])
        tensor = torch.tensor(flat, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        flat = tensor.cpu().tolist()

        result = {}
        for i, key in enumerate(self._KEYS):
            base = i * self._STATS_PER_KEY
            (
                ce_sum, ce_count,
                seg_sum, seg_count,
                focal_sum, focal_count,
                dice_sum, dice_count,
            ) = flat[base:base + self._STATS_PER_KEY]
            if key == "all":
                ce_key = "val/ce_loss"
                seg_key = "val/seg_loss"
                focal_key = "val/seg_focal_loss"
                dice_key = "val/seg_dice_loss"
            elif key in PART_SEG_REPORT_MODES:
                ce_key = f"val/ce_loss_{key}"
                seg_key = f"val/seg_loss_{key}"
                focal_key = None
                dice_key = None
            elif key.startswith("dataset_"):
                dataset = key[len("dataset_"):]
                ce_key = f"val_{dataset}/ce_loss"
                seg_key = f"val_{dataset}/seg_loss"
                focal_key = None
                dice_key = None
            else:
                ce_key = None
                seg_key = None
                focal_key = None
                dice_key = None
                for mode in PART_SEG_REPORT_PROMPT_MODES:
                    prefix = f"{mode}_dataset_"
                    if key.startswith(prefix):
                        dataset = key[len(prefix):]
                        seg_key = f"val_{dataset}/seg_loss_{mode}"
                        break
            if ce_count > 0 and ce_key is not None:
                result[ce_key] = ce_sum / ce_count
            if seg_count > 0 and seg_key is not None:
                result[seg_key] = seg_sum / seg_count
            if focal_key is not None and focal_count > 0:
                result[focal_key] = focal_sum / focal_count
            if dice_key is not None and dice_count > 0:
                result[dice_key] = dice_sum / dice_count
        return result


class _GPUKeepAlive:
    def __init__(self, rank: int):
        self.rank = rank


        self.enabled = os.getenv("VERL_GPU_KEEPALIVE_ENABLE", "0") == "1"
        matmul_size_env = os.getenv("VERL_GPU_KEEPALIVE_MATMUL_SIZE", "6144")
        try:
            self.matmul_size = int(matmul_size_env)
        except ValueError:
            logger.warning(f"[GPU keepalive] invalid VERL_GPU_KEEPALIVE_MATMUL_SIZE={matmul_size_env}, fallback to 6144")
            self.matmul_size = 6144

        util_target_env = os.getenv("VERL_GPU_KEEPALIVE_UTIL_TARGET", "70")
        try:
            self.util_target = int(util_target_env)
        except ValueError:
            logger.warning(f"[GPU keepalive] invalid VERL_GPU_KEEPALIVE_UTIL_TARGET={util_target_env}, fallback to 70")
            self.util_target = 70

        default_gemms = 8 if self.util_target >= 85 else (4 if self.util_target >= 70 else 2)
        gemms_env = os.getenv("VERL_GPU_KEEPALIVE_GEMMS_PER_LOOP", str(default_gemms))
        try:
            self.gemms_per_loop = max(1, int(gemms_env))
        except ValueError:
            logger.warning(f"[GPU keepalive] invalid VERL_GPU_KEEPALIVE_GEMMS_PER_LOOP={gemms_env}, fallback to {default_gemms}")
            self.gemms_per_loop = default_gemms

        self.dtype_name = os.getenv("VERL_GPU_KEEPALIVE_DTYPE", "bf16").lower()
        self._stop_event = threading.Event()
        self._thread = None
        self._device = None

    def _pick_dtype(self):
        if self.dtype_name == "fp16":
            return torch.float16
        if self.dtype_name == "bf16" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def start(self):
        if not self.enabled or not torch.cuda.is_available():
            return
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            return

        try:
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
        except ValueError:
            local_rank = 0
        local_rank = local_rank % device_count
        self._device = torch.device(f"cuda:{local_rank}")
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"gpu-keepalive-rank{self.rank}")
        self._thread.start()
        logger.warning(
            f"[GPU keepalive] started on {self._device}, matmul_size={self.matmul_size}, "
            f"gemms_per_loop={self.gemms_per_loop}, util_target={self.util_target}, dtype={self.dtype_name}"
        )

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=30)
        try:
            if self._device is not None:
                torch.cuda.synchronize(self._device)
        except Exception as e:
            logger.warning(f"[GPU keepalive] synchronize failed on rank {self.rank}: {e}")
        logger.warning(f"[GPU keepalive] stopped on rank {self.rank}")
        self._thread = None

    def _run(self):
        try:
            if self._device is None:
                return
            torch.cuda.set_device(self._device)
            dtype = self._pick_dtype()
            matmul_size = max(512, self.matmul_size)
            stream = torch.cuda.Stream(device=self._device)
            with torch.no_grad(), torch.cuda.stream(stream):
                a = torch.randn((matmul_size, matmul_size), device=self._device, dtype=dtype)
                b = torch.randn((matmul_size, matmul_size), device=self._device, dtype=dtype)
                while not self._stop_event.is_set():
                    for _ in range(self.gemms_per_loop):
                        if self._stop_event.is_set():
                            break
                        _ = a @ b
                    stream.synchronize()
        except Exception as e:
            logger.warning(f"[GPU keepalive] background loop failed on rank {self.rank}: {e}")


class SFTTrainer:
    def __init__(
        self,
        config,
    ):
        self.config = config

        self.rank = torch.distributed.get_rank()
        self._gpu_keepalive = _GPUKeepAlive(rank=self.rank)
        self._gpu_keepalive.start()
        atexit.register(self._gpu_keepalive.stop)

        self._build_config()
        self._build_dataset()

        self._build_engine()

        self._build_dataloader()

        self._init_engine()

        self._build_ckpt_handler()


        self.resume_global_step = self.ckpt_handler.load_checkpoint()

        self.device_name = self.config.trainer.device


        self.val_vis_samples = getattr(self.config.trainer, "val_vis_samples", 4)

        if self.rank == 0:
            print(self.config)

    def _build_ckpt_handler(self):
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)
        default_hdfs_dir = getattr(self.config.trainer, "default_hdfs_dir", None)

        self.ckpt_handler = CheckpointHandler(
            engine=self.engine,
            train_dataloader=self.train_dataloader,
            default_local_dir=self.config.trainer.default_local_dir,
            max_ckpt_to_keep=max_ckpt_to_keep,
            default_hdfs_dir=default_hdfs_dir,
            resume_mode=resume_mode,
            resume_from_path=resume_from_path,
        )

    def _build_config(self):
        from verl.utils.config import omega_conf_to_dataclass

        self.model_config = omega_conf_to_dataclass(self.config.model)
        self.engine_config = omega_conf_to_dataclass(self.config.engine)
        self.optimizer_config = omega_conf_to_dataclass(self.config.optim)
        self.checkpoint_config = omega_conf_to_dataclass(self.config.checkpoint)
        self.profiler_config = omega_conf_to_dataclass(self.config.profiler)


        self.profiler_interval = self.config.trainer.profile_interval
        self._validate_profiler_interval()

    def _stop_gpu_keepalive(self, reason: str):
        if self._gpu_keepalive is None:
            return
        logger.warning(f"[GPU keepalive] stopping before {reason} on rank {self.rank}")
        self._gpu_keepalive.stop()
        self._gpu_keepalive = None

    def _validate_profiler_interval(self):
        assert len(self.profiler_interval) == 2
        self.start_profile_step = self.profiler_interval[0]
        self.end_profile_step = self.profiler_interval[1]
        assert self.end_profile_step >= self.start_profile_step
        if self.start_profile_step < 0:
            assert self.end_profile_step < 0

    def _build_engine(self):
        from verl.workers.engine_workers import TrainingWorkerConfig
        from verl.workers.utils.losses import sft_loss

        self._stop_gpu_keepalive("distributed engine initialization")
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            logger.warning(f"[GPU keepalive] waiting for all ranks before device mesh init on rank {self.rank}")
            torch.distributed.barrier()

        self.loss_fn = partial(sft_loss, config=None)

        config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )

        self.training_client = TrainingWorker(config=config)
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)

        self.engine = self.training_client.engine

    def _init_engine(self):

        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = self.config.trainer.total_training_steps
        else:
            self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        self.optimizer_config.total_training_steps = self.total_training_steps

        self.steps_per_epoch = len(self.train_dataloader)


        self.save_freq = self.config.trainer.save_freq
        if self.save_freq == "after_each_epoch":
            self.save_freq = self.steps_per_epoch

        self.test_freq = self.config.trainer.test_freq
        if self.test_freq == "after_each_epoch":
            self.test_freq = self.steps_per_epoch

        self.training_client.reset()

    def _build_dataset(self):
        config = self.config
        tokenizer = self.model_config.tokenizer
        processor = self.model_config.processor
        train_dataset = create_sft_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        if config.data.val_files:
            val_dataset = create_sft_dataset(
                config.data.val_files,
                config.data,
                tokenizer,
                processor,
                max_samples=config.data.get("val_max_samples", -1),
                is_train=False,
            )
        else:
            val_dataset = None

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

    def _build_dataloader(self):

        config = self.config




        device_name = get_device_name()

        dp_rank = self.engine.get_data_parallel_rank()
        dp_size = self.engine.get_data_parallel_size()

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
        )

        self.global_batch_size = config.data.train_batch_size
        self.train_batch_size_per_dp = self.global_batch_size // dp_size
        self.collate_fn = SFTTensorCollator(config.data.pad_mode)

        dataloader_num_workers = config.data.get("dataloader_num_workers", 8)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=dataloader_num_workers,
            pin_memory=False,
            drop_last=True,
            pin_memory_device=device_name,
        )

        if self.val_dataset:
            self.val_sampler = DistributedSampler(
                self.val_dataset, shuffle=False, num_replicas=dp_size, rank=dp_rank, drop_last=True
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.val_sampler,
                collate_fn=self.collate_fn,
                num_workers=dataloader_num_workers,
                pin_memory=False,
                drop_last=False,
                pin_memory_device=device_name,
            )
        else:
            self.val_dataloader = None

    def _get_batch_seqlens(self, data):

        is_nested = data["input_ids"].is_nested
        if is_nested:
            batch_seqlens: torch.Tensor = data["input_ids"].offsets().diff()
        else:
            batch_seqlens: torch.Tensor = data["attention_mask"].sum(dim=-1)
        batch_seqlens = batch_seqlens.to(self.device_name)

        output_tensor = torch.empty(
            (batch_seqlens.shape[0] * self.engine.get_data_parallel_size(),),
            dtype=batch_seqlens.dtype,
            device=self.device_name,
        )

        torch.distributed.all_gather_into_tensor(
            output_tensor=output_tensor,
            input_tensor=batch_seqlens,
            group=self.engine.get_data_parallel_group(),
        )

        batch_seqlens = output_tensor.tolist()
        return batch_seqlens

    def fit(self):
        is_output_rank = self.engine.is_mp_src_rank_with_outputs()
        is_logging = self.rank == 0

        if is_logging:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step
        last_valid_metric = None

        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=0,
            log_only_rank_0=True,
        )



        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=0,
                log_only_rank_0=True,
            )


        start_epoch = global_step // self.steps_per_epoch

        meta_info = {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
            "seg_loss_weight": getattr(self.engine.model_config.hf_config, "seg_loss_weight", 1.0),




            "joint_focal_gamma": getattr(self.engine.model_config.hf_config, "joint_focal_gamma", 2.0),
            "joint_lambda_dice": getattr(self.engine.model_config.hf_config, "joint_lambda_dice", 2.0),
            "joint_exclude_bg": getattr(self.engine.model_config.hf_config, "joint_exclude_bg", True),
        }

        train_time = 0
        total_tokens = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=not is_logging,
                )
            ):
                global_step += 1

                if self._gpu_keepalive is not None:
                    self._gpu_keepalive.stop()
                    self._gpu_keepalive = None


                data = tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)
                batch_seqlens = self._get_batch_seqlens(data=data)

                batch_seqlens_ntd = NonTensorData(batch_seqlens)

                tu.assign_non_tensor(data, update_lr_scheduler=True, global_token_num=batch_seqlens_ntd)


                if global_step == self.start_profile_step:
                    self.training_client.start_profile()

                output = self.training_client.train_batch(data=data)

                if global_step == self.end_profile_step:
                    self.training_client.stop_profile()

                if is_output_rank:
                    metrics = _normalize_scalar_metrics(tu.get(output, "metrics"))

                    metrics["train/loss"] = metrics.pop("loss")
                    if "ce_loss" in metrics:
                        metrics["train/ce_loss"] = metrics.pop("ce_loss")

                    for _key in list(metrics.keys()):
                        if _key == "seg_loss" or (_key.startswith("seg_") and _key.endswith("_loss")):
                            metrics[f"train/{_key}"] = metrics.pop(_key)
                    if "seg_loss_weight" in metrics:
                        metrics["train/seg_loss_weight"] = metrics.pop("seg_loss_weight")
                    if "seg_loss_skipped" in metrics:
                        metrics["train/seg_loss_skipped"] = metrics.pop("seg_loss_skipped")
                    metrics["train/grad_norm"] = metrics.pop("grad_norm")


                    module_grad_norms = metrics.pop("module_grad_norms", None)
                    if module_grad_norms is not None:
                        for group_name, gn in module_grad_norms.items():
                            metrics[f"train/grad_norm_{group_name}"] = gn


                    lr_value = metrics.pop("lr")
                    if isinstance(lr_value, dict):
                        for group_name, group_lr in lr_value.items():
                            metrics[f"train/lr_{group_name}"] = group_lr

                        metrics["train/lr"] = lr_value.get("llm", next(iter(lr_value.values())))
                    else:
                        metrics["train/lr"] = lr_value

                    metrics["train/mfu"] = metrics.pop("mfu")
                    metrics["train/global_tokens"] = torch.sum(
                        torch.tensor(batch_seqlens, device=self.device_name)
                    ).item()
                    total_tokens += metrics["train/global_tokens"]
                    metrics["train/total_tokens(B)"] = total_tokens / 1e9

                    if is_logging:
                        tracking.log(data=metrics, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.test_freq == 0
                is_save_step = global_step % self.save_freq == 0


                if self.val_dataloader is not None and (is_last_step or (self.test_freq > 0 and is_valid_step)):

                    val_loss_accumulator = _ValLossAccumulator()


                    try:
                        from recipe.part_seg.eval_metrics import (
                            EvalAccumulator,
                            PROMPT_MODE_NAMES,
                            DATASET_TYPE_NAMES,
                            MACRO_CATEGORY_NAMES,
                            SUBMODE_NAMES,
                            compute_sample_miou,
                            save_classification_ply,
                        )
                        eval_accumulator = EvalAccumulator()
                        has_eval_metrics = True
                    except ImportError:
                        has_eval_metrics = False

                    vis_count = 0

                    for val_data in self.val_dataloader:
                        val_data = tu.get_tensordict(tensor_dict=val_data, non_tensor_dict=meta_info)
                        output = self.training_client.infer_batch(val_data)

                        if is_output_rank:
                            metrics = tu.get(output, "metrics")

                            if "macro_category_id" in output.keys():
                                macro_id_list = output["macro_category_id"].unbind()
                            else:
                                macro_id_list = []
                            macro_names = []
                            for macro_id_i in macro_id_list:
                                macro_idx = int(macro_id_i.reshape(-1)[0].item())
                                if has_eval_metrics and 0 <= macro_idx < len(MACRO_CATEGORY_NAMES):
                                    macro_names.append(MACRO_CATEGORY_NAMES[macro_idx])
                                else:
                                    macro_names.append("unknown")

                            if "dataset_type_id" in output.keys():
                                ds_id_list_for_loss = output["dataset_type_id"].unbind()
                            else:
                                ds_id_list_for_loss = []
                            dataset_names = []
                            for ds_id_i in ds_id_list_for_loss:
                                ds_idx = int(ds_id_i.reshape(-1)[0].item())
                                if has_eval_metrics and 0 <= ds_idx < len(DATASET_TYPE_NAMES):
                                    dataset_names.append(DATASET_TYPE_NAMES[ds_idx])
                                else:
                                    dataset_names.append("unknown")


                            if "log_probs" in output.keys():
                                log_probs = output["log_probs"]
                                loss_mask = val_data["loss_mask"]
                                if log_probs.is_nested:
                                    log_prob_samples = log_probs.unbind()
                                    loss_mask_samples = loss_mask.unbind()
                                else:
                                    log_prob_samples = log_probs
                                    loss_mask_samples = loss_mask
                                for sample_idx, (log_prob_i, loss_mask_i) in enumerate(
                                    zip(log_prob_samples, loss_mask_samples, strict=False)
                                ):
                                    shifted_mask = torch.roll(loss_mask_i.to(log_prob_i.device), shifts=-1, dims=0).float()
                                    token_count = float(shifted_mask.sum().item())
                                    if token_count <= 0:
                                        continue
                                    ce_i = -((log_prob_i.float() * shifted_mask).sum() / shifted_mask.sum()).item()
                                    category_name = macro_names[sample_idx] if sample_idx < len(macro_names) else None
                                    dataset_name = dataset_names[sample_idx] if sample_idx < len(dataset_names) else None
                                    val_loss_accumulator.update_ce(ce_i, token_count, dataset_type=dataset_name)


                            if (
                                has_eval_metrics
                                and "diffusion_outputs" in output.keys()
                                and "classification_labels" in output.keys()
                            ):
                                logits_list = output["diffusion_outputs"].unbind()
                                labels_list = output["classification_labels"].unbind()


                                if "prompt_mode_id" in output.keys():
                                    mode_id_list = output["prompt_mode_id"].unbind()
                                else:
                                    mode_id_list = [None] * len(logits_list)


                                if "dataset_type_id" in output.keys():
                                    ds_id_list = output["dataset_type_id"].unbind()
                                else:
                                    ds_id_list = [None] * len(logits_list)


                                if "submode_id" in output.keys():
                                    submode_id_list = output["submode_id"].unbind()
                                else:
                                    submode_id_list = [None] * len(logits_list)

                                if "macro_category_id" in output.keys():
                                    macro_id_list = output["macro_category_id"].unbind()
                                else:
                                    macro_id_list = [None] * len(logits_list)


                                if "val_coords" in output.keys():
                                    coords_list = output["val_coords"].unbind()
                                else:
                                    coords_list = [None] * len(logits_list)

                                for logits_i, labels_i, mode_id_i, coords_i, ds_id_i, submode_id_i, macro_id_i in zip(
                                    logits_list,
                                    labels_list,
                                    mode_id_list,
                                    coords_list,
                                    ds_id_list,
                                    submode_id_list,
                                    macro_id_list,
                                ):
                                    miou = compute_sample_miou(logits_i, labels_i)


                                    if mode_id_i is not None:
                                        mode_name = PROMPT_MODE_NAMES[mode_id_i.squeeze().item()]
                                    else:
                                        mode_name = "grounding"


                                    if ds_id_i is not None:
                                        ds_name = DATASET_TYPE_NAMES[ds_id_i.squeeze().item()]
                                    else:
                                        ds_name = None


                                    if submode_id_i is not None:
                                        submode_name = SUBMODE_NAMES[submode_id_i.squeeze().item()]
                                    else:
                                        submode_name = None
                                    report_mode_name = (
                                        "refine"
                                        if mode_name == "promptable" and submode_name == "refine"
                                        else mode_name
                                    )


                                    if macro_id_i is not None:
                                        macro_idx = int(macro_id_i.reshape(-1)[0].item())
                                        if 0 <= macro_idx < len(MACRO_CATEGORY_NAMES):
                                            macro_name = MACRO_CATEGORY_NAMES[macro_idx]
                                        else:
                                            macro_name = "unknown"
                                    else:
                                        macro_name = None

                                    eval_accumulator.update(miou, report_mode_name, None, ds_name)




                                    from verl.workers.utils.losses import compute_joint_mask_loss
                                    joint_total, joint_parts = compute_joint_mask_loss(
                                        logits_i.float(),
                                        labels_i.to(logits_i.device).long(),
                                    )
                                    val_loss_accumulator.update_seg(
                                        seg_loss=joint_total.detach().item(),
                                        prompt_mode=report_mode_name,
                                        focal_loss=joint_parts["focal"].detach().item() if "focal" in joint_parts else None,
                                        dice_loss=joint_parts["dice"].detach().item() if "dice" in joint_parts else None,
                                        dataset_type=ds_name,
                                    )


                                    if (
                                        self.rank == 0
                                        and vis_count < self.val_vis_samples
                                        and coords_i is not None
                                        and mode_name == "open"
                                    ):
                                        vis_dir = os.path.join(
                                            self.config.trainer.default_local_dir,
                                            "val_vis",
                                            f"step_{global_step}",
                                            f"sample_{vis_count}",
                                        )
                                        points = coords_i.squeeze(0).float().cpu().numpy()
                                        pred_labels = logits_i.squeeze(0).argmax(dim=0).cpu().numpy()
                                        gt_labels = labels_i.squeeze(0).long().cpu().numpy()

                                        save_classification_ply(points, pred_labels, os.path.join(vis_dir, "pred.ply"))
                                        save_classification_ply(points, gt_labels, os.path.join(vis_dir, "gt.ply"))
                                        vis_count += 1


                    seg_metrics = {}
                    loss_metrics = {}
                    if is_output_rank:

                        loss_metrics = val_loss_accumulator.compute(
                            dp_group=self.engine.get_data_parallel_group(),
                            device=self.device_name,
                        )
                        if has_eval_metrics:
                            seg_metrics = eval_accumulator.compute(
                                dp_group=self.engine.get_data_parallel_group(),
                                device=self.device_name,
                            )


                    if is_logging:
                        val_metric = {}


                        val_metric.update(loss_metrics)
                        val_metric.update(seg_metrics)


                        if bool(self.config.data.get("part_seg_report_21_metrics_only", False)):
                            val_metric = _filter_part_seg_metrics_for_current_eval(val_metric)


                        if self.rank == 0 and vis_count > 0:
                            try:
                                import wandb
                                for vi in range(vis_count):
                                    vis_dir = os.path.join(
                                        self.config.trainer.default_local_dir,
                                        "val_vis",
                                        f"step_{global_step}",
                                        f"sample_{vi}",
                                    )
                                    pred_path = os.path.join(vis_dir, "pred.ply")
                                    gt_path = os.path.join(vis_dir, "gt.ply")
                                    if os.path.exists(pred_path):
                                        val_metric[f"val/vis_sample_{vi}_pred"] = wandb.Object3D(pred_path)
                                    if os.path.exists(gt_path):
                                        val_metric[f"val/vis_sample_{vi}_gt"] = wandb.Object3D(gt_path)
                            except (ImportError, Exception):
                                pass

                        tracking.log(data=val_metric, step=global_step)
                        last_valid_metric = val_metric
                    torch.distributed.barrier()

                if is_last_step or (self.save_freq > 0 and is_save_step):
                    self.ckpt_handler.save_checkpoint(step=global_step)

                if is_last_step:
                    if is_logging:
                        print(f"Total time for train steps: {train_time:.2f}s")
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_sft(config):
    from verl.utils.distributed import initialize_global_process_group

    initialize_global_process_group()
    trainer = SFTTrainer(config=config)
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config):

    auto_set_device(config)
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer, processor, max_samples=-1, **kwargs):
    """Create a dataset.

    Extra kwargs (e.g. is_train) are forwarded to the dataset constructor
    only if the constructor accepts them (checked via inspect.signature).
    """
    import inspect



    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_object

        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
    else:

        dataset_cls = MultiTurnSFTDataset


    if kwargs:
        try:
            sig = inspect.signature(dataset_cls.__init__)
            accepted = set(sig.parameters.keys()) - {"self"}
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if not has_var_keyword:
                kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        except (ValueError, TypeError):
            kwargs = {}


    dataset = dataset_cls(
        parquet_files=data_paths, tokenizer=tokenizer, config=data_config, processor=processor, max_samples=max_samples, **kwargs
    )
    return dataset


if __name__ == "__main__":
    main()
