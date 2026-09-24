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
"""
The concrete Engine implementation using PyTorch FullyShardedDataParallel (FSDP)
"""

import gc
import logging
import os
import warnings
from contextlib import nullcontext
from typing import Callable, ContextManager, Optional

import torch
import torch.distributed
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.distributed import set_device_mesh_timeout
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    apply_fsdp2,
    collect_lora_params,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.model import convert_weight_keys, extract_multi_modal_inputs
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import enable_full_determinism, postprocess_batch_func, prepare_micro_batches
from .utils import create_device_mesh, get_sharding_strategy

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class FSDPEngine(BaseEngine):
    """
    Concrete Engine implementation using PyTorch FullyShardedDataParallel (FSDP).

    Supports model sharding, activation/optimizer offloading, LoRA, and sequence parallelism.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        """
        Initialize the FSDPEngine.

        Sets up distributed device meshes, LoRA, and offload policies based on config.

        Args:
            config: Configuration object with FSDP and model settings.
        """
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        self.mode = None

        self.rank = torch.distributed.get_rank()


        self.use_remove_padding = self.model_config.use_remove_padding

        self._init_device_mesh()

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)


        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._is_lora = self.model_config.lora_rank > 0

        if self.engine_config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.engine_config.use_torch_compile
            else entropy_from_logits
        )

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self):
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
        else:
            is_collect = True
        return is_collect

    def initialize(self):
        """
        Build the model, optimizer, and learning rate scheduler under FSDP.

        Applies device, dtype, and precision configurations, including mixed precision.
        Sets up checkpoint manager and FLOPs counter.
        """

        self._build_model_optimizer()

        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.module,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.model_config.get_processor(),
            checkpoint_config=self.checkpoint_config,
        )

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def _init_device_mesh(self):
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.engine_config.fsdp_size

        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
        set_device_mesh_timeout(self.device_mesh)
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.engine_config.ulysses_sequence_parallel_size
        dp_size = self.get_data_parallel_size()
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp_size, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )
            set_device_mesh_timeout(self.ulysses_device_mesh)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

    def _build_module(self):
        from verl.utils.model import get_hf_auto_model_class
        from verl.utils.torch_dtypes import PrecisionType

        torch_dtype = self.engine_config.model_dtype

        if torch_dtype is None:

            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16

        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not self.model_config.hf_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            auto_class = get_hf_auto_model_class(hf_config=self.model_config.hf_config)

            
            
            hf_cfg = self.model_config.hf_config
            text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
            target_vocab_size = text_cfg.vocab_size

            ckpt_vocab_size = None
            import json as _json, os as _os
            index_path = _os.path.join(self.model_config.local_path, "config.json")
            if _os.path.exists(index_path):
                with open(index_path) as _f:
                    ckpt_config = _json.load(_f)
                
                ckpt_vocab_size = ckpt_config.get("vocab_size") or ckpt_config.get("text_config", {}).get("vocab_size")

            need_resize = ckpt_vocab_size is not None and target_vocab_size != ckpt_vocab_size
            if need_resize:
                
                text_cfg.vocab_size = ckpt_vocab_size

            module = auto_class.from_pretrained(
                pretrained_model_name_or_path=self.model_config.local_path,
                torch_dtype=torch_dtype,
                config=self.model_config.hf_config,
                trust_remote_code=self.model_config.trust_remote_code,
                attn_implementation=self.model_config.override_config.get(
                    "attn_implementation", "flash_attention_2"
                ),
                ignore_mismatched_sizes=True,
            )

            if need_resize:
                
                module.resize_token_embeddings(target_vocab_size)
                text_cfg.vocab_size = target_vocab_size
                logger.info(f"Resized token embeddings: {ckpt_vocab_size} -> {target_vocab_size}")


            if self.model_config.drop_modules:
                from transformers.trainer_pt_utils import get_module_class_from_name

                for mod_name in self.model_config.drop_modules.split(","):
                    mod_name = mod_name.strip()

                    target = module
                    if hasattr(module, "model") and hasattr(module.model, mod_name):
                        target = module.model
                    if hasattr(target, mod_name):
                        delattr(target, mod_name)
                        logger.info(f"Dropped sub-module '{mod_name}' to save memory")
                    else:
                        logger.warning(f"Sub-module '{mod_name}' not found, skipping drop")


                no_split = getattr(module, "_no_split_modules", None)
                if no_split:
                    module._no_split_modules = [
                        cls_name for cls_name in no_split
                        if get_module_class_from_name(module, cls_name) is not None
                    ]
                    logger.info(f"_no_split_modules after drop: {module._no_split_modules}")



            _tokenizer_path = (
                self.model_config.local_tokenizer_path
                or self.model_config.tokenizer_path
                or self.model_config.local_path
            )
            from transformers import AutoTokenizer as _AutoTokenizer

            _tokenizer = _AutoTokenizer.from_pretrained(_tokenizer_path, trust_remote_code=True)
            _current_vocab_size = getattr(
                module.config, "vocab_size", getattr(module.config, "text_config", module.config).vocab_size
            )
            if len(_tokenizer) > _current_vocab_size:
                logger.info(
                    f"Resizing token embeddings: {_current_vocab_size} -> {len(_tokenizer)}"
                )
                module.resize_token_embeddings(len(_tokenizer))

                try:
                    from recipe.part_seg.loc_token_utils import init_loc_token_embeddings

                    init_loc_token_embeddings(module, _tokenizer)
                except ImportError:
                    logger.warning("loc_token_utils not found, skipping loc embedding init")
            del _tokenizer


            use_liger = self.model_config.use_liger

            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=module)

            fused_kernel_options = self.model_config.fused_kernel_options
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            use_fused_kernels = self.model_config.use_fused_kernels
            apply_monkey_patch(
                model=module,
                use_remove_padding=self.use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )


            module.to(torch_dtype)

            if self.model_config.enable_gradient_checkpointing:
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        
        
        device = torch.device(f"cuda:{torch.distributed.get_rank() % torch.cuda.device_count()}")
        encoder_type = getattr(self.model_config.hf_config, "point_cloud_encoder_type", "utonia")
        if encoder_type != "utonia":
            raise ValueError("PartLLM supports only the Utonia point-cloud encoder")
        utonia_ckpt = getattr(self.model_config.hf_config, "utonia_ckpt_path", None)
        module.model.load_utonia_weights(ckpt_path=utonia_ckpt)

        if os.environ.get("UTONIA_LOAD_FROM_HF_CKPT", "0") == "1":
            from safetensors import safe_open
            from glob import glob as _glob

            _hf_ckpt = self.model_config.local_path
            _shards = sorted(_glob(os.path.join(_hf_ckpt, "*.safetensors")))
            if not _shards:
                if self.rank == 0:
                    print(
                        f"[UtoniaOverride] no safetensors found in {_hf_ckpt}; keeping base Utonia weights",
                        flush=True,
                    )
            else:
                _prefix = "model.utonia_model."
                _sd = {}
                for _shard in _shards:
                    with safe_open(_shard, framework="pt", device="cpu") as _f:
                        for _key in _f.keys():
                            if _key.startswith(_prefix):
                                _sd[_key[len(_prefix) :]] = _f.get_tensor(_key)
                if not _sd:
                    if self.rank == 0:
                        print(
                            f"[UtoniaOverride] no '{_prefix}*' keys in {_hf_ckpt}; keeping base Utonia weights",
                            flush=True,
                        )
                else:
                    _missing, _unexpected = module.model.utonia_model.load_state_dict(_sd, strict=False)
                    if self.rank == 0:
                        print(
                            f"[UtoniaOverride] loaded {len(_sd)} keys from {_hf_ckpt}; "
                            f"missing={len(_missing)}, unexpected={len(_unexpected)}",
                            flush=True,
                        )
        module.model.utonia_model.eval()
        module.model.utonia_model.to(device)
        module.model.utonia_proj.to(device)
        if hasattr(module.model, "diffusion_model"):
            module.model.diffusion_model.to(device)
            module.model.diffusion_model.enable_xformers_memory_efficient_attention()

        return module

    def _build_lora_module(self, module):
        module.enable_input_require_grads()

        lora_adapter_path = getattr(self.model_config, "lora_adapter_path", None)
        if lora_adapter_path is not None:
            from peft import PeftModel

            from verl.utils.fs import copy_to_local

            print(f"Loading pre-trained LoRA adapter to from: {lora_adapter_path}")

            local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.model_config.use_shm)

            module = PeftModel.from_pretrained(module, local_adapter_path, is_trainable=True)
            peft_config = module.peft_config["default"]

            if isinstance(peft_config.task_type, str):
                peft_config.task_type = TaskType.CAUSAL_LM
        else:

            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": self.model_config.lora_rank,
                "lora_alpha": self.model_config.lora_alpha,
                "target_modules": convert_to_regular_types(self.model_config.target_modules),
                "exclude_modules": convert_to_regular_types(self.model_config.exclude_modules),
                "bias": "none",
            }
            module = get_peft_model(module, LoraConfig(**lora_config))

        return module

    def _build_fsdp_module(self, module):
        from torch.distributed.fsdp import CPUOffload, MixedPrecision

        from verl.utils.torch_dtypes import PrecisionType

        mixed_precision_config = self.engine_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=module,
            config=self.engine_config.wrap_policy,
            is_lora=self.model_config.lora_rank > 0,
        )

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)


        if self.engine_config.strategy == "fsdp":







            cpu_offload = None
            if self.engine_config.forward_only:
                cpu_offload = CPUOffload(offload_params=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False

            module = FSDP(
                module,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=self.engine_config.forward_prefetch,
                use_orig_params=self.engine_config.use_orig_params,
                cpu_offload=cpu_offload,
            )
        elif self.engine_config.strategy == "fsdp2":



            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            offload_policy = None
            if self.engine_config.offload_policy or self.engine_config.forward_only:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": self.engine_config.reshard_after_forward,
            }
            full_state = module.state_dict()
            apply_fsdp2(module, fsdp_kwargs, self.engine_config)
            fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {self.engine_config.strategy}")

        if self.model_config.enable_activation_offload:
            enable_gradient_checkpointing = self.model_config.enable_gradient_checkpointing
            enable_activation_offloading(module, self.engine_config.strategy, enable_gradient_checkpointing)

        if torch.distributed.get_world_size() == 1 and fsdp_version(module) == 1:
            FSDP.set_state_dict_type(
                module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(module) == 1:
            FSDP.set_state_dict_type(
                module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        return module

    def _build_optimizer(self, module):
        from verl.workers.config.optimizer import build_optimizer

        config = self.optimizer_config
        has_module_lr = any(
            getattr(config, f"lr_{m}", None) is not None for m in ["encoder", "mapping", "llm", "decoder"]
        )

        if has_module_lr:
            param_groups = self._build_param_groups(module)
            optimizer = build_optimizer(param_groups, config)
        else:
            optimizer = build_optimizer(module.parameters(), config)

        return optimizer

    def _build_param_groups(self, module):
        """Build parameter groups with per-module learning rates.

        Module matching rules:
          - mapping: Utonia `model.utonia_proj.`
          - encoder: Utonia `model.utonia_model.`
          - decoder: `model.mask_decoder.`
          - llm: everything else
        """
        config = self.optimizer_config
        base_lr = config.lr

        mapping_prefixes = ("model.utonia_proj.",)
        encoder_prefixes = ("model.utonia_model.",)
        decoder_prefixes = ("model.mask_decoder.",)

        groups = {
            "encoder": {"params": [], "lr": config.lr_encoder if config.lr_encoder is not None else base_lr},
            "mapping": {"params": [], "lr": config.lr_mapping if config.lr_mapping is not None else base_lr},
            "decoder": {"params": [], "lr": config.lr_decoder if config.lr_decoder is not None else base_lr},
            "llm": {"params": [], "lr": config.lr_llm if config.lr_llm is not None else base_lr},
        }

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue


            clean_name = name.replace("_fsdp_wrapped_module.", "")

            if any(clean_name.startswith(p) for p in mapping_prefixes):
                groups["mapping"]["params"].append(param)
            elif any(clean_name.startswith(p) for p in encoder_prefixes):
                groups["encoder"]["params"].append(param)
            elif any(clean_name.startswith(p) for p in decoder_prefixes):
                groups["decoder"]["params"].append(param)
            else:
                groups["llm"]["params"].append(param)


        param_groups = []
        for group_name, group_dict in groups.items():
            if len(group_dict["params"]) > 0:
                param_groups.append({
                    "params": group_dict["params"],
                    "lr": group_dict["lr"],
                    "group_name": group_name,
                })


        if self.rank == 0:
            for pg in param_groups:
                num_params = sum(p.numel() for p in pg["params"])
                print(f"[ParamGroup] {pg['group_name']}: {num_params:,} params, lr={pg['lr']}")

        return param_groups

    def _build_lr_scheduler(self, optimizer):
        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        optim_config = self.optimizer_config

        total_steps = optim_config.total_training_steps
        num_warmup_steps = optim_config.lr_warmup_steps
        lr_scheduler_type = optim_config.lr_scheduler_type
        min_lr_ratio = optim_config.min_lr_ratio
        num_cycles = optim_config.num_cycles
        if num_warmup_steps <= 0:
            num_warmup_steps_ratio = optim_config.lr_warmup_steps_ratio
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        if lr_scheduler_type == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
        elif lr_scheduler_type == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
            )
        else:
            raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")
        return lr_scheduler

    def _build_model_optimizer(self):
        from verl.utils.model import print_model_size


        module = self._build_module()

        if self._is_lora:
            module = self._build_lora_module(module)


        torch.distributed.barrier()
        if self.rank == 0:
            print_model_size(module)
        log_gpu_memory_usage("After init model from HF AutoModel", logger=logger)


        log_gpu_memory_usage("Before FSDP", logger=None)
        module = self._build_fsdp_module(module)
        log_gpu_memory_usage("After FSDP", logger=None)

        if not self.engine_config.forward_only:

            optimizer = self._build_optimizer(module)

            lr_scheduler = self._build_lr_scheduler(optimizer)
        else:
            optimizer = None
            lr_scheduler = None

        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train_mode(self, **kwargs):
        """
        Return a context manager that switches to training mode with FSDP-specific handling.

        Includes parameter and optimizer offload entry/exit.
        """
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """
        Return a context manager that switches to evaluation mode with FSDP-specific handling.

        Includes activation offload entry/exit.
        """
        return EngineEvalModeCtx(self, **kwargs)

    def get_data_parallel_rank(self):
        if self.ulysses_device_mesh is not None:
            return self.ulysses_device_mesh["dp"].get_local_rank()
        else:
            return torch.distributed.get_rank()

    def get_data_parallel_size(self):
        
        return torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size

    def get_data_parallel_group(self):
        if self.ulysses_device_mesh is not None:
            return self.ulysses_device_mesh.get_group(mesh_dim="dp")
        else:
            return torch.distributed.group.WORLD

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> list[TensorDict]:

        tu.assign_non_tensor(data, sp_size=self.ulysses_sequence_parallel_size)


        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        if len(data["multi_modal_inputs"]) > 0 and "diffusion_targets" in data["multi_modal_inputs"][0]:
            batch_decoder_tokens = torch.tensor(sum(f["diffusion_targets"].numel() for f in data["multi_modal_inputs"])).to(get_device_id())
            torch.distributed.all_reduce(batch_decoder_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group())
            tu.assign_non_tensor(data, batch_decoder_tokens=batch_decoder_tokens.item())

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )

        output_lst = []

        ctx = torch.no_grad() if forward_only else nullcontext()

        for micro_batch in micro_batches:
            with ctx:
                loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)

                if not forward_only:
                    loss.backward()

            output_lst.append(meta_info)


        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def optimizer_zero_grad(self):
        """
        Zero gradients and enforce FSDP grad-clipping logic.
        """
        self.optimizer.zero_grad()

    def optimizer_step(self):
        """
        Clip gradients, skip update if non-finite, and step optimizer.

        Returns:
            grad_norm (float): Norm of gradients before clipping.
            module_grad_norms (dict or None): Per-module grad norms if param groups are configured.
        """
        assert self.optimizer_config.clip_grad is not None


        module_grad_norms = self._compute_module_grad_norms()

        if isinstance(self.module, FSDP):
            grad_norm = self.module.clip_grad_norm_(self.optimizer_config.clip_grad)
        elif isinstance(self.module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.module.parameters(), max_norm=self.optimizer_config.clip_grad)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.module.parameters(), max_norm=self.optimizer_config.clip_grad
            )

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()


        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        return grad_norm.item(), module_grad_norms

    def _compute_module_grad_norms(self):
        """Compute per-module gradient L2 norms using optimizer param groups.

        Returns dict {group_name: grad_norm} if named groups exist, else None.
        """
        param_groups = self.optimizer.param_groups
        has_named_groups = any("group_name" in pg for pg in param_groups)
        if not has_named_groups:
            return None

        result = {}
        for pg in param_groups:
            group_name = pg.get("group_name", "unknown")
            total_norm_sq = 0.0
            for p in pg["params"]:
                if p.grad is not None:
                    g = p.grad.detach()
                    if isinstance(g, DTensor):

                        local_tensor = g._local_tensor
                        total_norm_sq += local_tensor.float().norm().item() ** 2
                    else:
                        total_norm_sq += g.float().norm().item() ** 2

            norm_tensor = torch.tensor(total_norm_sq, device="cuda")
            torch.distributed.all_reduce(norm_tensor, op=torch.distributed.ReduceOp.SUM)
            result[group_name] = norm_tensor.item() ** 0.5
        return result

    def lr_scheduler_step(self):
        """
        Advance FSDP scheduler and return updated learning rate(s).

        Returns a dict mapping group names to their LR when per-module LRs are configured,
        otherwise returns a single float for backward compatibility.
        """
        self.lr_scheduler.step()
        last_lrs = self.lr_scheduler.get_last_lr()


        param_groups = self.optimizer.param_groups
        has_named_groups = any("group_name" in pg for pg in param_groups)

        if has_named_groups and len(last_lrs) > 1:
            lr_dict = {}
            for i, pg in enumerate(param_groups):
                group_name = pg.get("group_name", f"group_{i}")
                lr_dict[group_name] = last_lrs[i]
            return lr_dict
        else:
            return last_lrs[0]

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move FSDP model and/or optimizer to CPU or GPU with offload support.
        Note that this function executes irrespective of offload config. It serves as manual control
        """
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        if self.engine_config.forward_only:

            return

        device_name = get_device_name()

        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_fsdp_model_to_gpu(self.module)
            if optimizer and self.optimizer is not None:
                load_fsdp_optimizer(self.optimizer, device)
            gc.collect()
        elif device == "cpu":
            if model:
                offload_fsdp_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_fsdp_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Save FSDP checkpoint, handling parameter offload as needed.
        """
        origin_module_device = next(self.module.parameters()).device.type
        if self._is_offload_param or origin_module_device == "cpu":
            load_fsdp_model_to_gpu(self.module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        """
        Load FSDP checkpoint, restoring parameters and optimizer state.
        """
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.optimizer)

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False):
        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)

        load_fsdp_model_to_gpu(self.module)

        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if hasattr(peft_model, "peft_config"):
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=self.module,
                layered_summon=layered_summon,
                base_sync_done=base_sync_done,
            )
            if not base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            params = self.module.state_dict()

        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        if peft_config is not None and base_sync_done:
            per_tensor_param = params
        else:
            device = get_device_id()
            per_tensor_param = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in params.items()
            )
        return per_tensor_param, peft_config

    def disable_adapter(self) -> ContextManager:
        return self.module.disable_adapter()


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: FSDPEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, FSDPEngine)
        super().__enter__()
        self.engine.ulysses_sharding_manager.__enter__()
        self.engine.module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, FSDPEngine)
        self.engine.ulysses_sharding_manager.__exit__(exc_type, exc_value, traceback)



        if self.engine.engine_config.fsdp_size > 1:
            if fsdp_version(self.engine.module) == 1:
                self.engine.module._handle.reshard(True)
            elif fsdp_version(self.engine.module) == 2:
                self.engine.module.reshard()

        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: FSDPEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, FSDPEngine)
        super().__enter__()
        self.engine.ulysses_sharding_manager.__enter__()
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, FSDPEngine)
        self.engine.ulysses_sharding_manager.__exit__(exc_type, exc_value, traceback)
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPEngineWithLMHead(FSDPEngine):
    def prepare_model_inputs(self, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        temperature = micro_batch["temperature"]
        temperature_item = temperature
        if use_fused_kernels:
            assert not isinstance(temperature, torch.Tensor), (
                "use_fused_kernels does not support per sample temperature yet"
            )
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        multi_modal_inputs = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]

        if not isinstance(temperature, torch.Tensor):
            temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

        temperature = temperature.to(torch.float32)
        assert temperature.shape[0] == input_ids.shape[0]


        output_args = {}

        if use_remove_padding:



            temperature_rmpad = verl_F.expand_as_nested(temperature, input_ids).values()
            temperature_rmpad = temperature_rmpad.unsqueeze(0)

            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids_rmpad = input_ids.values().unsqueeze(0)
                if position_ids.dim() == 3:
                    position_ids_rmpad = position_ids.values().unsqueeze(1)
                else:
                    position_ids_rmpad = position_ids.values().unsqueeze(0)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")


            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)


            if self.use_ulysses_sp:
                is_vlm_model = hasattr(getattr(self.module, "module", self.module).config, "vision_config")
                if is_vlm_model:

                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                else:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                        skip_position_ids_rmpad=True if self.__class__.__name__ == "VeOmniEngineWithLMHead" else False,
                    )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled,
                    position_ids_rmpad=None,
                    sp_size=self.ulysses_sequence_parallel_size,
                )

                temperature_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                    temperature_rmpad, position_ids_rmpad=None, sp_size=self.ulysses_sequence_parallel_size, pad_value=1
                )

                output_args["pad_size"] = pad_size

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
            temperature_rmpad = temperature_rmpad.squeeze(0)
            output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled
            output_args["temperature_rmpad"] = temperature_rmpad




            partllm_batch_input_ids = input_ids
            partllm_batch_position_ids = position_ids



            model_inputs = {
                "input_ids": input_ids_rmpad,
                "attention_mask": None,
                "position_ids": position_ids_rmpad.transpose(0, 2),  
                "partllm_batch_input_ids": partllm_batch_input_ids,
                "partllm_batch_position_ids": partllm_batch_position_ids,
            }

        else:
            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids = micro_batch["input_ids"]
                position_ids = micro_batch["position_ids"]
                loss_mask = micro_batch["loss_mask"]

                pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
                batch_size = micro_batch.batch_size[0]
                seq_len_effective = input_ids.offsets().diff()
                max_seq_len = max(seq_len_effective)

                input_ids_rmpad_rolled = torch.roll(input_ids.values(), shifts=-1, dims=0)
                output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled

                output_args["temperature"] = temperature

                input_ids = torch.nested.to_padded_tensor(
                    input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
                )

                if position_ids.dim() == 3:
                    position_ids = torch.nested.to_padded_tensor(
                        position_ids, padding=0, output_size=(batch_size, 4, max_seq_len)
                    ).transpose(0, 1)
                else:
                    position_ids = torch.nested.to_padded_tensor(
                        position_ids, padding=0, output_size=(batch_size, max_seq_len)
                    )

                attention_mask_list = [torch.ones_like(t, dtype=torch.int32) for t in loss_mask]
                attention_mask = torch.nested.as_nested_tensor(attention_mask_list, layout=torch.jagged)
                attention_mask = torch.nested.to_padded_tensor(
                    attention_mask, padding=0, output_size=(batch_size, max_seq_len)
                )

                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }

            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        extra_args = {}
        if use_fused_kernels:
            extra_args["temperature"] = temperature_item
            extra_args["return_dict"] = True


        _MODEL_EXCLUDED_KEYS = {
            "prompt_mode_id",
            "dataset_type_id",
            "submode_id",
            "macro_category_id",
            "part_count_bucket_id",
            "classification_labels",
            "diffusion_targets",
            "sigmas",
            "vertex",
        }
        model_inputs.update({k: v for k, v in multi_modal_inputs.items() if k not in _MODEL_EXCLUDED_KEYS})
        model_inputs.update(extra_args)

        return model_inputs, output_args

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)

        model_output = {}

        input_ids = micro_batch["input_ids"]

        if use_remove_padding:
            input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
            temperature_rmpad = output_args["temperature_rmpad"]

            if use_fused_kernels:

                log_probs = output.log_probs.squeeze(0)
                entropy_rmpad = output.entropy.squeeze(0)
            else:
                logits_rmpad = output.logits.squeeze(0)
                logits_rmpad.div_(temperature_rmpad.clamp(min=1e-8).unsqueeze(-1).to(logits_rmpad.dtype))


                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                    inplace_backward=inplace_backward,
                )


                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                    else:
                        entropy_rmpad = torch.utils.checkpoint.checkpoint(
                            self.compute_entropy_from_logits, logits_rmpad
                        )


            if self.use_ulysses_sp:
                pad_size = output_args["pad_size"]


                log_probs = gather_outputs_and_unpad(
                    log_probs,
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=pad_size,
                )
                if calculate_entropy:
                    entropy_rmpad = gather_outputs_and_unpad(
                        entropy_rmpad,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()

                log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                if calculate_entropy:
                    entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        else:
            response_length = tu.get_non_tensor_data(data=micro_batch, key="max_response_length", default=1024)
            if use_fused_kernels:
                log_probs = output.log_probs[:, -response_length - 1 : -1]
                entropy = output.entropy[:, -response_length - 1 : -1]

            else:
                logits = output.logits
                temperature = output_args["temperature"]
                temperature = temperature.unsqueeze(-1).unsqueeze(-1)
                logits.div_(temperature.clamp(min=1e-8).to(logits.dtype))

                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        entropy = verl_F.entropy_from_logits(logits)
                    else:
                        entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

                if pad_mode == DatasetPadMode.NO_PADDING:
                    cu_seqlens = input_ids.offsets()
                    seq_lengths = cu_seqlens.diff()
                    starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                    logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
                    logits_rmpad = torch.cat([t for t in logits.unbind()])
                    input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
                    log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                    log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                    if calculate_entropy:
                        entropy = torch.nested.narrow(entropy, 1, starts, seq_lengths, layout=torch.jagged)
                        entropy_rmpad = torch.cat([t for t in entropy.unbind()])
                        entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
                else:
                    raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        model_output["log_probs"] = log_probs
        if calculate_entropy:
            model_output["entropy"] = entropy
        
        if hasattr(self.module.model, "diffusion_model") and output.diffusion_outputs is not None:
            

            assert len(micro_batch["multi_modal_inputs"]) == 1, "multi_modal_inputs should be of length 1"
            model_output["diffusion_outputs"] = output.diffusion_outputs
            if "diffusion_targets" in micro_batch["multi_modal_inputs"][0]:
                model_output["diffusion_targets"] = micro_batch["multi_modal_inputs"][0]["diffusion_targets"]
                model_output["sigmas"] = micro_batch["multi_modal_inputs"][0]["sigmas"]


        if hasattr(output, "mask_decoder_outputs") and output.mask_decoder_outputs is not None:
            mm_inputs = [x.data if hasattr(x, "data") else x for x in micro_batch["multi_modal_inputs"]]
            model_output["diffusion_outputs"] = output.mask_decoder_outputs

            def _stack_if_all(key: str):
                vals = [sample[key] for sample in mm_inputs if sample is not None and key in sample]
                if len(vals) != len(mm_inputs):
                    return None
                try:
                    return torch.stack(vals, dim=0)
                except RuntimeError:
                    return None

            classification_labels = _stack_if_all("classification_labels")
            if classification_labels is not None:
                num_parts = getattr(output, "mask_decoder_num_parts", None)
                if num_parts is None:
                    raise ValueError(
                        "joint mask outputs require mask_decoder_num_parts so per-sample "
                        "background labels can be mapped after class padding"
                    )
                num_parts = num_parts.to(device=classification_labels.device, dtype=torch.long).reshape(-1)
                if num_parts.numel() != classification_labels.shape[0]:
                    raise ValueError(
                        "mask_decoder_num_parts batch mismatch: "
                        f"{num_parts.numel()} counts for {classification_labels.shape[0]} labels"
                    )
                if output.mask_decoder_outputs.dim() != 3:
                    raise ValueError(
                        "joint mask outputs must have shape [B,C,N], got "
                        f"{tuple(output.mask_decoder_outputs.shape)}"
                    )
                global_bg_index = output.mask_decoder_outputs.shape[1] - 1
                if int(num_parts.max().item()) != global_bg_index:
                    raise ValueError(
                        "joint mask class layout mismatch: max num_parts is "
                        f"{int(num_parts.max().item())}, but global BG index is {global_bg_index}"
                    )

                classification_labels = classification_labels.clone()
                for sample_idx, local_bg_index in enumerate(num_parts.tolist()):
                    sample_labels = classification_labels[sample_idx]
                    invalid = (sample_labels < 0) | (sample_labels > local_bg_index)
                    if invalid.any():
                        invalid_values = sample_labels[invalid].unique().tolist()
                        raise ValueError(
                            f"sample {sample_idx} has labels outside [0,{local_bg_index}]: "
                            f"{invalid_values}"
                        )
                    sample_labels[sample_labels == local_bg_index] = global_bg_index
                model_output["classification_labels"] = classification_labels


            for _meta_key in ("prompt_mode_id", "dataset_type_id", "submode_id", "macro_category_id", "part_count_bucket_id"):
                _meta_val = _stack_if_all(_meta_key)
                if _meta_val is not None:
                    model_output[_meta_key] = _meta_val.reshape(_meta_val.shape[0], -1)

        return model_output

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        device_name = get_device_name()

        micro_batch = micro_batch.to(get_device_id())
        model_inputs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            raw_output = self.module(
                **model_inputs,
                use_cache=False,
            )

            model_output = self.prepare_model_outputs(
                output=raw_output, output_args=output_args, micro_batch=micro_batch
            )

            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
                )
                for _consumed_key in ("diffusion_targets", "sigmas"):
                    model_output.pop(_consumed_key, None)
            else:
                assert forward_only, "forward_only must be True when loss_function is None"
                loss = torch.tensor(1.0, device=device_name)
                metrics = {}

            output = {
                "model_output": model_output,
                "loss": loss.detach().item(),
                "metrics": metrics,
            }

            return loss, output


@EngineRegistry.register(model_type="value_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPEngineWithValueHead(FSDPEngineWithLMHead):
    """
    The only difference between critic and actor is how the raw model output is processed
    """

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)

        input_ids = micro_batch["input_ids"]
        if use_remove_padding:
            if hasattr(self.module, "v_head"):

                values_rmpad = output[2].squeeze(0).unsqueeze(-1)
            else:
                values_rmpad = output.logits
                values_rmpad = values_rmpad.squeeze(0)


                values_rmpad = values_rmpad.squeeze(-1)


            if self.use_ulysses_sp:
                pad_size = output_args["pad_size"]
                values_rmpad = gather_outputs_and_unpad(values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()

                values = torch.nested.nested_tensor_from_jagged(values_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        else:
            if hasattr(self.module, "v_head"):

                values = output[2]
            else:
                values = output.logits

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                seq_lengths = cu_seqlens.diff()
                starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                values = torch.nested.narrow(values, 1, starts, seq_lengths, layout=torch.jagged)
                values_rmpad = torch.cat([t for t in values.unbind()])

                values = torch.nested.nested_tensor_from_jagged(values_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        return {"values": values}
