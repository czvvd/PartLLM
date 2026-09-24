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

import functools
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch




try:
    from torch_cluster import fps as torch_cluster_fps
except Exception as exc:
    raise RuntimeError(
        "point-token sampling requires a working torch_cluster.fps build; "
        "refusing to fall back to torch.randperm"
    ) from exc
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
)
from transformers.models.qwen3_vl.mask_decoder import _utonia_unpooling

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def sample_point_indices(
    coords: torch.Tensor,
    n_samples: int,
    indices_cache: Optional[dict] = None,
) -> torch.Tensor:
    """Sample point-token indices, optionally reusing an inference-only cache.

    Training never sets ``indices_cache`` and therefore keeps its original
    stochastic behavior.  Interactive inference explicitly installs an empty
    cache per shape so generate(), the final mask forward, and all correction
    rounds use the exact same point-token subset.
    """
    cache_key = (
        int(coords.shape[0]),
        int(n_samples),
        coords.device.type,
        coords.device.index,
    )
    if indices_cache is not None and cache_key in indices_cache:
        return indices_cache[cache_key].to(device=coords.device)

    fixed_seed = None if indices_cache is None else indices_cache.get("__seed__")
    if n_samples >= coords.shape[0]:
        indices = torch.arange(coords.shape[0], device=coords.device)
    else:
        if fixed_seed is None:
            indices = torch_cluster_fps(
                coords.float(), ratio=n_samples / coords.shape[0], random_start=True
            )
        else:


            devices = [coords.device.index] if coords.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(fixed_seed))
                if coords.is_cuda:
                    torch.cuda.manual_seed(int(fixed_seed))
                indices = torch_cluster_fps(
                    coords.float(), ratio=n_samples / coords.shape[0], random_start=True
                )
        indices = indices[:n_samples]

    if indices_cache is not None:
        indices_cache[cache_key] = indices.detach().clone()
    return indices





_spconv_patched = False


def _patch_spconv_for_bf16():
    """Monkey-patch spconv to handle bfloat16 by casting to float32 inside conv layers.

    spconv's CUDA kernels don't support bfloat16. This patch wraps
    SparseConvolution.forward so that if features are bf16, it casts
    features/weight/bias to fp32 for the kernel, then casts output back.

    Safe to call multiple times (no-op after first).
    Does NOT modify model weights or FSDP state — only temporary casts
    during forward, with proper autograd support.
    """
    global _spconv_patched
    if _spconv_patched:
        return
    _spconv_patched = True

    try:
        from spconv.pytorch.conv import SparseConvolution
        from spconv import ConvAlgo
    except ImportError:
        return

    _orig_conv_forward = SparseConvolution._conv_forward

    @functools.wraps(_orig_conv_forward)
    def _bf16_safe_conv_forward(self, training, input, weight, bias, add_input=None, *args, **kwargs):

        need_cast = input.features.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16
        if need_cast:
            input = input.replace_feature(input.features.float())
            weight = weight.float()
            if bias is not None:
                bias = bias.float()
        result = _orig_conv_forward(self, training, input, weight, bias, add_input, *args, **kwargs)
        if need_cast:
            result = result.replace_feature(result.features.to(torch.bfloat16))
        return result

    SparseConvolution._conv_forward = _bf16_safe_conv_forward


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Gets the position ids for Qwen3-VL, it should be generated before sharding the sequence.
    The batch dim has been removed and the input_ids should be a 1D tensor representing a single example.
    https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L916
    """
    spatial_merge_size = processor.image_processor.merge_size
    image_token_id = processor.image_token_id
    video_token_id = processor.video_token_id
    vision_start_token_id = processor.vision_start_token_id




    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(3, input_ids.shape[0], dtype=input_ids.dtype, device=input_ids.device)
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(input_ids.device)
        input_ids = input_ids[attention_mask == 1]
        image_nums, video_nums = 0, 0
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)



            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(attention_mask.device)
        else:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1).expand(3, -1)

    return position_ids



def _slice_utonia_point_dict(utonia_point_dict: dict, sample_idx: int) -> dict:
    """Slice a batched Utonia collate dict down to one sample.

    The dataset stores each sample as a collated one-item Utonia dict. The SFT
    collator keeps those dicts in a Python list for true micro-batches > 1, but
    this helper also handles an already-collated batched dict defensively.
    """
    if isinstance(utonia_point_dict, list):
        return utonia_point_dict[sample_idx]

    if not isinstance(utonia_point_dict, dict):
        return utonia_point_dict

    offsets = utonia_point_dict.get("offset", None)
    if not isinstance(offsets, torch.Tensor) or offsets.numel() <= 1:
        return utonia_point_dict

    start = 0 if sample_idx == 0 else int(offsets[sample_idx - 1].item())
    end = int(offsets[sample_idx].item())
    sliced = {}
    for key, value in utonia_point_dict.items():
        if not isinstance(value, torch.Tensor):
            sliced[key] = value
        elif key == "offset":
            sliced[key] = value.new_tensor([end - start])
        elif key == "batch":
            sliced[key] = value[start:end].new_zeros((end - start,))
        elif value.dim() > 0 and value.shape[0] == offsets[-1].item():
            sliced[key] = value[start:end]
        elif key == "inverse" and value.dim() > 0:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def _get_input_embeds(
    model: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    point_clouds: Optional[torch.FloatTensor] = None,
    utonia_point_dict: Optional[dict] = None,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_mask, video_mask = None, None
    pc_info = None
    point_cloud_embeds = None
    point_cloud_embeds_for_decoder = None
    utonia_point_out = None
    grid_sample_inverse = None
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds, deepstack_image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds, deepstack_video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = input_ids == model.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
    
    encoder_type = getattr(model, "point_cloud_encoder_type", "utonia")
    if encoder_type != "utonia":
        raise ValueError("PartLLM supports only the Utonia point-cloud encoder")
    pc_info = None
    utonia_feat_raw = None
    utonia_coords = None
    
    
    if utonia_point_dict is None:
        utonia_point_dict = getattr(model, "_utonia_point_dict_cache", None)
        if utonia_point_dict is not None:
            model._utonia_point_dict_cache = None  
    if utonia_point_dict is not None:

        device = model.utonia_proj.weight.device
        for key in utonia_point_dict:
            if isinstance(utonia_point_dict[key], torch.Tensor):
                if utonia_point_dict[key].is_floating_point():
                    utonia_point_dict[key] = utonia_point_dict[key].to(device=device, dtype=torch.float32)
                else:
                    utonia_point_dict[key] = utonia_point_dict[key].to(device=device)


        _patch_spconv_for_bf16()

        utonia_point = model.utonia_model(utonia_point_dict)


        feat_full = _utonia_unpooling(utonia_point, utonia_point.feat, num_concat_levels=4)
        inverse = utonia_point_dict["inverse"]
        feat_at_original = feat_full[inverse]
        coords_original = utonia_point_dict["coord"][inverse]


        n_fps = (input_ids == model.config.point_cloud_token_id).sum().item()
        fps_indices = sample_point_indices(
            coords_original,
            n_fps,
            getattr(model, "_partllm_fps_indices_cache", None),
        )
        centers = coords_original[fps_indices]
        embeddings_fps = feat_at_original[fps_indices]

        utonia_proj_input = embeddings_fps.to(model.utonia_proj.weight.dtype)
        point_cloud_embeds = model.utonia_proj(utonia_proj_input)
        feat_raw = feat_at_original

        mask = input_ids == model.config.point_cloud_token_id
        point_cloud_mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        point_cloud_embeds = point_cloud_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(point_cloud_mask, point_cloud_embeds)

        pc_info = [centers.unsqueeze(0)]
        utonia_feat_raw = feat_raw
        utonia_coords = coords_original

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:

        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds, strict=False):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    if pixel_values is None and pixel_values_videos is None and getattr(model, "visual", None) is not None and model.training:
        
        
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        image_embeds, dummy_deepstack_image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        inputs_embeds += 0.0 * image_embeds.mean()
        for emb in dummy_deepstack_image_embeds or []:
            inputs_embeds += 0.0 * emb.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "visual_pos_masks": visual_pos_masks,
        "deepstack_visual_embeds": deepstack_visual_embeds,
        "pc_info": pc_info,
        "point_cloud_embeddings": point_cloud_embeds,
        "utonia_feat_raw": utonia_feat_raw if encoder_type == "utonia" else None,
        "utonia_coords": utonia_coords if encoder_type == "utonia" else None,
    }


@dataclass
class Qwen3VLCausalLMOutputForPPO(Qwen3VLCausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None
    mask_decoder_outputs: Optional[torch.FloatTensor] = None
    mask_decoder_num_parts: Optional[torch.LongTensor] = None


def qwen3_vl_base_forward(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    point_clouds: Optional[torch.FloatTensor] = None,
    utonia_point_dict: Optional[dict] = None,
    vertex: Optional[torch.FloatTensor] = None,
    timesteps: Optional[torch.LongTensor] = None,
    **kwargs,
):
    if vertex is not None:
        raise ValueError(
            "The legacy vertex input has been removed. Pass geometry through "
            "point_clouds/utonia_point_dict only."
        )

    is_generation_forward = (
        bool(kwargs.get("use_cache", False))
        or kwargs.get("past_key_values", None) is not None
        or "cache_position" in kwargs
    )

    partllm_batch_input_ids = kwargs.pop("partllm_batch_input_ids", None)
    if partllm_batch_input_ids is not None:
        batch_size = len(partllm_batch_input_ids)
        token_ids_for_masks = [t for t in partllm_batch_input_ids.unbind()]
    else:
        batch_size = input_ids.shape[0]
        token_ids_for_masks = [input_ids[i] for i in range(batch_size)]

    input_embeds_list = []
    attention_mask_list = [] if attention_mask is not None else None
    mask_decoder_per_sample = []
    visual_pos_masks_list = []
    deepstack_per_sample = []
    image_grid_offset = 0
    image_pixel_offset = 0
    video_grid_offset = 0
    video_pixel_offset = 0

    for sample_idx in range(batch_size):
        sample_input_ids = token_ids_for_masks[sample_idx].unsqueeze(0)
        sample_attention_mask = attention_mask[sample_idx : sample_idx + 1] if attention_mask is not None else None
        sample_point_clouds = point_clouds[sample_idx : sample_idx + 1] if point_clouds is not None else None
        sample_utonia = _slice_utonia_point_dict(utonia_point_dict, sample_idx) if utonia_point_dict is not None else None





        def _count_visual_items(token_id):
            if token_id is None:
                return 0
            token_mask = sample_input_ids[0] == token_id
            previous = torch.cat(
                [token_mask.new_zeros(1), token_mask[:-1]], dim=0
            )
            return int((token_mask & ~previous).sum().item())

        sample_pixel_values = None
        sample_image_grid_thw = None
        if pixel_values is not None:
            n_images = _count_visual_items(getattr(self.config, "image_token_id", None))
            sample_image_grid_thw = image_grid_thw[
                image_grid_offset : image_grid_offset + n_images
            ]
            n_image_patches = int(sample_image_grid_thw.prod(dim=-1).sum().item())
            sample_pixel_values = pixel_values[
                image_pixel_offset : image_pixel_offset + n_image_patches
            ]
            image_grid_offset += n_images
            image_pixel_offset += n_image_patches

        sample_pixel_values_videos = None
        sample_video_grid_thw = None
        if pixel_values_videos is not None:
            n_videos = _count_visual_items(getattr(self.config, "video_token_id", None))
            sample_video_grid_thw = video_grid_thw[
                video_grid_offset : video_grid_offset + n_videos
            ]
            n_video_patches = int(sample_video_grid_thw.prod(dim=-1).sum().item())
            sample_pixel_values_videos = pixel_values_videos[
                video_pixel_offset : video_pixel_offset + n_video_patches
            ]
            video_grid_offset += n_videos
            video_pixel_offset += n_video_patches

        sample_input_kwargs = _get_input_embeds(
            self,
            sample_input_ids,
            sample_attention_mask,
            sample_pixel_values,
            sample_pixel_values_videos,
            sample_image_grid_thw,
            sample_video_grid_thw,
            sample_point_clouds,
            sample_utonia,
        )
        input_embeds_list.append(sample_input_kwargs.pop("inputs_embeds"))
        if attention_mask_list is not None:
            attention_mask_list.append(sample_input_kwargs.pop("attention_mask"))
        else:
            sample_input_kwargs.pop("attention_mask", None)
        visual_pos_masks_list.append(sample_input_kwargs.pop("visual_pos_masks"))
        deepstack_per_sample.append(sample_input_kwargs.pop("deepstack_visual_embeds"))
        mask_decoder_per_sample.append(sample_input_kwargs)

    if partllm_batch_input_ids is not None:
        kwargs.pop("partllm_batch_position_ids", None)
        kwargs.update({"inputs_embeds": torch.cat(input_embeds_list, dim=1)})
    else:
        kwargs.update({"inputs_embeds": torch.cat(input_embeds_list, dim=0)})
    if attention_mask_list is not None:
        kwargs["attention_mask"] = torch.cat(attention_mask_list, dim=0)

    active_visual_masks = [mask for mask in visual_pos_masks_list if mask is not None]
    active_deepstack = [embeds for embeds in deepstack_per_sample if embeds is not None]
    if active_visual_masks:
        mask_cat_dim = 1 if partllm_batch_input_ids is not None else 0
        kwargs["visual_pos_masks"] = torch.cat(active_visual_masks, dim=mask_cat_dim)
        layer_count = len(active_deepstack[0])
        if any(len(embeds) != layer_count for embeds in active_deepstack):
            raise ValueError("DeepStack visual layer count differs within a micro-batch")
        kwargs["deepstack_visual_embeds"] = [
            torch.cat([embeds[layer_idx] for embeds in active_deepstack], dim=0)
            for layer_idx in range(layer_count)
        ]

    outputs = self.language_model(input_ids=None, **kwargs)

    has_point_cloud = utonia_point_dict is not None
    if has_point_cloud and getattr(self.config, "use_mask_decoder", False):





        if is_generation_forward:
            return outputs






        if (not self.training) and not any((sample_ids == self.config.seg_token_id).any() for sample_ids in token_ids_for_masks):
            return outputs

        bg_token_id = getattr(self.config, "bg_token_id", None)
        sample_masks = []
        sample_num_parts = []

        for sample_idx, sample_kwargs in enumerate(mask_decoder_per_sample):
            sample_input_ids = token_ids_for_masks[sample_idx]
            if partllm_batch_input_ids is not None:
                offsets = partllm_batch_input_ids.offsets()
                sample_hidden = outputs.last_hidden_state[0, offsets[sample_idx] : offsets[sample_idx + 1]]
            else:
                sample_hidden = outputs.last_hidden_state[sample_idx]
            seg_token_mask = sample_input_ids == self.config.seg_token_id
            if not seg_token_mask.any():
                raise ValueError(f"sample {sample_idx} has no SEG token for mask decoding")

            seg_hidden_states = sample_hidden[seg_token_mask]
            num_parts = int(seg_hidden_states.shape[0])
            if bg_token_id is not None:
                bg_token_mask = sample_input_ids == bg_token_id
                if bg_token_mask.any():
                    bg_hidden_states = sample_hidden[bg_token_mask][-1:, :]
                else:
                    raise ValueError(f"sample {sample_idx} has no BG token for joint mask decoding")
            else:
                bg_hidden_states = torch.zeros(
                    1,
                    seg_hidden_states.shape[-1],
                    device=seg_hidden_states.device,
                    dtype=seg_hidden_states.dtype,
                )
            encoder_hidden_states = torch.cat([seg_hidden_states, bg_hidden_states], dim=0).unsqueeze(0)

            pc_embeddings = sample_kwargs["point_cloud_embeddings"][None]
            centers = sample_kwargs["pc_info"][0]
            coords = sample_kwargs["utonia_coords"].unsqueeze(0)
            _feat_raw = sample_kwargs.get("utonia_feat_raw")
            feat_raw = _feat_raw.unsqueeze(0) if _feat_raw is not None else None

            sample_mask = self.mask_decoder(
                hidden_states=pc_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                centers=centers,
                coords=coords,
                feat_raw=feat_raw,
            ).squeeze(0)
            expected_classes = num_parts + 1
            if sample_mask.shape[0] != expected_classes:
                raise ValueError(
                    f"sample {sample_idx} mask decoder returned {sample_mask.shape[0]} classes; "
                    f"expected {expected_classes} for {num_parts} parts in joint mode"
                )
            sample_masks.append(sample_mask)
            sample_num_parts.append(num_parts)

        if len(sample_masks) != batch_size:
            raise ValueError(
                f"mask decoder produced {len(sample_masks)} sample outputs for batch size {batch_size}"
            )

        if sample_masks:
            num_points = sample_masks[0].shape[1]
            if any(mask.shape[1] != num_points for mask in sample_masks):
                point_counts = [int(mask.shape[1]) for mask in sample_masks]
                raise ValueError(
                    "mask decoder point counts differ within a micro-batch, but labels are stacked "
                    f"without point padding: {point_counts}"
                )

            max_parts = max(sample_num_parts)
            max_classes = max_parts + 1
            padded_masks = []
            for mask, num_parts in zip(sample_masks, sample_num_parts, strict=True):
                padded = mask.new_full((max_classes, num_points), -1e4)
                padded[:num_parts] = mask[:num_parts]


                padded[max_parts] = mask[num_parts]
                padded_masks.append(padded)
            outputs.mask_decoder_outputs = torch.stack(padded_masks, dim=0)
            outputs.mask_decoder_num_parts = torch.tensor(
                sample_num_parts,
                device=outputs.mask_decoder_outputs.device,
                dtype=torch.long,
            )

    return outputs


def forward_with_normal_backend(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    return Qwen3VLCausalLMOutputForPPO(
        logits=logits,
        hidden_states=outputs.hidden_states,
        past_key_values=outputs.past_key_values if hasattr(outputs, "past_key_values") else None,
        mask_decoder_outputs=outputs.mask_decoder_outputs if hasattr(outputs, "mask_decoder_outputs") else None,
        mask_decoder_num_parts=(
            outputs.mask_decoder_num_parts if hasattr(outputs, "mask_decoder_num_parts") else None
        ),
    )


def forward_with_torch_backend(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]


    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )
    return Qwen3VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
        mask_decoder_outputs=outputs.mask_decoder_outputs if hasattr(outputs, "mask_decoder_outputs") else None,
        mask_decoder_num_parts=(
            outputs.mask_decoder_num_parts if hasattr(outputs, "mask_decoder_num_parts") else None
        ),
    )


def forward_with_triton_backend(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]


    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )
    return Qwen3VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
        mask_decoder_outputs=outputs.mask_decoder_outputs if hasattr(outputs, "mask_decoder_outputs") else None,
        mask_decoder_num_parts=(
            outputs.mask_decoder_num_parts if hasattr(outputs, "mask_decoder_num_parts") else None
        ),
    )


def patch_qwen3_vl_moe_sparse_moe_block_forward():
    """Patch router-weight allocation for compatible Qwen3-VL MoE releases."""
    try:
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock
    except ImportError:
        return

    original_forward = Qwen3VLMoeTextSparseMoeBlock.forward

    @functools.wraps(original_forward)
    def patched_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)
        router_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        routed_out = self.experts(hidden_states, router_weights, router_indices)
        return routed_out

    Qwen3VLMoeTextSparseMoeBlock.forward = patched_forward
    logger.info("Monkey patched Qwen3VLMoeTextSparseMoeBlock.forward to fix router_weights bug")
