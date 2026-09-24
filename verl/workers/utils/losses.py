# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig

from torchvision.ops import sigmoid_focal_loss

def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:


        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()


        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)




        ce_loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        ce_loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    metrics = {"ce_loss": ce_loss.detach().item()}

    loss = ce_loss

    if "diffusion_targets" in model_output:
        diffusion_outputs = model_output["diffusion_outputs"].float()
        diffusion_targets = model_output["diffusion_targets"].to(diffusion_outputs.device).float()
        batch_decoder_tokens = data["batch_decoder_tokens"]

        loss = torch.sum((diffusion_outputs.float() - diffusion_targets.float()) ** 2) / batch_decoder_tokens * dp_size
    elif "classification_labels" in model_output:
        diffusion_outputs = model_output["diffusion_outputs"].float()
        seg_loss_weight = tu.get_non_tensor_data(data=data, key="seg_loss_weight", default=1.0)



        joint_loss_cfg = dict(
            gamma=tu.get_non_tensor_data(data=data, key="joint_focal_gamma", default=2.0),
            lambda_dice=tu.get_non_tensor_data(data=data, key="joint_lambda_dice", default=2.0),
            exclude_bg=tu.get_non_tensor_data(data=data, key="joint_exclude_bg", default=True),
        )

        classification_labels = model_output["classification_labels"].to(diffusion_outputs.device)
        seg_loss, seg_parts = compute_joint_mask_loss(
            diffusion_outputs, classification_labels, **joint_loss_cfg
        )

        loss += seg_loss_weight * seg_loss
        metrics["seg_loss"] = seg_loss.detach().item()
        metrics["seg_loss_weight"] = seg_loss_weight
        for _name, _val in seg_parts.items():
            metrics[f"seg_{_name}_loss"] = _val.detach().item()
        
    return loss, metrics


def _slice_response_from_unpad_output(tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
    """Slice response from unpad model output.

    Args:
        tensor: model output tensor of shape [bsz, 1]
        data: TensorDict with "prompt_ids", "response_ids", "attention_mask"

    Returns:
        tensor: sliced response tensor of shape [bsz, max_response_len]
    """
    values = tensor.values() if tensor.is_nested else tensor
    prompt_ids = data["prompts"]
    response_ids = data["responses"]
    attention_mask = data["attention_mask"]

    if prompt_ids.is_nested:
        prompt_lens = prompt_ids.offsets().diff()
        response_lens = response_ids.offsets().diff()
        max_response_len = response_ids.offsets().max().item()
    else:
        assert not attention_mask.is_nested
        prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
        response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
        max_response_len = response_ids.shape[1]

    sequence_lens = prompt_lens + response_lens
    sequence_offsets = sequence_lens.cumsum(dim=0)
    assert sequence_offsets[-1].item() == values.shape[0]

    response_list = []
    for resp_len, seq_offset in zip(response_lens, sequence_offsets, strict=True):
        pad_size = max_response_len - resp_len

        response_list.append(F.pad(values[seq_offset - resp_len - 1 : seq_offset - 1], (0, pad_size)))

    output = torch.stack(response_list, dim=0)
    return output


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    log_prob = _slice_response_from_unpad_output(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = _slice_response_from_unpad_output(entropy, data)


    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor




    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    response_mask = data["response_mask"].to(bool)

    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )



    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss


    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)


    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]

        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = _slice_response_from_unpad_output(model_output["values"], data)

    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics


def _normalize_joint_logits_labels(logits: torch.Tensor, labels: torch.Tensor):
    """Normalize joint-mode logits/labels to canonical [B, C, N] / [B, N] shapes.

    Accepts:
        logits: [C, N] | [1, C, N] | [B, C, N]
        labels: [N] | [1, N] | [B, N] | [B, 1, N]

    Returns:
        (logits[B, C, N], labels[B, N] long)
    """
    labels = labels.long()
    if logits.dim() == 2:

        logits = logits.unsqueeze(0)
        labels = labels.reshape(1, -1)
        return logits, labels

    if logits.dim() != 3:
        raise ValueError(f"joint mask logits must have shape [C,N] or [B,C,N], got {tuple(logits.shape)}")

    if labels.dim() == 1:
        if logits.shape[0] != 1:
            raise ValueError(
                f"batched joint mask logits {tuple(logits.shape)} require batched labels, got {tuple(labels.shape)}"
            )
        labels = labels.unsqueeze(0)
    elif labels.dim() == 3 and labels.shape[1] == 1:
        labels = labels.squeeze(1)

    if labels.dim() != 2:
        raise ValueError(f"joint mask labels must have shape [N] or [B,N], got {tuple(labels.shape)}")
    if logits.shape[0] != labels.shape[0] or logits.shape[2] != labels.shape[1]:
        raise ValueError(f"joint mask logits/labels shape mismatch: {tuple(logits.shape)} vs {tuple(labels.shape)}")
    return logits, labels


def compute_joint_mask_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    gamma: float = 2.0,
    lambda_dice: float = 2.0,
    exclude_bg: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """Joint classification loss for mask prediction.

    Combines multi-class softmax focal loss and macro-averaged soft Dice
    over present classes (BG excluded by default). The two terms are
    summed as ``focal + lambda_dice * dice``.

    Args:
        logits: [C, N] | [1, C, N] | [B, C, N] per-point logits.
                Class C-1 is treated as background.
        labels: [N] | [1, N] | [B, N] ground-truth class labels in
                ``{0..C-1}``.
        gamma:  focal loss focusing parameter. ``gamma=0`` collapses to
                plain cross-entropy.
        lambda_dice: weight on the Dice term.
        exclude_bg:  if True, BG class is dropped from the per-sample
                macro Dice average. Recommended for highly imbalanced
                point clouds where BG dominates.
        eps:    Dice numerical stability.

    Returns:
        total_loss (scalar), parts (dict): ``{"focal": scalar tensor,
        "dice": scalar tensor}``. The parts tensors keep grad so callers
        can detach for metrics.
    """
    logits, labels = _normalize_joint_logits_labels(logits, labels)
    B, C, N = logits.shape
    bg_index = C - 1


    log_probs = F.log_softmax(logits, dim=1)
    ce_per_pt = F.nll_loss(log_probs, labels, reduction="none")
    p_t = (-ce_per_pt).exp()
    if gamma == 0.0:
        focal = ce_per_pt.mean()
    else:
        focal = ((1.0 - p_t).pow(gamma) * ce_per_pt).mean()


    probs = log_probs.exp()
    onehot = F.one_hot(labels, C).permute(0, 2, 1).to(probs.dtype)
    inter = (probs * onehot).sum(dim=-1)
    denom = probs.sum(dim=-1) + onehot.sum(dim=-1)
    dice_per_class = 1.0 - (2.0 * inter + eps) / (denom + eps)

    present = onehot.sum(dim=-1) > 0
    if exclude_bg and 0 <= bg_index < C:
        present = present.clone()
        present[..., bg_index] = False

    sample_dice = []
    for b in range(B):
        if present[b].any():
            sample_dice.append(dice_per_class[b][present[b]].mean())
        else:


            sample_dice.append(probs.new_zeros(()))
    dice = torch.stack(sample_dice).mean()

    total = focal + lambda_dice * dice
    return total, {"focal": focal, "dice": dice}


def compute_mask_loss(
    logits: torch.Tensor, labels: torch.Tensor, loss_weight_dice: float = 2
):
    """Loss for mask prediction.

    Args:
        logits: A float tensor of shape [B, C, N]. Multi-mask predicted logits.
        labels: A float tensor of shape [B, N]. Ground-truth binary masks.

    Returns:
        torch.Tensor: [B, C]. Mask loss
    """
    assert logits.dim() == 3, logits.shape

    logits = logits.clamp(-50.0, 50.0)
    _labels = labels.unsqueeze(1).expand_as(logits)
    _labels = _labels.to(dtype=logits.dtype)
    loss_ce = sigmoid_focal_loss(logits, _labels, alpha=-1, reduction="none")
    loss_dice = dice_loss(logits.sigmoid(), _labels, reduction="none")
    loss = loss_ce.mean(-1) + loss_weight_dice * loss_dice
    return loss

@torch.jit.script
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-3,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs: A float tensor of arbitrary shape, [B, ..., N].
                The (probability) predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        reduction: ``'none'`` | ``'mean'`` | ``'sum'``
        eps: A small epsilon value to avoid division by zero.

    Returns:
        torch.Tensor: If reduction is 'none', then the shape is [B, ...]. Otherwise, a scalar is returned.

    References:
        https://github.com/CoinCheung/pytorch-loss/blob/master/soft_dice_loss.py
        https://github.com/UX-Decoder/Semantic-SAM/blob/3d6a43a0f8e77167c0013d14067933a78e2d1f5a/semantic_sam/modules/criterion_interactive_many_to_many.py#L57
        https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/mmdet/models/losses/dice_loss.py#L9
    """
    assert inputs.shape == targets.shape, (inputs.shape, targets.shape)
    assert inputs.dtype == targets.dtype, (inputs.dtype, targets.dtype)

    numerator = 2 * (inputs * targets).sum(-1)

    denominator = inputs.square().sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)


    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss
