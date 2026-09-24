import numpy as np
import torch


def _to_numpy_logits(mask_logits):
    if isinstance(mask_logits, torch.Tensor):
        return mask_logits.detach().float().cpu().numpy()
    return np.asarray(mask_logits, dtype=np.float32)


def decode_joint_logits(mask_logits):
    logits = _to_numpy_logits(mask_logits)
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"Expected joint logits with shape [1, C, N], got {logits.shape}")
    class_logits = logits.squeeze(0)
    pred_labels = np.argmax(class_logits, axis=0).astype(np.int64)
    return {
        "mode": "joint",
        "class_logits": class_logits,
        "pred_labels": pred_labels,
    }


def decode_mask_logits(mask_logits, threshold: float = 0.5):
    del threshold
    return decode_joint_logits(mask_logits)
