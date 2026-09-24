"""PCA feature dump for inference.

Captures two per-point features through forward hooks:
  - encoder feature: the ``feat_raw`` argument of ``MaskDecoder.forward``
    (the full-resolution Utonia encoder output)
  - decoder pre-logits: the output of ``output_upscaling[_with_fusion]``

Runs PCA to three dimensions independently for each mesh and exports a colored PLY.

Usage:
    fc = FeatureCapture.attach(model)
    with torch.no_grad():
        outputs = model(**inputs)
    fc.dump(save_dir, point_cloud_xyz)
    fc.detach()
"""
from __future__ import annotations
import os
import numpy as np
import torch
from sklearn.decomposition import PCA


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _pca_to_rgb(feat: np.ndarray) -> np.ndarray:
    """feat [N, C] -> rgb uint8 [N, 3] via per-mesh PCA(3)."""
    N, C = feat.shape
    n_comp = min(3, C)
    pca = PCA(n_components=n_comp)
    proj = pca.fit_transform(feat)
    if n_comp < 3:
        pad = np.zeros((N, 3 - n_comp), dtype=proj.dtype)
        proj = np.concatenate([proj, pad], axis=1)
    lo = proj.min(axis=0, keepdims=True)
    hi = proj.max(axis=0, keepdims=True)
    rng = np.maximum(hi - lo, 1e-6)
    rgb = (proj - lo) / rng
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray):
    N = xyz.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            f.write(
                f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} "
                f"{int(rgb[i,0])} {int(rgb[i,1])} {int(rgb[i,2])}\n"
            )


class FeatureCapture:
    """Forward-hook based capture of utonia encoder feat + decoder pre-logits."""

    def __init__(self):
        self.encoder_feat: torch.Tensor | None = None
        self.decoder_feat: torch.Tensor | None = None
        self._handles: list = []

    @classmethod
    def attach(cls, model) -> "FeatureCapture":
        self = cls()

        try:
            mdec_outer = model.model.mask_decoder
            mdec_inner = mdec_outer.mask_decoder
        except AttributeError as e:
            raise RuntimeError(f"FeatureCapture: cannot locate mask_decoder on model: {e}")


        def _pre_hook(_module, args, kwargs):
            feat_raw = kwargs.get("feat_raw", None)
            if feat_raw is None and len(args) >= 5:
                feat_raw = args[4]
            if feat_raw is not None:
                self.encoder_feat = feat_raw[0].detach()

        self._handles.append(
            mdec_inner.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        )


        
        
        
        def _post_hook(_module, _inp, out):

            self.decoder_feat = out[0].detach()

        fusion = getattr(mdec_inner, "output_upscaling_with_fusion", None)
        plain = getattr(mdec_inner, "output_upscaling", None)
        hooked = False
        for seq in (fusion, plain):
            if seq is None or len(seq) == 0:
                continue
            self._handles.append(seq[-1].register_forward_hook(_post_hook))
            hooked = True
        if not hooked:
            raise RuntimeError("FeatureCapture: no upscaling sequential found on MaskDecoder")
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def dump(self, save_dir: str, point_cloud_xyz: np.ndarray, save_npz: bool = True):
        """Run PCA on captured feats, write PLYs (+ optional npz) to save_dir."""
        os.makedirs(save_dir, exist_ok=True)

        for tag, feat in (("encoder", self.encoder_feat), ("decoder", self.decoder_feat)):
            if feat is None:
                print(f"[PCA] {tag} feat not captured (hook never fired)")
                continue
            feat_np = _to_numpy(feat)
            if feat_np.shape[0] != point_cloud_xyz.shape[0]:
                print(
                    f"[PCA] {tag} feat N={feat_np.shape[0]} mismatches xyz N={point_cloud_xyz.shape[0]}, skip"
                )
                continue
            rgb = _pca_to_rgb(feat_np)
            ply_path = os.path.join(save_dir, f"pca_{tag}.ply")
            _write_ply(ply_path, point_cloud_xyz.astype(np.float32), rgb)
            print(f"[PCA] {tag}: feat shape {feat_np.shape} -> {ply_path}")
            if save_npz:
                np.savez_compressed(
                    os.path.join(save_dir, f"pca_{tag}.npz"),
                    feat=feat_np.astype(np.float16),
                    rgb=rgb,
                    xyz=point_cloud_xyz.astype(np.float32),
                )


def is_enabled() -> bool:
    return os.environ.get("SAVE_PCA_FEATS", "0") not in ("0", "", "false", "False")
