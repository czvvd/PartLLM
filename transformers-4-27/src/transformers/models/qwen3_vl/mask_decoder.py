import os
import yaml
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict, Type
from torch import Tensor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PreTrainedModel, PretrainedConfig
from diffusers.models.attention import AttentionMixin, FeedForward, JointTransformerBlock
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.modeling_utils import ModelMixin
import dataclasses


class MaskDecoderConfig(PretrainedConfig):
    model_type = "qwen3_vl_mask_decoder"
    base_config_key = "mask_decoder_config"

    def __init__(
        self,
        transformer_depth: int = 2,
        transformer_embedding_dim: int = 256,
        transformer_num_heads: int = 8,
        transformer_mlp_dim: int = 2048,
        transformer_attention_downsample_rate: int = 2,
        mask_decoder_transformer_dim: int = 256,
        mask_decoder_embedding_input_dim: int = 448,
        use_pos_emb: bool = True,
        input_size: int = 4096,
        utonia_enc_channels: list = None,
        **kwargs,
    ):
        self.transformer_depth = transformer_depth
        self.transformer_embedding_dim = transformer_embedding_dim
        self.transformer_num_heads = transformer_num_heads
        self.transformer_mlp_dim = transformer_mlp_dim
        self.transformer_attention_downsample_rate = transformer_attention_downsample_rate
        self.mask_decoder_transformer_dim = mask_decoder_transformer_dim
        self.mask_decoder_embedding_input_dim = mask_decoder_embedding_input_dim
        self.utonia_enc_channels = utonia_enc_channels or [54, 108, 216, 432, 576]
        super().__init__(**kwargs)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    return np.concatenate([emb_sin, emb_cos], axis=1)


def _utonia_unpooling(point, feat, num_concat_levels=4):
    """Unpool features through the pooling_parent/pooling_inverse chain WITHOUT modifying point.

    Args:
        point: Utonia Point object with pooling chain (read-only)
        feat: [N_bottleneck, C] features to unpool (can differ from point.feat)
        num_concat_levels: number of levels to concat (first N levels)

    Returns:
        Unpooled features tensor at grid_sample resolution [N_grid, C_out]
    """

    levels = []
    cur = point
    while "pooling_parent" in cur.keys():
        levels.append((cur["pooling_parent"], cur["pooling_inverse"]))
        cur = cur["pooling_parent"]


    for i, (parent, inverse) in enumerate(levels):
        if i < num_concat_levels:
            feat = torch.cat([parent.feat, feat[inverse]], dim=-1)
        else:
            feat = feat[inverse]
    return feat


class MaskDecoderModel(PreTrainedModel, ModelMixin):
    config: MaskDecoderConfig
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    
    def __init__(
        self,
        config: MaskDecoderConfig,
        **kwargs
    ):
        super().__init__(config)
        
        two_way_transformer = TwoWayTransformer(
            depth=config.transformer_depth,
            embedding_dim=config.transformer_embedding_dim,
            num_heads=config.transformer_num_heads,
            mlp_dim=config.transformer_mlp_dim, 
            attention_downsample_rate=config.transformer_attention_downsample_rate,
        )
        self.mask_decoder = MaskDecoder(
            transformer=two_way_transformer,
            transformer_dim=config.mask_decoder_transformer_dim,
            embedding_input_dim=config.mask_decoder_embedding_input_dim,
            utonia_enc_channels=getattr(config, "utonia_enc_channels", None),
        )

        self.pe_layer = PositionEmbeddingRandom(config.transformer_embedding_dim // 2)

    @staticmethod
    def _find_mismatched_keys(
        state_dict,
        model_state_dict,
        loaded_keys,
        ignore_mismatched_sizes,
    ):
        mismatched_keys = []
        if ignore_mismatched_sizes:
            for checkpoint_key in loaded_keys:
                model_key = checkpoint_key

                if (
                    model_key in model_state_dict
                    and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                ):
                    mismatched_keys.append(
                        (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                    )
                    del state_dict[checkpoint_key]
        return mismatched_keys

    def _init_weights(self, module: nn.Module) -> None:
        std = 0.02
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm) and module.bias is not None:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            module.weight.data.normal_(mean=0.0, std=std)
        elif module.__class__.__name__ == "RMSNorm":
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        centers: torch.Tensor = None,
        coords: torch.Tensor = None,
        feat_raw: torch.Tensor = None,
    ):

        coord_max = centers.abs().max()
        centers_normalized = centers / (coord_max + 1e-8)
        pc_pe = self.pe_layer(centers_normalized)
        with torch.no_grad():
            interp_index, interp_weight = compute_interp_weights(coords, centers)
        aux_inputs = AuxInputs(
            coords=coords, centers=centers,
            interp_index=interp_index, interp_weight=interp_weight,
        )
        masks = self.mask_decoder(
            hidden_states, pc_pe, encoder_hidden_states, aux_inputs,
            feat_raw=feat_raw,
        )

        return masks


def repeat_interleave(x: torch.Tensor, repeats: int, dim: int):
    if repeats == 1:
        return x
    shape = list(x.shape)
    shape.insert(dim + 1, 1)
    shape[dim + 1] = repeats
    x = x.unsqueeze(dim + 1).expand(shape).flatten(dim, dim + 1)
    return x

def compute_interp_weights(query: torch.Tensor, key: torch.Tensor, k=3, eps=1e-8, chunk_size=8192):
    """Compute interpolation weights for each query point, chunked to save memory.

    Args:
        query: [B, Nq, 3]. Query points.
        key: [B, Nk, 3]. Key points.
        k: int. The number of nearest neighbors.
        eps: float. A small value to avoid division by zero.
        chunk_size: int. Number of query points per chunk.

    Returns:
        torch.Tensor: [B, Nq, K], indices of the k nearest neighbors in the key.
        torch.Tensor: [B, Nq, K], interpolation weights.
    """
    B, Nq, _ = query.shape
    all_idx = []
    all_weight = []
    for start in range(0, Nq, chunk_size):
        end = min(start + chunk_size, Nq)
        dist, idx = knn_points(query[:, start:end, :], key, k)
        inv_dist = 1.0 / torch.clamp(dist.square(), min=eps)
        normalizer = torch.sum(inv_dist, dim=2, keepdim=True)
        weight = inv_dist / normalizer
        all_idx.append(idx)
        all_weight.append(weight)
    return torch.cat(all_idx, dim=1), torch.cat(all_weight, dim=1)

def interpolate_features(x: torch.Tensor, index: torch.Tensor, weight: torch.Tensor):
    """
    Interpolates features based on the given index and weight.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, num_keys, num_features).
        index (torch.Tensor): The index tensor of shape (batch_size, num_queries, K).
        weight (torch.Tensor): The weight tensor of shape (batch_size, num_queries, K).

    Returns:
        torch.Tensor: The interpolated features tensor of shape (batch_size, num_queries, num_features).
    """
    B, Nq, K = index.shape
    batch_offset = torch.arange(B, device=x.device).reshape(-1, 1, 1) * x.shape[1]
    index_flat = (index + batch_offset).flatten()
    _x = x.flatten(0, 1)[index_flat].reshape(B, Nq, K, x.shape[-1])
    return (_x * weight.unsqueeze(-1)).sum(-2)

def knn_points(
    query: torch.Tensor,
    key: torch.Tensor,
    k: int,
    sorted: bool = False,
    transpose: bool = False,
):
    """Compute k nearest neighbors.

    Args:
        query: [B, N1, D], query points. [B, D, N1] if @transpose is True.
        key:  [B, N2, D], key points. [B, D, N2] if @transpose is True.
        k: the number of nearest neighbors.
        sorted: whether to sort the results
        transpose: whether to transpose the last two dimensions.

    Returns:
        torch.Tensor: [B, N1, K], distances to the k nearest neighbors in the key.
        torch.Tensor: [B, N1, K], indices of the k nearest neighbors in the key.
    """
    if transpose:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)

    distance = torch.cdist(query, key)
    if k == 1:
        knn_dist, knn_ind = torch.min(distance, dim=2, keepdim=True)
    else:
        knn_dist, knn_ind = torch.topk(distance, k, dim=2, largest=False, sorted=sorted)
    return knn_dist, knn_ind

@dataclasses.dataclass
class AuxInputs:
    coords: torch.Tensor


    centers: torch.Tensor
    interp_index: torch.Tensor = None
    interp_weight: torch.Tensor = None


class MaskDecoder(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        embedding_input_dim: int = 448,
        utonia_enc_channels: list = None,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_mask_tokens = 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        self.encoder_mapper = MLP(embedding_input_dim, transformer_dim, transformer_dim, 3)
        self.seg_token_mapper = MLP(embedding_input_dim, transformer_dim, transformer_dim, 3)
        self.output_upscaling = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.GELU(),
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
        )



        utonia_enc_channels = tuple(utonia_enc_channels or [54, 108, 216, 432, 576])
        utonia_raw_dim = sum(utonia_enc_channels)
        self.output_upscaling_with_fusion = nn.Sequential(
            nn.Linear(transformer_dim + utonia_raw_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.GELU(),
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
        )

    def forward(
        self,
        pc_embeddings: torch.Tensor,
        pc_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        aux_inputs: AuxInputs,
        feat_raw: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        pc_embeddings = self.encoder_mapper(pc_embeddings)
        sparse_prompt_embeddings = self.seg_token_mapper(sparse_prompt_embeddings)

        masks = self.predict_masks_joint(
            pc_embeddings=pc_embeddings,
            pc_pe=pc_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            aux_inputs=aux_inputs,
            feat_raw=feat_raw,
        )

        return masks

    def predict_masks_joint(
        self,
        pc_embeddings: torch.Tensor,
        pc_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        aux_inputs: AuxInputs,
        feat_raw: torch.Tensor = None,
    ) -> torch.Tensor:
        """Joint classification: all seg tokens + BG in sequence dim.

        Args:
            pc_embeddings: [1, L, dim] point cloud embeddings (already mapped)
            pc_pe: [1, L, dim] positional encoding
            sparse_prompt_embeddings: [1, S+1, dim] all seg + BG tokens (already mapped)
            aux_inputs: KNN interpolation data
            feat_raw: [1, N, utonia_raw_dim] optional Utonia features

        Returns:
            masks: [1, S+1, N] per-point logits for each class
        """
        tokens = sparse_prompt_embeddings
        src = pc_embeddings
        pos_src = pc_pe


        interp_index = aux_inputs.interp_index
        interp_weight = aux_inputs.interp_weight
        if interp_index is None or interp_weight is None:
            with torch.no_grad():
                interp_index, interp_weight = compute_interp_weights(
                    aux_inputs.coords, aux_inputs.centers
                )
            aux_inputs.interp_index = interp_index
            aux_inputs.interp_weight = interp_weight


        hs, src = self.transformer(src, pos_src, tokens)



        interp_emb = interpolate_features(src, interp_index, interp_weight)


        if feat_raw is not None:
            first_linear = self.output_upscaling_with_fusion[0]
            W = first_linear.weight
            b = first_linear.bias
            W_interp = W[:, :pc_embeddings.shape[-1]]
            W_feat = W[:, pc_embeddings.shape[-1]:]
            feat_raw_contrib = feat_raw @ W_feat.T
            combined = interp_emb @ W_interp.T + feat_raw_contrib + b
            upscaled = combined
            for layer in self.output_upscaling_with_fusion[1:]:
                upscaled = layer(upscaled)
        else:
            upscaled = self.output_upscaling(interp_emb)



        num_tokens = hs.shape[1]
        hyper_in_list = []
        for i in range(num_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[0](hs[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        masks = hyper_in @ upscaled.transpose(-1, -2)

        return masks


# Adapted from https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x), inplace=True) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        pc_embedding: Tensor,
        pc_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          pc_embedding (torch.Tensor): point cloud to attend to. Should be shape
            B x N_pc_tokens x embedding_dim.
          pc_pe (torch.Tensor): the positional encoding to add to the point cloud. 
            Must have the same shape as pc_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed pc_embedding
        """

        queries = point_embedding
        keys = pc_embedding


        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=pc_pe,
            )

        


        q = queries + point_embedding
        k = keys + pc_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:

        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)


        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)


        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)


        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)


        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)


        out = F.scaled_dot_product_attention(q, k, v)


        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))



class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = 1.0) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [-1,1]."""
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: shape (..., coord_dim), normalized coordinates in [-1, 1].

        Returns:
            torch.Tensor: shape (..., num_pos_feats), positional encoding.
        """
        if (coords < -1.0 - 1e-6).any() or (coords > 1.0 + 1e-6).any():
            coord_min = coords.min().item()
            coord_max = coords.max().item()
            raise ValueError(
                f"Input coordinates must be normalized to [-1, 1], got "
                f"[{coord_min:.6g}, {coord_max:.6g}]"
            )
        return self._pe_encoding(coords)
