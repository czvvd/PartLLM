import os
import yaml
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PreTrainedModel, PretrainedConfig


class PointDiTConfig(PretrainedConfig):
    model_type = "qwen3_vl_point_dit"
    base_config_key = "point_dit_config"

    def __init__(self,
        input_size: int = 4096,
        in_channels: int = 64,
        out_channels: int = 2,
        hidden_size: int = 2048,
        context_dim: int = 1024,
        depth: int = 21,
        num_heads: int = 16,
        qk_norm: bool = True,
        text_len: int = 1370,
        with_decoupled_ca: bool = False,
        use_attention_pooling: bool = False,
        qk_norm_type: str = 'rms',
        qkv_bias: bool = False,
        use_pos_emb: bool = False,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
        mlp_ratio=4.0,
        norm_type='layer',
        additional_cond_hidden_state=768,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        guidance_cond_proj_dim=None,
        **kwargs,
    ):
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.context_dim = context_dim
        self.depth = depth
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.text_len = text_len
        self.with_decoupled_ca = with_decoupled_ca

        self.use_attention_pooling = use_attention_pooling
        self.qk_norm_type = qk_norm_type
        self.qkv_bias = qkv_bias
        self.use_pos_emb = use_pos_emb
        self.num_moe_layers = num_moe_layers
        self.num_experts = num_experts
        self.moe_top_k = moe_top_k

        self.norm_type = norm_type
        self.additional_cond_hidden_state = additional_cond_hidden_state
        self.decoupled_ca_dim = decoupled_ca_dim
        self.decoupled_ca_weight = decoupled_ca_weight
        self.guidance_cond_proj_dim = guidance_cond_proj_dim
        self.mlp_ratio = mlp_ratio

        super().__init__(**kwargs)


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss, 
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss

class MoEGate(nn.Module):
    def __init__(self, embed_dim, num_experts=16, num_experts_per_tok=2, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts

        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False


        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape    

        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator


        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k

            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1, 
                    topk_idx_for_aux_loss, 
                    torch.ones(
                        bsz, seq_len * aux_topk,
                        device=hidden_states.device
                    )
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean()
                aux_loss = aux_loss * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1),
                                    num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss

class MoEBlock(nn.Module):
    def __init__(self, dim, num_experts=8, moe_top_k=2,
                    activation_fn = "gelu", dropout=0.0, final_dropout = False, 
                    ff_inner_dim = None, ff_bias = True):
        super().__init__()
        self.moe_top_k = moe_top_k
        self.experts = nn.ModuleList([
                FeedForward(dim,dropout=dropout, 
                            activation_fn=activation_fn,  
                            final_dropout=final_dropout,  
                            inner_dim=ff_inner_dim,  
                            bias=ff_bias)
        for i in range(num_experts)])
        self.gate = MoEGate(embed_dim=dim, num_experts=num_experts, num_experts_per_tok=moe_top_k)

        self.shared_experts = FeedForward(dim,dropout=dropout, activation_fn=activation_fn,  
                                          final_dropout=final_dropout,  inner_dim=ff_inner_dim,  
                                          bias=ff_bias)

    def initialize_weight(self):
        pass
    
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states) 

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.moe_top_k, dim=0)
            y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
            for i, expert in enumerate(self.experts): 
                tmp = expert(hidden_states[flat_topk_idx == i])
                y[flat_topk_idx == i] = tmp.to(hidden_states.dtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y =  y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        y = y + self.shared_experts(identity)
        return y
    

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x) 
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.moe_top_k 
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]]) 
            

            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                                         expert_out, 
                                         reduce='sum')
        return expert_cache


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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


class Timesteps(nn.Module):
    def __init__(self,
                 num_channels: int,
                 downscale_freq_shift: float = 0.0,
                 scale: int = 1,
                 max_period: int = 10000
                 ):
        super().__init__()
        self.num_channels = num_channels
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        embedding_dim = self.num_channels
        half_dim = embedding_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = self.scale * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, cond_proj_dim=None, out_size=None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, frequency_embedding_size, bias=True),
            nn.GELU(),
            nn.Linear(frequency_embedding_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, frequency_embedding_size, bias=False)

        self.time_embed = Timesteps(hidden_size)

    def forward(self, t, condition):

        t_freq = self.time_embed(t).type(self.mlp[0].weight.dtype)

        if condition is not None:
            t_freq = t_freq + self.cond_proj(condition)

        t = self.mlp(t_freq)
        t = t.unsqueeze(dim=1)
        return t


class MLP(nn.Module):
    def __init__(self, *, width: int):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class CrossAttention(nn.Module):
    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        with_decoupled_ca=False,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        **kwargs,
    ):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "self.qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = nn.Linear(kdim, qdim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(qdim, qdim, bias=True)

        self.with_dca = with_decoupled_ca
        if self.with_dca:
            self.kv_proj_dca = nn.Linear(kdim, 2 * qdim, bias=qkv_bias)
            self.k_norm_dca = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
            self.dca_dim = decoupled_ca_dim
            self.dca_weight = decoupled_ca_weight

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen1, hidden_dim) (where hidden_dim = num heads * head dim)
        y: torch.Tensor
            (batch, seqlen2, hidden_dim2)
        freqs_cis_img: torch.Tensor
            (batch, hidden_dim // 2), RoPE for image
        """
        b, s1, c = x.shape

        if self.with_dca:
            token_len = y.shape[1]
            context_dca = y[:, -self.dca_dim:, :]
            kv_dca = self.kv_proj_dca(context_dca).view(b, self.dca_dim, 2, self.num_heads, self.head_dim)
            k_dca, v_dca = kv_dca.unbind(dim=2)
            k_dca = self.k_norm_dca(k_dca)
            y = y[:, :(token_len - self.dca_dim), :]

        _, s2, c = y.shape
        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        kv = torch.cat((k, v), dim=-1)
        split_size = kv.shape[-1] // self.num_heads // 2
        kv = kv.view(1, -1, self.num_heads, split_size * 2)
        k, v = torch.split(kv, split_size, dim=-1)

        q = q.view(b, s1, self.num_heads, self.head_dim)
        k = k.view(b, s2, self.num_heads, self.head_dim)
        v = v.view(b, s2, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True
        ):
            q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.num_heads), (q, k, v))
            context = F.scaled_dot_product_attention(
                q, k, v
            ).transpose(1, 2).reshape(b, s1, -1)

        if self.with_dca:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=True
            ):
                k_dca, v_dca = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.num_heads),
                                   (k_dca, v_dca))
                context_dca = F.scaled_dot_product_attention(
                    q, k_dca, v_dca).transpose(1, 2).reshape(b, s1, -1)

            context = context + self.dca_weight * context_dca

        out = self.out_proj(context)

        return out


class Attention(nn.Module):
    """
    We rename some layer names to align with flash attention
    """

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.head_dim = self.dim // num_heads

        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qkv = torch.cat((q, k, v), dim=-1)
        split_size = qkv.shape[-1] // self.num_heads // 3
        qkv = qkv.view(1, -1, self.num_heads, split_size * 3)
        q, k, v = torch.split(qkv, split_size, dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True
        ):
            x = F.scaled_dot_product_attention(q, k, v)
            x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.out_proj(x)
        return x


class HunYuanDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        c_emb_size,
        num_heads,
        text_states_dim=1024,
        use_flash_attn=False,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
        qk_norm_layer=nn.RMSNorm,
        with_decoupled_ca=False,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        init_scale=1.0,
        qkv_bias=True,
        skip_connection=True,
        timested_modulate=False,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.use_flash_attn = use_flash_attn
        use_ele_affine = True


        self.norm1 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)
        self.attn1 = Attention(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
                               norm_layer=qk_norm_layer)


        self.norm2 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)



        self.timested_modulate = timested_modulate
        if self.timested_modulate:
            self.default_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(c_emb_size, hidden_size, bias=True)
            )


        self.attn2 = CrossAttention(hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_norm=qk_norm, norm_layer=qk_norm_layer,
                                    with_decoupled_ca=with_decoupled_ca, decoupled_ca_dim=decoupled_ca_dim,
                                    decoupled_ca_weight=decoupled_ca_weight, init_scale=init_scale,
                                    )
        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)

        if skip_connection:
            self.skip_norm = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None

        self.use_moe = False
        if self.use_moe:

            print("using moe")
            self.moe = MoEBlock(
                hidden_size,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
                dropout=0.0,
                activation_fn="gelu",
                final_dropout=False,
                ff_inner_dim=int(hidden_size * 4.0),
                ff_bias=True,
            )
        else:
            self.mlp = MLP(width=hidden_size)

    def forward(self, x, c=None, text_states=None, skip_value=None):

        if self.skip_linear is not None:
            cat = torch.cat([skip_value, x], dim=-1)
            x = self.skip_linear(cat)
            x = self.skip_norm(x)


        if self.timested_modulate:
            shift_msa = self.default_modulation(c).unsqueeze(dim=1)
            x = x + shift_msa

        attn_out = self.attn1(self.norm1(x))

        x = x + attn_out


        x = x + self.attn2(self.norm2(x), text_states)


        mlp_inputs = self.norm3(x)

        if self.use_moe:
            x = x + self.moe(mlp_inputs)
        else:
            x = x + self.mlp(mlp_inputs)

        return x


class AttentionPool(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, attention_mask=None):
        x = x.permute(1, 0, 2)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).permute(1, 0, 2)
            global_emb = (x * attention_mask).sum(dim=0) / attention_mask.sum(dim=0)
            x = torch.cat([global_emb[None,], x], dim=0)

        else:
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class FinalLayer(nn.Module):
    """
    The final layer of HunYuanDiT.
    """

    def __init__(self, final_hidden_size, out_channels):
        super().__init__()
        self.final_hidden_size = final_hidden_size
        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(final_hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = x[:, 1:]
        x = self.linear(x)
        return x


class PointDiffusionModel(PreTrainedModel):
    config: PointDiTConfig
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    
    def __init__(
        self,
        config: PointDiTConfig,
        **kwargs
    ):
        super().__init__(config)
        self.input_size = config.input_size
        self.depth = config.depth
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_heads = config.num_heads

        self.hidden_size = config.hidden_size
        self.norm = nn.LayerNorm if config.norm_type == 'layer' else nn.RMSNorm
        self.qk_norm = nn.RMSNorm if config.qk_norm_type == 'rms' else nn.LayerNorm
        self.context_dim = config.context_dim

        self.with_decoupled_ca = config.with_decoupled_ca
        self.decoupled_ca_dim = config.decoupled_ca_dim
        self.decoupled_ca_weight = config.decoupled_ca_weight
        self.use_pos_emb = config.use_pos_emb
        self.use_attention_pooling = config.use_attention_pooling
        self.guidance_cond_proj_dim = config.guidance_cond_proj_dim

        self.text_len = config.text_len

        self.x_embedder = nn.Linear(config.in_channels, config.hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(config.hidden_size, config.hidden_size * 4, cond_proj_dim=config.guidance_cond_proj_dim)


        if self.use_pos_emb:
            self.register_buffer("pos_embed", torch.zeros(1, config.input_size, config.hidden_size))
            pos = np.arange(self.input_size, dtype=np.float32)
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], pos)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.use_attention_pooling = config.use_attention_pooling
        if config.use_attention_pooling:
            self.pooler = AttentionPool(self.text_len, config.context_dim, num_heads=8, output_dim=1024)
            self.extra_embedder = nn.Sequential(
                nn.Linear(1024, config.hidden_size * 4),
                nn.SiLU(),
                nn.Linear(config.hidden_size * 4, config.hidden_size, bias=True),
            )

        if config.with_decoupled_ca:
            self.additional_cond_hidden_state = config.additional_cond_hidden_state
            self.additional_cond_proj = nn.Sequential(
                nn.Linear(config.additional_cond_hidden_state, config.hidden_size * 4),
                nn.SiLU(),
                nn.Linear(config.hidden_size * 4, 1024, bias=True),
            )


        self.blocks = nn.ModuleList([
            HunYuanDiTBlock(hidden_size=config.hidden_size,
                            c_emb_size=config.hidden_size,
                            num_heads=config.num_heads,
                            mlp_ratio=config.mlp_ratio,
                            text_states_dim=config.context_dim,
                            qk_norm=config.qk_norm,
                            norm_layer=self.norm,
                            qk_norm_layer=self.qk_norm,
                            skip_connection=layer > config.depth // 2,
                            with_decoupled_ca=config.with_decoupled_ca,
                            decoupled_ca_dim=config.decoupled_ca_dim,
                            decoupled_ca_weight=config.decoupled_ca_weight,
                            qkv_bias=config.qkv_bias,
                            use_moe=True if config.depth - layer <= config.num_moe_layers else False,
                            num_experts=config.num_experts,
                            moe_top_k=config.moe_top_k
                            )
            for layer in range(config.depth)
        ])
        self.depth = config.depth

        self.final_layer = FinalLayer(config.hidden_size, self.out_channels)

        self.cond_in = nn.Linear(config.context_dim, self.hidden_size)
    

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
        if isinstance(module, (nn.Linear | nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
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

    def forward(self, x, t, context, **kwargs):

        cond = context

        t = self.t_embedder(t, condition=kwargs.get('guidance_cond'))
        x = self.x_embedder(x[None])

        if self.use_pos_emb:
            pos_embed = self.pos_embed[:, :x.shape[1]].to(x.dtype)
            x = x + pos_embed

        x = torch.cat([t, x], dim=1)

        skip_value_list = []
        for layer, block in enumerate(self.blocks):
            skip_value = None if layer <= self.depth // 2 else skip_value_list.pop()
            x = block(x, t, cond, skip_value=skip_value)
            if layer < self.depth // 2:
                skip_value_list.append(x)

        x = self.final_layer(x)
        return x
