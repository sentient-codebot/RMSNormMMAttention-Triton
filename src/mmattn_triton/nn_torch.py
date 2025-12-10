"This file defines the torch nn.Module"

from __future__ import annotations

from typing import Any, NamedTuple, TypeVar

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Complex, Float, Int
from torch import Tensor, nn


try:
    import triton  # noqa: F401

    from .fused_layernorm import layernorm_adaln
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    layernorm_adaln = None

T = TypeVar("T", bound=Any)


# helper functions
def default(value: T | None, default_value: T) -> T:
    if value is not None:
        return value
    else:
        # more efficient in that it does not evaluate default_value unless it is needed
        if callable(default_value):
            return default_value()
        else:
            return default_value


class AttentionOutput(NamedTuple):
    """namedtuple
    - x_out: Float[Tensor, "batch sequence dim"]
    - context_out: Float[Tensor, "batch seq2 dim"] | None = None
    """

    x_out: Float[Tensor, "batch sequence dim"]
    context_out: Float[Tensor, "batch seq2 dim"] | None = None


class RMSNormSelfAttention(nn.Module):
    """Self Attention with RMS Norm

    Arguments:
        dim (int): model dimension (==input output dimension)
        num_head (int): number of heads, `dim` must be divisible by `num_head`

    Input:
        x (Float[Tensor, "batch sequence dim"]): input tensor

    Output:
        x (Float[Tensor, "batch sequence dim"]): output tensor

    """

    def __init__(
        self, dim: int, num_head: int = 4, max_seq_len: int = 4096, linear: bool = False
    ):
        super().__init__()
        self.dim_model = dim
        self.num_head = num_head
        if dim % num_head != 0:
            raise ValueError("dim must be divisible by num_head")
        self.dim_head = dim // num_head
        self.scale = self.dim_head**-0.5
        self.to_qkv = nn.Linear(self.dim_model, self.dim_model * 3, bias=True)
        self.to_out = nn.Sequential(
            nn.Linear(self.dim_model, self.dim_model),
            nn.RMSNorm(self.dim_model),
        )
        self.rms_norm_q = nn.RMSNorm(self.dim_model)
        self.rms_norm_k = nn.RMSNorm(self.dim_model)
        if self.dim_head % 2 != 0:
            raise ValueError("dim_head must be even for rotary embedding")
        self.freqs_cis = precompute_freqs_cis(
            dim=self.dim_head,
            max_seq_len=max_seq_len,
        )
        self.linear = linear

    def forward(
        self,
        x: Float[Tensor, "batch sequence dim"],
        attn_mask: Float[Tensor, "batch sequence sequence"] | None = None,
        start_pos: Int | None = None,  # shape: (batch,)
    ) -> AttentionOutput:
        if start_pos is None:
            start_pos = torch.zeros(x.shape[0], device=x.device, dtype=torch.int64)
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 3 * (batch, sequence, dim)
        q, k, v = (t for t in qkv)  # shape: 3 * (batch, sequence, dim)
        q = self.rms_norm_q(q)  # shape: (batch, sequence, dim)
        k = self.rms_norm_k(k)  # shape: (batch, sequence, dim)
        q, k, v = (
            rearrange(
                t,
                "b L (num_head dim_head) -> b num_head L dim_head",
                num_head=self.num_head,
            )
            for t in (q, k, v)
        )  # shape: 3 * (batch, num_head, sequence, dim_head)
        self.freqs_cis = self.freqs_cis.to(q.device)
        # quickly calculate indices for freqs_cis
        _offset = torch.arange(
            q.shape[-2], device=start_pos.device, dtype=start_pos.dtype
        ).unsqueeze(0)  # (1, seq_len)
        pos_indices = start_pos.unsqueeze(1) + _offset  # (batch, seq_len)
        freqs_cis = self.freqs_cis[pos_indices]  # (batch, seq_len, dim//2)
        q, k = apply_rotary_emb(q, freqs_cis, k)

        if attn_mask is not None:
            attn_mask = rearrange(attn_mask, "b L1 L2 -> b 1 L1 L2").bool()

        if not self.linear:
            attn_out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                scale=self.scale,
                attn_mask=attn_mask,
            )  # shape: (batch, num_head, sequence, dim_head)
        else:
            attn_out = linear_attn_op(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
            )  # shape: (batch, num_head, seq_q, dim_value_head)
        attn_out = rearrange(
            attn_out, "b num_head L dim_head -> b L (num_head dim_head)"
        )  # shape: (batch, sequence, dim)
        final_out = AttentionOutput(
            self.to_out(attn_out)
        )  # shape: (batch, sequence, dim)
        return final_out


class RMSNormSelfCrossAttention(nn.Module):
    """Attention with RMS Norm and Cross Attention with context.
    NOTE: no positional embedding is applied on the context. assuming they are
    order-agnostic. each context element's representation is learned in previous
    layers.

    Arguments:
        dim (int): model dimension (==input output dimension)
        num_head (int): number of heads, `dim` must be divisible by `num_head`
        use_context (bool): whether to use context

    Input:
        x (Float[Tensor, "batch sequence dim"]): input tensor
        c (Float[Tensor, "batch seq2 dim"] | None): context tensor

    Output:
        x (Float[Tensor, "batch sequence dim"]): output tensor
        c (Float[Tensor, "batch seq2 dim"] | None): output tensor
    """

    def __init__(
        self,
        dim: int,
        num_head: int = 4,
        use_context: bool = False,
        max_seq_len: int = 4096,
        linear: bool = False,
    ):
        super().__init__()
        self.dim_model = dim
        self.num_head = num_head
        if dim % num_head != 0:
            raise ValueError("dim must be divisible by num_head")
        self.dim_head = dim // num_head
        self.scale = self.dim_head**-0.5
        self.to_qkv = nn.Linear(self.dim_model, self.dim_model * 3, bias=True)
        self.to_out = nn.Sequential(
            nn.Linear(self.dim_model, self.dim_model),
            nn.RMSNorm(self.dim_model),
        )
        self.rms_norm_q = nn.RMSNorm(self.dim_model)
        self.rms_norm_k = nn.RMSNorm(self.dim_model)

        self.use_context = use_context
        if self.use_context:
            self.c_to_qkv = nn.Linear(self.dim_model, self.dim_model * 3, bias=True)
            self.c_rms_norm_q = nn.RMSNorm(self.dim_model)
            self.c_rms_norm_k = nn.RMSNorm(self.dim_model)
            self.c_to_out = nn.Sequential(
                nn.Linear(self.dim_model, self.dim_model),
                nn.RMSNorm(self.dim_model),
            )
        if self.dim_head % 2 != 0:
            raise ValueError("dim_head must be even for rotary embedding")
        self.freqs_cis = precompute_freqs_cis(
            dim=self.dim_head,
            max_seq_len=max_seq_len,
        )
        self.linear = linear

    def _get_augmented_mask(
        self, attn_mask: Float[Tensor, "batch seq1 seq2"], context_length: int
    ):
        """`attn_mask` without context -> `attn_mask` with context.
        if sequence length == L, context length == C, then:
            (b L L) -> (b L+C L+C)
        """
        attn_mask = attn_mask.detach().bool()  # make sure no grad
        batch_size = attn_mask.shape[0]
        _seq_length = attn_mask.shape[1]  # [b, L, L]
        block_c_to_seq = attn_mask.any(dim=1, keepdim=True)  # [b, 1, L]
        block_c_to_seq = block_c_to_seq.expand(-1, context_length, -1)  # [b, C, L]
        block_seq_to_c = attn_mask.any(dim=2, keepdim=True).expand(
            -1, -1, context_length
        )  # [b, L, C]
        block_c_to_c = torch.ones(
            (batch_size, context_length, context_length),
            device=attn_mask.device,
            dtype=attn_mask.dtype,
        )  # [b, C, C]
        block_seq_to_both = torch.concat(
            [attn_mask, block_seq_to_c], dim=2
        )  # [b, L, L+C]
        block_c_to_both = torch.concat(
            [block_c_to_seq, block_c_to_c], dim=2
        )  # [b, C, L+C]
        full_mask = torch.concat(
            [block_seq_to_both, block_c_to_both], dim=1
        )  # [b, L+C, L+C]

        return full_mask

    def forward(
        self,
        x: Float[Tensor, "batch sequence dim"],
        context: Float[Tensor, "batch seq2 dim"] | None = None,
        attn_mask: Float[Tensor, "batch sequence sequence"] | None = None,
        start_pos: Int | None = None,  # shape: (batch,)
    ) -> AttentionOutput:
        if start_pos is None:
            start_pos = torch.zeros(x.shape[0], device=x.device, dtype=torch.int64)
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 3 * (batch, sequence, dim)
        q, k, v = (t for t in qkv)  # shape: 3 * (batch, sequence, dim)
        q = self.rms_norm_q(q)  # shape: (batch, sequence, dim)
        k = self.rms_norm_k(k)  # shape: (batch, sequence, dim)
        q, k, v = (
            rearrange(
                t,
                "b L (num_head dim_head) -> b num_head L dim_head",
                num_head=self.num_head,
            )
            for t in (q, k, v)
        )  # shape: 3 * (batch, num_head, sequence, dim_head)
        self.freqs_cis = self.freqs_cis.to(q.device)
        # quickly calculate indices for freqs_cis
        _offset = torch.arange(
            q.shape[-2], device=start_pos.device, dtype=start_pos.dtype
        ).unsqueeze(0)  # (1, seq_len)
        pos_indices = start_pos.unsqueeze(1) + _offset  # (batch, seq_len)
        freqs_cis_x = self.freqs_cis[pos_indices]  # (batch, seq_len, dim//2)
        q, k = apply_rotary_emb(q, freqs_cis_x, k)
        if self.use_context and context is not None:
            c_qkv = self.c_to_qkv(context).chunk(3, dim=-1)  # 3 * (batch, seq2, dim)
            c_q, c_k, c_v = (t for t in c_qkv)  # shape: 3 * (batch, seq2, dim)
            c_q = self.c_rms_norm_q(c_q)  # shape: (batch, seq2, dim)
            c_k = self.c_rms_norm_k(c_k)
            c_q, c_k, c_v = (
                rearrange(
                    t,
                    "b L (num_head dim_head) -> b num_head L dim_head",
                    num_head=self.num_head,
                )
                for t in (c_q, c_k, c_v)
            )  # shape: 3 * (batch, num_head, seq2, dim_head)
            q = torch.concat((q, c_q), dim=2)
            # shape: (batch, num_head, sequence+seq2, dim_head)
            k = torch.concat((k, c_k), dim=2)
            # shape: (batch, num_head, sequence+seq2, dim_head)
            v = torch.concat((v, c_v), dim=2)
            # shape: (batch, num_head, sequence+seq2, dim_head)

        if attn_mask is not None:
            if context is not None:
                attn_mask = self._get_augmented_mask(
                    attn_mask, context.shape[1]
                )  # [batch, L+C, L+C]
            attn_mask = rearrange(attn_mask, "b L1 L2 -> b 1 L1 L2").bool()

        if not self.linear:
            attn_out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                scale=self.scale,
                attn_mask=attn_mask,
            )  # shape: (B, #H, L, dim_head)
        else:
            attn_out = linear_attn_op(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
            )  # shape: (B, #H, L, dim_head)

        attn_out = rearrange(
            attn_out, "b num_head L dim_head -> b L (num_head dim_head)"
        )  # shape: (batch, sequence, dim)
        x_out = attn_out[:, : x.shape[1], :]  # shape: (batch, sequence, dim)
        if self.use_context and context is not None:
            c_out = attn_out[:, x.shape[1] :, :]  # shape: (batch, seq2, dim)
            return AttentionOutput(self.to_out(x_out), self.c_to_out(c_out))
        else:
            return AttentionOutput(self.to_out(x_out))


class GPT2Block(nn.Module):
    """GPT2 block for 1D data

    Forward:
        input:
            - x: (batch, sequence, channel)

        return:
            - x: (batch, sequence, channel)

    """

    def __init__(
        self,
        dim_base: int,  # model dimension
        num_attn_head: int = 4,
        dim_head: None | int = None,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        conditioning: bool = False,
        use_cross_attn: bool = False,
    ):
        self.linear_attn = False
        super().__init__()
        self.dim_base = dim_base
        self.num_attn_head = num_attn_head
        self.dim_head = default(dim_head, dim_base // num_attn_head)
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.conditioning = conditioning
        self.use_cross_attn = use_cross_attn
        if use_cross_attn and not conditioning:
            raise ValueError("cross attention must be used with conditioning.")

        if self.use_cross_attn:
            self.context_layers = nn.ModuleDict({})
            self.context_layers["ln_1"] = nn.LayerNorm(
                dim_base, elementwise_affine=not self.conditioning, eps=1e-6
            )
            self.context_layers["ln_2"] = nn.LayerNorm(
                dim_base, elementwise_affine=not self.conditioning, eps=1e-6
            )
            self.cross_attn = RMSNormSelfCrossAttention(
                dim_base,
                num_head=self.num_attn_head,
                use_context=self.use_cross_attn,
                linear=self.linear_attn,
            )
            self.context_layers["mlp"] = nn.Sequential(
                nn.Linear(dim_base, dim_feedforward),
                nn.SiLU(),
                nn.Linear(dim_feedforward, dim_base),
                nn.Dropout(self.dropout),
            )
            # assert conditioning
            self.context_layers["adaLN_modulation"] = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim_base, dim_base * 6),
            )
        else:
            self.attn = RMSNormSelfAttention(
                dim_base,
                num_head=self.num_attn_head,
                linear=self.linear_attn,
            )

        self.ln_1 = nn.LayerNorm(
            dim_base, elementwise_affine=not self.conditioning, eps=1e-6
        )
        self.ln_2 = nn.LayerNorm(
            dim_base, elementwise_affine=not self.conditioning, eps=1e-6
        )

        # no cross attention compared with the GPT2
        if not self.linear_attn:
            self.mlp = nn.Sequential(
                nn.Linear(dim_base, dim_feedforward),
                nn.SiLU(),
                nn.Linear(dim_feedforward, dim_base),
                nn.Dropout(self.dropout),
            )
        else:
            # use mixffn
            raise NotImplementedError("mixffn not implemented in this project.")

        # conditioning
        if self.conditioning:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim_base, dim_base * 6),
            )

    def forward(
        self,
        x: Float[Tensor, "batch sequence channel"],
        y: None | Float[Tensor, "batch 1 channel"],
        attn_mask: None | Float[Tensor, "batch sequence sequence"],
        context: None | Float[Tensor, "batch seq2 channel"] = None,
        start_pos: int = 0,
    ) -> AttentionOutput:
        """
        Input:
            x - input tensor (batch, sequence, channel)
            y - label tensor (batch, 1, channel)
            attn_mask - attention mask (batch, sequence, sequence)
            context - context tensor (batch, seq2, channel)

        Return:
            x - updated sequence (batch, sequence, channel)
            if use_cross_attn:
            context - updated context sequence (batch, seq2, channel)
        """
        if self.conditioning and y is None:
            raise ValueError("y must be provided when conditioning is True.")
        if self.use_cross_attn and context is None:
            raise ValueError("context must be provided when use_cross_attn is True.")
        if self.conditioning:
            cond_scale_shift_gate = self.adaLN_modulation(
                y
            )  # shape: (batch, 1, channel * 6)
            scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp = (
                cond_scale_shift_gate.chunk(6, dim=2)
            )
        else:
            scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp = [
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
            ]
        if self.use_cross_attn:
            cond_scale_shift_gate = self.context_layers["adaLN_modulation"](
                y
            )  # shape: (batch, 1, channel * 6)
            (
                context_scale_attn,
                context_shift_attn,
                context_gate_attn,
                context_scale_mlp,
                context_shift_mlp,
                context_gate_mlp,
            ) = cond_scale_shift_gate.chunk(6, dim=2)

        # copy + normalization
        x_copy = x.clone()
        if self.conditioning and HAS_TRITON and x.is_cuda:
            # Fused LayerNorm + AdaLN (only when conditioning is True)
            x = layernorm_adaln(x, scale_attn, shift_attn)
        else:
            x = self.ln_1(x)
            x = x * (1.0 + scale_attn) + shift_attn  # scale shift from adaLN (or 0 if not conditioning)
            
        if self.use_cross_attn:
            context_copy = context.clone()
            if self.conditioning and HAS_TRITON and context.is_cuda:
                 context = layernorm_adaln(context, context_scale_attn, context_shift_attn)
            else:
                context = self.context_layers["ln_1"](context)
                context = (
                    context * (1.0 + context_scale_attn) + context_shift_attn
                )  # scale shift

        # attention
        if not self.use_cross_attn:
            attn_output = self.attn(
                x,
                attn_mask=attn_mask,
                start_pos=start_pos,
            )  # shape: (batch, sequence, channel)
            attn_x = attn_output.x_out
            attn_context = attn_output.context_out
        else:
            attn_output = self.cross_attn(
                x, context=context, attn_mask=attn_mask, start_pos=start_pos
            )
            attn_x = attn_output.x_out
            attn_context = attn_output.context_out
        x = x_copy + attn_x * gate_attn  # residual connection
        if self.use_cross_attn:
            context = context_copy + attn_context * context_gate_attn
            # residual connection

        # mlp / feedforward
        x_copy = x.clone()
        if self.conditioning and HAS_TRITON and x.is_cuda:
             x = layernorm_adaln(x, scale_mlp, shift_mlp)
        else:
            x = self.ln_2(x)
            x = x * (1.0 + scale_mlp) + shift_mlp  # scale shift from adaLN

        x = self.mlp(x)
        x = x_copy + x * gate_mlp  # residual connection
        if self.use_cross_attn:
            context_copy = context.clone()
            if self.conditioning and HAS_TRITON and context.is_cuda:
                 context = layernorm_adaln(context, context_scale_mlp, context_shift_mlp)
            else:
                context = self.context_layers["ln_2"](context)
                context = context * (1.0 + context_scale_mlp) + context_shift_mlp
            
            context = self.context_layers["mlp"](context)
            context = context_copy + context * context_gate_mlp  # residual
            return AttentionOutput(x, context)  # (b, seq, c), (b, seq2, c)

        return AttentionOutput(x)  # shape: (batch, sequence, channel)


def linear_attn_op(
    query: Float[Tensor, "batch head seq_q dim"],
    key: Float[Tensor, "batch head seq_k dim"],
    value: Float[Tensor, "batch head seq_k dim"],
    attn_mask: Float[Tensor, "batch head seq_q seq_k"] | None = None,
):
    """Linear attention operation. replacement for
    `scaled_dot_product_attention` with linear complexity.

    Args:

    Returns:
    """
    # hparams
    kernel_func = nn.ReLU(inplace=False)
    eps = 1e-6
    pad_val = 1.0
    query = kernel_func(query)  # shape: (batch, head, seq_q, dim)
    key = kernel_func(key)  # shape: (batch, head, seq_k, dim)

    if attn_mask is not None:
        # q_mask = attn_mask[:, :, :, 0]  # shape: (B, H, L_q)
        k_mask = attn_mask[:, :, 0, :]  # shape: (B, H, L_k)
        # q_mask = q_mask.unsqueeze(-1)  # shape: (B, H, L_q, 1)
        k_mask = k_mask.unsqueeze(-1)  # shape: (B, H, L_k, 1)
        # query = query * q_mask  # shape: (B, H, L_q, D)
        key = key * k_mask  # shape: (B, H, L_k, D)

    value = F.pad(
        value,
        (0, 1),
        mode="constant",
        value=pad_val,
    )  # shape: (B, H, L_k, D) -> (B, H, L_k, D+1)

    vk = einsum(
        value,
        key,
        "b h L_k d_plus_one, b h L_k d -> b h d_plus_one d",
    )  # shape: (batch, head, D_v+1, D_k)
    result = einsum(
        vk,
        query,
        "b h d_plus_one d, b h L_q d -> b h L_q d_plus_one",
    )  # shape: (batch, head, seq_q, D_v+1)
    # normalization
    result = result[:, :, :, :-1] / (result[:, :, :, -1:] + eps)  # (B, H, seq_Q, D_v)

    return result  # shape: (batch, head, seq_q, dim)


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Complex[Tensor, "seq_len dim//2"]:
    """precompute the rotatory complex numbers for RoPE. origin: Llama

    Args:
        dim (int): dimension of the embedding, must be even
        max_seq_len (int): maximum sequence length
        theta (float): scaling factor for the frequency. default = 10000.0
    Returns:
        freqs_cis (Complex[Tensor, "seq_len dim//2"]): precomputed complex numbers

    """
    # shape: (dim//2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # shape: (max_seq_len)
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


@torch.compiler.disable(recursive=False)
def apply_rotary_emb(
    xq: Float[Tensor, "batch * seq_len dim"],
    freqs_cis: Complex[Tensor, "seq_len dim//2"] | Complex[Tensor, "batch seq_len dim"],
    xk: torch.Tensor | None = None,
):
    """Apply rotary embedding to the input tensor.

    Args:
        xq (Float[Tensor, "batch * dim"]): input tensor
        freqs_cis (Complex[Tensor, "seq_len dim//2"] |
            Complex[Tensor, "batch seq_len dim//2]): precomputed complex numbers.
            assumes on the same device as xq. Can be batched.
        xk (torch.Tensor | None): optional second input tensor to apply rotary embedding
    Returns: tuple[Tensor, Tensor]
        xq (Float[Tensor, "batch seq_len dim"]): output tensor with rotary embedding applied
        xk (torch.Tensor | None): optional second output tensor with rotary embedding applied

    """
    # splits the last dimension into pairs -> complex
    # shape [batch, *, seq_len, dim//2]
    assert xq.shape[-1] % 2 == 0, "last dimension must be even"
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    if xk is not None:
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    while freqs_cis.ndim < xq_.ndim:
        # batch dimension preserved if exists
        freqs_cis = freqs_cis.unsqueeze(-3)  # [..., seq_len, dim//2]
    # shape [batch, *, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2).type_as(xq)
    if xk is not None:
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2).type_as(xk)
    else:
        xk_out = None
    return xq_out, xk_out
