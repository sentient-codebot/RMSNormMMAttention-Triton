"This file defines the torch nn.Module"
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Complex, Float, Int
from torch import Tensor, nn


class AttentionOutput(NamedTuple):
    """namedtuple
    - x_out: Float[Tensor, "batch sequence dim"]
    - context_out: Float[Tensor, "batch seq2 dim"] | None = None
    """

    x_out: Float[Tensor, "batch sequence dim"]
    context_out: Float[Tensor, "batch seq2 dim"] | None = None


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
        start_pos: Int | None = None, # shape: (batch,)
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