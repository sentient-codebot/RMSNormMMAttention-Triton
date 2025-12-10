
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8), # Unlikely
    ],
    key=['N'] # Re-benchmarking if N (dimension) changes
)
@triton.jit
def _layernorm_adaln_fwd_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    y_ptr,
    stride_x_batch, stride_x_seq, stride_x_dim,
    stride_scale_batch, stride_scale_dim,
    stride_shift_batch, stride_shift_dim,
    stride_y_batch, stride_y_seq, stride_y_dim,
    N,  # dimension
    seq_len, # sequence length to map pid to batch_idx
    eps,
    BLOCK_SIZE: tl.constexpr
):
    """Fused LayerNorm + AdaLN modulation forward kernel.

    x: (Batch, Seq, Dim)
    scale: (Batch, 1, Dim)
    shift: (Batch, 1, Dim)
    y: (Batch, Seq, Dim)

    Process:
        - the grid is designed to be (Batch * Seq,)
    
    """
    pid = tl.program_id(0)
    
    # Map pid to batch and seq index
    batch_idx = pid // seq_len  # Which sequence are we in
    seq_idx = pid % seq_len  # Which token in the sequence
    
    # Calculate pointers
    x_row_ptr = x_ptr + batch_idx * stride_x_batch + seq_idx * stride_x_seq
    y_row_ptr = y_ptr + batch_idx * stride_y_batch + seq_idx * stride_y_seq
    
    scale_row_ptr = scale_ptr + batch_idx * stride_scale_batch
    shift_row_ptr = shift_ptr + batch_idx * stride_shift_batch
    
    # Load data
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # stride in feature dimension, should be 1 for row-major
    x_offset = cols * stride_x_dim  
    x = tl.load(x_row_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)
    
    # Load scale and shift
    scale_offset = cols * stride_scale_dim
    shift_offset = cols * stride_shift_dim
    scale = tl.load(scale_row_ptr + scale_offset, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_row_ptr + shift_offset, mask=mask, other=0.0).to(tl.float32)
    
    # LayerNorm Computation
    # mean = sum(x) / N
    # var = sum((x - mean)^2) / N
    # x_norm = (x - mean) / sqrt(var + eps)
    
    mean = tl.sum(x, axis=0) / N
    x_centered = tl.where(mask, x - mean, 0.0) # Masking is crucial here for correctness of var if BLOCK_SIZE > N
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd
    
    # AdaLN Modulation
    out = x_norm * (1.0 + scale) + shift
    
    # Store output
    y_offset = cols * stride_y_dim
    tl.store(y_row_ptr + y_offset, out, mask=mask)

def layernorm_adaln(x, scale, shift, eps=1e-6):
    """
    Fused LayerNorm + AdaLN modulation.
    
    Args:
        x: Input tensor of shape (Batch, Seq, Dim)
        scale: Scale tensor of shape (Batch, 1, Dim)
        shift: Shift tensor of shape (Batch, 1, Dim)
        eps: Epsilon for LayerNorm
        
    Returns:
        y: Output tensor of shape (Batch, Seq, Dim)
    """
    # Checks
    assert x.dim() == 3, "x must be 3D (Batch, Seq, Dim)"
    assert scale.dim() == 3 and scale.shape[1] == 1, "scale must be (Batch, 1, Dim)"
    assert shift.dim() == 3 and shift.shape[1] == 1, "shift must be (Batch, 1, Dim)"
    assert x.shape[0] == scale.shape[0] == shift.shape[0], "Batch dimension mismatch"
    assert x.shape[2] == scale.shape[2] == shift.shape[2], "Dimension mismatch"
    
    B, L, D = x.shape
    
    # Output tensor
    y = torch.empty_like(x)
    
    # Grid configuration
    grid = (B * L, )
    
    # Heuristic for block size
    BLOCK_SIZE = triton.next_power_of_2(D)
    
    # Launch kernel
    _layernorm_adaln_fwd_kernel[grid](
        x, scale, shift, y,
        x.stride(0), x.stride(1), x.stride(2),
        scale.stride(0), scale.stride(2),
        shift.stride(0), shift.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        N=D,
        seq_len=L,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y
