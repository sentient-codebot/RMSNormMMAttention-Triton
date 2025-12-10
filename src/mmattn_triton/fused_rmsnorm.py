
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_adaln_fwd_kernel(
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
    pid = tl.program_id(0)
    
    # Map pid to batch and seq index
    # We parallelize over (Batch * Seq) rows
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Calculate pointers for the specific row
    x_row_ptr = x_ptr + batch_idx * stride_x_batch + seq_idx * stride_x_seq
    y_row_ptr = y_ptr + batch_idx * stride_y_batch + seq_idx * stride_y_seq
    
    # Scale and Shift are (Batch, 1, Dim), so they only depend on batch_idx
    scale_row_ptr = scale_ptr + batch_idx * stride_scale_batch
    shift_row_ptr = shift_ptr + batch_idx * stride_shift_batch
    
    # Load data
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # Load x
    x_offset = cols * stride_x_dim
    x = tl.load(x_row_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)
    
    # Load scale and shift
    scale_offset = cols * stride_scale_dim
    shift_offset = cols * stride_shift_dim
    scale = tl.load(scale_row_ptr + scale_offset, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_row_ptr + shift_offset, mask=mask, other=0.0).to(tl.float32)
    
    # RMSNorm Computation
    # var = mean(x^2)
    x_sq = x * x
    # Sum over the block. Since we masked loaded with 0.0, the sum is correct for the first N elements.
    var = tl.sum(x_sq, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    x_norm = x * rstd
    
    # AdaLN Modulation
    # out = x_norm * (1 + scale) + shift
    out = x_norm * (1.0 + scale) + shift
    
    # Store output
    y_offset = cols * stride_y_dim
    tl.store(y_row_ptr + y_offset, out, mask=mask)

def rmsnorm_adaln(x, scale, shift, eps=1e-6):
    """
    Fused RMSNorm + AdaLN modulation.
    
    Args:
        x: Input tensor of shape (Batch, Seq, Dim)
        scale: Scale tensor of shape (Batch, 1, Dim)
        shift: Shift tensor of shape (Batch, 1, Dim)
        eps: Epsilon for RMSNorm
        
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
    # One kernel instance per row (Sequence element)
    grid = (B * L, )
    
    # Heuristic for block size
    BLOCK_SIZE = triton.next_power_of_2(D)
    
    # Launch kernel
    _rmsnorm_adaln_fwd_kernel[grid](
        x, scale, shift, y,
        x.stride(0), x.stride(1), x.stride(2),
        scale.stride(0), scale.stride(2), # stride_scale_dim corresponds to dim=2
        shift.stride(0), shift.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        N=D,
        seq_len=L,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y
