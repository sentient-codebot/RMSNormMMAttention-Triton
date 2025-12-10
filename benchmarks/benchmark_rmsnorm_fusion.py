
import torch
import torch.nn as nn
import triton
import triton.testing

from mmattn_triton.fused_rmsnorm import rmsnorm_adaln


# Reference PyTorch Implementation
class ReferenceRMSNormAdaLN(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.rms_norm = nn.RMSNorm(dim, eps=eps)

    def forward(self, x, scale, shift):
        # x: (batch, seq_len, dim)
        # scale: (batch, 1, dim) or (batch, dim) - broadcasted over seq_len
        # shift: (batch, 1, dim) or (batch, dim)
        
        # Standard RMSNorm
        x_norm = self.rms_norm(x)
        
        # AdaLN Modulation
        # x_mod = x_norm * (1 + scale) + shift
        return x_norm * (1.0 + scale) + shift

def benchmark_pytorch(dim=1024, seq_len=1024, batch_size=4, num_iters=100):
    device = torch.device("cuda")
    
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
    # scale and shift usually come from a conditioning vector, shape (batch, 1, dim)
    scale = torch.randn(batch_size, 1, dim, device=device, dtype=torch.float32)
    shift = torch.randn(batch_size, 1, dim, device=device, dtype=torch.float32)
    
    model = ReferenceRMSNormAdaLN(dim).to(device)
    
    # Verify Correctness
    print("Verifying correctness...")
    y_ref = model(x, scale, shift)
    y_triton = rmsnorm_adaln(x, scale, shift)
    
    if torch.allclose(y_ref, y_triton, atol=1e-5):
        print("Correctness check passed!")
    else:
        print("Correctness check FAILED!")
        diff = (y_ref - y_triton).abs().max().item()
        print(f"Max difference: {diff}")
    
    # Benchmark PyTorch
    pytorch_ms = triton.testing.do_bench(lambda: model(x, scale, shift))
    print(f"PyTorch (Batch={batch_size}, Seq={seq_len}, Dim={dim}): {pytorch_ms:.3f} ms")
    
    # Benchmark Triton


    triton_ms = triton.testing.do_bench(lambda: rmsnorm_adaln(x, scale, shift))
    print(f"Triton  (Batch={batch_size}, Seq={seq_len}, Dim={dim}): {triton_ms:.3f} ms")
    
    return pytorch_ms, triton_ms

if __name__ == "__main__":
    print("Benchmarking Reference PyTorch Implementation vs Triton...")
    benchmark_pytorch()
