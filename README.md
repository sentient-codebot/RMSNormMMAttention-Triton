# RMSNormMMAttention-Triton

**RMSNormMMAttention-Triton** is a Python project implementing Triton kernels for RMSNorm-based Multi Modal Attention (Self + Cross), as proposed in MMDiT. It leverages PyTorch and Triton for efficient attention mechanisms.

## ✅ Current Status

We have implemented and optimized key components:

*   **Core PyTorch Modules**: `GPT2Block`, `RMSNormSelfAttention`, and `RMSNormSelfCrossAttention` in `src/mmattn_triton/nn_torch.py`.
*   **Fused Triton Kernels**:
    *   **LayerNorm + AdaLN**: Fused Layer Normalization with Adaptive Layer Normalization (`src/mmattn_triton/fused_layernorm.py`).
    *   **RMSNorm + AdaLN**: Fused RMS Normalization with Adaptive Layer Normalization (`src/mmattn_triton/fused_rmsnorm.py`).
*   **Integration**: Fused LayerNorm is integrated into `GPT2Block` with PyTorch fallback.
*   **Benchmarking**: Performance comparison scripts for fused kernels are available in `benchmarks/`.

## 🎯 Target Configuration

We are primarily optimizing for the following model configuration. These dimensions significantly influence our kernel design choices (e.g., block sizes, memory access patterns).

*   **Hidden State Dimension (`dim_base`)**: 128
*   **Sequence Length**: 186
*   **Feedforward Dimension (`dim_feedforward`)**: 512
*   **Attention Heads (`num_attn_head`)**: 8
*   **Transformer Layers**: 12
*   **Conditioning**: True
*   **Use Cross Attention**: True

## 🚀 Goals (To Do)

- [ ] Implement robust PyTorch modules for Multi-Modal Attention, including support for both Self-Attention and Cross-Attention within a unified block.
- [x] Leverage Triton to fuse bandwidth-bound operations, specifically Normalization layers (LayerNorm/RMSNorm) with Adaptive Layer Normalization (AdaLN) modulation.
- [x] Ensure seamless integration, providing automatic fallbacks to standard PyTorch implementations when Triton is unavailable or on non-CUDA devices.
- [x] Rigorously benchmark the kernels to demonstrate speedups over vanilla PyTorch implementations.

## 🛠️ Installation & Usage

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run benchmarks
uv run python benchmarks/benchmark_layernorm_fusion.py
```