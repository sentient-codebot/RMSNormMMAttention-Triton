# Project Context: RMSNormMMAttention-Triton

## Project Overview
**RMSNormMMAttention-Triton** is a Python project implementing Triton kernels for RMSNorm-based Multi Modal Attention (Self + Cross), as proposed in MMDiT. It leverages PyTorch and Triton for efficient attention mechanisms.

**Key Technologies:**
- **Python:** Core language (requires >=3.14 per pyproject.toml, though likely compatible with older versions if adjusted).
- **PyTorch:** Deep learning framework.
- **Triton:** GPU programming for efficient kernels.
- **Einops:** Flexible tensor operations.
- **Jaxtyping:** Type hints for tensors.

### Target Optimization Configuration
The project specifically targets optimization for the following hyperparameter configuration. Kernel optimizations (e.g., block sizes) should be tuned with these dimensions in mind:
*   **Hidden State Dimension (`dim_base`)**: 128
*   **Sequence Length**: 186
*   **Feedforward Dimension (`dim_feedforward`)**: 512
*   **Attention Heads (`num_attn_head`)**: 8
*   **Transformer Layers**: 12
*   **Conditioning**: True
*   **Use Cross Attention**: True

## Building and Running

### Dependency Management
This project uses `uv` for dependency management and building.

*   **Install Dependencies:**
    ```bash
    uv pip install .
    # or
    uv sync
    ```

### Building
The project uses `uv_build` as the build backend.

*   **Build Project:**
    ```bash
    uv build
    ```

### Testing
This project uses `pytest` for unit and integration testing.

*   **Run Tests:**
    ```bash
    uv run pytest
    # To run specific tests, you can pass arguments to pytest:
    # uv run pytest tests/your_test_file.py
    ```

### Running Scripts
For running any Python script within the project's virtual environment, use `uv run`. This ensures that the script executes with all necessary dependencies properly loaded.

*   **Example:**
    ```bash
    uv run python your_script_name.py
    ```

## Development Conventions

### Code Style & Linting
The project uses `ruff` for linting and formatting.

*   **Lint:**
    ```bash
    ruff check .
    ```
*   **Format:**
    ```bash
    ruff format .
    ```

### Code Structure
*   `src/mmattn_triton/`: Main source code directory.
    *   `nn_torch.py`: PyTorch implementation of RMSNorm-based Self and Cross Attention, and GPT2Block. Includes RoPE (Rotary Positional Embeddings).

### Type Hinting
The codebase makes heavy use of `jaxtyping` for annotating tensor shapes (e.g., `Float[Tensor, "batch sequence dim"]`). Adhere to this convention when modifying or adding code.

### Optimization Guidelines

**1. Benchmarking:**
Prefer using Triton's built-in benchmarking tools (e.g., `triton.testing.do_bench`) for accurate performance measurements of kernels.

**2. Backward Compatibility:**
When introducing Triton kernels, ensure backward compatibility by keeping the original PyTorch implementation as a fallback. This supports environments without Triton (e.g., CPU, MPS).

**Pattern:**
```python
try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

class ExampleFusedAdaLN(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Keep the vanilla layer for fallback compatibility
        self.ln = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward_triton(self, x, scale, shift):
        # Call your fused kernel
        from .fused_module import fused_kernel
        return fused_kernel(x, scale, shift, self.eps)

    def forward_pytorch(self, x, scale, shift):
        # Exact logic from vanilla code
        x = self.ln(x)
        return x * (1.0 + scale) + shift

    def forward(self, x, scale, shift):
        # 1. Check if Triton is available
        # 2. Check if input is on a CUDA device
        if HAS_TRITON and x.is_cuda:
            return self.forward_triton(x, scale, shift)
        else:
            return self.forward_pytorch(x, scale, shift)
```

## General Practices

### Task Breakdown
Avoid attempting large, monolithic tasks in a single step. Instead, break down complex objectives into smaller, verifiable subtasks.

**Example: Implementing a new Triton Kernel**
1.  **Benchmark Baseline:** Create a script to reproduce and measure the performance of the existing PyTorch implementation.
2.  **Implement Kernel:** Write the raw Triton kernel logic.
3.  **Unit Test & Verify:** Create a targeted test to verify the kernel's output matches the PyTorch baseline (correctness is paramount).
4.  **Benchmark Kernel:** Measure the performance of the new kernel against the baseline.
5.  **Integrate:** Add the kernel to the main codebase, implementing fallback mechanisms for backward compatibility.
6.  **System Test:** Verify the integrated operation works correctly within the full model context.
7.  **AOT Compilation:** Configure Triton's Ahead-of-Time (AOT) compilation to export the kernel as a standalone binary/library. This enables usage in environments without the Triton compiler installed at runtime and reduces warmup latency.
