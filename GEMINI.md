# Project Context: RMSNormMMAttention-Triton

## Project Overview
**RMSNormMMAttention-Triton** is a Python project implementing Triton kernels for RMSNorm-based Multi Modal Attention (Self + Cross), as proposed in MMDiT. It leverages PyTorch and Triton for efficient attention mechanisms.

**Key Technologies:**
- **Python:** Core language (requires >=3.14 per pyproject.toml, though likely compatible with older versions if adjusted).
- **PyTorch:** Deep learning framework.
- **Triton:** GPU programming for efficient kernels.
- **Einops:** Flexible tensor operations.
- **Jaxtyping:** Type hints for tensors.

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
*   **Run Tests:**
    <!-- TODO: No explicit test commands or test directory found. -->
    *(Check if tests are added in `tests/` or inline.)*

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
    *   `nn_torch.py`: PyTorch implementation of RMSNorm-based Self and Cross Attention, and GPT2Block. Includes RoPE (Rotary Positional Embeddings) and linear attention operations.

### Type Hinting
The codebase makes heavy use of `jaxtyping` for annotating tensor shapes (e.g., `Float[Tensor, "batch sequence dim"]`). Adhere to this convention when modifying or adding code.
