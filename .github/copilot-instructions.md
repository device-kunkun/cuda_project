# Copilot / AI Agent Instructions for matmul_4070ti

Short, actionable guidance so an AI coding agent can be immediately productive.

1. Project purpose
- This repository implements multiple matrix-multiply kernels (simple, shared-memory, Tensor Core) optimized for an NVIDIA 4070 Ti (SM89). Key files: `main.cu`, `src/kernels/*`, `include/config.cuh`.

2. Build & run (Windows / PowerShell)
- Recommended (Visual Studio): open `cmake-build-debug-visual-studio/matmul_4070ti.sln` and build `matmul_test` in `Release`.
- CLI CMake (from repo root):
  - `mkdir build; cd build; cmake -S .. -B . -G "Visual Studio 17 2022" -A x64`
  - `cmake --build . --config Release`
  - Run: `.
    ${PWD}\bin\matmul_test.exe` (or open the `bin` output folder).
- Notes: CMake targets `CUDA_ARCHITECTURES=89` and links `CUDA::cudart` and `CUDA::cublas` (see `CMakeLists.txt`).

3. Tests & benchmarks
- Python PyTorch benchmark: `python benchmarks/benchmark.py` (requires `torch` with CUDA). This is a performance reference, not a unit test.
- Test patterns helper: `scripts/test_patterns.py` creates verification matrices (identity, sequence, checkerboard).

4. Important code conventions & patterns
- Types & macros: `include/config.cuh` defines `half_t`, `float_t`, `CUDA_CHECK`, `WMMA_M/N/K`, and tiling macros (`BLOCK_SIZE`, `TILE_M/N/K`). Always use these typedefs/macros when adding kernels.
- Kernel wrappers: public APIs are declared in `src/kernels/kernels.h` and implemented in `src/kernels/*.cu`. Each wrapper has signature like:
  `void tensor_core_matmul(const half_t* A, const half_t* B, float_t* C, int M, int N, int K, cudaStream_t stream = 0);`
  Prefer updating `kernels.h` when adding new kernels.
- Tensor Core usage: `src/kernels/tensor_core.cu` uses `nvcuda::wmma` (WMMA fragments, `load_matrix_sync`, `mma_sync`, `store_matrix_sync`). Memory layout expectations:
  - `A` stride is `K` (row-major). `B` loading uses `N` stride (row-major). Be careful with striding when writing tests.
- Error handling: use `CUDA_CHECK(...)` macro defined in `include/config.cuh` for runtime CUDA calls.
- Streams & timing: wrappers accept `cudaStream_t`; the project uses a `GpuTimer` in `src/utils/timer.cu` for timing. Use streams for non-blocking tests.

5. Build-time flags and debug workflow
- Debug builds: CMake sets CUDA debug flags in `CMakeLists.txt` (`--device-debug -O0 -g -G`) for MSVC. To debug kernels, build `Debug` and run under Visual Studio or Nsight.
- Release builds: enable `Release` for optimized tensor-core runs (`-O3 --use_fast_math`).

6. Adding new kernels / experiments (practical steps)
- Add implementation file to `src/kernels/` (e.g. `my_kernel.cu`).
- Declare the wrapper in `src/kernels/kernels.h` and implement it with an optional `cudaStream_t` argument.
- Add the `.cu` to the `add_executable(matmul_test ...)` list in `CMakeLists.txt`.
- Rebuild and call from `main.cu` (or write a small driver) following patterns in `test_matmul_implementations()`.

7. Performance & correctness tips specific to this repo
- Use matrix sizes divisible by WMMA tiles (16) when testing tensor_core paths (e.g. 256/512/1024) to exercise full Tensor Core tiling.
- Validate against the simple reference: `simple_matmul` produces the reference result; other kernels call `validate_results(...)` in `src/utils/validator.cu`.
- For profiling, prefer Release build + NVIDIA Nsight or `nvprof`/`nsys`.

8. Items NOT present / assumptions
- There is no repository-level README describing high-level goals; use this file for agent guidance.
- No CI/test harness is present; benchmarks are manual and Python-based.

If anything in these instructions is unclear or you want additional examples (e.g., a new kernel template, example `cmake` commands for non-MSVC generators, or a README), tell me which part to expand. 
