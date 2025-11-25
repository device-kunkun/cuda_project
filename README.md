# 矩阵乘法 Playground（RTX 4070 Ti Super）

CUDA GEMM/Softmax/LayerNorm 练习项目，包含多种核实现、统一基准与校验、绘图脚本以及可选的 Nsight Compute Profiling。默认目标架构 SM89（RTX 4070 Ti Super），也可按需调整。

## 功能概览
- **GEMM**：朴素、共享内存、向量化、向量化优化（64×64 tile）、Tensor Core/WMMA。
- **Batched GEMM**：共享内存版本，适合固定小矩阵批处理。
- **Softmax**：行归一化，warp/block 级归约。
- **LayerNorm**：行归一化（均值/方差归约）。
- **参考实现**：CPU GEMM/Softmax/LayerNorm，cuBLAS GEMM 参考，对比 GPU 输出。
- **基准与日志**：多尺寸 sweep（256/512/1024/2048/4096），输出 GFLOPS，追加到 `results.csv`。
- **绘图**：Python 脚本生成 GFLOPS 曲线 `gemm_perf.png`。
- **Profiling 钩子**：PowerShell 脚本可选调用 `ncu` 分析性能。

## 环境依赖
- Windows + Visual Studio 2022（MSVC）。
- CUDA Toolkit 12.x（含 NVCC、cuBLAS、Nsight Compute `ncu`）。
- Python 3.x（可选，仅用于绘图）。
- VSCode（可选），插件：CMake Tools、C/C++、CUDA（可选）。

## VSCode 任务（推荐）
`.vscode/tasks.json` 已配置（指向 VS 自带 cmake）：
1) `build-matmul`：配置并构建 Release。
2) `run-matmul`：构建后运行 `build/bin/matmul_test.exe`。
3) `run-matmul-and-plot`：构建、运行、再调用 `python scripts/plot.py`。
4) `sweep-and-plot`：构建后运行 `scripts/sweep.ps1`（基准+出图）。
5) `sweep-and-profile-both`：构建后运行 `scripts/sweep.ps1 -ProfileShared -ProfileTensorCore`（需 `ncu` 在 PATH）。

VSCode 菜单：`Terminal -> Run Task...` 选择对应任务即可。

## 命令行构建运行（PowerShell）
```pwsh
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -S . -B build -G "Visual Studio 17 2022" -A x64
"C:\Program Files\Microsoft Visual Studio\2022\Community\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Release --target matmul_test
.\build\bin\matmul_test.exe
```

## 基准与绘图
- 直接运行 `matmul_test.exe`：遍历多尺寸/算法，打印时间+GFLOPS，写入 `results.csv`，并运行附加 Demo（CPU/ cuBLAS 校验、Softmax、LayerNorm、Batched GEMM）。
- 绘图：`python scripts/plot.py` 生成 `gemm_perf.png`。
- 一键：`.\scripts\sweep.ps1`（有 Python 则自动出图）。
- Profiling：`.\scripts\sweep.ps1 -ProfileShared -ProfileTensorCore`（需 Nsight Compute）。

## 核实现
- **Naive**：基线 GEMM。
- **Shared-mem**：动态共享内存加载 A/B，tile/block 可配置。
- **Vectorized**：half2 向量化，2 列/线程。
- **Vectorized-opt**：64×64 tile，4×4 输出/线程，half2 加载。
- **TensorCore (WMMA)**：4 warp/块，32×32 tile，直接从全局加载 WMMA fragment。
- **Batched GEMM**：grid.z = batch 的共享内存实现。
- **Softmax**：行归一化，warp/block 归约。
- **LayerNorm**：行归一化，warp/block 归约均值/方差。

## 校验策略
- 基准 sweep：首个算法作为参考，其余与之对比（容差通常 1e-2）。
- 额外校验：CPU GEMM（256 尺寸）、cuBLAS GEMM（512 尺寸）对比 GPU 输出；CPU Softmax/LayerNorm 对比 GPU 输出。
- 容差：GEMM/LayerNorm/Softmax 视需求 1e-2~1e-4。

## 主要文件
- `main.cu`：驱动、基准、CSV 记录、Softmax/LayerNorm/Batched GEMM/ cuBLAS 校验。
- `include/config.cuh`：数据类型、默认块/Tile、架构（SM89）。
- `src/kernels/`：各类 GEMM、TensorCore、Softmax、LayerNorm、Batched、调度器。
- `src/utils/validator.cu`：数据生成、CPU 参考、校验函数。
- `scripts/plot.py`：读取 `results.csv` 绘制 GFLOPS 曲线。
- `scripts/sweep.ps1`：一键基准+绘图，可选 `ncu` Profiling。

## Profiling（Nsight Compute 示例）
```pwsh
ncu --set full --kernel-name "shared_mem_matmul_kernel" .\build\bin\matmul_test.exe
ncu --set full --kernel-name "tensor_core_kernel" .\build\bin\matmul_test.exe
```
若报 `ERR_NVGPUCTRPERM`，需在 NVIDIA 控制面板启用性能计数器，或用管理员权限/合适的驱动设置。

## 常见问题
- `cmake` 不在 PATH：使用 VSCode 任务（已写死 cmake 全路径），或命令行用完整路径。
- Python 不在 PATH：安装后重开终端；无 Python 仅会跳过绘图，`results.csv` 仍生成。
- `ncu` 未找到或权限错误：安装 Nsight Compute，加入 PATH，并在控制面板启用计数器访问。

## Git 工作流示例
```pwsh
git add .
git commit -m "Add GEMM variants, softmax/layernorm, benchmarks, plotting"
git push origin main
git checkout -b test_qa
git push origin test_qa
```
`test_qa` 可用于添加 CI/自动化测试。

## 扩展建议
- 新增核：在 `kernels.h` 声明、`src/kernels/` 实现，若是 GEMM 变体可接入 `dispatch`。
- 调整 sweep：修改 `main.cu` 中的 `configs` 与 `cases`。
- 添加 CI：在 `scripts/` 增加测试脚本，或配置 GitHub Actions 进行构建/运行。
