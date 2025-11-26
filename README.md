# 矩阵乘法 Playground（RTX 4070 Ti Super）

CUDA GEMM/Softmax/LayerNorm 练习项目，含多种核实现、统一基准/校验、绘图和可选 `ncu` Profiling。默认架构 SM89（RTX 4070 Ti Super），可按需调整。

## 功能概览
- **GEMM**：朴素、共享内存、向量化、向量化优化（64×64 tile）、Tensor Core/WMMA。
- **Batched GEMM**：共享内存版，适合小矩阵批处理。
- **Softmax / LayerNorm**：行归一化，warp/block 级归约。
- **参考实现**：CPU GEMM/Softmax/LayerNorm，cuBLAS GEMM 对比。
- **基准与日志**：多尺寸 sweep（256/512/1024/2048/4096），输出 GFLOPS，写入 `results.csv`。
- **绘图**：Python 脚本生成 GFLOPS 曲线 `gemm_perf.png`；`perf_runner` 汇总为 `perf_summary.json/csv`。
- **Profiling 钩子**：PowerShell/任务可调用 `ncu`。

## 环境依赖
- Windows + Visual Studio 2022（MSVC）。
- CUDA Toolkit 12.x（含 NVCC、cuBLAS、Nsight Compute `ncu`）。
- Python 3.x（可选，仅用于绘图/看板）。
- VSCode（可选），插件：CMake Tools、C/C++、CUDA（可选）。

## VSCode 任务（推荐）
`.vscode/tasks.json` 已写好（指向 VS 自带 cmake）：
1) `build-matmul`：配置并构建 Release。
2) `run-matmul`：构建后运行 `build/bin/matmul_test.exe`。
3) `run-matmul-and-plot`：构建、运行、再调用 `python scripts/plot.py`。
4) `sweep-and-plot`：构建后运行 `scripts/sweep.ps1`（基准+出图）。
5) `sweep-and-profile-both`：构建后运行 `scripts/sweep.ps1 -ProfileShared -ProfileTensorCore`（需 `ncu` 在 PATH）。
6) `perf-runner`：构建后运行 `python scripts/perf_runner.py`，生成 `scripts/perf_summary.json/csv`。
7) `perf-runner-ncu`：构建后运行 `python scripts/perf_runner.py --ncu shared_mem_matmul_kernel tensor_core_kernel`（需 `ncu`）。
8) `serve-dashboard`：启动简易 HTTP 服务（`python -m http.server 8000`），用于访问前端看板。
9) `perf-runner-and-serve`：构建+运行 `perf_runner`，再启动 HTTP 服务（组合式，一次完成）。

VSCode 菜单：`Terminal -> Run Task...` 选择对应任务即可。

## 命令行构建运行（PowerShell）
```pwsh
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -S . -B build -G "Visual Studio 17 2022" -A x64
"C:\Program Files\Microsoft Visual Studio\2022\Community\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Release --target matmul_test
.\build\bin\matmul_test.exe
```

## 基准、汇总、看板
- 运行 `matmul_test.exe`：遍历多尺寸/算法，打印时间/GFLOPS，写入 `results.csv`，并跑附加 Demo（CPU/cuBLAS 校验、Softmax、LayerNorm、Batched GEMM）。
- 绘图：`python scripts/plot.py` 生成 `gemm_perf.png`。
- 自动汇总：`python scripts/perf_runner.py`（或任务 `perf-runner`）生成 `scripts/perf_summary.json/csv`，可选 `--ncu kernel1 kernel2` 生成 `reports/ncu_*.txt`。
- 前端看板：运行 `perf-runner` 后，在仓库根启动 `python -m http.server 8000`（或任务 `serve-dashboard` / `perf-runner-and-serve` / `perf-runner-ncu-and-serve`），浏览器打开 `http://localhost:8000/frontend/index.html`，点击“刷新”加载最新数据（React + Chart.js）。务必在仓库根启动服务，以便访问 `scripts/` 和 `reports/`。
- 若 ncu 报告无法加载：确认已运行 `perf_runner.py --ncu ...` 生成 `reports/ncu_index.json`，并在仓库根启动 http 服务（而非 frontend 目录）；旧的索引含绝对路径时可重新生成。

## 核实现
- Naive / Shared-mem / Vectorized / Vectorized-opt（64×64 tile，4×4 输出/线程，half2）/ TensorCore (WMMA, 4 warp/块)。
- Batched GEMM：grid.z=batch 的共享内存实现。
- Softmax：行归一化，warp/block 归约。
- LayerNorm：行归一化，warp/block 归约均值/方差。

## 校验策略
- 基准 sweep：首个算法作为参考，其余对比（容差通常 1e-2）。
- 额外校验：CPU GEMM（256 尺寸）、cuBLAS GEMM（512 尺寸）对比；CPU Softmax/LayerNorm 对比。
- 容差：GEMM/LayerNorm/Softmax 视需求 1e-2~1e-4。

## 主要文件
- `main.cu`：驱动、基准、CSV 记录、Softmax/LayerNorm/Batched GEMM/cuBLAS 校验。
- `include/config.cuh`：数据类型、默认块/Tile、架构（SM89）。
- `src/kernels/`：各类 GEMM、TensorCore、Softmax、LayerNorm、Batched、调度器。
- `src/utils/validator.cu`：数据生成、CPU 参考、校验函数。
- `scripts/plot.py`：读取 `results.csv` 绘制 GFLOPS 曲线。
- `scripts/perf_runner.py`：自动运行/汇总基准，输出 perf_summary.json/csv，可选调用 ncu。
- `scripts/sweep.ps1`：一键基准+绘图，可选 `ncu`。
- `frontend/index.html`：React + Chart.js 性能看板。

## Profiling（Nsight Compute 示例）
```pwsh
ncu --set full --kernel-name "shared_mem_matmul_kernel" .\build\bin\matmul_test.exe
ncu --set full --kernel-name "tensor_core_kernel" .\build\bin\matmul_test.exe
```
如遇 `ERR_NVGPUCTRPERM`，在 NVIDIA 控制面板启用性能计数器，或用管理员权限/合适驱动设置。

## 常见问题
- `cmake` 不在 PATH：用 VSCode 任务（已写死 cmake 路径）或命令行全路径。
- Python 不在 PATH：安装后重开终端；无 Python 会跳过绘图，看板无法加载数据。
- `ncu` 未找到/权限错误：安装 Nsight Compute，加入 PATH，并在控制面板启用计数器访问。

## Git 工作流示例
```pwsh
git add .
git commit -m "Add GEMM variants, softmax/layernorm, benchmarks, plotting"
git push origin main
git checkout -b test_qa
git push origin test_qa
```
`test_qa` 可继续添加 CI/自动化测试或更多算子。
