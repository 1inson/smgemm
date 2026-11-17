# SM120 NVFP4 GEMM：学习项目

目标是在 RTX 5090 上研究量化和矩阵乘法。当前先完成一个可在普通电脑运行的 **CPU 数值参考**，以及官方 SM120 GEMM 的复现脚本；二者目前是独立的，还没有接成 GPU 线性层。

## 当前完成到哪里

| 内容 | 状态 |
| --- | --- |
| 1D block NVFP4 的量化、打包与反量化 | 已实现 CPU 参考版，可在本机运行 |
| CUTLASS 79a 官方 SM120 GEMM | 已准备 Linux 复现脚本；需要租到 5090 后运行 |
| 将量化结果连接到 CUTLASS | 尚未实现，需要处理矩阵布局和 scale swizzle |
| CUDA 量化 kernel 与性能优化 | 尚未实现，CPU 版用于学习和作为正确性参考 |

本机的 RTX 4060 不支持运行 SM120 专用 GEMM；不能用 CPU 版本推断 5090 的性能。

## 1. 在本地运行量化演示

需要 Python 3.10 或更新，不需要安装 PyTorch、CUDA 或第三方依赖。在 PowerShell 进入项目目录：

```powershell
cd D:\GEMM
python examples/demo_quantization.py
python -m unittest discover -s tests -v
```

演示会打印每个数值的原始值、FP4 编码、反量化值和误差，以及每个 block 的 scale。代码从 [examples/demo_quantization.py](examples/demo_quantization.py) 开始读，再看 [src/nvfp4_reference.py](src/nvfp4_reference.py) 中的 `quantize`、`dequantize`、编码和打包函数。

### 量化到底做了什么？

本参考实现选择一种**一维、确定性舍入**的 NVFP4 配方。把输入看作一维序列，每 16 个连续值共用一个 scale；末尾不足 16 个的 block 也支持，方便观察边界。它使用：

- 4 bit E2M1 表示数据：正值可表示 `0, 0.5, 1, 1.5, 2, 3, 4, 6`，另有符号位。
- 8 bit E4M3FN 表示每个 block 的 scale；每两个 FP4 编码打包进 1 byte，先低 4 bit、后高 4 bit。
- 一个 FP32 global scale。设全局最大绝对值为 `amax`，取 `global = FP32(amax / (448 * 6))`；全零输入使用 1。
- 每块的 scale 近似为 `block_amax / (6 * global)`，再舍入为 E4M3FN；对非零块，如果 scale 小到舍入成零，就用最小正 E4M3 次正规数，避免除零。全零块使用 scale 1。
- 量化值按最接近可表示值舍入，恰好在中点时选偶数编码；超出范围饱和。非有限输入会报错。

反量化关系是 `x_hat = E2M1值 × block_scale × global_scale`。结果一般不会和原始输入完全一致：FP4 可表示的数值很少，误差正是需要研究的内容。该版本使用 Python 双精度完成中间运算、明确将 global scale 舍入到 FP32；它是教学参考配方，不承诺和某个 GPU kernel 的中间运算逐位一致。

注意参考实现把输入视作一个一维序列，block 不会自动在矩阵行边界重置。后续矩阵量化需要明确 K 方向、每行 padding 和布局，不能直接把这里的顺序 scale 数据当作 CUTLASS 的 SFA/SFB 布局。

## 2. 租到 5090 后运行官方 GEMM

在 Linux 服务器上准备支持 SM120 的 CUDA Toolkit（12.8+，包括 `nvcc`）、驱动、Git、CMake、C++ 编译器，确认网络能够访问 GitHub。把本项目同步到服务器后，在项目根目录运行：

```bash
bash scripts/run_cutlass_79a.sh
```

脚本会检查工具，下载固定版本的 NVIDIA CUTLASS 到 `third_party/cutlass`，只编译 79a 示例并运行。它会检查输出包含 `Disposition: Passed` 和 `Avg runtime:`；如果已有同路径的不同版本 checkout，脚本会停止，不修改它。可以用 `BUILD_JOBS=1 bash scripts/run_cutlass_79a.sh` 降低编译并行度。脚本尚未在本项目的云 GPU 上实测。

官方示例使用独立生成的量化格式数据和 scale，验证的是 **NVFP4 GEMM**；它不执行本项目的 BF16→NVFP4 量化，也不说明端到端量化误差或速度。它报告的时间用于初步复现，不能直接视作端到端推理耗时。

## 3. 后续要做的事

1. 在 5090 上记录工具链和官方示例的验证输出。
2. 将一维参考扩展为按矩阵 K 方向分块；处理 FP4 打包、矩阵转置和 scale swizzle，对照 CUTLASS 的布局约束。
3. 用相同量化输入对比：CUTLASS GEMM 输出 vs 反量化后的高精度矩阵乘法，再与原始输入的高精度结果对比，分别定位实现误差和量化误差。
4. 实现 GPU 量化 kernel，测量量化、格式转换、GEMM 和完整线性层耗时，再针对实测瓶颈优化。

## 参考资料

- [NVIDIA Transformer Engine：NVFP4 格式与缩放](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)
- [CUTLASS：SM120 NVFP4 到 BF16 GEMM 示例](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)
- [CUTLASS：SM120 GEMM 示例构建配置](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/CMakeLists.txt)
