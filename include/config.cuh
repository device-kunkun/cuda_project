//
// Created by Admin on 2025/11/23.
//

#ifndef CONFIG_CUH
#define CONFIG_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>  // 必须包含这个头文件

// 硬件配置 - 4070 Ti Super (SM89)
#define SM_ARCH 89
#define WARP_SIZE 32

// 矩阵分块配置 - 调整为更合理的值
#define BLOCK_SIZE 32  // 改为32，更好的GPU利用率
#define TILE_M 32      // 与BLOCK_SIZE保持一致
#define TILE_N 32      // 与BLOCK_SIZE保持一致
#define TILE_K 32      // 保持32

// Tensor Core 配置
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 数据类型 - 使用 CUDA 的标准 half 类型
typedef __half half_t;
typedef float float_t;

// 错误检查宏
#define CUDA_CHECK(cmd) { \
cudaError_t error = cmd; \
if (error != cudaSuccess) { \
printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
exit(1); \
} \
}


#endif