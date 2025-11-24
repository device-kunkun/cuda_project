//
// Created by Admin on 2025/11/23.
//

#include <cuda_runtime.h>
#include "../../include/config.cuh"
#include "kernels.h"

// 最简单的矩阵乘法 - 用于验证正确性
__global__ void simple_matmul_kernel(
    const half_t* A, const half_t* B, float_t* C,
    int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float_t sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // 将half类型转换为float类型进行计算
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = sum;
    }
}

void simple_matmul(const half_t* A, const half_t* B, float_t* C,
                   int M, int N, int K,
                   const GemmConfig& cfg, cudaStream_t stream) {
    const int bx = cfg.block_x > 0 ? cfg.block_x : BLOCK_SIZE;
    const int by = cfg.block_y > 0 ? cfg.block_y : BLOCK_SIZE;
    dim3 block(bx, by);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    simple_matmul_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
