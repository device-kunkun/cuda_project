//
// Created by Admin on 2025/11/23.
//

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include "../../include/config.cuh"
#include "kernels.h"

// 共享内存 GEMM，使用可配置 tile_k 和 block 尺寸
__global__ void shared_mem_matmul_kernel(
    const half_t* A, const half_t* B, float_t* C,
    int M, int N, int K,
    int tile_k, int block_x, int block_y) {

    extern __shared__ float_t shared[];
    float_t* As = shared;                      // block_y x tile_k
    float_t* Bs = As + block_y * tile_k;       // tile_k x block_x

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * block_y + ty;
    int col = bx * block_x + tx;

    float_t sum = 0.0f;

    for (int k_base = 0; k_base < K; k_base += tile_k) {
        // 加载 A
        int a_row = row;
        int a_col = k_base + tx;
        if (a_row < M && a_col < K && tx < tile_k) {
            As[ty * tile_k + tx] = __half2float(A[a_row * K + a_col]);
        } else if (tx < tile_k) {
            As[ty * tile_k + tx] = 0.0f;
        }

        // 加载 B (tile_k 行需要多个ty轮流加载)
        for (int kk = ty; kk < tile_k; kk += block_y) {
            int b_row = k_base + kk;
            int b_col = col;
            if (b_row < K && b_col < N) {
                Bs[kk * block_x + tx] = __half2float(B[b_row * N + b_col]);
            } else {
                Bs[kk * block_x + tx] = 0.0f;
            }
        }

        __syncthreads();

        const int k_limit = min(tile_k, K - k_base);
#pragma unroll
        for (int k = 0; k < k_limit; k++) {
            sum += As[ty * tile_k + k] * Bs[k * block_x + tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void shared_mem_matmul(const half_t* A, const half_t* B, float_t* C,
                       int M, int N, int K,
                       const GemmConfig& cfg, cudaStream_t stream) {
    const int bx = cfg.block_x > 0 ? cfg.block_x : BLOCK_SIZE;
    const int by = cfg.block_y > 0 ? cfg.block_y : BLOCK_SIZE;
    const int tile_k = cfg.tile_k > 0 ? cfg.tile_k : TILE_K;

    dim3 block_size(bx, by);
    dim3 grid((N + block_size.x - 1) / block_size.x,
              (M + block_size.y - 1) / block_size.y);

    size_t shared_mem_size = (by * tile_k + tile_k * bx) * sizeof(float_t);

    shared_mem_matmul_kernel<<<grid, block_size, shared_mem_size, stream>>>(
        A, B, C, M, N, K, tile_k, bx, by);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Shared memory kernel error: %s\n", cudaGetErrorString(err));
    }
}
