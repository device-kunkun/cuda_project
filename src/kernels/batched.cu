//
// Batched GEMM: simple shared-memory version + wrapper using shared kernel per batch.
//

#include <cuda_runtime.h>
#include <cstdio>
#include "../../include/config.cuh"
#include "kernels.h"

// Shared-memory matmul per batch (reuses BLOCK_SIZE, TILE_K from config)
__global__ void batched_shared_kernel(
    const half_t* A, const half_t* B, float_t* C,
    int M, int N, int K, int strideA, int strideB, int strideC) {

    int batch = blockIdx.z;
    const half_t* Ab = A + batch * strideA;
    const half_t* Bb = B + batch * strideB;
    float_t* Cb = C + batch * strideC;

    __shared__ float As[BLOCK_SIZE][TILE_K];
    __shared__ float Bs[TILE_K][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;
    for (int kb = 0; kb < K; kb += TILE_K) {
        if (row < M && kb + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = __half2float(Ab[row * K + kb + threadIdx.x]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (kb + threadIdx.y < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = __half2float(Bb[(kb + threadIdx.y) * N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        Cb[row * N + col] = sum;
    }
}

void batched_shared_matmul(const half_t* A, const half_t* B, float_t* C,
                           int batch_count,
                           int M, int N, int K,
                           const GemmConfig& cfg, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE,
              batch_count);
    int strideA = M * K;
    int strideB = K * N;
    int strideC = M * N;
    batched_shared_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, strideA, strideB, strideC);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Batched shared kernel error: %s\n", cudaGetErrorString(err));
    }
}
