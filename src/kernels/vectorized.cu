//
// Vectorized GEMM: each thread computes two columns using half2 loads for B.
//

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include "../../include/config.cuh"
#include "kernels.h"

__global__ void vectorized_matmul_kernel(
    const half_t* A, const half_t* B, float_t* C,
    int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col0 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;  // two columns/thread
    int col1 = col0 + 1;

    if (row >= M || col0 >= N) return;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const bool col1_valid = (col1 < N);

    for (int k = 0; k < K; ++k) {
        float a = __half2float(A[row * K + k]);

        if (col1_valid && ((col0 & 1) == 0) && (reinterpret_cast<uintptr_t>(B + k * N + col0) % 4 == 0)) {
            // Aligned half2 load when possible
            half2 b2 = *reinterpret_cast<const half2*>(B + k * N + col0);
            sum0 += a * __half2float(b2.x);
            sum1 += a * __half2float(b2.y);
        } else {
            sum0 += a * __half2float(B[k * N + col0]);
            if (col1_valid) {
                sum1 += a * __half2float(B[k * N + col1]);
            }
        }
    }

    C[row * N + col0] = sum0;
    if (col1_valid) {
        C[row * N + col1] = sum1;
    }
}

void vectorized_matmul(const half_t* A, const half_t* B, float_t* C,
                       int M, int N, int K,
                       const GemmConfig& cfg, cudaStream_t stream) {
    const int bx = cfg.block_x > 0 ? cfg.block_x : 32;  // threads in x (each 2 cols)
    const int by = cfg.block_y > 0 ? cfg.block_y : 8;

    dim3 block(bx, by);
    dim3 grid((N + block.x * 2 - 1) / (block.x * 2),
              (M + block.y - 1) / block.y);

    vectorized_matmul_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Vectorized kernel error: %s\n", cudaGetErrorString(err));
    }
}
