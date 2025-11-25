//
// Vectorized optimized GEMM: 64x64 output tile, TILE_K=32, block (16,16).
// Each thread computes a 4x4 patch using shared tiles and half2-friendly loads.
//

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include "../../include/config.cuh"
#include "kernels.h"

// Tile sizes for this kernel
#define VOPT_TILE_M 64
#define VOPT_TILE_N 64
#define VOPT_TILE_K 32

// Shared tiles: A (64x32), B (32x64)
__global__ void vectorized_opt_kernel(
    const half_t* __restrict__ A,
    const half_t* __restrict__ B,
    float_t* __restrict__ C,
    int M, int N, int K) {

    __shared__ half As[VOPT_TILE_M][VOPT_TILE_K];
    __shared__ half Bs[VOPT_TILE_K][VOPT_TILE_N];

    const int tx = threadIdx.x; // 0..15
    const int ty = threadIdx.y; // 0..15

    const int block_row = blockIdx.y * VOPT_TILE_M;
    const int block_col = blockIdx.x * VOPT_TILE_N;

    // Each thread computes a 4x4 patch
    const int row_base = block_row + ty * 4;
    const int col_base = block_col + tx * 4;

    float acc[4][4] = {0.0f};

    // Loop over K tiles
    for (int kb = 0; kb < K; kb += VOPT_TILE_K) {
        // Load A tile: cover 64x32
        for (int c = tx; c < VOPT_TILE_K; c += blockDim.x) {
            int gcol = kb + c;
            int grow0 = row_base + 0;
            int grow1 = row_base + 1;
            int grow2 = row_base + 2;
            int grow3 = row_base + 3;
            As[ty * 4 + 0][c] = (grow0 < M && gcol < K) ? A[grow0 * K + gcol] : __float2half(0.0f);
            As[ty * 4 + 1][c] = (grow1 < M && gcol < K) ? A[grow1 * K + gcol] : __float2half(0.0f);
            As[ty * 4 + 2][c] = (grow2 < M && gcol < K) ? A[grow2 * K + gcol] : __float2half(0.0f);
            As[ty * 4 + 3][c] = (grow3 < M && gcol < K) ? A[grow3 * K + gcol] : __float2half(0.0f);
        }

        // Load B tile: cover 32x64
        for (int r = ty; r < VOPT_TILE_K; r += blockDim.y) {
            int grow = kb + r;
            // load 4 columns per thread using two half2 loads when aligned
            int gcol0 = col_base + 0;
            int gcol2 = col_base + 2;
            half2 v0 = __float2half2_rn(0.0f);
            half2 v1 = __float2half2_rn(0.0f);
            if (grow < K && gcol0 + 1 < N) {
                v0 = *reinterpret_cast<const half2*>(B + grow * N + gcol0);
            } else {
                if (grow < K && gcol0 < N) v0.x = B[grow * N + gcol0];
                if (grow < K && gcol0 + 1 < N) v0.y = B[grow * N + gcol0 + 1];
            }
            if (grow < K && gcol2 + 1 < N) {
                v1 = *reinterpret_cast<const half2*>(B + grow * N + gcol2);
            } else {
                if (grow < K && gcol2 < N) v1.x = B[grow * N + gcol2];
                if (grow < K && gcol2 + 1 < N) v1.y = B[grow * N + gcol2 + 1];
            }
            Bs[r][tx * 4 + 0] = v0.x;
            Bs[r][tx * 4 + 1] = v0.y;
            Bs[r][tx * 4 + 2] = v1.x;
            Bs[r][tx * 4 + 3] = v1.y;
        }

        __syncthreads();

        // Compute on this tile
        #pragma unroll
        for (int k = 0; k < VOPT_TILE_K; ++k) {
            float b0 = __half2float(Bs[k][tx * 4 + 0]);
            float b1 = __half2float(Bs[k][tx * 4 + 1]);
            float b2 = __half2float(Bs[k][tx * 4 + 2]);
            float b3 = __half2float(Bs[k][tx * 4 + 3]);

            float a0 = __half2float(As[ty * 4 + 0][k]);
            float a1 = __half2float(As[ty * 4 + 1][k]);
            float a2 = __half2float(As[ty * 4 + 2][k]);
            float a3 = __half2float(As[ty * 4 + 3][k]);

            acc[0][0] += a0 * b0; acc[0][1] += a0 * b1; acc[0][2] += a0 * b2; acc[0][3] += a0 * b3;
            acc[1][0] += a1 * b0; acc[1][1] += a1 * b1; acc[1][2] += a1 * b2; acc[1][3] += a1 * b3;
            acc[2][0] += a2 * b0; acc[2][1] += a2 * b1; acc[2][2] += a2 * b2; acc[2][3] += a2 * b3;
            acc[3][0] += a3 * b0; acc[3][1] += a3 * b1; acc[3][2] += a3 * b2; acc[3][3] += a3 * b3;
        }

        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int r = row_base + i;
        if (r < M) {
            if (col_base + 0 < N) C[r * N + (col_base + 0)] = acc[i][0];
            if (col_base + 1 < N) C[r * N + (col_base + 1)] = acc[i][1];
            if (col_base + 2 < N) C[r * N + (col_base + 2)] = acc[i][2];
            if (col_base + 3 < N) C[r * N + (col_base + 3)] = acc[i][3];
        }
    }
}

void vectorized_opt_matmul(const half_t* A, const half_t* B, float_t* C,
                           int M, int N, int K,
                           const GemmConfig& cfg, cudaStream_t stream) {
    dim3 block(16, 16); // 256 threads
    dim3 grid((N + VOPT_TILE_N - 1) / VOPT_TILE_N,
              (M + VOPT_TILE_M - 1) / VOPT_TILE_M);
    vectorized_opt_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Vectorized opt kernel error: %s\n", cudaGetErrorString(err));
    }
}
