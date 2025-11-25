//
// Tensor Core (WMMA) GEMM - multi-warp per block, no shared memory staging.
// Assumes M, N, K are multiples of WMMA tile (16). Current benchmarks satisfy this.
//

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include "../../include/config.cuh"
#include "kernels.h"

#define div_ceil(a,b) (((a)+(b)-1)/(b))

// block has 4 warps -> 128 threads, covers a 32x32 output tile (2x2 warps).
__global__ void tensor_core_kernel(
    const half_t* __restrict__ A,
    const half_t* __restrict__ B,
    float_t* __restrict__ C,
    int M, int N, int K) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda::wmma;

    const int warp_id = threadIdx.x >> 5;      // 0..3
    const int lane_id = threadIdx.x & 31;

    const int warp_m = warp_id >> 1;           // 0..1
    const int warp_n = warp_id & 1;            // 0..1

    const int block_tile_m = WMMA_M * 2;       // 32
    const int block_tile_n = WMMA_N * 2;       // 32

    const int tile_row = blockIdx.y * block_tile_m + warp_m * WMMA_M;
    const int tile_col = blockIdx.x * block_tile_n + warp_n * WMMA_N;

    if (tile_row >= M || tile_col >= N) return;

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    fill_fragment(c_frag, 0.0f);

    const int k_tiles = div_ceil(K, WMMA_K);

    for (int kt = 0; kt < k_tiles; ++kt) {
        const half* a_ptr = A + tile_row * K + kt * WMMA_K;
        const half* b_ptr = B + (kt * WMMA_K) * N + tile_col;

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;

        load_matrix_sync(a_frag, a_ptr, K);
        load_matrix_sync(b_frag, b_ptr, N);

        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // store
    store_matrix_sync(C + tile_row * N + tile_col, c_frag, N, mem_row_major);
#else
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        C[row * N + col] = sum;
    }
#endif
}

void tensor_core_matmul(const half_t* A, const half_t* B, float_t* C,
                        int M, int N, int K,
                        const GemmConfig& cfg, cudaStream_t stream) {
    dim3 block(128); // 4 warps
    dim3 grid(div_ceil(N, WMMA_N * 2), div_ceil(M, WMMA_M * 2));

    tensor_core_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Tensor Core kernel error: %s\n", cudaGetErrorString(err));
    }
}
