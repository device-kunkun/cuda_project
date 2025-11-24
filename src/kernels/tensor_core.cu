//
// Tensor Core (WMMA) GEMM
//

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include "../../include/config.cuh"
#include "kernels.h"

#define div_ceil(a,b) (((a)+(b)-1)/(b))

__global__ void tensor_core_kernel(
    const half_t* A, const half_t* B, float_t* C,
    int M, int N, int K) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda::wmma;

    const int warp_row = blockIdx.y * WMMA_M;
    const int warp_col = blockIdx.x * WMMA_N;

    if (warp_row >= M || warp_col >= N) return;

    __shared__ half As[WMMA_M * WMMA_K];
    __shared__ half Bs[WMMA_K * WMMA_N];

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    fill_fragment(c_frag, 0.0f);

    const int k_tiles = div_ceil(K, WMMA_K);

    for (int tile = 0; tile < k_tiles; ++tile) {
        // cooperative load A tile
        int linear_tid = threadIdx.x;
        for (int idx = linear_tid; idx < WMMA_M * WMMA_K; idx += blockDim.x) {
            int i = idx / WMMA_K;
            int j = idx - i * WMMA_K;
            int r = warp_row + i;
            int c = tile * WMMA_K + j;
            if (r < M && c < K) {
                As[idx] = A[r * K + c];
            } else {
                As[idx] = __float2half(0.0f);
            }
        }

        // cooperative load B tile
        for (int idx = linear_tid; idx < WMMA_K * WMMA_N; idx += blockDim.x) {
            int i = idx / WMMA_N;
            int j = idx - i * WMMA_N;
            int r = tile * WMMA_K + i;
            int c = warp_col + j;
            if (r < K && c < N) {
                Bs[idx] = B[r * N + c];
            } else {
                Bs[idx] = __float2half(0.0f);
            }
        }

        __syncthreads();

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;

        load_matrix_sync(a_frag, As, WMMA_K);
        load_matrix_sync(b_frag, Bs, WMMA_N);

        mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // store
    if (warp_row + WMMA_M <= M && warp_col + WMMA_N <= N) {
        store_matrix_sync(C + warp_row * N + warp_col, c_frag, N, mem_row_major);
    } else {
        float tmp[WMMA_M * WMMA_N];
        store_matrix_sync(tmp, c_frag, WMMA_N, mem_row_major);
        for (int i = 0; i < WMMA_M; ++i) {
            for (int j = 0; j < WMMA_N; ++j) {
                int r = warp_row + i;
                int c = warp_col + j;
                if (r < M && c < N) {
                    C[r * N + c] = tmp[i * WMMA_N + j];
                }
            }
        }
    }
#else
    // Fallback: simple multiply on older GPUs
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
    dim3 block(32);  // one warp
    dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));

    tensor_core_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Tensor Core kernel error: %s\n", cudaGetErrorString(err));
    }
}
