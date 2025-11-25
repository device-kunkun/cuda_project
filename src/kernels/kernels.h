//
// Created by Admin on 2025/11/23.
//

#ifndef MATMUL_4070TI_KERNELS_H
#define MATMUL_4070TI_KERNELS_H

#include "../../include/config.cuh"
#include <cuda_runtime.h>

enum class GemmAlgo {
    Naive = 0,
    SharedMem = 1,
    Vectorized = 2,
    TensorCore = 3,
    VectorizedOpt = 4
};

struct GemmConfig {
    GemmAlgo algo{GemmAlgo::Naive};
    int block_x{16};   // threads in X
    int block_y{16};   // threads in Y
    int tile_k{16};    // K tile (used by shared/vec). Set 0 to use defaults.
};

void simple_matmul(const half_t* A, const half_t* B, float_t* C,
                   int M, int N, int K,
                   const GemmConfig& cfg, cudaStream_t stream = 0);

void shared_mem_matmul(const half_t* A, const half_t* B, float_t* C,
                       int M, int N, int K,
                       const GemmConfig& cfg, cudaStream_t stream = 0);

void vectorized_matmul(const half_t* A, const half_t* B, float_t* C,
                       int M, int N, int K,
                       const GemmConfig& cfg, cudaStream_t stream = 0);

void vectorized_opt_matmul(const half_t* A, const half_t* B, float_t* C,
                           int M, int N, int K,
                           const GemmConfig& cfg, cudaStream_t stream = 0);

void tensor_core_matmul(const half_t* A, const half_t* B, float_t* C,
                        int M, int N, int K,
                        const GemmConfig& cfg, cudaStream_t stream = 0);

// batched GEMM (shared-memory version)
void batched_shared_matmul(const half_t* A, const half_t* B, float_t* C,
                           int batch_count,
                           int M, int N, int K,
                           const GemmConfig& cfg, cudaStream_t stream = 0);

// softmax (row-wise, float input/output)
void softmax_rowwise(const float* X, float* Y, int rows, int cols, cudaStream_t stream = 0);

void launch_gemm(const half_t* A, const half_t* B, float_t* C,
                 int M, int N, int K,
                 const GemmConfig& cfg, cudaStream_t stream = 0);

// LayerNorm (row-wise, float input/output)
void layernorm_rowwise(const float* X, float* Y, int rows, int cols, float eps, cudaStream_t stream = 0);

#endif // MATMUL_4070TI_KERNELS_H
