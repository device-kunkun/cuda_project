//
// Unified GEMM launcher
//

#include <cuda_runtime.h>
#include <cstdio>
#include "../../include/config.cuh"
#include "kernels.h"

void launch_gemm(const half_t* A, const half_t* B, float_t* C,
                 int M, int N, int K,
                 const GemmConfig& cfg, cudaStream_t stream) {
    switch (cfg.algo) {
        case GemmAlgo::Naive:
            simple_matmul(A, B, C, M, N, K, cfg, stream);
            break;
        case GemmAlgo::SharedMem:
            shared_mem_matmul(A, B, C, M, N, K, cfg, stream);
            break;
        case GemmAlgo::Vectorized:
            vectorized_matmul(A, B, C, M, N, K, cfg, stream);
            break;
        case GemmAlgo::TensorCore:
            tensor_core_matmul(A, B, C, M, N, K, cfg, stream);
            break;
        default:
            printf("Unknown GEMM algorithm\n");
            break;
    }
}
