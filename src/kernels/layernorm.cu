//
// Row-wise LayerNorm kernel (float input/output).
//

#include <cuda_runtime.h>
#include <cstdio>
#include "../../include/config.cuh"
#include "kernels.h"

// One block per row
__global__ void layernorm_row_kernel(const float* X, float* Y, int cols, float eps) {
    int row = blockIdx.x;
    const float* x = X + row * cols;
    float* y = Y + row * cols;

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_count = (blockDim.x + 31) >> 5;

    // mean
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        local_sum += x[c];
    }
    // warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    __shared__ float partial[32];
    if (lane == 0) partial[warp_id] = local_sum;
    __syncthreads();
    float mean = 0.0f;
    if (threadIdx.x == 0) {
        for (int i = 0; i < warp_count; ++i) mean += partial[i];
        partial[0] = mean / cols;
    }
    __syncthreads();
    mean = partial[0];

    // variance
    float local_var = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float d = x[c] - mean;
        local_var += d * d;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_var += __shfl_down_sync(0xffffffff, local_var, offset);
    if (lane == 0) partial[warp_id] = local_var;
    __syncthreads();
    float var = 0.0f;
    if (threadIdx.x == 0) {
        for (int i = 0; i < warp_count; ++i) var += partial[i];
        partial[0] = var / cols;
    }
    __syncthreads();
    var = partial[0];
    float inv_std = rsqrtf(var + eps);

    // normalize
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        y[c] = (x[c] - mean) * inv_std;
    }
}

void layernorm_rowwise(const float* X, float* Y, int rows, int cols, float eps, cudaStream_t stream) {
    int threads = (cols >= 512) ? 512 : (cols >= 256 ? 256 : 128);
    layernorm_row_kernel<<<rows, threads, 0, stream>>>(X, Y, cols, eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("LayerNorm kernel error: %s\n", cudaGetErrorString(err));
    }
}
