//
// Row-wise softmax kernel (float input/output) with block-level reduction.
//

#include <cuda_runtime.h>
#include <cstdio>
#include "../../include/config.cuh"
#include "kernels.h"

// Warp reduce helper
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Each block handles one row
__global__ void softmax_row_kernel(const float* X, float* Y, int cols) {
    int row = blockIdx.x;
    const float* x = X + row * cols;
    float* y = Y + row * cols;

    // step 1: block max
    float local_max = -1e20f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        local_max = fmaxf(local_max, x[c]);
    }
    float warp_max = warp_reduce_max(local_max);
    __shared__ float smax[32]; // enough for up to 1024 threads
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) smax[warp_id] = warp_max;
    __syncthreads();
    float block_max = (lane < (blockDim.x + 31) / 32) ? smax[lane] : -1e20f;
    block_max = warp_reduce_max(block_max);
    block_max = __shfl_sync(0xffffffff, block_max, 0);

    // step 2: exp and block sum
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = expf(x[c] - block_max);
        y[c] = v;
        local_sum += v;
    }
    float warp_sum = warp_reduce_sum(local_sum);
    if (lane == 0) smax[warp_id] = warp_sum; // reuse shared buffer
    __syncthreads();
    float block_sum = (lane < (blockDim.x + 31) / 32) ? smax[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    block_sum = __shfl_sync(0xffffffff, block_sum, 0);

    // step 3: normalize
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        y[c] = y[c] / block_sum;
    }
}

void softmax_rowwise(const float* X, float* Y, int rows, int cols, cudaStream_t stream) {
    int threads = (cols >= 512) ? 512 : (cols >= 256 ? 256 : 128);
    softmax_row_kernel<<<rows, threads, 0, stream>>>(X, Y, cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Softmax kernel error: %s\n", cudaGetErrorString(err));
    }
}
