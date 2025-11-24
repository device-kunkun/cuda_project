// //
// // Created by device on 2025/11/23.
// //
//
// #include <cuda_runtime.h>
// #include <cuda_fp16.h>
// #include "../../include/config.cuh"
//
// // 高性能共享内存矩阵乘法
// __global__ void high_performance_matmul_kernel(
//     const half_t* A, const half_t* B, float_t* C,
//     int M, int N, int K) {
//
//     // 使用128x128的共享内存块，但每个线程处理多个元素
//     const int TILE_SIZE = 128;
//     const int SUB_TILE = 32;  // 子分块大小
//     __shared__ half_t As[SUB_TILE][TILE_SIZE];
//     __shared__ half_t Bs[TILE_SIZE][SUB_TILE];
//
//     int bx = blockIdx.x, by = blockIdx.y;
//     int tx = threadIdx.x, ty = threadIdx.y;
//
//     // 每个块处理128x128的输出块
//     int row = by * TILE_SIZE + ty * 4;  // 每个线程处理4行
//     int col = bx * TILE_SIZE + tx * 4;  // 每个线程处理4列
//
//     // 每个线程计算4x4的子块
//     float sum[4][4] = {0};
//
//     // 在K维度上分块
//     for (int k_base = 0; k_base < K; k_base += TILE_SIZE) {
//         // 协作加载A的分块到共享内存 (32x128)
//         for (int i = 0; i < 4; i++) {
//             int load_row = by * TILE_SIZE + ty * 4 + i;
//             int load_col = k_base + tx;
//             if (load_row < M && load_col < K) {
//                 As[ty * 4 + i][tx] = A[load_row * K + load_col];
//             } else {
//                 As[ty * 4 + i][tx] = __float2half(0.0f);
//             }
//         }
//
//         // 协作加载B的分块到共享内存 (128x32)
//         for (int i = 0; i < 4; i++) {
//             int load_row = k_base + ty * 4 + i;
//             int load_col = bx * TILE_SIZE + tx;
//             if (load_row < K && load_col < N) {
//                 Bs[ty * 4 + i][tx] = B[load_row * N + load_col];
//             } else {
//                 Bs[ty * 4 + i][tx] = __float2half(0.0f);
//             }
//         }
//
//         __syncthreads();
//
//         // 计算部分和 - 使用循环展开优化
//         for (int k = 0; k < TILE_SIZE; k++) {
//             // 预取4个A值和4个B值
//             half_t a_vals[4], b_vals[4];
//
//             for (int i = 0; i < 4; i++) {
//                 a_vals[i] = As[ty * 4 + i][k];
//             }
//
//             for (int j = 0; j < 4; j++) {
//                 b_vals[j] = Bs[k][tx * 4 + j];
//             }
//
//             // 计算4x4子块
//             for (int i = 0; i < 4; i++) {
//                 for (int j = 0; j < 4; j++) {
//                     sum[i][j] += __half2float(a_vals[i]) * __half2float(b_vals[j]);
//                 }
//             }
//         }
//
//         __syncthreads();
//     }
//
//     // 写入4x4的结果子块
//     for (int i = 0; i < 4; i++) {
//         for (int j = 0; j < 4; j++) {
//             int out_row = row + i;
//             int out_col = col + j;
//             if (out_row < M && out_col < N) {
//                 C[out_row * N + out_col] = sum[i][j];
//             }
//         }
//     }
// }
//
// void tensor_core_matmul(const half_t* A, const half_t* B, float_t* C,
//                        int M, int N, int K, cudaStream_t stream) {
//     // 配置：32x8线程块，每个线程处理4x4=16个输出元素
//     // 每个块处理 32*4 = 128 行 x 8*4 = 32 列
//     dim3 block(8, 32);  // 256 threads
//     dim3 grid((N + 31) / 32, (M + 127) / 128);
//
//     high_performance_matmul_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
// }