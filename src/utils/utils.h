//
// Created by Admin on 2025/11/23.
//

#ifndef MATMUL_4070TI_UTILS_H
#define MATMUL_4070TI_UTILS_H

#include "../../include/config.cuh"
#include <cuda_runtime.h>
#include <chrono>

class GpuTimer {
private:
    cudaEvent_t start_, stop_;
public:
    GpuTimer();
    ~GpuTimer();
    void start();
    void stop();
    float elapsed();
};

class CpuTimer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;

public:
    void start();
    float stop();
};

void generate_test_matrix(float* A, int M, int K);
void generate_test_matrix_half(half_t* A, int M, int K);
bool validate_results(float* ref, float* test, int M, int N, float tolerance = 1e-3f);
void cpu_softmax_rowwise(const float* x, float* y, int rows, int cols);
bool validate_softmax(const float* ref, const float* test, int rows, int cols, float tol = 1e-4f);
void cpu_gemm(const half_t* A, const half_t* B, float* C, int M, int N, int K);
void cpu_layernorm_rowwise(const float* x, float* y, int rows, int cols, float eps = 1e-5f);
bool validate_layernorm(const float* ref, const float* test, int rows, int cols, float tol = 1e-4f);

#endif //MATMUL_4070TI_UTILS_H
