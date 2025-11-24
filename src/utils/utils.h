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

#endif //MATMUL_4070TI_UTILS_H