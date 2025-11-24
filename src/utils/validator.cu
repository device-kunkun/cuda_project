//
// Created by Admin on 2025/11/23.
//

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <iostream>
#include "../../include/config.cuh"

void generate_test_matrix(float* A, int M, int K) {
    for (int i = 0; i < M * K; i++) {
        A[i] = static_cast<float>(i % 17 - 8) / 8.0f;  // 有规律的值便于调试
    }
}

void generate_test_matrix_half(half_t* A, int M, int K) {
    for (int i = 0; i < M * K; i++) {
        A[i] = __float2half(static_cast<float>(i % 17 - 8) / 8.0f);
    }
}

bool validate_results(float* ref, float* test, int M, int N, float tolerance = 1e-3f) {
    float max_error = 0.0f;
    float max_value = 0.0f;
    int error_count = 0;

    for (int i = 0; i < M * N; i++) {
        float error = std::abs(ref[i] - test[i]);
        max_error = std::max(max_error, error);
        max_value = std::max(max_value, std::abs(ref[i]));

        if (error > tolerance) {
            error_count++;
            if (error_count < 10) {  // 只打印前10个错误
                printf("Error at %d: ref=%.6f, test=%.6f, error=%.6f\n",
                       i, ref[i], test[i], error);
            }
        }
    }

    printf("Validation: max_error=%.6f, max_value=%.6f, error_count=%d/%d\n",
           max_error, max_value, error_count, M * N);

    return error_count == 0;
}
