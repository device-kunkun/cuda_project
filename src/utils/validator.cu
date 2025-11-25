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

// CPU softmax (row-wise)
void cpu_softmax_rowwise(const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float* yr = y + r * cols;
        float maxv = xr[0];
        for (int c = 1; c < cols; ++c) maxv = std::max(maxv, xr[c]);
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            yr[c] = std::exp(xr[c] - maxv);
            sum += yr[c];
        }
        for (int c = 0; c < cols; ++c) {
            yr[c] /= sum;
        }
    }
}

bool validate_softmax(const float* ref, const float* test, int rows, int cols, float tol = 1e-4f) {
    int errors = 0;
    float max_err = 0.0f;
    for (int i = 0; i < rows * cols; ++i) {
        float e = std::abs(ref[i] - test[i]);
        max_err = std::max(max_err, e);
        if (e > tol) {
            errors++;
            if (errors < 10) {
                printf("Softmax error at %d: ref=%.6f test=%.6f err=%.6f\n", i, ref[i], test[i], e);
            }
        }
    }
    printf("Softmax validation: max_err=%.6f, errors=%d/%d\n", max_err, errors, rows * cols);
    return errors == 0;
}

// CPU GEMM reference (row-major): C[MxN] = A[MxK] * B[KxN]
void cpu_gemm(const half_t* A, const half_t* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

// CPU layernorm (row-wise)
void cpu_layernorm_rowwise(const float* x, float* y, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float* yr = y + r * cols;
        float mean = 0.0f;
        for (int c = 0; c < cols; ++c) mean += xr[c];
        mean /= cols;
        float var = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float d = xr[c] - mean;
            var += d * d;
        }
        var /= cols;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < cols; ++c) {
            yr[c] = (xr[c] - mean) * inv_std;
        }
    }
}

bool validate_layernorm(const float* ref, const float* test, int rows, int cols, float tol) {
    int errors = 0;
    float max_err = 0.0f;
    for (int i = 0; i < rows * cols; ++i) {
        float e = std::abs(ref[i] - test[i]);
        max_err = std::max(max_err, e);
        if (e > tol) {
            errors++;
            if (errors < 10) {
                printf("LayerNorm error at %d: ref=%.6f test=%.6f err=%.6f\n", i, ref[i], test[i], e);
            }
        }
    }
    printf("LayerNorm validation: max_err=%.6f, errors=%d/%d\n", max_err, errors, rows * cols);
    return errors == 0;
}
