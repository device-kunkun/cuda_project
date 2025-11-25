//
// GEMM playground: multiple implementations with unified interface.
//

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <cublas_v2.h>

#include "include/config.cuh"
#include "src/kernels/kernels.h"
#include "src/utils/utils.h"

struct GemmCase {
    int M, N, K;
};

static std::string algo_name(GemmAlgo algo) {
    switch (algo) {
        case GemmAlgo::Naive: return "naive";
        case GemmAlgo::SharedMem: return "shared";
        case GemmAlgo::Vectorized: return "vectorized";
        case GemmAlgo::VectorizedOpt: return "vectorized_opt";
        case GemmAlgo::TensorCore: return "tensor_core";
        default: return "unknown";
    }
}

void run_bench_case(const GemmCase& cs,
                    const std::vector<std::pair<std::string, GemmConfig>>& configs,
                    std::ofstream* csv) {
    const int M = cs.M, N = cs.N, K = cs.K;
    std::cout << "\n=== Case M=" << M << " N=" << N << " K=" << K << " ===\n";

    std::vector<half_t> h_A(M * K);
    std::vector<half_t> h_B(K * N);
    std::vector<float_t> h_C(M * N);
    std::vector<float_t> h_ref(M * N);

    generate_test_matrix_half(h_A.data(), M, K);
    generate_test_matrix_half(h_B.data(), K, N);

    half_t* d_A = nullptr;
    half_t* d_B = nullptr;
    float_t* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(half_t) * M * K));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(half_t) * K * N));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(float_t) * M * N));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(half_t) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeof(half_t) * K * N, cudaMemcpyHostToDevice));

    GpuTimer timer;
    bool reference_ready = false;

    for (size_t i = 0; i < configs.size(); ++i) {
        const auto& name = configs[i].first;
        const auto& cfg = configs[i].second;
        std::cout << "-> " << name
                  << " [algo=" << algo_name(cfg.algo)
                  << ", block=" << cfg.block_x << "x" << cfg.block_y
                  << ", tile_k=" << cfg.tile_k << "]" << std::endl;

        CUDA_CHECK(cudaMemset(d_C, 0, sizeof(float_t) * M * N));

        timer.start();
        launch_gemm(d_A, d_B, d_C, M, N, K, cfg, 0);
        timer.stop();
        float ms = timer.elapsed();
        double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        double gflops = ms > 0 ? flops / (ms * 1e6) : 0.0;

        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeof(float_t) * M * N, cudaMemcpyDeviceToHost));

        if (!reference_ready) {
            h_ref = h_C;
            reference_ready = true;
            std::cout << "   time: " << ms << " ms, GFLOPS: " << gflops << " (reference)\n";
            if (csv && csv->is_open()) {
                *csv << name << "," << M << "," << N << "," << K << "," << ms << "," << gflops << "," << "yes" << "\n";
            }
        } else {
            bool ok = validate_results(h_ref.data(), h_C.data(), M, N, 1e-2f);
            std::cout << "   time: " << ms << " ms, GFLOPS: " << gflops
                      << ", valid=" << (ok ? "yes" : "NO") << "\n";
            if (csv && csv->is_open()) {
                *csv << name << "," << M << "," << N << "," << K << "," << ms << "," << gflops << "," << (ok ? "yes" : "NO") << "\n";
            }
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void print_device_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Multi-processors: " << prop.multiProcessorCount << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "Matrix Multiplication Playground (4070 Ti Super)" << std::endl;
    std::cout << "===============================================" << std::endl;
    print_device_info();

    std::ofstream csv;
    const std::string csv_path = "results.csv";
    bool need_header = true;
    if (std::filesystem::exists(csv_path)) {
        need_header = std::filesystem::file_size(csv_path) == 0;
    }
    csv.open(csv_path, std::ios::app);
    if (csv.is_open() && need_header) {
        csv << "algo,M,N,K,ms,gflops,valid\n";
    }

    // Sweep configs (first entry acts as reference)
    std::vector<std::pair<std::string, GemmConfig>> configs = {
        {"naive 16x16", {GemmAlgo::Naive, 16, 16, TILE_K}},
        {"shared 32x8 tk32", {GemmAlgo::SharedMem, 32, 8, 32}},
        {"vectorized 32x8", {GemmAlgo::Vectorized, 32, 8, 0}},
        {"vec-opt 64x64", {GemmAlgo::VectorizedOpt, 16, 16, 32}},
        {"tensorcore wmma16", {GemmAlgo::TensorCore, 32, 1, WMMA_K}}
    };

    // Sweep matrix sizes
    std::vector<GemmCase> cases = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096}
    };

    for (const auto& cs : cases) {
        run_bench_case(cs, configs, csv.is_open() ? &csv : nullptr);
    }

    // CPU reference vs GPU for a small size
    {
        const int M = 256, N = 256, K = 256;
        std::vector<half_t> hA(M * K), hB(K * N);
        std::vector<float> hC(M * N), hRef(M * N);
        generate_test_matrix_half(hA.data(), M, K);
        generate_test_matrix_half(hB.data(), K, N);
        cpu_gemm(hA.data(), hB.data(), hRef.data(), M, N, K);

        half_t *dA, *dB;
        float *dC;
        CUDA_CHECK(cudaMalloc(&dA, sizeof(half_t) * M * K));
        CUDA_CHECK(cudaMalloc(&dB, sizeof(half_t) * K * N));
        CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * M * N));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(half_t) * M * K, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(half_t) * K * N, cudaMemcpyHostToDevice));

        GemmConfig cfg{GemmAlgo::SharedMem, 32, 8, 32};
        launch_gemm(dA, dB, dC, M, N, K, cfg, 0);
        CUDA_CHECK(cudaMemcpy(hC.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
        validate_results(hRef.data(), hC.data(), M, N, 1e-2f);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    // cuBLAS reference for a medium size (row-major via transposed call)
    {
        const int M = 512, N = 512, K = 512;
        std::vector<half_t> hA(M * K), hB(K * N);
        std::vector<float> hC(M * N), hRef(M * N);
        const size_t sizeC = static_cast<size_t>(M) * N;
        generate_test_matrix_half(hA.data(), M, K);
        generate_test_matrix_half(hB.data(), K, N);

        // Host column-major copies for cuBLAS (col-major expects leading dimension = rows)
        std::vector<half_t> hA_col(M * K);
        std::vector<half_t> hB_col(K * N);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < K; ++j)
                hA_col[i + j * M] = hA[i * K + j]; // row-major -> col-major
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < N; ++j)
                hB_col[i + j * K] = hB[i * N + j];

        half_t *dA_col, *dB_col;
        float *dC_col;
        CUDA_CHECK(cudaMalloc(&dA_col, sizeof(half_t) * M * K));
        CUDA_CHECK(cudaMalloc(&dB_col, sizeof(half_t) * K * N));
        CUDA_CHECK(cudaMalloc(&dC_col, sizeof(float) * M * N));
        CUDA_CHECK(cudaMemcpy(dA_col, hA_col.data(), sizeof(half_t) * M * K, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB_col, hB_col.data(), sizeof(half_t) * K * N, cudaMemcpyHostToDevice));

        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t stat = cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            dA_col, CUDA_R_16F, M,
            dB_col, CUDA_R_16F, K,
            &beta,
            dC_col, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "cuBLAS gemm failed, status " << stat << std::endl;
        }
        // dC_col is column-major M x N -> convert to row-major hRef
        std::vector<float> hTmp(sizeC);
        CUDA_CHECK(cudaMemcpy(hTmp.data(), dC_col, sizeof(float) * sizeC, cudaMemcpyDeviceToHost));
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                hRef[i * N + j] = hTmp[i + j * M];
            }
        }

        // compare to one of our GPU kernels (e.g., vectorized opt)
        half_t *dA_rm, *dB_rm;
        float *dC_rm;
        CUDA_CHECK(cudaMalloc(&dA_rm, sizeof(half_t) * M * K));
        CUDA_CHECK(cudaMalloc(&dB_rm, sizeof(half_t) * K * N));
        CUDA_CHECK(cudaMalloc(&dC_rm, sizeof(float) * M * N));
        CUDA_CHECK(cudaMemcpy(dA_rm, hA.data(), sizeof(half_t) * M * K, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB_rm, hB.data(), sizeof(half_t) * K * N, cudaMemcpyHostToDevice));
        GemmConfig cfg{GemmAlgo::VectorizedOpt, 16, 16, 32};
        launch_gemm(dA_rm, dB_rm, dC_rm, M, N, K, cfg, 0);
        CUDA_CHECK(cudaMemcpy(hC.data(), dC_rm, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
        validate_results(hRef.data(), hC.data(), M, N, 1e-2f);

        cublasDestroy(handle);
        cudaFree(dA_col); cudaFree(dB_col); cudaFree(dC_col);
        cudaFree(dA_rm); cudaFree(dB_rm); cudaFree(dC_rm);
    }

    // Softmax demo
    {
        const int rows = 1024;
        const int cols = 512;
        std::vector<float> h_x(rows * cols);
        std::vector<float> h_y(rows * cols);
        std::vector<float> h_ref(rows * cols);
        for (int i = 0; i < rows * cols; ++i) h_x[i] = static_cast<float>((i % 17) - 8) * 0.1f;
        cpu_softmax_rowwise(h_x.data(), h_ref.data(), rows, cols);

        float *d_x = nullptr, *d_y = nullptr;
        CUDA_CHECK(cudaMalloc(&d_x, sizeof(float) * rows * cols));
        CUDA_CHECK(cudaMalloc(&d_y, sizeof(float) * rows * cols));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), sizeof(float) * rows * cols, cudaMemcpyHostToDevice));
        GpuTimer t;
        t.start();
        softmax_rowwise(d_x, d_y, rows, cols, 0);
        t.stop();
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost));
        validate_softmax(h_ref.data(), h_y.data(), rows, cols, 1e-3f);
        std::cout << "Softmax: rows=" << rows << " cols=" << cols << ", time=" << t.elapsed() << " ms" << std::endl;
        cudaFree(d_x); cudaFree(d_y);
    }

    // LayerNorm demo (row-wise)
    {
        const int rows = 1024;
        const int cols = 512;
        std::vector<float> h_x(rows * cols);
        std::vector<float> h_y(rows * cols);
        std::vector<float> h_ref(rows * cols);
        for (int i = 0; i < rows * cols; ++i) h_x[i] = static_cast<float>((i % 31) - 15) * 0.05f;
        cpu_layernorm_rowwise(h_x.data(), h_ref.data(), rows, cols, 1e-5f);

        float *d_x = nullptr, *d_y = nullptr;
        CUDA_CHECK(cudaMalloc(&d_x, sizeof(float) * rows * cols));
        CUDA_CHECK(cudaMalloc(&d_y, sizeof(float) * rows * cols));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), sizeof(float) * rows * cols, cudaMemcpyHostToDevice));
        GpuTimer t;
        t.start();
        layernorm_rowwise(d_x, d_y, rows, cols, 1e-5f, 0);
        t.stop();
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost));
        validate_layernorm(h_ref.data(), h_y.data(), rows, cols, 1e-3f);
        std::cout << "LayerNorm: rows=" << rows << " cols=" << cols << ", time=" << t.elapsed() << " ms" << std::endl;
        cudaFree(d_x); cudaFree(d_y);
    }

    // Batched GEMM demo
    {
        const int batch = 64;
        const int M = 64, N = 64, K = 64;
        size_t sizeA = batch * M * K;
        size_t sizeB = batch * K * N;
        size_t sizeC = batch * M * N;
        std::vector<half_t> hA(sizeA), hB(sizeB);
        std::vector<float_t> hC(sizeC), hRef(sizeC);
        generate_test_matrix_half(hA.data(), batch * M, K); // reuse pattern
        generate_test_matrix_half(hB.data(), batch * K, N);
        half_t *dA, *dB;
        float_t *dC;
        CUDA_CHECK(cudaMalloc(&dA, sizeof(half_t) * sizeA));
        CUDA_CHECK(cudaMalloc(&dB, sizeof(half_t) * sizeB));
        CUDA_CHECK(cudaMalloc(&dC, sizeof(float_t) * sizeC));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(half_t) * sizeA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(half_t) * sizeB, cudaMemcpyHostToDevice));

        GemmConfig cfg{GemmAlgo::SharedMem, BLOCK_SIZE, BLOCK_SIZE, TILE_K};
        GpuTimer t;
        t.start();
        batched_shared_matmul(dA, dB, dC, batch, M, N, K, cfg, 0);
        t.stop();
        CUDA_CHECK(cudaMemcpy(hC.data(), dC, sizeof(float_t) * sizeC, cudaMemcpyDeviceToHost));
        std::cout << "Batched GEMM: batch=" << batch << " M=" << M << " N=" << N << " K=" << K
                  << ", time=" << t.elapsed() << " ms" << std::endl;
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    return 0;
}
