//
// GEMM playground: multiple implementations with unified interface.
//

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>

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
        case GemmAlgo::TensorCore: return "tensor_core";
        default: return "unknown";
    }
}

void run_bench_case(const GemmCase& cs,
                    const std::vector<std::pair<std::string, GemmConfig>>& configs) {
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

        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeof(float_t) * M * N, cudaMemcpyDeviceToHost));

        if (!reference_ready) {
            h_ref = h_C;
            reference_ready = true;
            std::cout << "   time: " << ms << " ms (reference)\n";
        } else {
            bool ok = validate_results(h_ref.data(), h_C.data(), M, N, 1e-2f);
            std::cout << "   time: " << ms << " ms, valid=" << (ok ? "yes" : "NO") << "\n";
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

    // Sweep configs (first entry acts as reference)
    std::vector<std::pair<std::string, GemmConfig>> configs = {
        {"naive 16x16", {GemmAlgo::Naive, 16, 16, TILE_K}},
        {"shared 32x8 tk32", {GemmAlgo::SharedMem, 32, 8, 32}},
        {"vectorized 32x8", {GemmAlgo::Vectorized, 32, 8, 0}},
        {"tensorcore wmma16", {GemmAlgo::TensorCore, 32, 1, WMMA_K}}
    };

    // Sweep matrix sizes
    std::vector<GemmCase> cases = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024}
    };

    for (const auto& cs : cases) {
        run_bench_case(cs, configs);
    }

    return 0;
}
