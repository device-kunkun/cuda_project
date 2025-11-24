// //
// // Created by Admin on 2025/11/23.
// //
//
// #include <cuda_runtime.h>
// #include <chrono>
// #include "utils.h"
//
// class GpuTimer {
// private:
//     cudaEvent_t start_, stop_;
//
// public:
//     GpuTimer() {
//         cudaEventCreate(&start_);
//         cudaEventCreate(&stop_);
//     }
//
//     ~GpuTimer() {
//         cudaEventDestroy(start_);
//         cudaEventDestroy(stop_);
//     }
//
//     void start() {
//         cudaEventRecord(start_);
//     }
//
//     void stop() {
//         cudaEventRecord(stop_);
//         cudaEventSynchronize(stop_);
//     }
//
//     float elapsed() {
//         float ms;
//         cudaEventElapsedTime(&ms, start_, stop_);
//         return ms;
//     }
// };
//
// class CpuTimer {
// private:
//     std::chrono::time_point<std::chrono::high_resolution_clock> start_;
//
// public:
//     void start() {
//         start_ = std::chrono::high_resolution_clock::now();
//     }
//
//     float stop() {
//         auto end = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<float, std::milli> duration = end - start_;
//         return duration.count();
//     }
// };


#include "utils.h"
#include <cuda_runtime.h>
#include <chrono>

// GpuTimer 实现
GpuTimer::GpuTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
}

GpuTimer::~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

void GpuTimer::start() {
    cudaEventRecord(start_);
}

void GpuTimer::stop() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
}

float GpuTimer::elapsed() {
    float ms;
    cudaEventElapsedTime(&ms, start_, stop_);
    return ms;
}

// CpuTimer 实现
void CpuTimer::start() {
    start_ = std::chrono::high_resolution_clock::now();
}

float CpuTimer::stop() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start_;
    return duration.count();
}