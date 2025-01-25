/*
Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
...
*/

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "helper.hpp"

// Constants
constexpr int M = 32;
constexpr int N = 32;
constexpr int K = 32;

constexpr int LDA = K;
constexpr int LDB = N;
constexpr int LDD = N;
constexpr unsigned int compute_repetitions = 20000;

constexpr int A_size = M * LDA;
constexpr int B_size = K * LDB;
constexpr int D_size = M * LDD;

__global__ void sgemm_32x32x32(const float16_t* A, const float16_t* B, float* D) {
#if __gfx90a__ || __gfx908__
    using float16x4 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
    using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
    floatx16 d = {0}; // Zero out 16 VGPRs

    for (int iter = 0; iter < compute_repetitions; ++iter) {
        for (int k = 0; k < 4; ++k) {
            float16x4 a;
            float16x4 b;
            for (int i = 0; i < 4; ++i) {
                const int a_idx = threadIdx.x * LDA + i + threadIdx.y * 4 + k * 8;
                a[i] = A[a_idx];

                const int b_idx = threadIdx.x + i * LDB + threadIdx.y * 4 * LDB + k * 8 * LDB;
                b[i] = B[b_idx];
            }

            d = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, d, 0, 0, 0);
        }
    }

    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            const int d_idx = threadIdx.x + i * LDD + threadIdx.y * 4 * LDD + j * 2 * 4 * LDD;
            D[d_idx] = d[i + 4 * j];
        }
    }
#endif
}

int main() {
    if (!gpuArchCheck("gfx90a") && !gpuArchCheck("gfx908")) {
        std::cout << "mfma_f32_32x32x8f16 instruction only available on gfx908 or later."
                  << std::endl;
        exit(-1);
    }

    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(-1, 1);

    // Make and populate host matrices
    std::vector<float16_t> A_h(A_size);
    for (auto& val : A_h) val = static_cast<float16_t>(dist(gen));
    std::vector<float16_t> B_h(B_size);
    for (auto& val : B_h) val = static_cast<float16_t>(dist(gen));

    // Calculate reference D on host
    std::vector<float> Dref_h(D_size);
    gemm_host(A_h, B_h, Dref_h, M, N, K, LDA, LDB, LDD);

    // Make and populate device buffers
    float16_t *A_d, *B_d;
    float *D_d;
    HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(float16_t)));
    HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(float)));
    HIP_CHECK(hipMemcpy(A_d, A_h.data(), A_size * sizeof(float16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_d, B_h.data(), B_size * sizeof(float16_t), hipMemcpyHostToDevice));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    auto overall_start = std::chrono::high_resolution_clock::now();
    double runtime = 0.0;
    int kernel_runs = 0;

    while (runtime < 5.0) {
        auto t1 = std::chrono::high_resolution_clock::now();
        sgemm_32x32x32<<<dim3(128, 64, 64), dim3(32, 2)>>>(A_d, B_d, D_d);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        auto t2 = std::chrono::high_resolution_clock::now();

        runtime += std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        ++kernel_runs;
    }

    auto overall_end = std::chrono::high_resolution_clock::now();
    double overall_runtime = std::chrono::duration_cast<std::chrono::duration<double>>(overall_end - overall_start).count();

    HIP_CHECK(hipStreamDestroy(stream));

    // Print timing results
    std::cout << "Kernel was executed " << kernel_runs << " times in " << runtime << " seconds.\n";
    std::cout << "Average kernel execution time: " << (runtime / kernel_runs) << " seconds.\n";
    std::cout << "Overall elapsed time (including loop overhead): " << overall_runtime << " seconds.\n";


    // Copy result back to host
    std::vector<float> D_h(D_size);
    HIP_CHECK(hipMemcpy(D_h.data(), D_d, D_size * sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "Sum of squared differences of host/device result matrices: "
              << compute_l2_error(Dref_h, D_h, M, N, LDD, LDD)
              << std::endl;

    HIP_CHECK(hipFree(D_d));
    HIP_CHECK(hipFree(B_d));
    HIP_CHECK(hipFree(A_d));
    return 0;
}

