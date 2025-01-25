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
constexpr int M = 4;
constexpr int N = 4;
constexpr int K = 4;
constexpr int nBatch = 16;
constexpr unsigned int compute_repetitions = 1;

constexpr int LDA = K;
constexpr int LDB = N;
constexpr int LDD = N;

constexpr int batchStrideA = M * LDA;
constexpr int batchStrideB = K * LDB;
constexpr int batchStrideD = M * LDD;

constexpr int A_size = batchStrideA * nBatch;
constexpr int B_size = batchStrideB * nBatch;
constexpr int D_size = batchStrideD * nBatch;

__global__ void sgemm_4x4x4_batch(const float* A, const float* B, float* D) {
#if __gfx90a__ || __gfx908__
    using float4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    float4 d = {0}; // Zero out 4 VGPRs

    for (int iter = 0; iter < compute_repetitions; ++iter) {
        for (int k = 0; k < 4; ++k) {
            int a_idx = LDA * threadIdx.x + batchStrideA * threadIdx.y;
            int b_idx = threadIdx.x + batchStrideB * threadIdx.y;

            for (int i = 0; i < 4; ++i) {
                const float a = A[a_idx];
                const float b = B[b_idx];

                d = __builtin_amdgcn_mfma_f32_4x4x1f32(a, b, d, 0, 0, 0);
                //                                     ^  ^  ^
                //D(=C)                                |  |  C(=D)
                //            one column from each A---|  |--- one row from each B
                a_idx += 1;   // Move one column to the right
                b_idx += LDB; // Move one row down
            }
        }
    }

    for (int i = 0; i < 4; ++i) {
        const int d_idx = threadIdx.x + i * LDD + threadIdx.y * batchStrideD;
        D[d_idx] = d[i];
    }
#endif
}

int main() {
    if (!gpuArchCheck("gfx90a") && !gpuArchCheck("gfx908")) {
        std::cout << "mfma_f32_4x4x1f32 instruction only available on gfx908 or later."
                  << std::endl;
        exit(-1);
    }

    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(-1, 1);

    // Make and populate host matrices
    std::vector<float> A_h(A_size);
    for (auto& val : A_h) val = dist(gen);
    std::vector<float> B_h(B_size);
    for (auto& val : B_h) val = dist(gen);

    // Calculate reference D on host
    std::vector<float> Dref_h(D_size);
    gemm_host_batch(A_h, B_h, Dref_h, M, N, K, nBatch, LDA, LDB, LDD, batchStrideA, batchStrideB, batchStrideD);

    // Make and populate device buffers
    float *A_d, *B_d, *D_d;
    HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(float)));
    HIP_CHECK(hipMemcpy(A_d, A_h.data(), A_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_d, B_h.data(), B_size * sizeof(float), hipMemcpyHostToDevice));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    auto overall_start = std::chrono::high_resolution_clock::now();
    double runtime = 0.0;
    int kernel_runs = 0;

    while (runtime < 5.0) {
        auto t1 = std::chrono::high_resolution_clock::now();
        sgemm_4x4x4_batch<<<dim3(128,64,64), dim3(4,16)>>>(A_d, B_d, D_d);
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
              << compute_l2_error_batch(Dref_h, D_h, M, N, nBatch, LDD, LDD, batchStrideD, batchStrideD)
              << std::endl;

    HIP_CHECK(hipFree(D_d));
    HIP_CHECK(hipFree(B_d));
    HIP_CHECK(hipFree(A_d));
    return 0;
}

