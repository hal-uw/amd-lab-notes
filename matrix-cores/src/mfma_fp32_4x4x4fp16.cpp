/*
Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono> // For timing
#include "helper.hpp"

/*
This example code uses the mfma intrinsic __builtin_amdgcn_mfma_f32_4x4x4f16 to
compute a batch of 16 4x4x4 matrix multiplications.

Input:
  A : 4 x 4 x 16 float16s (16 4x4 matrices)
  B : 4 x 4 x 16 float16s (16 4x4 matrices)

Output:
  D : 4 x 4 x 16 floats (16 4x4 matrices)
*/

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

__global__ void sgemm_4x4x4_batch(const float16_t *A, const float16_t *B, float *D)
{

#if __gfx90a__ || __gfx908__
  using float16x4 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  floatx4 d = {0}; // zero out 4 vanilla VGPRs

  for (int iter = 0; iter < compute_repetitions; ++iter) {
    float16x4 a;
    float16x4 b;
    for (int i = 0; i < 4; ++i) {
      const int a_idx = threadIdx.x * LDA + i + threadIdx.y * batchStrideA;
      a[i] = A[a_idx];

      const int b_idx = threadIdx.x + i * LDB + threadIdx.y * batchStrideB;
      b[i] = B[b_idx];
    }

    d = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, d, 0, 0, 0);
  }

  for (int i = 0; i < 4; ++i) {
    const int d_idx = threadIdx.x + i * LDD + threadIdx.y * batchStrideD;
    D[d_idx] = d[i];
  }
#endif
}

int main() {
  if (!gpuArchCheck("gfx90a") && !gpuArchCheck("gfx908")) {
    std::cout << "mfma_f32_4x4x4f16 instruction only available on gfx908 or later."
              << std::endl;
    exit(-1);
  }

  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dist(-1, 1);

  std::vector<float16_t> A_h(A_size);
  for (int i = 0; i < A_h.size(); ++i) {
    A_h[i] = static_cast<float16_t>(dist(gen));
  }

  std::vector<float16_t> B_h(B_size);
  for (int i = 0; i < B_h.size(); ++i) {
    B_h[i] = static_cast<float16_t>(dist(gen));
  }

  std::vector<float> Dref_h(D_size);
  gemm_host_batch(A_h, B_h, Dref_h, M, N, K, nBatch, LDA, LDB, LDD, batchStrideA, batchStrideB, batchStrideD);

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
    sgemm_4x4x4_batch<<<dim3(128, 64, 64), dim3(4, 16)>>>(A_d, B_d, D_d);
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

