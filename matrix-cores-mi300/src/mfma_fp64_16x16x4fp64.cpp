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
This example code uses the mfma intrinsic __builtin_amdgcn_mfma_f64_16x16x4f64 to
compute a 16x16x16 matrix multiplication.

Input:
  A : 16 x 16 doubles (a 16x16 matrix)
  B : 16 x 16 doubles (a 16x16 matrix)

Output:
  D : 16 x 16 doubles (a 16x16 matrix)
*/

constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;
constexpr unsigned int compute_repetitions = 1;

constexpr int LDA = K;
constexpr int LDB = N;
constexpr int LDD = N;

constexpr int A_size = M * LDA;
constexpr int B_size = K * LDB;
constexpr int D_size = M * LDD;

__global__ void dgemm_16x16x16(const double* A, const double* B, double* D)
{
#if __gfx90a__
  using double4 = __attribute__((__vector_size__(4 * sizeof(double)))) double;
  double4 d = {0}; // zero out 4 * 2 vanilla VGPRs

  int a_idx = LDA * threadIdx.x + threadIdx.y;
  int b_idx = threadIdx.x + LDB * threadIdx.y;

  for (int iter = 0; iter < compute_repetitions; ++iter) {
    for (int i = 0; i < 4; ++i) {
      const double a = A[a_idx];
      const double b = B[b_idx];

      d = __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, d, 0, 0, 0);
      a_idx += 4;     // move four columns to the right
      b_idx += 4 * LDB; // move four rows down
    }
  }

  for (int i = 0; i < 4; ++i) {
    const int d_idx = threadIdx.x + 4 * LDD * i + LDD * threadIdx.y;
    D[d_idx] = d[i];
  }
#endif
}

int main() {
  if (!gpuArchCheck("gfx90a")) {
    std::cout << "mfma_f64_16x16x4f64 instruction only available on gfx90a or later."
              << std::endl;
    exit(-1);
  }

  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(-1, 1);

  std::vector<double> A_h(A_size);
  for (int i = 0; i < A_h.size(); ++i) {
    A_h[i] = dist(gen);
  }
  std::vector<double> B_h(B_size);
  for (int i = 0; i < B_h.size(); ++i) {
    B_h[i] = dist(gen);
  }

  std::vector<double> Dref_h(D_size);
  gemm_host(A_h, B_h, Dref_h, M, N, K, LDA, LDB, LDD);

  double *A_d, *B_d, *D_d;
  HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(double)));
  HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(double)));
  HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(double)));
  HIP_CHECK(hipMemcpy(A_d, A_h.data(), A_size * sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h.data(), B_size * sizeof(double), hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  auto overall_start = std::chrono::high_resolution_clock::now();
  double runtime = 0.0;
  int kernel_runs = 0;

  while (runtime < 5.0) {
    auto t1 = std::chrono::high_resolution_clock::now();
    dgemm_16x16x16<<<dim3(128, 64, 64), dim3(16, 4)>>>(A_d, B_d, D_d);
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

  std::vector<double> D_h(D_size);
  HIP_CHECK(hipMemcpy(D_h.data(), D_d, D_size * sizeof(double), hipMemcpyHostToHost));

  std::cout << "Sum of squared differences of host/device result matrices: "
            << compute_l2_error(Dref_h, D_h, M, N, LDD, LDD)
            << std::endl;

  HIP_CHECK(hipFree(D_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(A_d));
  return 0;
}

