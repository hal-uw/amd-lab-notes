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
#include "helper.hpp"

/*
This example code uses the mfma intrinsic __builtin_amdgcn_mfma_f32_4x4x1f32 to
compute a batch of 16 4x4x4 matrix multiplications.

Input:
  A : 4 x 4 x 16 floats (16 4x4 matrices)
  B : 4 x 4 x 16 floats (16 4x4 matrices)

Output:
  D : 4 x 4 x 16 floats (16 4x4 matrices)
*/

constexpr int M = 4;
constexpr int N = 4;
constexpr int K = 4;
constexpr int nBatch = 16;
constexpr unsigned int compute_repetitions = 10000;

constexpr int LDA = K;
constexpr int LDB = N;
constexpr int LDD = N;

constexpr int batchStrideA = M * LDA;
constexpr int batchStrideB = K * LDB;
constexpr int batchStrideD = M * LDD;

constexpr int A_size = batchStrideA * nBatch;
constexpr int B_size = batchStrideB * nBatch;
constexpr int D_size = batchStrideD * nBatch;

__global__ void sgemm_4x4x4_batch(const float *A, const float *B, float *D)
{
#if __gfx90a__ || __gfx908__ || __gfx942__
  // Using float4 type for vectorized operations
  using float4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  float4 d = {0}; // Zero out 4 vanilla VGPRs

  /*
  This kernel has been optimized to:
  - Remove unnecessary nested loops (`j`, `k`, and `i` inside `compute_repetitions`).
  - Preload `A` and `B` before the compute loop to minimize global memory access.
  */

  // Load input matrices A and B into registers before compute loop
  float a, b;

  const int a_idx = threadIdx.x * LDA + threadIdx.y * batchStrideA;
  const int b_idx = threadIdx.x + threadIdx.y * batchStrideB;

  a = A[a_idx];
  b = B[b_idx];

  // Perform matrix multiplication repetitions without nested loops
  for (int rep_i = 0; rep_i < compute_repetitions; ++rep_i) {
    d = __builtin_amdgcn_mfma_f32_4x4x1f32(a, b, d, 0, 0, 0);
  }

  /*
  Matrix D is a batch of 16 4 x 4 matrices that are stored in 4 AccVGPRs as follows:
    d[0] covers row 0
    d[1] covers row 1
    d[2] covers row 2
    d[3] covers row 3
  */

  // Write results to output D matrix
  for (int i = 0; i < 4; ++i) {
    const int d_idx =   threadIdx.x                 // consecutive threads cover 4 consecutive columns
                      + i * LDD                     // consecutive registers take consecutive rows
                      + threadIdx.y * batchStrideD; // groups of 4 lanes cover each matrix in batch
    D[d_idx] = d[i];
  }
#endif
}




int main() {
  if (!gpuArchCheck("gfx90a") && !gpuArchCheck("gfx908") && !gpuArchCheck("gfx942")) {
    std::cout << "mfma_f32_4x4x1f32 instruction only available on gfx908 or later."
              << std::endl;
    exit(-1);
  }

  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dist(-1, 1);

  // Make and populate some host matrices
  std::vector<float> A_h(A_size);
  for(int i = 0; i < A_h.size(); ++i){
    A_h[i] = dist(gen);
  }
  std::vector<float> B_h(B_size);
  for(int i = 0; i < B_h.size(); ++i){
    B_h[i] = dist(gen);
  }

  // Calculate reference D on host
  std::vector<float> Dref_h(D_size);
  gemm_host_batch(A_h, B_h, Dref_h, M, N, K, nBatch,
                  LDA, LDB, LDD,
                  batchStrideA, batchStrideB, batchStrideD);

  // Make and populate device buffers
  float *A_d, *B_d, *D_d;
  HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(float)));
  HIP_CHECK(hipMemcpy(A_d, A_h.data(), A_size * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h.data(), B_size * sizeof(float), hipMemcpyHostToDevice));

  // Launch GEMM kernel
  sgemm_4x4x4_batch<<<dim3(128,64,64), dim3(4, 16)>>>(A_d, B_d, D_d);
  HIP_CHECK(hipGetLastError());

  // Copy result back to host
  std::vector<float> D_h(D_size);
  HIP_CHECK(hipMemcpy(D_h.data(), D_d, D_size * sizeof(float), hipMemcpyDeviceToHost));

  std::cout << "Sum of squared differences of host/device result matrices: "
            << compute_l2_error_batch(Dref_h, D_h, M, N, nBatch,
                                      LDD, LDD, batchStrideD, batchStrideD)
            << std::endl;

  HIP_CHECK(hipFree(D_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(A_d));
  return 0;
}
