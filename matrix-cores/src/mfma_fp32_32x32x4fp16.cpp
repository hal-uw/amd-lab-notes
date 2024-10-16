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
This example code uses the mfma intrinsic __builtin_amdgcn_mfma_f32_32x32x4f16 to
compute a batch of two 32x32x32 matrix multiplications.

Input:
  A : 32 x 32 x 2 float16s (two 32x32 matrices)
  B : 32 x 32 x 2 float16s (two 32x32 matrices)

Output:
  D : 32 x 32 x 2 floats (two 32x32 matrices)
*/

constexpr int M = 32;
constexpr int N = 32;
constexpr int K = 32;
constexpr int nBatch = 2;
constexpr unsigned int compute_repetitions = 20000;

constexpr int LDA = K;
constexpr int LDB = N;
constexpr int LDD = N;

constexpr int batchStrideA = M * LDA;
constexpr int batchStrideB = K * LDB;
constexpr int batchStrideD = M * LDD;

constexpr int A_size = batchStrideA * nBatch;
constexpr int B_size = batchStrideB * nBatch;
constexpr int D_size = batchStrideD * nBatch;


__global__ void sgemm_32x32x32_batch(const float16_t* A, const float16_t* B, float* D)
{

#if __gfx90a__ || __gfx908__
  // This kernel computes a batch of two 32x32x32 matrix multiplications using a single wavefront.
  using float16x4 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
  using floatx32 = __attribute__((__vector_size__(32 * sizeof(float)))) float;
  floatx32 d = {0}; // zero out 32 vanilla VGPRs

  /*
  One invocation of v_mfma_f32_32x32x4f16 accumulates two batches of 4 outer products,
  four columns of each A with four rows of each B, into a batch of two result matrices D.
  So we need 8 iterations to compute the full batch of matrix multiplications,
  starting with the leftmost columns of each A and the topmost column of each B,
  and then moving four columns to the right for each A, and down four rows for each B,
  for every iteration.

  For the four columns of each A, and the four rows of each B, we use a single VGPR pair.
  With 64 lanes, and 4 Float16 values per lane, that covers the 8 columns of A and 8
  rows of B.
  Matrix A is a batch of two 32 x 4 matrices stored in 2 VGPRs as follows:
    lanes 0-32     contain the first  A matrix
    lanes 32-63    contain the second A matrix
  Within a block of 32 lanes, e.g. lanes 0-31:
    a[0] covers column 0
    a[1] covers column 1
    a[2] covers column 2
    a[3] covers column 3
  Matrix B is a batch of two 4 x 32 matrices stored in 2 VGPRs as follows:
    lanes 0-32     contain the first  B matrix
    lanes 32-63    contain the second B matrix
  Within a block of 32 lanes, e.g. lanes 0-31:
    b[0] covers row 0
    b[1] covers row 1
    b[2] covers row 2
    b[3] covers row 3
  Note that each A and B are in row-major order.

  This kernel is called with a single wavefront in dim3(32, 2) layout
  */

  for(int k = 0; k < 8; ++k){
    float16x4 a;
    float16x4 b;
    for(int i = 0; i < 4; ++i){
      const int a_idx =  threadIdx.x * LDA          // consecutive threads cover 32 consecutive rows
                       + i                          // consecutive registers take consecutive columns
                       + threadIdx.y * batchStrideA // groups of 32 lanes cover each matrix in batch
                       + k * 4;                     // 4 columns fetched in each iteration
      a[i] = A[a_idx];

      const int b_idx =  threadIdx.x                // consecutive threads cover 32 consecutive columns
                       + i * LDB                    // consecutive registers take consecutive rows
                       + threadIdx.y * batchStrideB // groups of 32 lanes cover each matrix in batch
                       + k * 4 * LDB;               // 4 rows fetched in each iteration
      b[i] = B[b_idx];
    }

    d = __builtin_amdgcn_mfma_f32_32x32x4f16(a, b, d, 0, 0, 0);
    //                                       ^  ^  ^
    //D(=C)                                  |  |  C(=D)
    //                 4 columns of each A---|  |--- 4 rows of each B
  }

  /*
  Matrix D is a batch of two 32 x 32 matrices that are stored in 32 AccVGPRs as follows:
    d[0:15]     contain the first  D matrix
    d[16:31]    contain the second D matrix
  Within a block of 16 AccVGPRs, e.g. d[0:15]:
    d[0:3]   cover rows 0-7
    d[4:7]   cover rows 8-15
    d[8:11]  cover rows 16-23
    d[12:15] cover rows 24-31
  Within each block of 4 AccVGPRs, e.g. d[0:3]:
    d[0] covers rows 0, 4, 8, and 12
    d[1] covers rows 1, 5, 9, and 13
    d[2] covers rows 2, 6, 10, and 14
    d[3] covers rows 3, 7, 11, and 15
    first 16 lanes of d[0] cover row 0 -  last 16 lanes of d[0] cover row 12
    first 16 lanes of d[1] cover row 1 -  last 16 lanes of d[1] cover row 13
    first 16 lanes of d[2] cover row 2 -  last 16 lanes of d[2] cover row 14
    first 16 lanes of d[3] cover row 3 -  last 16 lanes of d[3] cover row 15
  */
  for (int b = 0; b < 2; ++b) {
    for(int j = 0; j < 4; ++j){
      for(int i = 0; i < 4; ++i){
        const int d_idx =  threadIdx.x            // consecutive threads cover 32 consecutive columns
                         + i * LDD                // consecutive registers take consecutive rows of 32 floats
                         + threadIdx.y * 4 * LDD  // last 32 lanes skip 4 rows
                         + j * 2 * 4 * LDD        // blocks of 4 registers cover 8 rows
                         + b * batchStrideD;      // groups of 16 registers cover each matrix in batch

        D[d_idx] = d[i + 4 * j + 16 * b];
      }
    }
  }
#endif
}


int main() {
  if (!gpuArchCheck("gfx90a") && !gpuArchCheck("gfx908")) {
    std::cout << "mfma_f32_32x32x4f16 instruction only available on gfx908 or later."
              << std::endl;
    exit(-1);
  }

  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dist(-1, 1);

  // Make and populate some host matrices
  std::vector<float16_t> A_h(A_size);
  for(int i = 0; i < A_h.size(); ++i){
    A_h[i] = static_cast<float16_t>(dist(gen));
  }
  std::vector<float16_t> B_h(B_size);
  for(int i = 0; i < B_h.size(); ++i){
    B_h[i] = static_cast<float16_t>(dist(gen));
  }

  // Calculate reference D on host
  std::vector<float> Dref_h(D_size);
  gemm_host_batch(A_h, B_h, Dref_h, M, N, K, nBatch,
                  LDA, LDB, LDD,
                  batchStrideA, batchStrideB, batchStrideD);

  // Make and populate device buffers
  float16_t *A_d, *B_d;
  float *D_d;
  HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(float16_t)));
  HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(float16_t)));
  HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(float)));
  HIP_CHECK(hipMemcpy(A_d, A_h.data(), A_size * sizeof(float16_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h.data(), B_size * sizeof(float16_t), hipMemcpyHostToDevice));

  // Launch GEMM kernel
  sgemm_32x32x32_batch<<<1, dim3(32, 2)>>>(A_d, B_d, D_d);
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
