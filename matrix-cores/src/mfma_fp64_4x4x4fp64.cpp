#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono> // For timing
#include "helper.hpp"

/*
This example code uses the mfma intrinsic __builtin_amdgcn_mfma_f64_4x4x4f64 to
compute a batch of 4 4x4x4 matrix multiplications.

Input:
  A : 4 x 4 x 4 doubles (four 4x4 matrices)
  B : 4 x 4 x 4 doubles (four 4x4 matrices)

Output:
  D : 4 x 4 x 4 doubles (four 4x4 matrices)
*/

constexpr int M = 4;
constexpr int N = 4;
constexpr int K = 4;
constexpr int nBatch = 4;
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

__global__ void dgemm_4x4x4_batch(const double *A, const double *B, double *D)
{
  double d = {0};

  int a_idx = LDA * threadIdx.x + threadIdx.z + batchStrideA * threadIdx.y;
  int b_idx = threadIdx.x + LDB * threadIdx.z + batchStrideB * threadIdx.y;

  const double a = A[a_idx];
  const double b = B[b_idx];

  for (int i = 0; i < compute_repetitions; ++i) {
    for (int j = 0; j < compute_repetitions; ++j) {
      d = __builtin_amdgcn_mfma_f64_4x4x4f64(a, b, d, 0, 0, 0);
    }
  }

  const int d_idx = threadIdx.x + threadIdx.y * batchStrideD + threadIdx.z * LDD;
  D[d_idx] = d;
}

int main() {
  if (!gpuArchCheck("gfx90a")) {
    std::cout << "mfma_f64_4x4x4f64 instruction only available on gfx90a or later."
              << std::endl;
    return -1;
  }

  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(-1, 1);

  // Make and populate some host matrices
  std::vector<double> A_h(A_size), B_h(B_size);
  for (int i = 0; i < A_h.size(); ++i) A_h[i] = dist(gen);
  for (int i = 0; i < B_h.size(); ++i) B_h[i] = dist(gen);

  // Calculate reference D on host
  std::vector<double> Dref_h(D_size);
  gemm_host_batch(A_h, B_h, Dref_h, M, N, K, nBatch, LDA, LDB, LDD,
                  batchStrideA, batchStrideB, batchStrideD);

  // Allocate device buffers
  double *A_d, *B_d, *D_d;
  HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(double)));
  HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(double)));
  HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(double)));

  // Copy data to device
  HIP_CHECK(hipMemcpy(A_d, A_h.data(), A_size * sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h.data(), B_size * sizeof(double), hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  auto overall_start = std::chrono::high_resolution_clock::now();
  double runtime = 0.0;
  int kernel_runs = 0;

  while (runtime < 5.0) {
    auto t1 = std::chrono::high_resolution_clock::now();
    dgemm_4x4x4_batch<<<dim3(128, 64, 64), dim3(4, 4, 4)>>>(A_d, B_d, D_d);
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
  std::vector<double> D_h(D_size);
  HIP_CHECK(hipMemcpy(D_h.data(), D_d, D_size * sizeof(double), hipMemcpyDeviceToHost));

  // Compute L2 error
  std::cout << "Sum of squared differences of host/device result matrices: "
            << compute_l2_error_batch(Dref_h, D_h, M, N, nBatch,
                                      LDD, LDD, batchStrideD, batchStrideD)
            << std::endl;

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(D_d));

  return 0;
}

