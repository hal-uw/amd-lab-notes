#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono> // For timing
#include "helper.hpp"

/*
This example code uses the mfma intrinsic __builtin_amdgcn_mfma_i32_16x16x16i8 to
compute a 16x16x16 matrix multiplication.

Input:
  A : 16 x 16 int8s (a 16x16 matrix)
  B : 16 x 16 int8s (a 16x16 matrix)

Output:
  D : 16 x 16 int32s (a 16x16 matrix)
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

__global__ void igemm_16x16x16(const int8_t* A, const int8_t* B, int32_t* D)
{
#if __gfx90a__ || __gfx908__
  using int32x4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
  int32x4 d = {0}; // Zero out 4 vanilla VGPRs

  int8_t a[4];
  int8_t b[4];
  for (int i = 0; i < 4; ++i) {
    const int a_idx = threadIdx.x * LDA + i + threadIdx.y * 4;
    const int b_idx = threadIdx.x + i * LDB + threadIdx.y * LDB * 4;
    a[i] = A[a_idx];
    b[i] = B[b_idx];
  }

  // Wrap the builtin_amdgcn_mfma with nested loops
  for (int i = 0; i < compute_repetitions; ++i) {
    for (int j = 0; j < compute_repetitions; ++j) {
      d = __builtin_amdgcn_mfma_i32_16x16x16i8(*reinterpret_cast<int32_t*>(a), *reinterpret_cast<int32_t*>(b), d, 0, 0, 0);
    }
  }

  for (int i = 0; i < 4; ++i) {
    const int d_idx = threadIdx.x + i * LDD + threadIdx.y * 4 * LDD;
    D[d_idx] = d[i];
  }
#endif
}

int main() {
  if (!gpuArchCheck("gfx90a") && !gpuArchCheck("gfx908")) {
    std::cout << "mfma_i32_16x16x16i8 instruction only available on gfx908 or later." << std::endl;
    return -1;
  }

  std::mt19937 gen(0);
  std::uniform_int_distribution<int8_t> dist(-100, 100);

  // Make and populate host matrices
  std::vector<int8_t> A_h(A_size), B_h(B_size);
  for (int i = 0; i < A_h.size(); ++i) A_h[i] = dist(gen);
  for (int i = 0; i < B_h.size(); ++i) B_h[i] = dist(gen);

  // Compute reference result on host
  std::vector<int32_t> Dref_h(D_size);
  gemm_host(A_h, B_h, Dref_h, M, N, K, LDA, LDB, LDD);

  // Allocate device buffers
  int8_t *A_d, *B_d;
  int32_t *D_d;
  HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(int8_t)));
  HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(int8_t)));
  HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(int32_t)));

  // Copy data to device
  HIP_CHECK(hipMemcpy(A_d, A_h.data(), A_size * sizeof(int8_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h.data(), B_size * sizeof(int8_t), hipMemcpyHostToDevice));

  // Measure kernel execution time and track total runs
  //dim3 gridDim(128, 64, 64); 
  //dim3 threadsPerBlock(16, 4);
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  auto overall_start = std::chrono::high_resolution_clock::now();
  double runtime = 0.0;
  int kernel_runs = 0;

  while (runtime < 5.0) {
    auto t1 = std::chrono::high_resolution_clock::now();
    igemm_16x16x16<<<dim3(128, 64, 64), dim3(16, 4)>>>(A_d, B_d, D_d);
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
  std::vector<int32_t> D_h(D_size);
  HIP_CHECK(hipMemcpy(D_h.data(), D_d, D_size * sizeof(int32_t), hipMemcpyDeviceToHost));

  std::cout << "Sum of squared differences of host/device result matrices: "
            << compute_l2_error(Dref_h, D_h, M, N, LDD, LDD)
            << std::endl;

  std::cout << "Measured runtime: " << overall_runtime << " seconds" << std::endl;

  // Free device memory
  HIP_CHECK(hipFree(D_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(A_d));

  return 0;
}

