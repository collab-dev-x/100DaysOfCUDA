#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    // Problem size
    int N = 1 << 24;              // ~16 million elements
    size_t bytes = N * sizeof(float);

    // ----------------------------------------------------------------
    // 1) Unified Memory version
    // ----------------------------------------------------------------
    float *A_um, *B_um, *C_um;
    cudaMallocManaged(&A_um, bytes);
    cudaMallocManaged(&B_um, bytes);
    cudaMallocManaged(&C_um, bytes);

    // Initialize on host (automatically visible to GPU)
    for (int i = 0; i < N; ++i) {
        A_um[i] = static_cast<float>(i);
        B_um[i] = static_cast<float>(i) * 2.0f;
    }
    cudaDeviceSynchronize();  // ensure host writes are visible

    // Time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vectorAdd<<<(N + 255) / 256, 256>>>(A_um, B_um, C_um, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_um = 0.0f;
    cudaEventElapsedTime(&time_um, start, stop);
    std::cout << "[Unified Memory] Kernel time: " << time_um << " ms\n";

    // Cleanup Unified Memory
    cudaFree(A_um);
    cudaFree(B_um);
    cudaFree(C_um);

    // ----------------------------------------------------------------
    // 2) Explicit Memory Management version
    // ----------------------------------------------------------------
    // Host buffers
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i) * 2.0f;
    }

    // Device buffers
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Time host-to-device copy
    auto t0 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Time kernel
    cudaEventRecord(start);
    vectorAdd<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_kernel = 0.0f;
    cudaEventElapsedTime(&time_kernel, start, stop);

    // Time device-to-host copy
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    auto t2 = std::chrono::high_resolution_clock::now();

    // Report timings
    auto h2d_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto d2h_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    auto total_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();

    std::cout << "[Explicit Copy] H2D copy:    " << h2d_ms     << " ms\n";
    std::cout << "[Explicit Copy] Kernel time: " << time_kernel << " ms\n";
    std::cout << "[Explicit Copy] D2H copy:    " << d2h_ms     << " ms\n";
    std::cout << "[Explicit Copy] Total CUDA:  " << total_ms   << " ms\n";

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}