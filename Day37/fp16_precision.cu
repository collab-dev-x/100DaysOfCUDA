#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <chrono>

__global__ void vectorAdd_FP32(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vectorAdd_FP16(__half* a, __half* b, __half* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

__global__ void convert_fp32_to_fp16(float* src, __half* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}


int main() {
    const int N = 1024 * 1024;
    const int bytes_fp32 = N * sizeof(float);
    const int bytes_fp16 = N * sizeof(__half);
    
    float* h_a = (float*)malloc(bytes_fp32);
    float* h_b = (float*)malloc(bytes_fp32);
    float* h_c_fp32 = (float*)malloc(bytes_fp32);
    __half* h_c_fp16 = (__half*)malloc(bytes_fp16);
    
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i) * 0.001f;
        h_b[i] = static_cast<float>(i) * 0.002f;
    }
    
    float *d_a_fp32, *d_b_fp32, *d_c_fp32;
    __half *d_a_fp16, *d_b_fp16, *d_c_fp16;
    
    cudaMalloc(&d_a_fp32, bytes_fp32);
    cudaMalloc(&d_b_fp32, bytes_fp32);
    cudaMalloc(&d_c_fp32, bytes_fp32);

    cudaMalloc(&d_a_fp16, bytes_fp16);
    cudaMalloc(&d_b_fp16, bytes_fp16);
    cudaMalloc(&d_c_fp16, bytes_fp16);
    
    cudaMemcpy(d_a_fp32, h_a, bytes_fp32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_fp32, h_b, bytes_fp32, cudaMemcpyHostToDevice);
    
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    
    convert_fp32_to_fp16<<<gridSize, blockSize>>>(d_a_fp32, d_a_fp16, N);
    convert_fp32_to_fp16<<<gridSize, blockSize>>>(d_b_fp32, d_b_fp16, N);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;
    
    std::cout << "Running Mixed Precision Vector Addition Demo\n";
    std::cout << "Vector size: " << N << " elements\n\n";
    
    cudaEventRecord(start);
    vectorAdd_FP32<<<gridSize, blockSize>>>(d_a_fp32, d_b_fp32, d_c_fp32, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    cudaMemcpy(h_c_fp32, d_c_fp32, bytes_fp32, cudaMemcpyDeviceToHost);
    std::cout << "FP32 computation time: " << elapsed_time << " ms\n";
    std::cout << "Memory usage (FP32): " << (3 * bytes_fp32) / (1024*1024) << " MB\n";
    
    cudaEventRecord(start);
    vectorAdd_FP16<<<gridSize, blockSize>>>(d_a_fp16, d_b_fp16, d_c_fp16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    cudaMemcpy(h_c_fp16, d_c_fp16, bytes_fp16, cudaMemcpyDeviceToHost);
    std::cout << "FP16 computation time: " << elapsed_time << " ms\n";
    std::cout << "Memory usage (FP16): " << (3 * bytes_fp16) / (1024*1024) << " MB\n";
    
    cudaFree(d_a_fp32); cudaFree(d_b_fp32); cudaFree(d_c_fp32);
    cudaFree(d_a_fp16); cudaFree(d_b_fp16); cudaFree(d_c_fp16);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_a);
    free(h_b);
    free(h_c_fp32);
    free(h_c_fp16);
    
    return 0;
}