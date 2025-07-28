#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>

__global__ void dotProductKernel(const float* x, const float* y, float* result, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? x[i] * y[i] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

int main() {
    int n = 10000000;
    size_t size = n * sizeof(float);

    float *h_x = (float*)malloc(size);
    float *h_y = (float*)malloc(size);
    float result;

    for (int i = 0; i < n; i++) {
        h_x[i] = static_cast<float>(i);
        h_y[i] = static_cast<float>(i*2);
    }

    float *d_x, *d_y, *d_result;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    float kernalTime, cuBLAS_Time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    dotProductKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_x, d_y, d_result, n);
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernalTime, start, stop);

    std::cout<<"Kernal Execution Time: "<<kernalTime<<std::endl;
    std::cout<<"Kernal Result: "<<result<<std::endl;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEventRecord(start);
    cublasSdot(handle, n, d_x, 1, d_y, 1, &result);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuBLAS_Time, start, stop);

    std::cout<<"cuBLAS Execution Time: "<<cuBLAS_Time<<std::endl;
    std::cout<<"cuBLAS Result: "<<result<<std::endl;

    free(h_x);
    free(h_y);


    return 0;
}