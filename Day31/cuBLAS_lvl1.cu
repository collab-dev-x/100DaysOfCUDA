#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " << status << std::endl; \
        exit(1); \
    } \
} while(0)


__global__ void normSquaredKernel(const float* x, float* result, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (i < n) ? x[i] : 0.0f;
    sdata[tid] = val * val;
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

__global__ void axpyKernel(int n, float alpha, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i] + y[i];
    }
}

__global__ void swap_kernel(int n, float* x, int incx, float* y, int incy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x_pos = idx * incx;
    int y_pos = idx * incy;
    float temp = x[x_pos];
    x[x_pos] = y[y_pos];
    y[y_pos] = temp;

}

__global__ void scal_kernel(int n, float alpha, float* x, int incx) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x_pos = idx * incx;
    x[x_pos] *= alpha;

}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);

    float *h_x = (float*)malloc(size);
    float *h_y = (float*)malloc(size);
    float *h_result = (float*)malloc(size);
    float *h_x_swap = (float*)malloc(size);
    float *h_y_swap = (float*)malloc(size);
    float result;

    for (int i = 0; i < n; i++) {
        h_x[i] = static_cast<float>(i);
        h_y[i] = static_cast<float>(i*2);
    }

    float *d_x, *d_y, *d_result, alpha = 2.0f;
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
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::cout<<"Euclidean Norm\n";
    cudaEventRecord(start);
    normSquaredKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_x, d_result, n);
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernalTime, start, stop);

    std::cout<<"Kernal Execution Time: "<<kernalTime<<std::endl;
    std::cout<<"Kernal Result: "<<result<<std::endl;

    cudaEventRecord(start);
    CHECK_CUBLAS(cublasSnrm2(handle, n, d_x, 1, &result));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuBLAS_Time, start, stop);

    std::cout<<"cuBLAS Execution Time: "<<cuBLAS_Time<<std::endl;
    std::cout<<"cuBLAS Result: "<<result*result<<std::endl;

    std::cout<<"Vector Addition (alpha * x + y)\n";
    cudaEventRecord(start);
    axpyKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(n,alpha,d_x,d_y);
    cudaMemcpy(h_result, d_y, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernalTime, start, stop);

    std::cout<<"Kernal Execution Time: "<<kernalTime<<std::endl;
    std::cout<<"Kernal Result: "<<std::endl;
    for(int i=0;i<5;i++){
        std::cout<<h_result[i]<<" ";
    }
    std::cout<<"\n";

    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    CHECK_CUBLAS(cublasSaxpy(handle,n,&alpha,d_x,1,d_y,1));
    cudaMemcpy(h_result, d_y, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuBLAS_Time, start, stop);

    std::cout<<"cuBLAS Execution Time: "<<cuBLAS_Time<<std::endl;
    std::cout<<"cuBLAS Result: "<<std::endl;
    for(int i=0;i<5;i++){
        std::cout<<h_result[i]<<" ";
    }
    std::cout<<"\n";

    std::cout<<"Swap\n";
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    swap_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(n,d_x,1,d_y,1);
    cudaMemcpy(h_y_swap, d_y, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x_swap, d_x, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernalTime, start, stop);

    std::cout<<"Kernal Execution Time: "<<kernalTime<<std::endl;
    std::cout<<"Kernal Result: "<<std::endl;
    std::cout<<"Before Swapping: \n";
    for(int i=0;i<5;i++){
        std::cout<<h_x[i]<<" "<<h_y[i]<<"\n";
    }
    std::cout<<"After Swapping: \n";
    for(int i=0;i<5;i++){
        std::cout<<h_x_swap[i]<<" "<<h_y_swap[i]<<"\n";
    }

    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    CHECK_CUBLAS(cublasSswap(handle,n,d_x,1,d_y,1));
    cudaMemcpy(h_y_swap, d_y, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x_swap, d_x, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuBLAS_Time, start, stop);

    std::cout<<"cuBLAS Execution Time: "<<cuBLAS_Time<<std::endl;
    std::cout<<"cuBLAS Result: "<<std::endl;
    std::cout<<"Before Swapping: \n";
    for(int i=0;i<5;i++){
        std::cout<<h_x[i]<<" "<<h_y[i]<<"\n";
    }
    std::cout<<"After Swapping: \n";
    for(int i=0;i<5;i++){
        std::cout<<h_x_swap[i]<<" "<<h_y_swap[i]<<"\n";
    }

    std::cout<<"Scaling\n";
    cudaEventRecord(start);
    scal_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(n,alpha,d_x,1);
    cudaMemcpy(h_result, d_x, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernalTime, start, stop);

    std::cout<<"Kernal Execution Time: "<<kernalTime<<std::endl;
    std::cout<<"Kernal Result: "<<std::endl;
    for(int i=0;i<5;i++){
        std::cout<<h_result[i]<<" ";
    }
    std::cout<<"\n";

    cudaEventRecord(start);
    CHECK_CUBLAS(cublasSscal(handle,n,&alpha,d_x,1));
    cudaMemcpy(h_result, d_x, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuBLAS_Time, start, stop);

    std::cout<<"cuBLAS Execution Time: "<<cuBLAS_Time<<std::endl;
    std::cout<<"cuBLAS Result: "<<std::endl;
    for(int i=0;i<5;i++){
        std::cout<<h_result[i]<<" ";
    }
    std::cout<<"\n";

    free(h_x);
    free(h_y);

    
    return 0;
}