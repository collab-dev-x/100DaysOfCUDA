#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    const int M = 3, N = 5;
    const float alpha = 1.0f, beta = 0.5f;

    size_t matrix_A_size = M * N * sizeof(float);
    size_t matrix_B_size = M * N * sizeof(float);
    size_t matrix_C_size = M * M * sizeof(float);
    
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, matrix_A_size);
    cudaMalloc(&d_B, matrix_B_size);
    cudaMalloc(&d_C, matrix_C_size);
    
    std::vector<float> h_A_data(M * N, 1.0f);
    std::vector<float> h_B_data(M * N, 2.0f);
    std::vector<float> h_C_data(M * M, 0.0f);
    std::vector<float> h_C_result(M * M);
    
    cudaMemcpy(d_A, h_A_data.data(), matrix_A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_data.data(), matrix_B_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C_data.data(), matrix_C_size, cudaMemcpyHostToDevice);

    float cuBLAS_Time1;

    cudaEventRecord(start1);
    cublasSsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, M, N, &alpha, d_A, M, d_B, M, &beta, d_C, M);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&cuBLAS_Time1, start1, stop1);
    cudaMemcpy(h_C_result.data(), d_C, matrix_C_size, cudaMemcpyDeviceToHost);
    
    std::cout<<"CUBLAS Symmetric Matrix Multiplication Execution Time: "<<cuBLAS_Time1<<std::endl;

    for(int i=0;i<M*M;i++){
        std::cout<<h_C_result[i]<<" ";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cublasDestroy(handle);
    
    return 0;
}