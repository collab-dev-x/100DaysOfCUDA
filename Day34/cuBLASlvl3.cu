#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEvent_t start1, stop1, start2, stop2;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    const int M = 256, N = 256, K = 256;
    const int batch_size = 8;
    const float alpha = 1.0f, beta = 0.5f;
    
    size_t matrix_size = M * N * sizeof(float);
    size_t batch_matrix_size = batch_size * M * N * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    float *d_batch_A, *d_batch_B, *d_batch_C;
    float **d_A_array, **d_B_array, **d_C_array;
    
    cudaMalloc(&d_A, matrix_size);
    cudaMalloc(&d_B, matrix_size);
    cudaMalloc(&d_C, matrix_size);
    cudaMalloc(&d_batch_A, batch_matrix_size);
    cudaMalloc(&d_batch_B, batch_matrix_size);
    cudaMalloc(&d_batch_C, batch_matrix_size);
    cudaMalloc(&d_A_array, batch_size * sizeof(float*));
    cudaMalloc(&d_B_array, batch_size * sizeof(float*));
    cudaMalloc(&d_C_array, batch_size * sizeof(float*));
    
    std::vector<float> h_data(M * N, 1.0f);
    cudaMemcpy(d_A, h_data.data(), matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_data.data(), matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_data.data(), matrix_size, cudaMemcpyHostToDevice);
    
    for(int i = 0; i < batch_size; i++) {
        cudaMemcpy(d_batch_A + i * M * N, h_data.data(), matrix_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_batch_B + i * M * N, h_data.data(), matrix_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_batch_C + i * M * N, h_data.data(), matrix_size, cudaMemcpyHostToDevice);
    }
    
    std::vector<float*> h_A_array(batch_size), h_B_array(batch_size), h_C_array(batch_size);
    for(int i = 0; i < batch_size; i++) {
        h_A_array[i] = d_batch_A + i * M * N;
        h_B_array[i] = d_batch_B + i * M * N;
        h_C_array[i] = d_batch_C + i * M * N;
    }
    cudaMemcpy(d_A_array, h_A_array.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_array, h_B_array.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_array, h_C_array.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice);
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M); //C = α × A × B + β × C

    float cuBLAS_Time1, cuBLAS_Time2;

    cudaEventRecord(start1);
    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, 
                       (const float**)d_A_array, M, (const float**)d_B_array, K, &beta, d_C_array, M, batch_size);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&cuBLAS_Time1, start1, stop1);
    cudaEventRecord(start2);
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, 
                              d_batch_A, M, M*N, d_batch_B, K, M*N, &beta, d_batch_C, M, M*N, batch_size);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&cuBLAS_Time2, start2, stop2);
    
    std::cout<<"CUBLAS Batched Execution Time: "<<cuBLAS_Time1<<std::endl;
    std::cout<<"CUBLAS Batched Strided Execution Time: "<<cuBLAS_Time2<<std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_batch_A);
    cudaFree(d_batch_B);
    cudaFree(d_batch_C);
    cudaFree(d_A_array);
    cudaFree(d_B_array);
    cudaFree(d_C_array);
    
    cublasDestroy(handle);
    
    return 0;
}