#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// y = αAx + β*y
__global__ void matVecMulKernel(const float* A, const float* x, float* y, float alpha, float beta, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m) {
        float dot = 0.0f;
        for (int col = 0; col < n; ++col) {
            dot += A[row * n + col] * x[col]; 
        }
        y[row] = alpha * dot + beta * y[row];
    }

}

int main() {
    const int m = 10;
    const int n = 8;

    const int num_mat = m * n;
    float h_a[num_mat];

    for(int i=0;i<num_mat;i++){
        h_a[i]=static_cast<float>(i%100);
        std::cout<<h_a[i]<<" ";
    }
    std::cout<<"\n";

    float h_x[n];
    for(int i=0;i<n;i++){
        h_x[i]=1.0f;
    }
    float h_y[m];
    for(int i=0;i<m;i++){
        h_y[i]=1.0f;
    }

    float h_y_result[m];
    float alpha = 2.0f;
    float beta = 0.6f;

    float *d_a, *d_x, *d_y;
    cudaMalloc(&d_a, m * n * sizeof(float));
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, m * sizeof(float));

    cudaMemcpy(d_a, h_a, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, m * sizeof(float), cudaMemcpyHostToDevice);

    float kernalTime, cuBLAS_Time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cublasHandle_t handle;
    cublasCreate(&handle);

    int threadsPerBlock = 256;
    int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    cublasSgemv(handle, CUBLAS_OP_T, n, m, &alpha, d_a, m, d_x, 1, &beta, d_y, 1);
    cudaMemcpy(h_y_result, d_y, m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuBLAS_Time, start, stop);

    std::cout<<"cuBLAS Execution Time: "<<cuBLAS_Time<<std::endl;
    std::cout<<"cuBLAS Result: "<<std::endl;
    for(int i=0;i<5;i++){
        std::cout<<h_y_result[i]<<" ";
    }
    std::cout<<"\n";

    cudaMemcpy(d_y, h_y, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    matVecMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_x, d_y, alpha, beta, m, n);
    cudaMemcpy(h_y_result, d_y, m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernalTime, start, stop);

    std::cout<<"Kernal Execution Time: "<<kernalTime<<std::endl;
    std::cout<<"Kernal Result: "<<std::endl;
    for(int i=0;i<5;i++){
        std::cout<<h_y_result[i]<<" ";
    }
    std::cout<<"\n";


    


    cudaFree(d_a);
    cudaFree(d_x);
    cudaFree(d_y);
    delete [] h_a;
    delete [] h_x;
    delete [] h_y;

    return 0;
}