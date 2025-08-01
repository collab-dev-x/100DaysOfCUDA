#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    const int m = 10;
    const int n = 8;

    const int num_mat = m * n;
    float h_a[num_mat];

    // for(int i=0;i<num_mat;i++){
    //     h_a[i]=static_cast<float>(i%100);
    //     std::cout<<h_a[i]<<" ";
    // }
    // std::cout<<"\n";

    float h_x[n];
    for(int i=0;i<n;i++){
        h_x[i]=1.0f;
    }
    float h_y[m];
    for(int i=0;i<m;i++){
        h_y[i]=1.0f;
    }

    float h_y_result[m];
    float h_a_result[m*n];
    float h_x_result[n];
    float alpha = 2.0f;

    float *d_a, *d_x, *d_y;
    cudaMalloc(&d_a, m * n * sizeof(float));
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, m * sizeof(float));

    cudaMemcpy(d_a, h_a, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, m * sizeof(float), cudaMemcpyHostToDevice);

    float cuBLAS_Time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEventRecord(start);
    cublasSger(handle, m, n, &alpha, d_x, 1, d_y, 1, d_a, m);
    cudaMemcpy(h_a_result, d_a, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuBLAS_Time, start, stop);

    std::cout<<"General Rank1 Update Execution Time: "<<cuBLAS_Time<<std::endl;
    std::cout<<"A = alpha * x * y^T + A, Result: "<<std::endl;
    for(int i=0;i<5;i++){
        std::cout<<h_a_result[i]<<" ";
    }
    std::cout<<"\n";

    cudaMemcpy(d_a, h_a, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    cublasStrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, d_a, n, d_x, 1);
    cudaMemcpy(h_x_result, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuBLAS_Time, start, stop);

    std::cout<<"CUBLAS STRMV (Triangular Matrix-Vector Multiplication) Execution Time: "<<cuBLAS_Time<<std::endl;
    std::cout<<"x = A * x (A is upper triangular), Result: "<<std::endl;
    for(int i=0;i<5;i++){
        std::cout<<h_x_result[i]<<" ";
    }
    std::cout<<"\n";
    
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    cublasStrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, d_a, n, d_x, 1);
    cudaMemcpy(h_x_result, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuBLAS_Time, start, stop);

    std::cout<<"CUBLAS STRSV (Triangular System Solve) Execution Time: "<<cuBLAS_Time<<std::endl;
    std::cout<<"A * x = b for x (A is lower triangular), Result: "<<std::endl;
    for(int i=0;i<5;i++){
        std::cout<<h_x_result[i]<<" ";
    }
    std::cout<<"\n";

    cudaFree(d_a);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}