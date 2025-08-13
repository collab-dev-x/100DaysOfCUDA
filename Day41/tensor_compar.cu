#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

#define CHECK_CUBLAS(call) \
    if ((call) != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

void fill_half_matrix(__half* mat, int size, float val) {
    for (int i = 0; i < size; ++i)
        mat[i] = __float2half(val);
}

void fill_matrix(float* mat, int size, float val){
    for (int i = 0; i < size; ++i)
        mat[i] = val;
}
int main() {
    int m = 1024, n = 1024, k = 1024;
    int lda = m, ldb = k, ldc = m;
    float alpha = 1.0f, beta = 0.0f;

    __half *d_A_16, *d_B_16, *d_C_16;
    float *d_A_32, *d_B_32, *d_C_32;

    CHECK_CUDA(cudaMalloc((void**)&d_A_16, m * k * sizeof(__half)));
    CHECK_CUDA(cudaMalloc((void**)&d_B_16, k * n * sizeof(__half)));
    CHECK_CUDA(cudaMalloc((void**)&d_C_16, m * n * sizeof(__half)));

    CHECK_CUDA(cudaMalloc((void**)&d_A_32, m * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B_32, k * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C_32, m * n * sizeof(float)));

    __half *h_A_16 = (__half*)malloc(m * k * sizeof(__half));
    __half *h_B_16 = (__half*)malloc(k * n * sizeof(__half));
    __half *h_C_16 = (__half*)malloc(m * n * sizeof(__half));

    float *h_A_32 = (float*)malloc(m * k * sizeof(float));
    float *h_B_32 = (float*)malloc(k * n * sizeof(float));
    float *h_C_32 = (float*)malloc(m * n * sizeof(float));

    fill_half_matrix(h_A_16, m * k, 1.0f);
    fill_half_matrix(h_B_16, k * n, 1.0f);
    
    fill_matrix(h_A_32, m * k, 1.0f);
    fill_matrix(h_B_32, k * n, 1.0f);

    CHECK_CUDA(cudaMemcpy(d_A_16, h_A_16, m * k * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_16, h_B_16, k * n * sizeof(__half), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_A_32, h_A_32, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_32, h_B_32, k * n * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A_16, CUDA_R_16F, lda, d_B_16, CUDA_R_16F, ldb, &beta, d_C_16, CUDA_R_16F, ldc, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A_16, CUDA_R_16F, lda, d_B_16, CUDA_R_16F, ldb, &beta, d_C_32, CUDA_R_32F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A_16, CUDA_R_16F, lda, d_B_16, CUDA_R_16F, ldb, &beta, d_C_32, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A_16, CUDA_R_16F, lda, d_B_16, CUDA_R_16F, ldb, &beta, d_C_16, CUDA_R_16F, ldc, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float _16f_ms;
    CHECK_CUDA(cudaEventElapsedTime(&_16f_ms, start, stop));

    float gflops = 2.0f * m * n * k / (_16f_ms * 1e6f);

    printf("Time: %.3f ms\n", _16f_ms);
    printf("Performance: %.2f GFLOPS\n", gflops);

    CHECK_CUDA(cudaEventRecord(start));
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A_16, CUDA_R_16F, lda, d_B_16, CUDA_R_16F, ldb, &beta, d_C_32, CUDA_R_32F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float _32f_ms;
    CHECK_CUDA(cudaEventElapsedTime(&_32f_ms, start, stop));

    gflops = 2.0f * m * n * k / (_32f_ms * 1e6f);

    printf("Time: %.3f ms\n", _32f_ms);
    printf("Performance: %.2f GFLOPS\n", gflops);


    CHECK_CUDA(cudaEventRecord(start));
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A_16, CUDA_R_16F, lda, d_B_16, CUDA_R_16F, ldb, &beta, d_C_32, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float _TF32f_ms;
    CHECK_CUDA(cudaEventElapsedTime(&_TF32f_ms, start, stop));

    gflops = 2.0f * m * n * k / (_TF32f_ms * 1e6f);

    printf("Time: %.3f ms\n", _TF32f_ms);
    printf("Performance: %.2f GFLOPS\n", gflops);


    CHECK_CUBLAS(cublasDestroy(handle));
    cudaFree(d_A_16); cudaFree(d_B_16); cudaFree(d_C_16);
    cudaFree(d_A_32); cudaFree(d_B_32); cudaFree(d_C_32);
    free(h_A_16); free(h_B_16), free(h_C_16);
    free(h_A_32); free(h_B_32), free(h_C_32);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
