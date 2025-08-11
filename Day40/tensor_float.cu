#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>

__global__ void matmul_fp32(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 1024;  
    const int size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_fp32 = (float*)malloc(size);
    float *h_C_tf32 = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C_fp32, *d_C_tf32;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C_fp32, size);
    cudaMalloc(&d_C_tf32, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f, beta = 0.0f;
    
    printf("=== CUDA Tensor Float vs Float32 Comparison ===\n\n");

    printf("1. **Float32 (FP32) Characteristics:**\n");
    printf("   - Precision: 23-bit mantissa + 1 implicit bit\n");
    printf("   - Range: ~1.4e-45 to ~3.4e38\n");
    printf("   - Memory: 32 bits per number\n");
    printf("   - Accuracy: Full precision\n\n");
    
    auto start_fp32 = std::chrono::high_resolution_clock::now();

    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C_fp32, N);
    cudaDeviceSynchronize();
    
    auto end_fp32 = std::chrono::high_resolution_clock::now();
    auto duration_fp32 = std::chrono::duration_cast<std::chrono::microseconds>(end_fp32 - start_fp32);

    printf("2. **Tensor Float (TF32) Characteristics:**\n");
    printf("   - Precision: 10-bit mantissa (reduced from 23-bit)\n");
    printf("   - Range: Same as FP32 (8-bit exponent)\n");
    printf("   - Memory: Still 32 bits, but reduced internal precision\n");
    printf("   - Accuracy: Slightly reduced for speed\n\n");
    
    auto start_tf32 = std::chrono::high_resolution_clock::now();

    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C_tf32, N);
    cudaDeviceSynchronize();
    
    auto end_tf32 = std::chrono::high_resolution_clock::now();
    auto duration_tf32 = std::chrono::duration_cast<std::chrono::microseconds>(end_tf32 - start_tf32);

    cudaMemcpy(h_C_fp32, d_C_fp32, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_tf32, d_C_tf32, size, cudaMemcpyDeviceToHost);

    printf("3. **Performance Results:**\n");

    printf("   - FP32 Time: %lld microseconds\n", duration_fp32.count());
    printf("   - TF32 Time: %lld microseconds\n", duration_tf32.count());

    printf("   - Speed improvement: %.2fx\n", 
           (float)duration_fp32.count() / duration_tf32.count());
    printf("\n");

    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int sample_size = std::min(100, N*N);
    
    for (int i = 0; i < sample_size; i++) {
        float diff = fabs(h_C_fp32[i] - h_C_tf32[i]);
        max_diff = std::max(max_diff, diff);
        avg_diff += diff;
    }
    avg_diff /= sample_size;
    
    printf("4. **Accuracy Comparison (Sample of %d elements):**\n", sample_size);
    printf("   - Maximum difference: %.8f\n", max_diff);
    printf("   - Average difference: %.8f\n", avg_diff);
    printf("   - Relative error: %.6f%%\n", (avg_diff / h_C_fp32[0]) * 100);
    printf("\n");

    printf("5. **Key Differences Summary:**\n");
    printf("   |-----------------|-------------|-------------|\n");
    printf("   | Aspect          | Float32     | TensorFloat |\n");
    printf("   |-----------------|-------------|-------------|\n");
    printf("   | Mantissa bits   | 23          | 10          |\n");
    printf("   | Exponent bits   | 8           | 8           |\n");
    printf("   | Total bits      | 32          | 32          |\n");
    printf("   | Precision       | High        | Reduced     |\n");
    printf("   | Speed           | Standard    | ~1.5-2x     |\n");
    printf("   | Hardware        | All GPUs    | Ampere+     |\n");
    printf("   |-----------------|-------------|-------------|\n");
    printf("\n");
    
    printf("6. **When to use each:**\n");
    printf("   - **Float32**: When you need maximum precision\n");
    printf("   - **TensorFloat**: For deep learning where slight precision\n");
    printf("     loss is acceptable for significant speed gains\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_fp32);
    cudaFree(d_C_tf32);
    free(h_A);
    free(h_B);
    free(h_C_fp32);
    free(h_C_tf32);
    cublasDestroy(handle);
    
    return 0;
}
