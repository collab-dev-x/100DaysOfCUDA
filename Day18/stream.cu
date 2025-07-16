#include <cuda_runtime.h>
#include <iostream>

#define N (1 << 24)
#define BLOCK_SIZE 256

__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

void check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A1, *d_B1, *d_C1;
    float *d_A2, *d_B2, *d_C2;

    size_t size = N * sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f; h_B[i] = 2.0f;
    }

    cudaMalloc(&d_A1, size); cudaMalloc(&d_B1, size); cudaMalloc(&d_C1, size);
    cudaMalloc(&d_A2, size); cudaMalloc(&d_B2, size); cudaMalloc(&d_C2, size);

    // Copy input to both sets
    cudaMemcpy(d_A1, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B1, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A2, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2, h_B, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float elapsed = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- Sequential
    cudaEventRecord(start);
    vectorAdd<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A1, d_B1, d_C1, N);
    vectorAdd<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A2, d_B2, d_C2, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    std::cout << "Sequential execution: " << elapsed << " ms\n";

    // --- Concurrent with Streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEventRecord(start);
    vectorAdd<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream1>>>(d_A1, d_B1, d_C1, N);
    vectorAdd<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream2>>>(d_A2, d_B2, d_C2, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    std::cout << "Concurrent execution (streams): " << elapsed << " ms\n";

    // Cleanup
    cudaFree(d_A1); cudaFree(d_B1); cudaFree(d_C1);
    cudaFree(d_A2); cudaFree(d_B2); cudaFree(d_C2);
    free(h_A); free(h_B); free(h_C);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
