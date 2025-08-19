#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <iostream>
#include <vector>

__global__ void int8_kernel(int8_t* input, int8_t* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Simple int8 arithmetic operations
        output[idx] = input[idx] * 2 + 10;
    }
}

__global__ void int4_kernel(int4* input, int4* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx].x = input[idx].x + 100;
        output[idx].y = input[idx].y + 200;
        output[idx].z = input[idx].z + 300;
        output[idx].w = input[idx].w + 400;
    }
}

int main() {
    const int n = 1024;

    printf("=== INT8 Example ===\n");

    std::vector<int8_t> h_int8_input(n);
    std::vector<int8_t> h_int8_output(n);

    for (int i = 0; i < n; i++) {
        h_int8_input[i] = static_cast<int8_t>(i % 100);
    }

    int8_t *d_int8_input, *d_int8_output;
    cudaMalloc(&d_int8_input, n * sizeof(int8_t));
    cudaMalloc(&d_int8_output, n * sizeof(int8_t));

    cudaMemcpy(d_int8_input, h_int8_input.data(), n * sizeof(int8_t), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    int8_kernel<<<gridSize, blockSize>>>(d_int8_input, d_int8_output, n);

    cudaMemcpy(h_int8_output.data(), d_int8_output, n * sizeof(int8_t), cudaMemcpyDeviceToHost);

    printf("First 10 int8 results: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", static_cast<int>(h_int8_output[i]));
    }
    printf("\n\n");

    printf("=== INT4 Example ===\n");
    
    const int n_int4 = 256;

    std::vector<int4> h_int4_input(n_int4);
    std::vector<int4> h_int4_output(n_int4);

    for (int i = 0; i < n_int4; i++) {
        h_int4_input[i] = make_int4(i, i+1, i+2, i+3);
    }

    int4 *d_int4_input, *d_int4_output;
    cudaMalloc(&d_int4_input, n_int4 * sizeof(int4));
    cudaMalloc(&d_int4_output, n_int4 * sizeof(int4));

    cudaMemcpy(d_int4_input, h_int4_input.data(), n_int4 * sizeof(int4), cudaMemcpyHostToDevice);

    dim3 gridSize_int4((n_int4 + blockSize.x - 1) / blockSize.x);
    int4_kernel<<<gridSize_int4, blockSize>>>(d_int4_input, d_int4_output, n_int4);

    cudaMemcpy(h_int4_output.data(), d_int4_output, n_int4 * sizeof(int4), cudaMemcpyDeviceToHost);

    printf("First 5 int4 results:\n");
    for (int i = 0; i < 5; i++) {
        printf("  [%d]: (%d, %d, %d, %d)\n", i, 
               h_int4_output[i].x, h_int4_output[i].y, 
               h_int4_output[i].z, h_int4_output[i].w);
    }
    printf("\n");

    cudaFree(d_int8_input);
    cudaFree(d_int8_output);
    cudaFree(d_int4_input);
    cudaFree(d_int4_output);
    return 0;
}
