#include <iostream>
#include <cuda_runtime.h>

#define N 16  // total number of elements
#define TILE_SIZE 16  // size of tile = blockDim.x

__global__ void sharedMemoryCopy(float *input, float *output) {
    __shared__ float tile[TILE_SIZE];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {

        tile[threadIdx.x] = input[tid];

        __syncthreads();

        float modified = tile[threadIdx.x] * 2.0f;

        output[tid] = modified;
    }
}

// No synchronization (to demonstrate incorrect results)
__global__ void withoutSync(float *input, float *output) {
    __shared__ float tile[TILE_SIZE];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        tile[threadIdx.x] = input[tid];

        // no __syncthreads()

        float modified = tile[threadIdx.x] * 2.0f;
        output[tid] = modified;
    }
}

int main() {
    float h_input[N], h_output[N];

    for (int i = 0; i < N; i++) h_input[i] = i;

    float *d_input, *d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE);

    std::cout << "With __syncthreads():" << std::endl;
    sharedMemoryCopy<<<gridSize, blockSize>>>(d_input, d_output);
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) std::cout << h_output[i] << " ";
    std::cout << "\n";

    std::cout << "Without __syncthreads():" << std::endl;
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    withoutSync<<<gridSize, blockSize>>>(d_input, d_output);
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) std::cout << h_output[i] << " ";
    std::cout << "\n";

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
