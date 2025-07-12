#include <iostream>
#include <cuda_runtime.h>

#define MAX_TILE_SIZE 32  // max tile size to prevent overuse of shared memory

// CUDA kernel with dynamic tile size
__global__ void matrixMultiplySharedDynamic(
    float* A, float* B, float* C,
    int M, int N, int K, int tileSize)
{
    // Allocate dynamic shared memory
    extern __shared__ float shared[];

    float* As = shared;                                // first tile for A
    float* Bs = (float*)&As[tileSize * tileSize];      // second tile for B

    int row = blockIdx.y * tileSize + threadIdx.y;
    int col = blockIdx.x * tileSize + threadIdx.x;

    float sum = 0.0f;

    // Iterate over tiles
    for (int t = 0; t < (K + tileSize - 1) / tileSize; ++t) {
        int tiledRow = row;
        int tiledCol = t * tileSize + threadIdx.x;

        // Load A tile to shared memory
        if (tiledRow < M && tiledCol < K)
            As[threadIdx.y * tileSize + threadIdx.x] = A[tiledRow * K + tiledCol];
        else
            As[threadIdx.y * tileSize + threadIdx.x] = 0.0f;

        tiledRow = t * tileSize + threadIdx.y;
        tiledCol = col;

        // Load B tile to shared memory
        if (tiledRow < K && tiledCol < N)
            Bs[threadIdx.y * tileSize + threadIdx.x] = B[tiledRow * N + tiledCol];
        else
            Bs[threadIdx.y * tileSize + threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial sum
        for (int i = 0; i < tileSize; ++i) {
            sum += As[threadIdx.y * tileSize + i] * Bs[i * tileSize + threadIdx.x];
        }

        __syncthreads();  // wait before overwriting shared memory
    }

    // Write result
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Host code to run the kernel
void runMatrixMultiply(int M, int N, int K, int tileSize) {
    if (tileSize > MAX_TILE_SIZE) {
        std::cerr << "Tile size too large!\n";
        return;
    }

    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 2.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

    dim3 block(tileSize, tileSize);
    dim3 grid((N + tileSize - 1) / tileSize, (M + tileSize - 1) / tileSize);
    size_t sharedSize = 2 * tileSize * tileSize * sizeof(float);  // for A and B

    matrixMultiplySharedDynamic<<<grid, block, sharedSize>>>(d_A, d_B, d_C, M, N, K, tileSize);
    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);

    std::cout << "Sample result C[0]: " << h_C[0] << std::endl;

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Main
int main() {
    int M = 128, N = 128, K = 128;
    int tileSize = 16;  // you can test with 8, 16, 32 (if resources allow)

    runMatrixMultiply(M, N, K, tileSize);
    return 0;
}
