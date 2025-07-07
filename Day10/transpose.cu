#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Matrix size N x N
#define TILE_DIM 32

__global__ void transposeNaive(float* in, float* out, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // Column index
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // Row index

    if (x < width && y < width) {
        out[x * width + y] = in[y * width + x];  // Transpose write
    }
}

__global__ void transposeShared(float* in, float* out, int width) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 avoids bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < width && y < width)
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < width && y < width)
        out[y * width + x] = tile[threadIdx.x][threadIdx.y];
}


void fillMatrix(float* mat, int width) {
    for (int i = 0; i < width * width; ++i)
        mat[i] = static_cast<float>(i);
}

void printMatrix(const float* mat, int width, int limit = 8) {
    for (int i = 0; i < limit; ++i) {
        for (int j = 0; j < limit; ++j)
            std::cout << mat[i * width + j] << "\t";
        std::cout << "\n";
    }
}

int main() {
    int size = N * N * sizeof(float);
    float *h_in = new float[N * N];
    float *h_out = new float[N * N];

    fillMatrix(h_in, N);

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 threads(32, 32);
    dim3 blocks(N / threads.x, N / threads.y);
    transposeNaive<<<blocks, threads>>>(d_in, d_out, N);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    std::cout << "Original:\n";
    printMatrix(h_in, N);
    std::cout << "\nTransposed:\n";
    printMatrix(h_out, N);

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}
