#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                        \
  do {                                                                           \
    cudaError_t err = call;                                                      \
    if (err != cudaSuccess) {                                                    \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " : "     \
                << cudaGetErrorString(err) << std::endl;                         \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

// Matrix dimensions
constexpr int WIDTH  = 1024;
constexpr int HEIGHT = 1024;

// Coalesced: row-major access
__global__ void rowMajorKernel(float* data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row index
    if (x < width && y < height) {
        int idx = y * width + x;    // contiguous in x-direction
        data[idx] = 1.0f;           // arbitrary write
    }
}

// Non-coalesced: column-major access
__global__ void colMajorKernel(float* data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row index
    if (x < width && y < height) {
        int idx = x * height + y;   // stride = height between consecutive x
        data[idx] = 2.0f;
    }
}

int main() {
    size_t numElements = size_t(WIDTH) * HEIGHT;
    size_t bytes = numElements * sizeof(float);

    // Allocate device buffer (we'll reuse it)
    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));

    dim3 block(32, 16); // 512 threads per block
    dim3 grid((WIDTH + block.x - 1) / block.x,
              (HEIGHT + block.y - 1) / block.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float timeRow = 0.0f, timeCol = 0.0f;

    // ——— Time the row-major (coalesced) kernel ———
    CHECK_CUDA(cudaEventRecord(start));
    rowMajorKernel<<<grid, block>>>(d_data, WIDTH, HEIGHT);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&timeRow, start, stop));

    // Clear (optional)
    CHECK_CUDA(cudaMemset(d_data, 0, bytes));

    // ——— Time the column-major (non-coalesced) kernel ———
    CHECK_CUDA(cudaEventRecord(start));
    colMajorKernel<<<grid, block>>>(d_data, WIDTH, HEIGHT);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&timeCol, start, stop));

    // Print results
    std::cout << "Row-major (coalesced) kernel time:    " << timeRow << " ms\n";
    std::cout << "Column-major (non-coalesced) kernel time: " << timeCol << " ms\n";

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}
