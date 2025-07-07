# Matrix Multiplication using Shared Memory Tiling

## 1. The Problem with Naive Matrix Multiplication

A naive GPU matrix multiplication assigns one thread to compute one element of the output matrix `C`. Below is an example of such a kernel:

```cpp
// Naive Kernel: One thread computes one element of C
__global__ void matMulBasic(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float val = 0.0f;
        for (int i = 0; i < n; ++i) {
            val += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = val;
    }
}
```

### Performance Bottleneck: Global Memory Access

The naive approach is inefficient due to:

- **High Latency**: Global memory (arrays `d_A`, `d_B`, `d_C` in GPU DRAM) is slow, taking hundreds of clock cycles per access.
- **Redundant Reads**: Threads in the same row or column redundantly read the same elements from matrices `A` and `B`.
- **Memory Bandwidth Saturation**: For an `N x N` matrix, the naive kernel performs `2 * N^3` global memory reads. For a 1024x1024 matrix, this translates to over 2 billion reads, saturating the memory bus and causing compute cores to idle.

The core issue is the lack of data reuse from global memory.

## 2. The Solution: Shared Memory Tiling

The CUDA memory hierarchy provides:

- **Global Memory**: Large, slow, persistent across kernel launches.
- **Shared Memory**: Small (~48-96 KB per SM), fast (on-chip), scoped to a thread block.
- **Registers**: Very fast, private to each thread.

The tiling strategy leverages shared memory to minimize global memory accesses by:

### Why is this Faster?

Tiling reduces global memory reads from `2 * N^3` to approximately `2 * N^3 / TILE_SIZE`. For a `TILE_SIZE` of 32, this results in a ~32x reduction in global memory accesses, significantly boosting performance.

## 3. Code Walkthrough: `matMulTiled` Kernel

Below is the tiled matrix multiplication kernel:

```cpp
__global__ void matMulTiled(float* A, float* B, float* C, int n) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float val = 0.0f;
    for (int t = 0; t < n / TILE_SIZE; ++t) {
        tileA[threadIdx.y][threadIdx.x] = A[row * n + (t * TILE_SIZE + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; ++i) {
            val += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < n) {
        C[row * n + col] = val;
    }
}
```

### Key Steps

1. **Shared Memory Tiles**: Declare `tileA` and `tileB` as shared memory arrays, sized to match block dimensions.
2. **Thread Position**: Compute the thread's target element in `C` using `blockIdx` and `threadIdx`.
3. **Tile Loop**: Iterate over tiles of `A` and `B` needed for one tile of `C`.
4. **Cooperative Loading**: Each thread loads one element from `A` and `B` into shared memory.
5. **Synchronization**: Use `__syncthreads()` to ensure all threads complete loading before computation.
6. **Shared Memory Computation**: Perform sub-matrix multiplication using fast shared memory.
7. **Second Synchronization**: Ensure computations are complete before loading new tiles.
8. **Final Write**: Write the accumulated result to `C` in global memory.