# CUDA Matrix Transposition

Matrix transposition is a fundamental operation in linear algebra where the rows and columns of a matrix are swapped. For a matrix \( A \), its transpose, denoted as \( A^T \), is formed such that \( A^T[i][j] = A[j][i] \).

![Matrix Transpose](https://upload.wikimedia.org/wikipedia/commons/e/e4/Matrix_transpose.gif)

This operation is essential in numerical algorithms, including solving systems of linear equations, certain Fourier transforms, and as a building block for complex algorithms like matrix-matrix multiplication, where it can assist in coalesced memory access. Sometimes called "corner turning," matrix transposition is ideal for massive parallelization on a GPU due to the independence of each element's calculation.

## Implementation Approaches

This project implements two CUDA kernels to demonstrate this challenge and its solution.

### The Simple Implementation

The naive implementation is straightforward: each thread handles one element of the matrix, reading it from the input matrix and writing it to its transposed position in the output matrix.

```cpp
__global__ void transposeNaive(float* in, float* out, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;  

    if (x < width && y < width) {
        out[x * width + y] = in[y * width + x];
    }
}
```

- **Memory Reads**: Reads from `in[y * width + x]` are coalesced. As `threadIdx.x` increments across a warp, threads access consecutive memory addresses (`in[y*width + 0]`, `in[y*width + 1]`, ...), which is efficient.
- **Memory Writes**: Writes to `out[x * width + y]` are strided. As `threadIdx.x` increments, threads write to memory locations separated by `width` elements (`out[0*width + y]`, `out[1*width + y]`, ...), making this the primary performance bottleneck.

### The Optimized Approach (transposeShared)

The optimized approach uses fast, on-chip shared memory to address the strided write problem by processing the matrix in smaller square tiles.

```cpp
__global__ void transposeShared(float* in, float* out, int width) {
    // +1 in the second dimension to avoid shared memory bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    if (x < width && y < width)
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    if (x < width && y < width)
        out[y * width + x] = tile[threadIdx.x][threadIdx.y];
}
```

The process for each thread block is:
1. **Coalesced Read**: Each thread block reads a `TILE_DIM x TILE_DIM` tile from global memory into shared memory, maintaining coalesced reads as in the naive kernel.
2. **Synchronize**: `__syncthreads()` ensures all data is loaded into the tile before proceeding.
3. **Transpose and Coalesced Write**: Threads read from the shared memory tile in a transposed pattern (`tile[threadIdx.x][threadIdx.y]`) and write to global memory with recalculated indices, ensuring coalesced writes.

## Performance Analysis: Coalescing is Key

| Operation             | transposeNaive       | transposeShared      |
|-----------------------|----------------------|----------------------|
| Global Memory Read    | Coalesced (Good)     | Coalesced (Good)     |
| Global Memory Write   | Strided (Bad)        | Coalesced (Good)     |

By using shared memory as a staging area, the optimized kernel performs two coalesced global memory operations instead of one coalesced and one strided operation. Since shared memory is significantly faster than global memory, the overhead of additional shared memory operations is negligible compared to the performance gain from eliminating strided writes.

## Shared Memory Bank Conflicts

The declaration `__shared__ float tile[TILE_DIM][TILE_DIM + 1];` includes a critical optimization to avoid shared memory bank conflicts. Shared memory is divided into 32 physical banks. If multiple threads in a warp access different data words in the same bank, a *bank conflict* occurs, serializing the accesses and reducing performance.

In `transposeShared`, when writing to global memory, threads read from `tile[threadIdx.x][threadIdx.y]`. For a warp, `threadIdx.y` is constant while `threadIdx.x` increments, meaning threads access different rows of the tile.

- **Without padding (`tile[32][32]`)**: Thread 0 accesses `tile[0][y]`, Thread 1 accesses `tile[1][y]`, etc. The address of `tile[i][y]` is `i * 32 + y`. For Threads 0 and 1, addresses are `y` and `32 + y`, falling into the same bank (`y % 32`), causing a conflict.
- **With padding (`tile[32][33]`)**: The address of `tile[i][y]` is `i * 33 + y`. For Threads 0 and 1, addresses are `y` and `33 + y`, falling into different banks (`y % 32` and `(y+1) % 32`), avoiding conflicts.

This padding ensures conflict-free row accesses, preserving shared memory's high speed.