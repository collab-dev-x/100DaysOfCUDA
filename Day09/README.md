# CUDA Memory Coalescing

Modern GPUs achieve high memory bandwidth by reading and writing to global memory in large, wide chunks. The GPU hardware is designed to service memory requests for a group of threads (a **warp**, typically 32 threads) as efficiently as possible.

**Memory coalescing** occurs when consecutive threads in a warp access consecutive memory locations. When this happens, the GPU can satisfy all 32 memory requests with a single, large memory transaction. This is the ideal scenario, as it makes the most efficient use of the available memory bandwidth.

**Non-coalesced** (or strided) access occurs when consecutive threads access memory locations that are far apart from each other. In this case, the GPU cannot group the requests into a single transaction. Instead, it must issue multiple, smaller, and less efficient memory transactions to satisfy the requests for the entire warp. This leads to wasted memory bandwidth and significantly slower kernel execution.

## Implemetation

The provided code benchmarks two kernels that perform the same amount of work but use different memory access patterns. Both kernels write to every element of a 2D matrix laid out in linear memory. The matrix is stored in **row-major** order, meaning elements of the same row are contiguous in memory.

### The Coalesced Kernel (`rowMajorKernel`)

This kernel demonstrates a perfectly coalesced access pattern.

```cpp
// Coalesced: row-major access
__global__ void rowMajorKernel(float* data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row index
    if (x < width && y < height) {
        int idx = y * width + x;    // contiguous in x-direction
        data[idx] = 1.0f;
    }
}
```

> **Note:** Coalesced memory access is does not equate row major access. Here row major access leads to coalesced memory access because the array is flattened in row major format.

## Performance Results

The measured execution times demonstrate the impact of memory access patterns on kernel performance. Replace the placeholders below with your actual timing results:

| Kernel                                 | Time (ms)             |
|----------------------------------------|-----------------------|
| Row-major (coalesced)                  | `0.150048 ms`        |
| Column-major (non-coalesced)           | `0.200704 ms`        |

**Placeholder Notice**: Insert the recorded times from your program output in place of `0.150048` and `0.200704`.
