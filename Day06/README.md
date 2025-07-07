# Shared Memory in CUDA

## Overview

In the CUDA programming model, GPUs have a hierarchy of different memory spaces, each with different characteristics in terms of scope, size, and speed.

---

## What Is Shared Memory?  
- **On-chip memory** accessible by all threads within a thread block.  
- **Low latency** and **high bandwidth** compared to global device memory.  
- **Scoped** to a block: data stored in shared memory by one block is invisible to other blocks.  

---

## Why Use Shared Memory?  
1. **Data Reuse**  
   - Load data once from global memory into shared memory; reuse it many times within the block.  
   - Reduces costly global memory accesses, improving throughput.  
2. **Thread Cooperation**  
   - Threads in the same block can collaboratively load and process data.  
   - Enables tiling and other optimization patterns (e.g., matrix multiplication tiling).  
3. **Performance Gains**  
   - Lower access latency (tens of cycles vs. hundreds for global memory).  
   - Higher effective memory bandwidth when accesses are coalesced into shared memory.  

---

## How to Use Shared Memory?  

A common and effective pattern is:

1. **Stage Data**: Each thread in a block loads a piece of data from global memory into shared memory. This is a coordinated, one-time read from slow memory.
2. **Process**: All threads in the block perform computations, repeatedly accessing the data from the fast shared memory, avoiding multiple high-latency trips to global memory.
3. **Write Back**: Once the computation is complete, the threads write the final results from shared memory back to global memory.

This pattern, known as "tiling," significantly improves performance by maximizing data reuse and leveraging the low latency of shared memory.

## Code Walkthrough

The code demonstrates shared memory usage by performing a simple operation: multiplying each element of an array by 2. It includes two implementations: one correct with synchronization and one incorrect without it.

### The `sharedMemoryCopy` Kernel (Correct Implementation)

This kernel demonstrates the proper use of shared memory with synchronization.

```cpp
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
```

**Step-by-step breakdown**:
1. `__shared__ float tile[TILE_SIZE];`: Declares an array named `tile` in shared memory. Each thread block gets its own private copy of this array.
2. `int tid = ...`: Calculates a unique global ID for each thread.
3. `tile[threadIdx.x] = input[tid];`: Each thread reads one element from the input array in slow global memory and writes it to its corresponding position in the fast shared memory tile.
4. `__syncthreads();`: A critical barrier that forces every thread in the block to wait until all threads have reached this point, ensuring the entire `tile` array is filled with data before proceeding.
5. `float modified = tile[threadIdx.x] * 2.0f;`: Each thread reads its value from the fast `tile` array, performs the multiplication, and writes the result to the output array in global memory.

### The `withoutSync` Kernel (Incorrect Implementation)

This kernel is identical to the first but omits the synchronization step.

```cpp
__global__ void withoutSync(float *input, float *output) {
    __shared__ float tile[TILE_SIZE];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        tile[threadIdx.x] = input[tid];

        // NO __syncthreads() HERE!

        float modified = tile[threadIdx.x] * 2.0f;
        output[tid] = modified;
    }
}
```

This code demonstrates the consequences of failing to synchronize.

## The Critical Role of `__syncthreads()`

`__syncthreads()` is a synchronization primitive that ensures order within a thread block. Without it, a **race condition** occurs.

**What is a race condition?**

A race condition occurs when the outcome of an operation depends on the unpredictable sequence or timing of threads. In CUDA, threads within a block do not execute in a guaranteed order, and some threads may run faster than others.

In the `withoutSync` kernel:
- A "fast" thread might execute `tile[threadIdx.x] = input[tid]` and immediately proceed to read `tile[threadIdx.x]`.
- **The Problem**: There is no guarantee that the write operation has completed and is visible to the read operation. The thread might read an old, uninitialized "garbage" value.

`__syncthreads()` prevents this race condition by acting as a barrier, ensuring all write operations to shared memory are complete before any read operations begin.
