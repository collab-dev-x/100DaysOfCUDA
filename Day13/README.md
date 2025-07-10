# CUDA Pitched Memory


## 1. The Problem: Memory Alignment and Coalescing

GPUs achieve their massive parallelism by having thousands of threads. However, their memory system is most efficient when these threads access memory in a specific, predictable pattern.

### What is Memory Alignment?

Modern computer memory is accessed in chunks, not byte by byte. The GPU's memory controller fetches data in segments (e.g., 32, 64, or 128 bytes). An access is *aligned* if the memory address being requested is a multiple of the segment size.

- **Aligned Access (Good):** A request for 128 bytes starting at address 0x1000 is efficient.
- **Unaligned Access (Bad):** A request for 128 bytes starting at address 0x1004 is inefficient. The hardware might need to perform two memory transactions and discard the unwanted bytes, wasting bandwidth.

### What is Coalesced Memory Access?(Also discussed before)

In CUDA, threads are grouped into warps (typically 32 threads). A memory access is *coalesced* when all threads in a warp access a contiguous, aligned block of memory. This allows the GPU to satisfy the memory requests of all 32 threads in a single memory transaction, maximizing memory bandwidth.

If the threads access scattered memory locations, the GPU must issue multiple memory transactions, leading to significant performance degradation. This is known as *uncoalesced* or *non-coalesced* access.

### The 2D Array Dilemma

Consider a 2D matrix (or image) stored linearly in memory. We typically calculate an element's index as `idx = row * width + col`.

Now, imagine a `float` matrix with a width of 1023 (as in our example):
- The size of one row is `1023 * sizeof(float) = 1023 * 4 = 4092` bytes.
- The address of `matrix[0][0]` is, let's say, `0x0000`. This is perfectly aligned.
- The address of `matrix[1][0]` is `0x0000 + 4092`.
- The number 4092 is *not* a multiple of the typical GPU alignment requirement (e.g., 256 or 512 bytes). This means that the start of Row 1 is not aligned. When a warp of threads tries to access the beginning of Row 1, Row 2, etc., it results in an unaligned memory access, hurting performance.

This is the exact problem that pitched memory solves.

## 2. The Solution: Pitched Memory

### What is Pitch?

*Pitch* is the width of a 2D array in bytes, including any padding added by the CUDA driver to ensure proper alignment. It is the number of bytes you must step forward in memory to get from the start of one row to the start of the next.

Let's visualize this:

**Non-Pitched Memory (Potentially Unaligned):**

```
Row 0: [D][D][D]...[D]
Row 1:                [D][D][D]...[D]
Row 2:                               [D][D][D]...[D]
```

(Where each new row starts immediately after the previous one)

**Pitched Memory (Guaranteed Alignment):**

```
<-- pitch (in bytes) -->
[D][D][D]...[D][P][P][P]  <-- Row 0 starts at an aligned address
[D][D][D]...[D][P][P][P]  <-- Row 1 starts at an aligned address
[D][D][D]...[D][P][P][P]  <-- Row 2 starts at an aligned address
```

`D` = Data, `P` = Padding

The *pitch* will always be greater than or equal to the actual data width (`width * sizeof(element)`). The extra bytes are padding, which are unused but ensure that the start of every row falls on a memory address that is optimal for the hardware.

### How cudaMallocPitch Works

Instead of manually calculating the required padding (which can vary between different GPU architectures), CUDA provides a convenient function: `cudaMallocPitch()`.

When you call `cudaMallocPitch(&devPtr, &pitch, widthInBytes, height)`, the CUDA driver:
1. Receives your requested row width (`widthInBytes`).
2. Determines the optimal alignment boundary for the specific GPU it's running on.
3. Calculates the smallest multiple of that alignment that is greater than or equal to `widthInBytes`. This value becomes the *pitch*.
4. Allocates `pitch * height` bytes of memory.
5. Returns a pointer to the allocated memory and the calculated pitch value.

You must then use this pitch value in your kernels and memory copy functions to correctly access the data.

## 3. How to Implement Pitched Memory: A Code Walkthrough

Let's break down the key parts of the provided `pitched_vs_nonpitched.cu` code.

### Step 1: Allocation

**Non-Pitched (Standard):**

Simple and direct. You only care about the total size.

```cpp
float* d_matrix_nonpitched;
size_t matrix_size = width * height * sizeof(float);
CUDA_CHECK(cudaMalloc(&d_matrix_nonpitched, matrix_size));
```

**Pitched:**

Here, we use `cudaMallocPitch`. It returns both the pointer (`d_matrix_pitched`) and the calculated pitch (`pitch`).

```cpp
float* d_matrix_pitched;
size_t pitch;
CUDA_CHECK(cudaMallocPitch(&d_matrix_pitched, &pitch, width * sizeof(float), height));
```

Notice we pass the width in bytes (`width * sizeof(float)`) as the third argument.

### Step 2: Data Transfer

**Non-Pitched (Standard):**

A single `cudaMemcpy` call for the entire block is sufficient.

```cpp
CUDA_CHECK(cudaMemcpy(d_matrix_nonpitched, h_matrix, matrix_size, cudaMemcpyHostToDevice));
```

**Pitched:**

You cannot use a simple `cudaMemcpy` because the source (host memory) is contiguous, while the destination (device memory) has padding between rows. We must use `cudaMemcpy2D`.

```cpp
// h_matrix is a contiguous host array, so its pitch is simply width * sizeof(float)
CUDA_CHECK(cudaMemcpy2D(d_matrix_pitched,       // Destination pointer
                       pitch,                  // Destination pitch
                       h_matrix,               // Source pointer
                       width * sizeof(float),  // Source pitch
                       width * sizeof(float),  // Width of data to copy (in bytes)
                       height,                 // Height of data to copy (in rows)
                       cudaMemcpyHostToDevice));
```

`cudaMemcpy2D` intelligently copies the data row by row, respecting the different pitches of the source and destination memory.

### Step 3: Kernel Access

This is the most critical difference. You must use the pitch to calculate memory addresses correctly.

**Non-Pitched Kernel:**

The familiar linear indexing scheme.

```cpp
__global__ void processMatrix_NonPitched(float* matrix, int width, int height) {
    // ...
    int idx = row * width + col;
    matrix[idx] = ...;
}
```

**Pitched Kernel:**

The indexing logic must account for the padded row width (`pitch`).

```cpp
__global__ void processMatrix_Pitched(float* matrix, size_t pitch, int width, int height) {
    // ...
    
    // 1. Cast the base pointer to char* for byte-level arithmetic.
    // 2. Calculate the start of the correct row: base_address + row * pitch_in_bytes.
    // 3. Cast this row pointer back to the actual data type (float*).
    float* row_ptr = (float*)((char*)matrix + row * pitch);
    
    // 4. Access the element by its column index within that row.
    row_ptr[col] = ...;
}
```

This calculation correctly skips over the padding between rows to find the start of the `row`-th row.

## Performance and Overhead

When you run the example code, you will likely observe:

- **Performance:** The pitched memory approach is significantly faster. This is the direct result of improved memory access patterns (coalescing and alignment), which leads to higher effective memory bandwidth. The speedup can be substantial, often 1.5x or more, depending on the GPU architecture and matrix dimensions.
- **Memory Overhead:** The pitched memory approach uses slightly more GPU memory due to the padding. The code calculates this overhead:

```cpp
Memory overhead: ((float)(pitch * height) / (width * height * sizeof(float)) - 1.0f) * 100.0f
```

This is the classic space-time tradeoff. You use a little more memory to gain a lot more speed. For most applications, this is a very worthwhile trade.
