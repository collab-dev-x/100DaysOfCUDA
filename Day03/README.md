# Matrix Multiplication

## Definition

Given two square matrices **A** and **B** of size `n x n`, the matrix multiplication result **C** is also an `n x n` matrix where each element `C[i][j]` is computed as:

```
C[i][j] = ∑ (from k = 0 to n-1) A[i][k] * B[k][j]
```
This operation has a time complexity of \(O(n^3)\) using the naive method.

---

## CUDA Implementation Overview

In C/C++, multi-dimensional arrays (like 2D or 3D) are stored in memory as contiguous blocks. However, CUDA kernels cannot directly accept multi-dimensional arrays (like int[][]) unless you use pointer-to-pointer constructs, which are inefficient and error-prone.

✅ Solution: Use 1D arrays (flattened matrices)
Flatten a 2D matrix M[i][j] into a 1D array using:

Row-major order (default in C/C++):
```
M[i][j] → M_flat[i * n + j]
```

Column-major order (used in Fortran, MATLAB):
```
M[i][j] → M_flat[j * n + i]
```

## Flattened Matrix Multiplication

My implementation considers matrix to be row-order flattened. Thus flattened representation of matrix multiplication is:

```
C[i * n + j] = ∑ (from k = 0 to n-1) A[i * n + k] * B[k * n + j]
```

## Implementation

Each CUDA thread computes one element of the resulting matrix by identifying its row and column from a single thread index. Inside the kernel, each thread calculates the dot product of the corresponding row from matrix A and the column from matrix B. Memory is dynamically allocated on both host and device, and values are transferred accordingly using cudaMemcpy. The kernel is launched with a suitable number of blocks and threads to cover all elements. Finally, the result matrix is copied back from device to host and printed in a 2D format. This basic implementation demonstrates how 1D thread indexing can be used to parallelize matrix operations, making it suitable for small to moderately sized matrices.
