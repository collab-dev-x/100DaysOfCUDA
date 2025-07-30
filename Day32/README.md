# cuBLAS Sgemv Documentation

The `cublasSgemv` function from the NVIDIA cuBLAS library performs a matrix-vector operation on single-precision floating-point data. It computes the operation:

**y = α ⋅ op(A) ⋅ x + β ⋅ y**

where `op(A)` is either the matrix `A` or its transpose `Aᵗ`, depending on the operation mode.

## Function Prototype
```c
cublasStatus_t cublasSgemv(
    cublasHandle_t handle,
    cublasOperation_t trans,
    int m, int n,
    const float *alpha,
    const float *A, int lda,
    const float *x, int incx,
    const float *beta,
    float *y, int incy
);
```

## Parameters
| Parameter | Description |
|-----------|-------------|
| `handle`  | cuBLAS context handle initialized by `cublasCreate`. |
| `trans`   | Operation type on matrix `A`: <br> - `CUBLAS_OP_N`: Use `A` (no transpose). <br> - `CUBLAS_OP_T`: Use `Aᵗ` (transpose). |
| `m`       | Number of rows of matrix `A`. If transposed, becomes number of columns. |
| `n`       | Number of columns of matrix `A`. If transposed, becomes number of rows. |
| `alpha`   | Pointer to scalar multiplier `α` (single-precision float). |
| `A`       | Pointer to matrix `A` in device memory, stored in column-major (Fortran-style) format. |
| `lda`     | Leading dimension of `A`. For non-transposed `A`, this is `m`. For transposed `A`, this is `n`. |
| `x`       | Pointer to input vector `x` in device memory. |
| `incx`    | Increment for elements of `x`. Typically set to `1` for contiguous memory. |
| `beta`    | Pointer to scalar multiplier `β` (single-precision float). |
| `y`       | Pointer to output vector `y` in device memory. |
| `incy`    | Increment for elements of `y`. Typically set to `1` for contiguous memory. |

## Operation Details
- **Operation**: Computes `y = α ⋅ op(A) ⋅ x + β ⋅ y`, where `op(A)` is either `A` (no transpose) or `Aᵗ` (transpose).
- **Matrix Layout**: cuBLAS uses column-major (Fortran-style) storage, unlike the row-major layout common in C/C++.
- **Transpose Mode**: When `trans = CUBLAS_OP_T`, the operation uses the transpose of `A`. This swaps the roles of `m` and `n` in the matrix dimensions.

## Example Usage
```cpp
cublasSgemv(handle, CUBLAS_OP_T, n, m, &alpha, d_a, m, d_x, 1, &beta, d_y, 1);
```
In this example:
- `CUBLAS_OP_T` specifies that the transpose of `A` (`Aᵗ`) is used.
- The operation performed is: **y = α ⋅ Aᵗ ⋅ x + β ⋅ y**.
- For instance, if `alpha = 2.0` and `beta = 0.6`, the operation becomes:
  **y = 2.0 ⋅ Aᵗ ⋅ x + 0.6 ⋅ y**.