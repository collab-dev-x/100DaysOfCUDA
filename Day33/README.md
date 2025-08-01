# cuBLAS Level 2 Functions

Functions used in the code: `cublasSger`, `cublasStrmv`, and `cublasStrsv`. These functions perform linear algebra operations on single-precision floating-point data using NVIDIA's cuBLAS library.

## 1. cublasSger

### Overview
The `cublasSger` function performs a rank-1 update of a general matrix, computing the operation:

**A = α ⋅ x ⋅ yᵗ + A**

where `A` is a matrix, `x` and `y` are vectors, `α` is a scalar, and `yᵗ` is the transpose of vector `y`.

### Function Prototype
```c
cublasStatus_t cublasSger(
    cublasHandle_t handle,
    int m, int n,
    const float *alpha,
    const float *x, int incx,
    const float *y, int incy,
    float *A, int lda
);
```

### Parameters
| Parameter | Description |
|-----------|-------------|
| `handle`  | cuBLAS context handle initialized by `cublasCreate`. |
| `m`       | Number of rows of matrix `A`. |
| `n`       | Number of columns of matrix `A`. |
| `alpha`   | Pointer to scalar multiplier `α` (single-precision float). |
| `x`       | Pointer to input vector `x` in device memory. |
| `incx`    | Increment for elements of `x`. Typically set to `1` for contiguous memory. |
| `y`       | Pointer to input vector `y` in device memory. |
| `incy`    | Increment for elements of `y`. Typically set to `1` for contiguous memory. |
| `A`       | Pointer to matrix `A` in device memory, stored in column-major format. |
| `lda`     | Leading dimension of `A`. Typically `m` for column-major storage. |

### Operation Details
- **Operation**: Computes `A = α ⋅ x ⋅ yᵗ + A`, updating matrix `A` in place.
- **Matrix Layout**: Uses column-major (Fortran-style) storage.
- **Example in Code**:
  ```cpp
  cublasSger(handle, m, n, &alpha, d_x, 1, d_y, 1, d_a, m);
  ```
  - With `alpha = 2.0f`, `m = 10`, `n = 8`, this performs: **A = 2.0 ⋅ x ⋅ yᵗ + A**.
  - `x` is a vector of length `m`, `y` is a vector of length `n`, and `A` is an `m × n` matrix.

---

## 2. cublasStrmv

### Overview
The `cublasStrmv` function performs a triangular matrix-vector multiplication, computing:

**x = op(A) ⋅ x**

where `A` is a triangular matrix (upper or lower), and `op(A)` is either `A` or its transpose.

### Function Prototype
```c
cublasStatus_t cublasStrmv(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t diag,
    int n,
    const float *A, int lda,
    float *x, int incx
);
```

### Parameters
| Parameter | Description |
|-----------|-------------|
| `handle`  | cuBLAS context handle initialized by `cublasCreate`. |
| `uplo`    | Specifies whether `A` is upper or lower triangular: <br> - `CUBLAS_FILL_MODE_UPPER`: Upper triangular. <br> - `CUBLAS_FILL_MODE_LOWER`: Lower triangular. |
| `trans`   | Operation type on matrix `A`: <br> - `CUBLAS_OP_N`: No transpose. <br> - `CUBLAS_OP_T`: Transpose. |
| `diag`    | Specifies whether the diagonal of `A` is unit or non-unit: <br> - `CUBLAS_DIAG_UNIT`: Unit diagonal (diagonal elements assumed to be 1). <br> - `CUBLAS_DIAG_NON_UNIT`: Non-unit diagonal. |
| `n`       | Number of rows and columns of matrix `A` (square matrix). |
| `A`       | Pointer to triangular matrix `A` in device memory, stored in column-major format. |
| `lda`     | Leading dimension of `A`. Typically `n` for column-major storage. |
| `x`       | Pointer to input/output vector `x` in device memory. |
| `incx`    | Increment for elements of `x`. Typically `1` for contiguous memory. |

### Operation Details
- **Operation**: Computes `x = op(A) ⋅ x`, overwriting `x` with the result.
- **Matrix Layout**: Uses column-major storage. Only the specified triangular part (upper or lower) of `A` is used.
- **Example in Code**:
  ```cpp
  cublasStrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, d_a, n, d_x, 1);
  ```
  - Uses the upper triangular part of `A` with no transpose and non-unit diagonal.
  - Performs: **x = A ⋅ x**, where `A` is an `n × n` upper triangular matrix.

---

## 3. cublasStrsv

### Overview
The `cublasStrsv` function solves a triangular system of equations with a single right-hand side:

**A ⋅ x = b**

where `A` is a triangular matrix (upper or lower), and `x` overwrites `b` with the solution.

### Function Prototype
```c
cublasStatus_t cublasStrsv(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t diag,
    int n,
    const float *A, int lda,
    float *x, int incx
);
```

### Parameters
| Parameter | Description |
|-----------|-------------|
| `handle`  | cuBLAS context handle initialized by `cublasCreate`. |
| `uplo`    | Specifies whether `A` is upper or lower triangular: <br> - `CUBLAS_FILL_MODE_UPPER`: Upper triangular. <br> - `CUBLAS_FILL_MODE_LOWER`: Lower triangular. |
| `trans`   | Operation type on matrix `A`: <br> - `CUBLAS_OP_N`: No transpose (solve `A ⋅ x = b`). <br> - `CUBLAS_OP_T`: Transpose (solve `Aᵗ ⋅ x = b`). |
| `diag`    | Specifies whether the diagonal of `A` is unit or non-unit: <br> - `CUBLAS_DIAG_UNIT`: Unit diagonal. <br> - `CUBLAS_DIAG_NON_UNIT`: Non-unit diagonal. |
| `n`       | Number of rows and columns of matrix `A` (square matrix). |
| `A`       | Pointer to triangular matrix `A` in device memory, stored in column-major format. |
| `lda`     | Leading dimension of `A`. Typically `n` for column-major storage. |
| `x`       | Pointer to input/output vector `x` in device memory. On input, contains `b`; on output, contains the solution `x`. |
| `incx`    | Increment for elements of `x`. Typically `1` for contiguous memory. |

### Operation Details
- **Operation**: Solves `A ⋅ x = b` (or `Aᵗ ⋅ x = b` if transposed), overwriting `x` with the solution.
- **Matrix Layout**: Uses column-major storage. Only the specified triangular part of `A` is used.
- **Example in Code**:
  ```cpp
  cublasStrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, d_a, n, d_x, 1);
  ```
  - Solves: **A ⋅ x = b**, where `A` is an `n × n` lower triangular matrix with non-unit diagonal.
  - `x` initially contains `b` and is overwritten with the solution.


---

## Common Notes for All Functions
- **Column-Major Storage**: All matrices are stored in column-major (Fortran-style) format, unlike C/C++ row-major.
- **Device Memory**: All input/output arrays (`A`, `x`, `y`) must reside in GPU device memory.
