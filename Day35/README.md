# cuBLAS Symmetric Matrix Functions
## 1. cublasSsymm

### Overview
The `cublasSsymm` function performs a symmetric matrix-matrix multiplication, computing one of the following operations:

- **If side = CUBLAS_SIDE_LEFT**: **C = α ⋅ A ⋅ B + β ⋅ C**
- **If side = CUBLAS_SIDE_RIGHT**: **C = α ⋅ B ⋅ A + β ⋅ C**

where `A` is a symmetric matrix, `B` and `C` are general matrices, and `α` and `β` are scalars.

### Function Prototype
```c
cublasStatus_t cublasSsymm(
    cublasHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    int m, int n,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc
);
```

### Parameters
| Parameter | Description |
|-----------|-------------|
| `handle`  | cuBLAS context handle initialized by `cublasCreate`. |
| `side`    | Specifies the side of the symmetric matrix `A`: <br> - `CUBLAS_SIDE_LEFT`: `A` is on the left (`C = α ⋅ A ⋅ B + β ⋅ C`). <br> - `CUBLAS_SIDE_RIGHT`: `A` is on the right (`C = α ⋅ B ⋅ A + β ⋅ C`). |
| `uplo`    | Specifies whether the upper or lower triangular part of `A` is used: <br> - `CUBLAS_FILL_MODE_UPPER`: Upper triangular. <br> - `CUBLAS_FILL_MODE_LOWER`: Lower triangular. |
| `m`       | Number of rows of matrix `C` and matrix `A` (if `side = CUBLAS_SIDE_LEFT`) or matrix `B` (if `side = CUBLAS_SIDE_RIGHT`). |
| `n`       | Number of columns of matrix `C` and matrix `B` (if `side = CUBLAS_SIDE_LEFT`) or matrix `A` (if `side = CUBLAS_SIDE_RIGHT`). |
| `alpha`   | Pointer to scalar multiplier `α` (single-precision float). |
| `A`       | Pointer to symmetric matrix `A` in device memory, stored in column-major format. Only the specified triangular part is used. |
| `lda`     | Leading dimension of `A`. Typically `m` for `CUBLAS_SIDE_LEFT` or `n` for `CUBLAS_SIDE_RIGHT`. |
| `B`       | Pointer to matrix `B` in device memory, stored in column-major format. |
| `ldb`     | Leading dimension of `B`. Typically `m` for `CUBLAS_SIDE_LEFT` or `k` (where `k` is the other dimension of `B`). |
| `beta`    | Pointer to scalar multiplier `β` (single-precision float). |
| `C`       | Pointer to matrix `C` in device memory, stored in column-major format. Updated in place. |
| `ldc`     | Leading dimension of `C`. Typically `m`. |

### Operation Details
- **Operation**:
  - For `CUBLAS_SIDE_LEFT`: Computes `C = α ⋅ A ⋅ B + β ⋅ C`.
  - For `CUBLAS_SIDE_RIGHT`: Computes `C = α ⋅ B ⋅ A + β ⋅ C`.
- **Matrix Layout**: Uses column-major (Fortran-style) storage. Only the specified triangular part (upper or lower) of `A` is used.

---

## 2. cublasSsyrk

### Overview
The `cublasSsyrk` function performs a symmetric rank-k update, computing one of the following operations:

- **C = α ⋅ A ⋅ Aᵗ + β ⋅ C** (if `trans = CUBLAS_OP_N`)
- **C = α ⋅ Aᵗ ⋅ A + β ⋅ C** (if `trans = CUBLAS_OP_T`)

where `A` is a general matrix, `C` is a symmetric matrix, and `α` and `β` are scalars.

### Function Prototype
```c
cublasStatus_t cublasSsyrk(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n, int k,
    const float *alpha,
    const float *A, int lda,
    const float *beta,
    float *C, int ldc
);
```

### Parameters
| Parameter | Description |
|-----------|-------------|
| `handle`  | cuBLAS context handle initialized by `cublasCreate`. |
| `uplo`    | Specifies whether the upper or lower triangular part of `C` is used: <br> - `CUBLAS_FILL_MODE_UPPER`: Upper triangular. <br> - `CUBLAS_FILL_MODE_LOWER`: Lower triangular. |
| `trans`   | Operation type for matrix `A`: <br> - `CUBLAS_OP_N`: `C = α ⋅ A ⋅ Aᵗ + β ⋅ C`. <br> - `CUBLAS_OP_T`: `C = α ⋅ Aᵗ ⋅ A + β ⋅ C`. |
| `n`       | Number of rows and columns of matrix `C`. Also, rows of `op(A)`. |
| `k`       | Number of columns of `op(A)`. |
| `alpha`   | Pointer to scalar multiplier `α` (single-precision float). |
| `A`       | Pointer to matrix `A` in device memory, stored in column-major format. |
| `lda`     | Leading dimension of `A`. Typically `n` for `CUBLAS_OP_N` or `k` for `CUBLAS_OP_T`. |
| `beta`    | Pointer to scalar multiplier `β` (single-precision float). |
| `C`       | Pointer to symmetric matrix `C` in device memory, stored in column-major format. Updated in place. |
| `ldc`     | Leading dimension of `C`. Typically `n`. |

### Operation Details
- **Operation**:
  - For `trans = CUBLAS_OP_N`: Computes `C = α ⋅ A ⋅ Aᵗ + β ⋅ C`.
  - For `trans = CUBLAS_OP_T`: Computes `C = α ⋅ Aᵗ ⋅ A + β ⋅ C`.
- **Matrix Layout**: Uses column-major storage. Only the specified triangular part of `C` is updated.

---

## 3. cublasSsyr2k

### Overview
The `cublasSsyr2k` function performs a symmetric rank-2k update, computing one of the following operations:

- **C = α ⋅ A ⋅ Bᵗ + α ⋅ B ⋅ Aᵗ + β ⋅ C** (if `trans = CUBLAS_OP_N`)
- **C = α ⋅ Aᵗ ⋅ B + α ⋅ Bᵗ ⋅ A + β ⋅ C** (if `trans = CUBLAS_OP_T`)

where `A` and `B` are general matrices, `C` is a symmetric matrix, and `α` and `β` are scalars.

### Function Prototype
```c
cublasStatus_t cublasSsyr2k(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n, int k,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc
);
```

### Parameters
| Parameter | Description |
|-----------|-------------|
| `handle`  | cuBLAS context handle initialized by `cublasCreate`. |
| `uplo`    | Specifies whether the upper or lower triangular part of `C` is used: <br> - `CUBLAS_FILL_MODE_UPPER`: Upper triangular. <br> - `CUBLAS_FILL_MODE_LOWER`: Lower triangular. |
| `trans`   | Operation type for matrices `A` and `B`: <br> - `CUBLAS_OP_N`: `C = α ⋅ A ⋅ Bᵗ + α ⋅ B ⋅ Aᵗ + β ⋅ C`. <br> - `CUBLAS_OP_T`: `C = α ⋅ Aᵗ ⋅ B + α ⋅ Bᵗ ⋅ A + β ⋅ C`. |
| `n`       | Number of rows and columns of matrix `C`. Also, rows of `op(A)` and `op(B)`. |
| `k`       | Number of columns of `op(A)` and `op(B)`. |
| `alpha`   | Pointer to scalar multiplier `α` (single-precision float). |
| `A`       | Pointer to matrix `A` in device memory, stored in column-major format. |
| `lda`     | Leading dimension of `A`. Typically `n` for `CUBLAS_OP_N` or `k` for `CUBLAS_OP_T`. |
| `B`       | Pointer to matrix `B` in device memory, stored in column-major format. |
| `ldb`     | Leading dimension of `B`. Typically `n` for `CUBLAS_OP_N` or `k` for `CUBLAS_OP_T`. |
| `beta`    | Pointer to scalar multiplier `β` (single-precision float). |
| `C`       | Pointer to symmetric matrix `C` in device memory, stored in column-major format. Updated in place. |
| `ldc`     | Leading dimension of `C`. Typically `n`. |

### Operation Details
- **Operation**:
  - For `trans = CUBLAS_OP_N`: Computes `C = α ⋅ A ⋅ Bᵗ + α ⋅ B ⋅ Aᵗ + β ⋅ C`.
  - For `trans = CUBLAS_OP_T`: Computes `C = α ⋅ Aᵗ ⋅ B + α ⋅ Bᵗ ⋅ A + β ⋅ C`.
- **Matrix Layout**: Uses column-major storage. Only the specified triangular part of `C` is updated.
