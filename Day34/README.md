# cuBLAS Level 3 Matrix Multiplication Functions


## 1. cublasSgemm

### Overview
The `cublasSgemm` function performs a general matrix-matrix multiplication, computing the operation:

**C = α ⋅ op(A) ⋅ op(B) + β ⋅ C**

where `A`, `B`, and `C` are matrices, `α` and `β` are scalars, and `op(A)` and `op(B)` can be either the matrices themselves or their transposes.

### Function Prototype
```c
cublasStatus_t cublasSgemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
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
| `transa`  | Operation type for matrix `A`: <br> - `CUBLAS_OP_N`: No transpose (use `A`). <br> - `CUBLAS_OP_T`: Transpose (use `Aᵗ`). |
| `transb`  | Operation type for matrix `B`: <br> - `CUBLAS_OP_N`: No transpose (use `B`). <br> - `CUBLAS_OP_T`: Transpose (use `Bᵗ`). |
| `m`       | Number of rows of matrix `op(A)` and matrix `C`. |
| `n`       | Number of columns of matrix `op(B)` and matrix `C`. |
| `k`       | Number of columns of `op(A)` and rows of `op(B)`. |
| `alpha`   | Pointer to scalar multiplier `α` (single-precision float). |
| `A`       | Pointer to matrix `A` in device memory, stored in column-major format. |
| `lda`     | Leading dimension of `A`. Typically `m` for `CUBLAS_OP_N` or `k` for `CUBLAS_OP_T`. |
| `B`       | Pointer to matrix `B` in device memory, stored in column-major format. |
| `ldb`     | Leading dimension of `B`. Typically `k` for `CUBLAS_OP_N` or `n` for `CUBLAS_OP_T`. |
| `beta`    | Pointer to scalar multiplier `β` (single-precision float). |
| `C`       | Pointer to matrix `C` in device memory, stored in column-major format. Updated in place. |
| `ldc`     | Leading dimension of `C`. Typically `m`. |

### Operation Details
- **Operation**: Computes `C = α ⋅ op(A) ⋅ op(B) + β ⋅ C`, where `op(A)` and `op(B)` depend on `transa` and `transb`.
- **Matrix Layout**: Uses column-major (Fortran-style) storage.
- **Example in Code**:
  ```cpp
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
  ```
  - With `M = 256`, `N = 256`, `K = 256`, `alpha = 1.0f`, `beta = 0.5f`, performs: **C = 1.0 ⋅ A ⋅ B + 0.5 ⋅ C**.
  - `A` is an `M × K` matrix, `B` is a `K × N` matrix, and `C` is an `M × N` matrix.

---

## 2. cublasSgemmBatched

### Overview
The `cublasSgemmBatched` function performs a batch of matrix-matrix multiplications for multiple pairs of matrices, computing:

**C[i] = α ⋅ op(A[i]) ⋅ op(B[i]) + β ⋅ C[i]** for each batch index `i`.

### Function Prototype
```c
cublasStatus_t cublasSgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float **Aarray, int lda,
    const float **Barray, int ldb,
    const float *beta,
    float **Carray, int ldc,
    int batchCount
);
```

### Parameters
| Parameter   | Description |
|-------------|-------------|
| `handle`    | cuBLAS context handle initialized by `cublasCreate`. |
| `transa`    | Operation type for matrices `A[i]`: <br> - `CUBLAS_OP_N`: No transpose. <br> - `CUBLAS_OP_T`: Transpose. |
| `transb`    | Operation type for matrices `B[i]`: <br> - `CUBLAS_OP_N`: No transpose. <br> - `CUBLAS_OP_T`: Transpose. |
| `m`         | Number of rows of `op(A[i])` and `C[i]`. |
| `n`         | Number of columns of `op(B[i])` and `C[i]`. |
| `k`         | Number of columns of `op(A[i])` and rows of `op(B[i])`. |
| `alpha`     | Pointer to scalar multiplier `α` (single-precision float). |
| `Aarray`    | Array of pointers to matrices `A[i]` in device memory, stored in column-major format. |
| `lda`       | Leading dimension of `A[i]`. Typically `m` for `CUBLAS_OP_N` or `k` for `CUBLAS_OP_T`. |
| `Barray`    | Array of pointers to matrices `B[i]` in device memory, stored in column-major format. |
| `ldb`       | Leading dimension of `B[i]`. Typically `k` for `CUBLAS_OP_N` or `n` for `CUBLAS_OP_T`. |
| `beta`      | Pointer to scalar multiplier `β` (single-precision float). |
| `Carray`    | Array of pointers to matrices `C[i]` in device memory, stored in column-major format. Updated in place. |
| `ldc`       | Leading dimension of `C[i]`. Typically `m`. |
| `batchCount`| Number of matrix multiplications to perform (size of the batch). |

### Operation Details
- **Operation**: For each `i` from 0 to `batchCount-1`, computes `C[i] = α ⋅ op(A[i]) ⋅ op(B[i]) + β ⋅ C[i]`.
- **Matrix Layout**: Uses column-major storage.
- **Example in Code**:
  ```cpp
  cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, 
                     (const float**)d_A_array, M, (const float**)d_B_array, K, &beta, d_C_array, M, batch_size);
  ```
  - With `batch_size = 8`, performs 8 matrix multiplications, each computing: **C[i] = 1.0 ⋅ A[i] ⋅ B[i] + 0.5 ⋅ C[i]**.
  - Each `A[i]` is `M × K`, `B[i]` is `K × N`, and `C[i]` is `M × N`.

---

## 3. cublasSgemmStridedBatched

### Overview
The `cublasSgemmStridedBatched` function performs a batch of matrix-matrix multiplications for matrices stored with a fixed stride in memory, computing:

**C[i] = α ⋅ op(A[i]) ⋅ op(B[i]) + β ⋅ C[i]** for each batch index `i`.

Unlike `cublasSgemmBatched`, this function assumes matrices are stored contiguously in memory with a specified stride, reducing the need for an array of pointers.

### Function Prototype
```c
cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda, long long int strideA,
    const float *B, int ldb, long long int strideB,
    const float *beta,
    float *C, int ldc, long long int strideC,
    int batchCount
);
```

### Parameters
| Parameter   | Description |
|-------------|-------------|
| `handle`    | cuBLAS context handle initialized by `cublasCreate`. |
| `transa`    | Operation type for matrices `A[i]`: <br> - `CUBLAS_OP_N`: No transpose. <br> - `CUBLAS_OP_T`: Transpose. |
| `transb`    | Operation type for matrices `B[i]`: <br> - `CUBLAS_OP_N`: No transpose. <br> - `CUBLAS_OP_T`: Transpose. |
| `m`         | Number of rows of `op(A[i])` and `C[i]`. |
| `n`         | Number of columns of `op(B[i])` and `C[i]`. |
| `k`         | Number of columns of `op(A[i])` and rows of `op(B[i])`. |
| `alpha`     | Pointer to scalar multiplier `α` (single-precision float). |
| `A`         | Pointer to the first matrix `A[0]` in device memory, stored in column-major format. Subsequent matrices are at `A + i * strideA`. |
| `lda`       | Leading dimension of `A[i]`. Typically `m` for `CUBLAS_OP_N` or `k` for `CUBLAS_OP_T`. |
| `strideA`   | Stride (in elements) between consecutive matrices `A[i]` and `A[i+1]`. |
| `B`         | Pointer to the first matrix `B[0]` in device memory, stored in column-major format. Subsequent matrices are at `B + i * strideB`. |
| `ldb`       | Leading dimension of `B[i]`. Typically `k` for `CUBLAS_OP_N` or `n` for `CUBLAS_OP_T`. |
| `strideB`   | Stride (in elements) between consecutive matrices `B[i]` and `B[i+1]`. |
| `beta`      | Pointer to scalar multiplier `β` (single-precision float). |
| `C`         | Pointer to the first matrix `C[0]` in device memory, stored in column-major format. Subsequent matrices are at `C + i * strideC`. Updated in place. |
| `ldc`       | Leading dimension of `C[i]`. Typically `m`. |
| `strideC`   | Stride (in elements) between consecutive matrices `C[i]` and `C[i+1]`. |
| `batchCount`| Number of matrix multiplications to perform (size of the batch). |

### Operation Details
- **Operation**: For each `i` from 0 to `batchCount-1`, computes `C[i] = α ⋅ op(A[i]) ⋅ op(B[i]) + β ⋅ C[i]`, where `A[i] = A + i * strideA`, `B[i] = B + i * strideB`, and `C[i] = C + i * strideC`.
- **Matrix Layout**: Uses column-major storage.
- **Example in Code**:
  ```cpp
  cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, 
                            d_batch_A, M, M*N, d_batch_B, K, M*N, &beta, d_batch_C, M, M*N, batch_size);
  ```
  - With `batch_size = 8`, `M = N = K = 256`, performs 8 matrix multiplications, each computing: **C[i] = 1.0 ⋅ A[i] ⋅ B[i] + 0.5 ⋅ C[i]**.
  - Matrices are stored contiguously with strides `strideA = strideB = strideC = M * N`.


---