# Matrix Multiplication with 2D Execution Grid

Unlike a 1D approach that launches threads along a single dimension and flattens matrix indices, this implementation maps each thread to a specific row and column in the output matrix, improving readability and aligning computation with data layout.


- **2D Execution Grid**: Uses `dim3 dimGrid((col2+31)/32, (row1+15)/16)` and `dim3 dimBlock(32,16)` to launch blocks and threads in two dimensions.
- **2D Matrix Indexing**: Each thread computes the element at `(r, c)` using:
  ```cpp
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  ```
- **General Dimensions**: Supports arbitrary `row1×col1` and `row2×col2` matrices (with `col1 == row2`).



## 2D vs 1D Access Patterns

| Aspect              | 1D Execution & Indexing               | 2D Execution & Indexing                  |
| ------------------- | -------------------------------------- | ----------------------------------------- |
| Grid Configuration  | `<<<(N*N+T-1)/T, T>>>` (1D blocks)     | `<<<(col2+Bx-1)/Bx, (row1+By-1)/By>>>` (2D) |
| Thread Mapping      | Single thread ID `idx` → `(row, col)`  | Direct `(r, c)` from `blockIdx` & `threadIdx` |
| Index Calculations  | `row = idx/n; col = idx%n`             | `r = blockIdx.y*By + threadIdx.y`         |
| Code Clarity        | More arithmetic to compute indices     | Aligns with 2D data for maintainability  |
| Memory Coalescing   | Similar at large scales, but 2D maps naturally to row-major layout |

## Performance Considerations

- **Coalescing**: Both approaches can coalesce memory when accessing row-major data, but 2D indexing makes the pattern explicit.
- **Occupancy**: Choosing `blockDim` to match warp sizes (e.g., 32×16) can improve occupancy and resource utilization.
- **Scalability**: 2D grids scale more naturally for rectangular matrices.

