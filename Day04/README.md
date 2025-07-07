# 2D and 3D Flattened Array

In CUDA, multidimensional arrays (like 2D or 3D arrays) are often **flattened** to 1D arrays for efficient memory access on the device.

Instead of using `array[i][j]` or `array[i][j][k]`, we use a 1D array with index computed manually.

## Row-Major Memory Layout

In **row-major order** (default in C/C++), the last index changes fastest.

- **2D Array**: `arr[row][col]` is stored as `arr[row * num_columns + col]`
- **3D Array**: `arr[depth][row][col]` is stored as `arr[depth * row_size * col_size + row * col_size + col]`

---

## Column-Major Memory Layout

In **column-major order**, **columns are stored consecutively** in memory. This is the opposite of **row-major**, where rows are stored consecutively.


- **2D Array**: `arr[row][col]` is stored as `arr[row + col * num_rows]`
- **3D Array**: `arr[depth][row][col]` is stored as `arr[z + y * depth + x * depth * height]`

---

## CUDA Execution Model

### 1. **Grid** (Entire Launch Domain)

A **grid** is the top-level structure that contains all blocks. When you launch a kernel like this:

```cpp
kernel<<<gridDim, blockDim>>>();
```

You are specifying:

- `gridDim`: Number of blocks in the grid (can be 1D, 2D, or 3D)
- `blockDim`: Number of threads per block (can also be 1D, 2D, or 3D)

---

### 2. **Block** (Group of Threads)

Each block:

- Contains a fixed number of threads
- Executes independently and can **share memory**

You access the block's index using:

```cpp
blockIdx.x, blockIdx.y, blockIdx.z
```

You access a thread's index **within a block** using:

```cpp
threadIdx.x, threadIdx.y, threadIdx.z
```

---

### 3. **Thread** (Unit of Execution)

Each thread:

- Has a unique index **within its block**
- Executes the kernel independently

You **compute its global index** using:

```cpp
globalIdx = threadIdx + blockIdx * blockDim;
```

This ensures **each thread is uniquely identified across the entire grid**.

---

## Thread Index Calculation

For 3D kernels, compute the global position of a thread like so:

```cpp
int row    = threadIdx.x + blockIdx.x * blockDim.x;
int column = threadIdx.y + blockIdx.y * blockDim.y;
int depth  = threadIdx.z + blockIdx.z * blockDim.z;
```

This is the **standard and correct way** to calculate each thread’s **global** index in a 3D computational space.

---
