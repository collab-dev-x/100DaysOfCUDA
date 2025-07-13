# Parallel Reduction in CUDA

Reduction is an operation that collapses a collection of input values into a single result by repeatedly applying a binary operator.

- **Input**: An array of values
- **Operator**: A binary operation like sum (+), min, max, product (*), etc.
- **Output**: A single value

The key mathematical property enabling parallelization is **associativity**. This allows operations to be grouped in any order, forming the basis for a "divide and conquer" strategy in parallel computing.

## 2. Serial Reduction

A serial reduction processes the input array sequentially:

```cpp
float acc = identity; // 0 for sum, +∞ for min, -∞ for max
for (int i = 0; i < N; ++i) {
    acc = acc ⊕ a[i];
}
```

- **Time Complexity**: \( O(N) \) operations
- **Memory Traffic**: \( N \) reads, 1 write
- **Limitation**: Executes on a single CPU core, underutilizing multi-core CPUs or GPUs unless explicitly parallelized.

## 3. Parallel (Tree-Based) Reduction

Parallel reduction organizes computation as a tree:

1. Pairwise combine adjacent elements in parallel → \( N/2 \) results.
2. Repeat on partial results → \( N/4 \), then \( N/8 \), and so on.
3. After log_2 N steps, one result remains.

- **Time Complexity**: O(log N) parallel steps
- **Work**: Approximately \( N \) total operations, distributed across threads
- **Ideal for**: SIMD/vector units, multi-core CPUs, GPUs

## 4. CUDA Execution Model

CUDA's programming model is hierarchical:

- **Grid**: Composed of many blocks
- **Block**: Up to 1,024 threads (typically 128–512 in practice)
- **Shared Memory**: Threads in a block share fast, on-chip memory and synchronize using `__syncthreads()`.

**Mapping Reduction to CUDA**:
- Each block processes a chunk of the input array, performing an intra-block tree-based reduction to produce one partial result.
- A second kernel (or the CPU) reduces the per-block results into a final value.

## 5. Warp and Warp-Level Optimization

A **warp** is a group of 32 threads executing in lock-step on NVIDIA GPUs.

- **Warp Divergence**: Conditional branches (e.g., `if-then-else`) where threads take different paths cause serialization, reducing performance.

### 5.1 Warp-Synchronous Shuffle

CUDA's shuffle intrinsics allow threads within a warp to exchange data directly via registers, bypassing shared memory:

```cpp
// Example: Sum across a warp
float val = ...; // Thread's local value
for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
    val += other;
}
```

- **Benefits**:
  - No shared memory or `__syncthreads()` required (shuffles are implicitly warp-synchronous).
  - Reduces latency and eliminates shared-memory bank conflicts.

### 5.2 Combining Shuffle and Shared Memory

A hybrid approach optimizes performance:

1. Each thread loads one element from global memory into a register.
2. Perform an in-warp shuffle reduction to produce one value per warp.
3. Thread 0 of each warp writes its partial result to shared memory.
4. The first warp reads the warp-level results from shared memory and performs a final shuffle reduction.

This minimizes shared memory usage (one value per warp, max 32 per block) while leveraging fast warp shuffles.

## 6. Avoiding Bank Conflicts in Shared Memory

When shared memory is used for intra-block reductions:

- **Bank Conflicts**: Occur when multiple threads access the same memory bank simultaneously, serializing access.
- **Solution**: Use indexing to ensure consecutive threads access different banks. For example, pad data arrays by one element to align thread accesses to distinct banks.

**Example**:
```cpp
__shared__ float partialSums[33]; // Pad to avoid bank conflicts
```

This ensures threads access distinct banks, maximizing memory bandwidth.

## 7. Summary

Parallel reduction in CUDA transforms a serial \( O(N) \) operation into an \( O(\log N) \) parallel algorithm by leveraging the GPU’s thread hierarchy, warp-level optimizations, and careful memory management. Key techniques include:

- Tree-based reduction for logarithmic parallel steps.
- Warp shuffles to minimize shared memory and synchronization overhead.
- Careful indexing to avoid shared memory bank conflicts.

By combining these strategies, CUDA enables highly efficient reductions for large datasets, fully utilizing the GPU’s parallel architecture.