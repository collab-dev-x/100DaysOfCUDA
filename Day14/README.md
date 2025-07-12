# Dynamic Tiling in CUDA


Dynamic tiling generalizes the tiling approach by:

1. **Parameterizing** the tile dimensions (`tileSize`) on the host.
2. **Allocating** shared memory dynamically using `extern __shared__ float shared[];`.
3. **Partitioning** that shared block into two sub-tiles (`As` and `Bs`) for input matrices A and B.

This flexibility allows tuning the tile size to match different GPU architectures or problem sizes without recompiling the code.

---

## 3. Implementation Details

- **Dynamic shared allocation:** `sharedSize = 2 * tileSize * tileSize * sizeof(float)` on kernel launch.
- **Flexible tileSize:** the same kernel can run with `tileSize = 8, 16, 32`, etc., depending on available shared memory and occupancy.

---

## 4. Optimization Benefits

1. **Adaptable Resource Usage:** Tune `tileSize` at runtime to fit different GPU SM capabilities (shared memory size, register limits).
2. **Performance Exploration:** Quickly experiment with different tile shapes to find the optimal configuration for a given matrix size.
3. **Code Maintenance:** Single kernel supports multiple tile sizes, reducing code duplication compared to multiple static-kernel variants.

---

## 5. Why Dynamic Tiling Is Needed

- **Diverse Hardware:** Different GPU architectures (e.g., older vs. newer) have varying shared memory limits; dynamic tiling helps adapt without code changes.
- **Mixed Problem Sizes:** When input dimensions change, dynamic tiling ensures optimal block utilization across a range of M, N, K values.
- **Resource Constrained:** For very large tile sizes, static allocation may exceed available shared memory; dynamic tiling can guard against this with host-side checks.

---

## 6. Comparison to Static Shared-Memory Tiling

Before diving into implementation, here’s how static and dynamic tiling differ:

| Aspect                | Static Tiling                        | Dynamic Tiling                                  |
| --------------------- | ------------------------------------ | ----------------------------------------------- |
| **Tile size**         | Compile-time constant (`TILE_SIZE`)  | Runtime parameter (`tileSize`)                  |
| **Shared memory use** | Fixed size (2 × `TILE_SIZE²` floats) | Flexible size (2 × `tileSize²` floats)          |
| **Kernel variants**   | One kernel per `TILE_SIZE` (duplication) | Single kernel supports all sizes           |
| **Recompilation**     | Required for each different tile size | Not required; adjust at launch                  |
| **Hardware tuning**   | Manual code changes                  | Launch-time checks against SM limits            |

**Key differences:**
- **Flexibility:** Dynamic tiling adapts to varying GPU architectures and matrix dimensions without code changes.
- **Maintainability:** Only one kernel to manage; static tiling requires separate kernels for each tile size.
- **Resource Safety:** Host-side checks can prevent exceeding shared-memory limits at runtime.

---

## 7. Conclusion

Dynamic tiling assists in:

- **Rapidly prototype** and **tune** performance across GPU architectures.
- **Reduce code bloat** by maintaining one versatile kernel.
- **Ensure safety** by checking `tileSize` against hardware limits at runtime.

