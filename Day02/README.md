# Day 02

## Device Properties

The `device_properties.cu` program queries all CUDA-capable GPUs in the system and prints a range of key hardware characteristics for each one. It does this by:

1. Calling `cudaGetDeviceCount()` to find out how many CUDA devices are present.  
2. For each device index `i` in `[0, device_count)`, calling `cudaGetDeviceProperties(&prop, i)` to fill a `cudaDeviceProp` struct with that device’s attributes.  
3. Printing out each field of `prop` and what it means:

| Property                      | Code member                          | What it tells you & why it matters                                                           |
| ----------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------- |
| Device name                   | `prop.name`                          | Human-readable identifier (e.g. “Tesla V100”). Helps you distinguish among multiple GPUs.    |
| Compute capability            | `prop.major`, `prop.minor`           | GPU architecture version (e.g. 7.0). Determines supported features and PTX capabilities.    |
| Clock rate                    | `prop.clockRate`                     | Core clock frequency in kHz. Higher → faster arithmetic throughput.                          |
| Device copy overlap           | `prop.deviceOverlap`                 | Whether the device can overlap host ↔ device transfers with kernel execution.               |
| Number of copy engines        | `prop.asyncEngineCount`              | How many simultaneous memcpy engines are available. More engines → more overlap potential.  |
| Concurrent kernels            | `prop.concurrentKernels`             | Whether multiple kernels can run at the same time on the GPU.                                |
| Kernel execution timeout      | `prop.kernelExecTimeoutEnabled`      | If enabled, long-running kernels may time out (useful on display GPUs).                      |
| Global memory size            | `prop.totalGlobalMem`                | Total device DRAM in bytes. Limits maximum data you can store on the GPU.                   |
| Constant memory size          | `prop.totalConstMem`                 | Amount of read-only constant cache in bytes.                                                |
| Max memory pitch              | `prop.memPitch`                      | Maximum row size (in bytes) for 2D memory copies.                                          |
| Texture alignment             | `prop.textureAlignment`              | Alignment requirement for texture references.                                               |
| Multiprocessor count          | `prop.multiProcessorCount`           | Number of SMs (Streaming Multiprocessors). More SMs → higher parallelism.                   |
| Max blocks per SM             | `prop.maxBlocksPerMultiProcessor`    | How many blocks can reside on one SM concurrently.                                          |
| Shared memory per SM          | `prop.sharedMemPerMultiprocessor`    | Total shared memory available to each SM (in bytes).                                        |
| Registers per block           | `prop.regsPerBlock`                  | Number of registers available to a block.                                                  |
| Warp size                     | `prop.warpSize`                      | Number of threads in a warp (typically 32). Influences divergence behavior.                 |
| Max threads per block         | `prop.maxThreadsPerBlock`            | Upper limit on threads you can launch in a block.                                          |
| Max thread-block dimensions   | `prop.maxThreadsDim[3]`              | Maximum size in each dimension (x,y,z) for a block.                                        |
| Max grid dimensions           | `prop.maxGridSize[3]`                | Maximum grid size in each dimension (x,y,z).                                               |
| ECC (Error-Correcting Code)   | `prop.ECCEnabled`                    | Whether ECC is enabled on device memory (helps reliability at small performance cost).      |

By examining these, you can tailor your kernel launches (block/grid sizes), memory usage, and feature usage to the specific hardware.

---

## Vector Addition

Two implementations of element-wise addition of two large integer arrays (`A` + `B` → `C`) were written:

### 2.1 CPU Version (`vectadd_cpu`)
- Uses C++ `<chrono>`:
  ```cpp
  auto start = high_resolution_clock::now();
  for (int i = 0; i < n; i++)
      C[i] = A[i] + B[i];
  auto end   = high_resolution_clock::now();
  cpu_time_ms = duration<double, milli>(end - start).count();
  ```
- Measures elapsed time in milliseconds on the host.

### GPU Version
- Allocates device arrays d_a, d_b, d_c with cudaMalloc().

- Copies inputs with cudaMemcpy(HostToDevice).

- Uses CUDA events to time the kernel launch:
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);

// launch with, e.g., <<< (n+255)/256, 256 >>>
vectadd_kernel<<<(n+255)/256,256>>>(d_a, d_b, d_c, n);

cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&gpu_time_ms, start, stop);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

## Performance Results

- When run with n = 10,000,000 elements:

```yaml
GPU Time:  5000 ms
CPU Time: 25000 ms
Result correct: Yes
```
- Speedup = CPU time / GPU time = 25 000 ms / 5 000 ms = 5× faster on the GPU.