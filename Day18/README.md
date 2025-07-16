# CUDA Streams: Concurrent Execution on the GPU

A CUDA stream is a sequence of commands (like kernel launches and memory copies) that execute in order on the GPU. By using multiple streams, you can run multiple sequences concurrently, overlapping operations to maximize GPU utilization and improve application throughput.

By default, all CUDA operations are sent to a single, synchronous queue known as the *default stream* (stream 0).

## Kernel Launch Syntax

To launch a kernel on a specific stream, use the fourth parameter in the `<<<...>>>` execution configuration:

```cpp
kernel_name<<<gridDim, blockDim, sharedMemBytes, stream>>>(...);
```

| Parameter        | Type          | Description                                      |
|------------------|---------------|--------------------------------------------------|
| gridDim          | dim3          | The dimensions of the grid of thread blocks.     |
| blockDim         | dim3          | The dimensions of the thread blocks.             |
| sharedMemBytes   | size_t        | Optional: Bytes of dynamic shared memory per block. |
| stream           | cudaStream_t  | The stream to which this kernel is enqueued.     |

## Core Benefits of Using Streams

- **Concurrent Kernel Execution**: Independent kernels launched in different streams can execute in parallel, provided the GPU has sufficient resources.
- **Overlap Data Transfer and Computation**: This is the most powerful feature. You can start a data copy in one stream while a kernel processes data from a previous copy in another stream. This hides data transfer latency and keeps the GPU busy.

Example workflow:
- `cudaMemcpyAsync()` in streamA (copy chunk N)
- `kernel_launch()` in streamB (process chunk N-1)


## Key Considerations and Pitfalls

- **Data Races**: If two streams access the same memory location without proper synchronization, and at least one access is a write, you will have a data race. Ensure operations are correctly ordered using events or stream synchronization.
- **Synchronization**: Managing dependencies between streams can be complex. Use `cudaStreamSynchronize()` to block the CPU until a stream is finished, or use `cudaEvents` for fine-grained, inter-stream dependencies.
- **Hardware Dependency**: True concurrency depends on the GPU's architecture and available resources. A device with few Streaming Multiprocessors (SMs) may serialize tasks that would run concurrently on a larger GPU.
- **Resource Contention**: Kernels in different streams might still execute serially if they compete for limited resources, such as memory bandwidth or specific execution units.
- **Pinned Memory**: To overlap host-to-device memory copies with kernel execution, you must use pinned (non-pageable) host memory. Pinned memory is a limited resource and should be managed carefully.