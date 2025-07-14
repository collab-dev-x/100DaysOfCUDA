# A Guide to Robust CUDA Error Handling

Effective error handling is critical for developing stable and debuggable CUDA applications. CUDA API calls can produce two types of errors:

- **Synchronous Errors**: Most API calls return an error code immediately, making them easy to check.
- **Asynchronous Errors**: Kernel launches and some memory operations return immediately. Errors from these operations are reported later, during a synchronizing event.

## CUDA_CHECK(call)

A macro for fatal error checking. It wraps a CUDA API call and terminates the program if the call fails, printing a detailed error message with the file and line number.

- **Purpose**: Ensures mandatory operations succeed. Use this for critical setup steps like memory allocation or context creation, where the program cannot continue after a failure.

## CUDA_TRY(call, operation)

A macro for non-fatal error handling. It uses the underlying `checkCudaError` function to check a CUDA call but allows the program to continue execution. It returns `true` on success and `false` on failure.

- **Purpose**: Handles recoverable errors or logs failures for optional operations without crashing the application.

## getAndClearLastError(context)

Checks for and clears any pending asynchronous error. This is a lightweight way to check the CUDA runtime's "sticky" error state after a kernel launch.

- **Purpose**: Perform a quick, non-blocking check for errors from recent asynchronous operations. The context string helps identify which operation failed.

## safeCudaDeviceSync(context)

Synchronizes the entire device by calling `cudaDeviceSynchronize()` and checks its return value. This is the most reliable way to confirm that all preceding asynchronous operations, including kernels, have completed successfully.

- **Purpose**: Forces the CPU to wait for the GPU to finish its work, turning any latent asynchronous error into a synchronous, detectable one. Essential for debugging kernel failures.

## Summary & Best Practices

| Utility                   | Use Case                          | Behavior on Error                              |
|---------------------------|-----------------------------------|-----------------------------------------------|
| `CUDA_CHECK(call)`        | Critical Setup (e.g., `cudaMalloc`) | Prints error and exits program.               |
| `CUDA_TRY(call, op)`      | Recoverable Operations            | Prints error and returns `false`.             |
| `getAndClearLastError(ctx)` | Quick Post-Kernel Check          | Prints error and returns the `cudaError_t`.   |
| `safeCudaDeviceSync(ctx)`  | Reliable Kernel Debugging         | Blocks execution, then prints error and returns `false`. |