# CudaLogger Technical Summary

CudaLogger is a C++ logging utility designed for CUDA applications, providing persistent, timestamped logging for both standard messages and API errors.

## Initialization & Lifecycle
- **Single Global Instance**: Instantiated at program startup.
- **RAII Utilization**: Opens `cuda_errors.log` in append mode (`std::ios::app`) upon construction and automatically closes the file stream upon destruction.

## Logging Operations
- **`logInfo(message)`**:
  - Prepends a `[timestamp] INFO:` prefix to the message.
  - Writes output to both `std::cout` and the log file.
- **`logError(err, context, file, line)`**:
  - Formats a detailed error message using `cudaGetErrorString(err)`.
  - Includes a `[timestamp] CUDA ERROR:` prefix, context, file, and line number.
  - Writes output to `std::cerr` and the log file.

## Data Persistence
- Every write operation is immediately followed by a `flush()` call to the file stream.
- Ensures logs are committed to disk instantly, critical for capturing data from unexpected application terminations.