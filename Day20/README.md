# Measuring Kernel Performance.

## 1. Prerequisites and Setup

I am using NVML and CUDA for profiling, include the necessary headers and link against the NVML library.

### Headers
```cpp
#include <nvml.h>
#include <cuda_runtime.h>
```

### Linker Configuration
Link against the NVML library. For a typical CUDA installation on Windows, add:
```
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\x64\nvml.lib"
```
*Note*: Replace `v12.x` with your CUDA Toolkit version.

## 2. NVML Initialization and Baseline Metrics

Before launching the kernel, initialize NVML and capture the GPU's baseline state to measure its idle or background activity.

```cpp
// Initialize NVML
nvmlInit();

// Get handle to GPU (device 0)
nvmlDevice_t device;
nvmlDeviceGetHandleByIndex(0, &device);

// Capture baseline metrics
nvmlUtilization_t preKernelUtilization;
nvmlDeviceGetUtilizationRates(device, &preKernelUtilization);

nvmlMemory_t preKernelMemoryInfo;
nvmlDeviceGetMemoryInfo(device, &preKernelMemoryInfo);

unsigned int preKernelTemp;
nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &preKernelTemp);

unsigned int preKernelPower;
nvmlDeviceGetPowerUsage(device, &preKernelPower);
```

These metrics include GPU utilization, memory usage, temperature, and power draw.

## 3. Kernel Execution and Timing

Use CUDA Events to measure the kernel's execution time accurately, accounting for the asynchronous nature of GPU operations.

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);

vecAdd<<<blocks, threads>>>(/* arguments */);

cudaEventRecord(stop);

cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

### Calculate Bandwidth
For a vector addition kernel (`c = a + b`), which involves two reads (vectors `a` and `b`) and one write (vector `c`), calculate the effective memory bandwidth:

```cpp
float time_s = milliseconds / 1000.0f; // Convert to seconds
double bandwidth_GBps = (3.0 * (size_of_vector)) / (time_s * (1 << 30)); // Convert to GB/s
```

Here, `(1 << 30)` converts bytes to gigabytes (2³⁰).

## 4. Post-Kernel NVML Metrics

After the kernel completes, capture the same NVML metrics to assess the workload's impact on the GPU.

```cpp
// Capture post-kernel metrics
nvmlUtilization_t postKernelUtilization;
nvmlDeviceGetUtilizationRates(device, &postKernelUtilization);

nvmlMemory_t postKernelMemoryInfo;
nvmlDeviceGetMemoryInfo(device, &postKernelMemoryInfo);

unsigned int postKernelTemp;
nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &postKernelTemp);

unsigned int postKernelPower;
nvmlDeviceGetPowerUsage(device, &postKernelPower);
```

Compare these metrics with the baseline to quantify changes in utilization, memory, temperature, and power.

## 5. NVML Cleanup

Release NVML resources to prevent memory leaks.

```cpp
nvmlShutdown();
```
