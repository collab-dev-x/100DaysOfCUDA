// file: measure.cu
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <nvml.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                          \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err)                    \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";               \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) C[idx] = A[idx] + B[idx];
}

void checkNvml(nvmlReturn_t result, const char* msg) {
  if (result != NVML_SUCCESS) {
    std::cerr << "NVML Error: " << msg << ": " 
              << nvmlErrorString(result) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

int main() {
  const int N = 100000000;
  const size_t bytes = N * sizeof(float);

  // 1) Initialize NVML
  checkNvml(nvmlInit(), "Failed to init NVML");
  nvmlDevice_t dev;
  checkNvml(nvmlDeviceGetHandleByIndex(0, &dev), "GetHandleByIndex");

  // 2) Allocate and initialize host data
  std::vector<float> hA(N, 1.0f), hB(N, 2.0f), hC(N);
  float *dA, *dB, *dC;
  CHECK_CUDA(cudaMalloc(&dA, bytes));
  CHECK_CUDA(cudaMalloc(&dB, bytes));
  CHECK_CUDA(cudaMalloc(&dC, bytes));
  CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

  // 3) Query NVML metrics before kernel
  nvmlUtilization_t util;
  nvmlMemory_t mem;
  unsigned int temp, power_mW;
  checkNvml(nvmlDeviceGetUtilizationRates(dev, &util), "GetUtilizationRates");
  checkNvml(nvmlDeviceGetMemoryInfo(dev, &mem), "GetMemoryInfo");
  checkNvml(nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp), "GetTemp");
  checkNvml(nvmlDeviceGetPowerUsage(dev, &power_mW), "GetPowerUsage");

  std::cout << "Before Kernel:\n"
            << " GPU Utilization: " << util.gpu << "%\n"
            << " Memory Utilization: " << (mem.used * 100 / mem.total) << "%\n"
            << " Temperature: " << temp << " C\n"
            << " Power Draw: " << (power_mW / 1000.0f) << " W\n\n";

  // 4) Time the kernel
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  int threads = 256;
  int blocks  = (N + threads - 1) / threads;

  CHECK_CUDA(cudaEventRecord(start));
  vecAdd<<<blocks, threads>>>(dA, dB, dC, N);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  std::cout << "Kernel execution time: " << ms << " ms\n";
  std::cout << "Approx. Bandwidth: " 
            << (bytes * 3) / (ms / 1e3) / (1<<30) 
            << " GB/s\n\n";  // 2 reads + 1 write

  // 5) Query NVML metrics after kernel
  checkNvml(nvmlDeviceGetUtilizationRates(dev, &util), "GetUtilizationRates");
  checkNvml(nvmlDeviceGetMemoryInfo(dev, &mem), "GetMemoryInfo");
  checkNvml(nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp), "GetTemp");
  checkNvml(nvmlDeviceGetPowerUsage(dev, &power_mW), "GetPowerUsage");

  std::cout << "After Kernel:\n"
            << " GPU Utilization: " << util.gpu << "%\n"
            << " Memory Utilization: " << (mem.used * 100 / mem.total) << "%\n"
            << " Temperature: " << temp << " C\n"
            << " Power Draw: " << (power_mW / 1000.0f) << " W\n";

  // 6) Cleanup
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  nvmlShutdown();
  return 0;
}
