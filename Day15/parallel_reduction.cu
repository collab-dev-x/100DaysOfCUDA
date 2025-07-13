#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <climits>
#include <cfloat>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

// Warp-level reduction using shuffle intrinsics
__device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ float warpReduceMin(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Block-level reduction kernel with warp optimization
__global__ void reductionKernel(float* input, float* sum_result, float* min_result, float* max_result, int n) {
    __shared__ float sdata_sum[BLOCK_SIZE/WARP_SIZE];
    __shared__ float sdata_min[BLOCK_SIZE/WARP_SIZE];
    __shared__ float sdata_max[BLOCK_SIZE/WARP_SIZE];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    
    // Initialize values for this thread
    float sum_val = 0.0f;
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    
    // Grid-stride loop with loop unrolling
    while (i < n) {
        float val = input[i];
        sum_val += val;
        min_val = fminf(min_val, val);
        max_val = fmaxf(max_val, val);
        
        // Unroll by processing multiple elements if possible
        if (i + blockDim.x * gridDim.x < n) {
            float val2 = input[i + blockDim.x * gridDim.x];
            sum_val += val2;
            min_val = fminf(min_val, val2);
            max_val = fmaxf(max_val, val2);
        }
        
        i += 2 * blockDim.x * gridDim.x;
    }
    
    // Warp-level reduction
    sum_val = warpReduceSum(sum_val);
    min_val = warpReduceMin(min_val);
    max_val = warpReduceMax(max_val);
    
    // Store warp results in shared memory
    if (laneId == 0) {
        sdata_sum[warpId] = sum_val;
        sdata_min[warpId] = min_val;
        sdata_max[warpId] = max_val;
    }
    
    __syncthreads();
    
    // Final reduction within first warp
    if (warpId == 0) {
        sum_val = (laneId < blockDim.x / WARP_SIZE) ? sdata_sum[laneId] : 0.0f;
        min_val = (laneId < blockDim.x / WARP_SIZE) ? sdata_min[laneId] : FLT_MAX;
        max_val = (laneId < blockDim.x / WARP_SIZE) ? sdata_max[laneId] : -FLT_MAX;
        
        sum_val = warpReduceSum(sum_val);
        min_val = warpReduceMin(min_val);
        max_val = warpReduceMax(max_val);
        
        if (laneId == 0) {
            atomicAdd(sum_result, sum_val);
            atomicMin((int*)min_result, __float_as_int(min_val));
            atomicMax((int*)max_result, __float_as_int(max_val));
        }
    }
}

// CPU sequential implementation for comparison
void cpuReduction(const std::vector<float>& input, float& sum, float& min_val, float& max_val) {
    sum = 0.0f;
    min_val = FLT_MAX;
    max_val = -FLT_MAX;
    
    for (float val : input) {
        sum += val;
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
}

int main() {
    const int N = 1000000;
    std::vector<float> h_input(N);
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
    }
    
    // CPU timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    float cpu_sum, cpu_min, cpu_max;
    cpuReduction(h_input, cpu_sum, cpu_min, cpu_max);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    
    // GPU implementation
    float *d_input, *d_sum, *d_min, *d_max;
    float h_sum = 0.0f, h_min = FLT_MAX, h_max = -FLT_MAX;
    
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    cudaMalloc(&d_min, sizeof(float));
    cudaMalloc(&d_max, sizeof(float));
    
    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_sum, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min, &h_min, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &h_max, sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch configuration
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    numBlocks = std::min(numBlocks, 65535); // Limit grid size
    
    // GPU timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    
    cudaEventRecord(start_gpu);
    reductionKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_sum, d_min, d_max, N);
    cudaEventRecord(stop_gpu);
    
    cudaEventSynchronize(stop_gpu);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    
    // Copy results back
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Results
    std::cout << "=== Parallel Reduction Results ===" << std::endl;
    std::cout << "CPU Sum: " << cpu_sum << ", Min: " << cpu_min << ", Max: " << cpu_max << std::endl;
    std::cout << "GPU Sum: " << h_sum << ", Min: " << h_min << ", Max: " << h_max << std::endl;
    std::cout << "CPU Time: " << cpu_time.count() << " μs" << std::endl;
    std::cout << "GPU Time: " << gpu_time * 1000 << " μs" << std::endl;
    std::cout << "Speedup: " << static_cast<float>(cpu_time.count()) / (gpu_time * 1000) << "x" << std::endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_sum);
    cudaFree(d_min);
    cudaFree(d_max);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    
    return 0;
}