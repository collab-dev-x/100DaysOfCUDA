#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Kernel WITHOUT atomic operations (demonstrates race conditions)
__global__ void reductionWithoutAtomics(int* input, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localId = threadIdx.x;
    
    // Shared memory for block-level reduction
    extern __shared__ int sdata[];
    
    // Load data into shared memory
    if (tid < n) {
        sdata[localId] = input[tid];
    } else {
        sdata[localId] = 0;
    }
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            sdata[localId] += sdata[localId + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 of each block writes partial sum to global memory
    // WITHOUT atomic operation - this causes race conditions!
    if (localId == 0) {
        *output += sdata[0];  // RACE CONDITION HERE!
    }
}

// Kernel WITH atomic operations (safe)
__global__ void reductionWithAtomics(int* input, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localId = threadIdx.x;
    
    // Shared memory for block-level reduction
    extern __shared__ int sdata[];
    
    // Load data into shared memory
    if (tid < n) {
        sdata[localId] = input[tid];
    } else {
        sdata[localId] = 0;
    }
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            sdata[localId] += sdata[localId + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 of each block atomically adds partial sum to global result
    if (localId == 0) {
        atomicAdd(output, sdata[0]);  // SAFE with atomic operation
    }
}

// Optimized version using multiple atomic operations
__global__ void reductionMultipleAtomics(int* input, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localId = threadIdx.x;
    
    extern __shared__ int sdata[];
    
    if (tid < n) {
        sdata[localId] = input[tid];
    } else {
        sdata[localId] = 0;
    }
    __syncthreads();
    
    // Reduce to warp level first
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (localId < stride) {
            sdata[localId] += sdata[localId + stride];
        }
        __syncthreads();
    }
    
    // Final warp reduction with multiple atomics
    if (localId < 32) {
        volatile int* vsdata = sdata;
        vsdata[localId] += vsdata[localId + 32];
        vsdata[localId] += vsdata[localId + 16];
        vsdata[localId] += vsdata[localId + 8];
        vsdata[localId] += vsdata[localId + 4];
        vsdata[localId] += vsdata[localId + 2];
        vsdata[localId] += vsdata[localId + 1];
        
        // Multiple threads can use atomics (demonstrates contention)
        if (localId < 8) {
            atomicAdd(output, vsdata[localId]);
        }
    }
}

// Demonstration of different atomic operations
__global__ void demonstrateAtomicOperations(int* data, int* results, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Different atomic operations
        atomicAdd(&results[0], data[tid]);           // Sum
        atomicMax(&results[1], data[tid]);           // Maximum
        atomicMin(&results[2], data[tid]);           // Minimum
        atomicAnd(&results[3], data[tid]);           // Bitwise AND
        atomicOr(&results[4], data[tid]);            // Bitwise OR
        atomicXor(&results[5], data[tid]);           // Bitwise XOR
        
        // Atomic compare-and-swap example
        int old_val, new_val;
        do {
            old_val = results[6];
            new_val = old_val + 1;
        } while (atomicCAS(&results[6], old_val, new_val) != old_val);
    }
}

// CPU reference implementation
int cpuSum(int* array, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += array[i];
    }
    return sum;
}

// Utility function to initialize array
void initializeArray(int* array, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        array[i] = rand() % 100 + 1;  // Values between 1-100
    }
}

// Performance timing utility
double getTime() {
    return (double)clock() / CLOCKS_PER_SEC;
}

int main() {
    printf("=== CUDA Atomic Operations Demonstration ===\n\n");
    
    const int N = 1024 * 1024;  // 1M elements
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Host memory allocation
    int* h_input = (int*)malloc(N * sizeof(int));
    int* h_output = (int*)malloc(sizeof(int));
    
    // Device memory allocation
    int* d_input;
    int* d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, sizeof(int));
    
    // Initialize input data
    initializeArray(h_input, N);
    
    // Calculate CPU reference
    printf("1. Computing CPU reference...\n");
    double start_time = getTime();
    int cpu_result = cpuSum(h_input, N);
    double cpu_time = getTime() - start_time;
    printf("   CPU Result: %d (Time: %.4f seconds)\n\n", cpu_result, cpu_time);
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Test 1: Reduction WITHOUT atomics (demonstrates race conditions)
    printf("2. Testing reduction WITHOUT atomics (race conditions):\n");
    for (int trial = 0; trial < 5; trial++) {
        *h_output = 0;
        cudaMemcpy(d_output, h_output, sizeof(int), cudaMemcpyHostToDevice);
        
        start_time = getTime();
        reductionWithoutAtomics<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
            d_input, d_output, N);
        cudaDeviceSynchronize();
        double gpu_time = getTime() - start_time;
        
        cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
        printf("   Trial %d: %d (Expected: %d) - %s (Time: %.4f seconds)\n", 
               trial + 1, *h_output, cpu_result, 
               (*h_output == cpu_result) ? "CORRECT" : "INCORRECT", gpu_time);
    }
    printf("\n");
    
    // Test 2: Reduction WITH atomics (correct results)
    printf("3. Testing reduction WITH atomics (correct):\n");
    for (int trial = 0; trial < 5; trial++) {
        *h_output = 0;
        cudaMemcpy(d_output, h_output, sizeof(int), cudaMemcpyHostToDevice);
        
        start_time = getTime();
        reductionWithAtomics<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
            d_input, d_output, N);
        cudaDeviceSynchronize();
        double gpu_time = getTime() - start_time;
        
        cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
        printf("   Trial %d: %d (Expected: %d) - %s (Time: %.4f seconds)\n", 
               trial + 1, *h_output, cpu_result, 
               (*h_output == cpu_result) ? "CORRECT" : "INCORRECT", gpu_time);
    }
    printf("\n");
    
    // Test 3: Multiple atomics (demonstrates contention)
    printf("4. Testing multiple atomics (contention demonstration):\n");
    *h_output = 0;
    cudaMemcpy(d_output, h_output, sizeof(int), cudaMemcpyHostToDevice);
    
    start_time = getTime();
    reductionMultipleAtomics<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
        d_input, d_output, N);
    cudaDeviceSynchronize();
    double contention_time = getTime() - start_time;
    
    cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    printf("   Result: %d (Expected: %d) - %s (Time: %.4f seconds)\n", 
           *h_output, cpu_result, 
           (*h_output == cpu_result) ? "CORRECT" : "INCORRECT", contention_time);
    printf("   Note: Higher contention due to multiple atomic operations per block\n\n");
    
    // Test 4: Demonstrate various atomic operations
    printf("5. Demonstrating various atomic operations:\n");
    int* d_atomic_results;
    int* h_atomic_results = (int*)calloc(7, sizeof(int));
    cudaMalloc(&d_atomic_results, 7 * sizeof(int));
    
    // Initialize results array
    h_atomic_results[1] = INT_MIN;  // For atomicMax
    h_atomic_results[2] = INT_MAX;  // For atomicMin
    h_atomic_results[3] = 0xFFFFFFFF;  // For atomicAnd
    cudaMemcpy(d_atomic_results, h_atomic_results, 7 * sizeof(int), cudaMemcpyHostToDevice);
    
    demonstrateAtomicOperations<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_atomic_results, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_atomic_results, d_atomic_results, 7 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("   atomicAdd (Sum):     %d\n", h_atomic_results[0]);
    printf("   atomicMax (Maximum): %d\n", h_atomic_results[1]);
    printf("   atomicMin (Minimum): %d\n", h_atomic_results[2]);
    printf("   atomicAnd (Bitwise): %d\n", h_atomic_results[3]);
    printf("   atomicOr  (Bitwise): %d\n", h_atomic_results[4]);
    printf("   atomicXor (Bitwise): %d\n", h_atomic_results[5]);
    printf("   atomicCAS (Counter): %d\n", h_atomic_results[6]);
    printf("\n");
    
    // Performance comparison
    printf("6. Performance Analysis:\n");
    printf("   CPU Time:                %.4f seconds\n", cpu_time);
    printf("   GPU with Atomics:        ~%.4f seconds\n", gpu_time);
    printf("   GPU with Contention:     %.4f seconds\n", contention_time);
    printf("   Speedup over CPU:        %.2fx\n", cpu_time / gpu_time);
    printf("\n");
    
    // Use cases and best practices
    printf("7. Atomic Operations - Use Cases & Best Practices:\n");
    printf("   Use Cases:\n");
    printf("   - Parallel reductions (sum, min, max)\n");
    printf("   - Histogram computation\n");
    printf("   - Global counters and accumulators\n");
    printf("   - Lock-free data structures\n");
    printf("   - Sparse matrix operations\n");
    printf("\n");
    printf("   Best Practices:\n");
    printf("   - Minimize atomic contention (fewer threads per memory location)\n");
    printf("   - Use local reduction first, then atomic for global updates\n");
    printf("   - Consider memory coalescing with atomic operations\n");
    printf("   - Use appropriate atomic operation for your data type\n");
    printf("   - Profile to identify atomic bottlenecks\n");
    printf("\n");
    
    printf("8. Key Takeaways:\n");
    printf("   - Atomics ensure correctness but can impact performance\n");
    printf("   - Race conditions without atomics produce incorrect results\n");
    printf("   - High contention on same memory location creates bottlenecks\n");
    printf("   - Block-level reduction + atomic global update is efficient\n");
    printf("   - Choose atomic operations based on your specific needs\n");
    
    // Cleanup
    free(h_input);
    free(h_output);
    free(h_atomic_results);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_atomic_results);
    
    return 0;
}