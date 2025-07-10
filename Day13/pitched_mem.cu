#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Kernel for non-pitched memory access
__global__ void processMatrix_NonPitched(float* matrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;
        // Simple processing: multiply by 2 and add row+col
        matrix[idx] = matrix[idx] * 2.0f + (row + col);
    }
}

// Kernel for pitched memory access
__global__ void processMatrix_Pitched(float* matrix, size_t pitch, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        // Calculate the actual memory address using pitch
        float* row_ptr = (float*)((char*)matrix + row * pitch);
        // Simple processing: multiply by 2 and add row+col
        row_ptr[col] = row_ptr[col] * 2.0f + (row + col);
    }
}

// Initialize matrix with test data
void initializeMatrix(float* matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            matrix[i * width + j] = static_cast<float>(i * width + j);
        }
    }
}

// Verify results are identical
bool verifyResults(float* result1, float* result2, int width, int height, float tolerance = 1e-5f) {
    for (int i = 0; i < height * width; i++) {
        if (std::abs(result1[i] - result2[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " 
                      << result1[i] << " vs " << result2[i] << std::endl;
            return false;
        }
    }
    return true;
}

void printMatrixInfo(int width, int height, size_t pitch = 0) {
    std::cout << std::setfill('=') << std::setw(60) << "" << std::endl;
    std::cout << "Matrix dimensions: " << width << " x " << height << std::endl;
    std::cout << "Total elements: " << width * height << std::endl;
    std::cout << "Memory size (non-pitched): " << width * height * sizeof(float) << " bytes" << std::endl;
    if (pitch > 0) {
        std::cout << "Pitch: " << pitch << " bytes" << std::endl;
        std::cout << "Memory size (pitched): " << pitch * height << " bytes" << std::endl;
        std::cout << "Memory efficiency: " << 
            (float)(width * sizeof(float)) / pitch * 100.0f << "%" << std::endl;
    }
    std::cout << std::setfill('=') << std::setw(60) << "" << std::endl;
}

int main() {
    // Matrix dimensions - try different sizes to see the effect
    const int width = 1023;  // Intentionally not a power of 2
    const int height = 1000;
    const size_t matrix_size = width * height * sizeof(float);
    
    std::cout << "CUDA Pitched Memory vs Non-Pitched Memory Comparison\n";
    printMatrixInfo(width, height);
    
    // Host memory allocation
    float* h_matrix = new float[width * height];
    float* h_result_nonpitched = new float[width * height];
    float* h_result_pitched = new float[width * height];
    
    // Initialize test data
    initializeMatrix(h_matrix, width, height);
    
    // ===== NON-PITCHED MEMORY APPROACH =====
    std::cout << "\n1. NON-PITCHED MEMORY APPROACH:" << std::endl;
    std::cout << "   - Uses regular cudaMalloc()" << std::endl;
    std::cout << "   - Row width = " << width * sizeof(float) << " bytes" << std::endl;
    std::cout << "   - May have poor memory alignment" << std::endl;
    
    float* d_matrix_nonpitched;
    CUDA_CHECK(cudaMalloc(&d_matrix_nonpitched, matrix_size));
    CUDA_CHECK(cudaMemcpy(d_matrix_nonpitched, h_matrix, matrix_size, cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Warm-up run
    processMatrix_NonPitched<<<gridSize, blockSize>>>(d_matrix_nonpitched, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing non-pitched approach
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 100;
    
    for (int i = 0; i < iterations; i++) {
        processMatrix_NonPitched<<<gridSize, blockSize>>>(d_matrix_nonpitched, width, height);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_nonpitched = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    CUDA_CHECK(cudaMemcpy(h_result_nonpitched, d_matrix_nonpitched, matrix_size, cudaMemcpyDeviceToHost));
    
    std::cout << "   Time for " << iterations << " iterations: " 
              << duration_nonpitched.count() << " microseconds" << std::endl;
    std::cout << "   Average time per iteration: " 
              << duration_nonpitched.count() / iterations << " microseconds" << std::endl;
    
    // ===== PITCHED MEMORY APPROACH =====
    std::cout << "\n2. PITCHED MEMORY APPROACH:" << std::endl;
    std::cout << "   - Uses cudaMallocPitch() for optimal alignment" << std::endl;
    std::cout << "   - CUDA driver chooses optimal pitch for hardware" << std::endl;
    
    float* d_matrix_pitched;
    size_t pitch;
    CUDA_CHECK(cudaMallocPitch(&d_matrix_pitched, &pitch, width * sizeof(float), height));
    
    std::cout << "   - Actual pitch chosen by CUDA: " << pitch << " bytes" << std::endl;
    std::cout << "   - Row width requested: " << width * sizeof(float) << " bytes" << std::endl;
    std::cout << "   - Extra padding per row: " << pitch - width * sizeof(float) << " bytes" << std::endl;
    
    // Copy data using pitched memory copy
    CUDA_CHECK(cudaMemcpy2D(d_matrix_pitched, pitch, 
                           h_matrix, width * sizeof(float),
                           width * sizeof(float), height,
                           cudaMemcpyHostToDevice));
    
    // Warm-up run
    processMatrix_Pitched<<<gridSize, blockSize>>>(d_matrix_pitched, pitch, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing pitched approach
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        processMatrix_Pitched<<<gridSize, blockSize>>>(d_matrix_pitched, pitch, width, height);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    end = std::chrono::high_resolution_clock::now();
    auto duration_pitched = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Copy result back using pitched memory copy
    CUDA_CHECK(cudaMemcpy2D(h_result_pitched, width * sizeof(float),
                           d_matrix_pitched, pitch,
                           width * sizeof(float), height,
                           cudaMemcpyDeviceToHost));
    
    std::cout << "   Time for " << iterations << " iterations: " 
              << duration_pitched.count() << " microseconds" << std::endl;
    std::cout << "   Average time per iteration: " 
              << duration_pitched.count() / iterations << " microseconds" << std::endl;
    
    // ===== PERFORMANCE COMPARISON =====
    std::cout << "\n" << std::setfill('=') << std::setw(60) << "" << std::endl;
    std::cout << "PERFORMANCE COMPARISON:" << std::endl;
    std::cout << std::setfill('=') << std::setw(60) << "" << std::endl;
    
    float speedup = (float)duration_nonpitched.count() / duration_pitched.count();
    std::cout << "Non-pitched time: " << duration_nonpitched.count() << " μs" << std::endl;
    std::cout << "Pitched time:     " << duration_pitched.count() << " μs" << std::endl;
    
    if (speedup > 1.0f) {
        std::cout << "Pitched memory is " << std::fixed << std::setprecision(2) 
                  << speedup << "x FASTER" << std::endl;
    } else {
        std::cout << "Non-pitched memory is " << std::fixed << std::setprecision(2) 
                  << 1.0f/speedup << "x faster" << std::endl;
    }
    
    // Verify results are identical
    if (verifyResults(h_result_nonpitched, h_result_pitched, width, height)) {
        std::cout << "✓ Results verification: PASSED (results are identical)" << std::endl;
    } else {
        std::cout << "✗ Results verification: FAILED (results differ)" << std::endl;
    }
    
    // ===== TECHNICAL EXPLANATION =====
    std::cout << "\n" << std::setfill('=') << std::setw(60) << "" << std::endl;
    std::cout << "WHY PITCHED MEMORY MATTERS:" << std::endl;
    std::cout << std::setfill('=') << std::setw(60) << "" << std::endl;
    std::cout << "1. MEMORY ALIGNMENT:\n";
    std::cout << "   - GPU memory controllers work optimally with aligned access\n";
    std::cout << "   - Pitched memory ensures rows start at optimal boundaries\n";
    std::cout << "   - Non-pitched memory may cause unaligned access penalties\n\n";
    
    std::cout << "2. COALESCED ACCESS:\n";
    std::cout << "   - GPU threads in a warp access consecutive memory locations\n";
    std::cout << "   - Proper alignment enables coalesced memory transactions\n";
    std::cout << "   - Poor alignment can reduce memory bandwidth utilization\n\n";
    
    std::cout << "3. CACHE EFFICIENCY:\n";
    std::cout << "   - Aligned access patterns improve cache hit rates\n";
    std::cout << "   - Better cache utilization = fewer memory stalls\n\n";
    
    std::cout << "4. WHEN TO USE PITCHED MEMORY:\n";
    std::cout << "   - 2D arrays and matrices\n";
    std::cout << "   - When row width is not a multiple of optimal alignment\n";
    std::cout << "   - Performance-critical applications\n";
    std::cout << "   - Large datasets with regular access patterns\n";
    
    // Memory usage analysis
    size_t nonpitched_memory = matrix_size;
    size_t pitched_memory = pitch * height;
    float memory_overhead = ((float)pitched_memory / nonpitched_memory - 1.0f) * 100.0f;
    
    std::cout << "\nMEMORY OVERHEAD ANALYSIS:\n";
    std::cout << "Non-pitched memory: " << nonpitched_memory << " bytes\n";
    std::cout << "Pitched memory:     " << pitched_memory << " bytes\n";
    std::cout << "Memory overhead:    " << std::fixed << std::setprecision(1) 
              << memory_overhead << "%\n";
    
    // Cleanup
    delete[] h_matrix;
    delete[] h_result_nonpitched;  
    delete[] h_result_pitched;
    CUDA_CHECK(cudaFree(d_matrix_nonpitched));
    CUDA_CHECK(cudaFree(d_matrix_pitched));
    
    std::cout << "\nProgram completed successfully!" << std::endl;
    return 0;
}