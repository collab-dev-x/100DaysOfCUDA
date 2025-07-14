#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <cassert>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

bool checkCudaError(cudaError_t err, const std::string& operation, 
                   const char* file, int line, bool printError = true) {
    if (err != cudaSuccess) {
        if (printError) {
            std::cerr << "CUDA Error in " << operation << ": " 
                      << cudaGetErrorString(err) << " (" << file << ":" << line << ")" 
                      << std::endl;
        }
        return false;
    }
    return true;
}

#define CUDA_TRY(call, operation) \
    checkCudaError(call, operation, __FILE__, __LINE__)

cudaError_t getAndClearLastError(const std::string& context = "") {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess && !context.empty()) {
        std::cerr << "Last CUDA Error after " << context << ": " 
                  << cudaGetErrorString(err) << std::endl;
    }
    return err;
}

bool safeCudaDeviceSync(const std::string& context = "") {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Sync Error";
        if (!context.empty()) {
            std::cerr << " after " << context;
        }
        std::cerr << ": " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

__global__ void validKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

__global__ void kernelWithError(float* data, int size, bool causeError) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (causeError && idx==0) {
        data[size + 1000] = 999.0f;
    }
    
    // if (idx < size) {
    //     data[idx] = sqrtf(data[idx]);
    // }
}

int main() {
    try {
        int deviceCount;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            std::cerr << "No CUDA devices found or CUDA not available!" << std::endl;
            return 1;
        }

        const int size = 100;
        const size_t bytes = size * sizeof(float);
        float *h_data = new float[size];
        float *d_data = nullptr;

        for (int i = 0; i < size; i++) {
            h_data[i] = static_cast<float>(i + 1);
        }

        CUDA_CHECK(cudaMalloc(&d_data, bytes));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
        dim3 blockSize(256);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
        validKernel<<<gridSize, blockSize>>>(d_data, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

        kernelWithError<<<gridSize, blockSize>>>(d_data, size, true);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float* nullPtr = nullptr;
        if (!CUDA_TRY(cudaMemset(nullPtr, 0, 1024), "Null pointer operation")) {
            std::cout << "Handled null pointer error" << std::endl;
        }
        CUDA_CHECK(cudaFree(d_data));
        delete[] h_data;



    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}