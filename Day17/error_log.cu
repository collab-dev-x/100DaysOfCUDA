#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

class CudaLogger {
private:
    std::ofstream logFile;
    
public:
    CudaLogger(const std::string& filename) {
        logFile.open(filename, std::ios::app);
        if (!logFile.is_open()) {
            std::cerr << "Failed to open log file: " << filename << std::endl;
        }
    }
    
    ~CudaLogger() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }
    
    void logError(cudaError_t err, const std::string& context, 
                  const std::string& file, int line) {
        std::string timestamp = getCurrentTime();
        std::string errorMsg = "[" + timestamp + "] CUDA ERROR: " + 
                              cudaGetErrorString(err) + " | Context: " + context +
                              " | File: " + file + ":" + std::to_string(line);

        if (logFile.is_open()) {
            logFile << errorMsg << std::endl;
            logFile.flush();
        }

        std::cerr << errorMsg << std::endl;
    }
    
    void logInfo(const std::string& message) {
        std::string timestamp = getCurrentTime();
        std::string infoMsg = "[" + timestamp + "] INFO: " + message;
        
        if (logFile.is_open()) {
            logFile << infoMsg << std::endl;
            logFile.flush();
        }
        
        std::cout << infoMsg << std::endl;
    }
    
private:
    std::string getCurrentTime() {
        time_t now = time(0);
        char* timeStr = ctime(&now);
        std::string result(timeStr);
        result.pop_back();
        return result;
    }
};

CudaLogger logger("cuda_errors.log");

void checkLastCudaError(const std::string& context) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        logger.logError(err, "Last error after " + context, __FILE__, __LINE__);
        exit(1);
    }
}

__global__ void sampleKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * 2.0f;
    }
}

void logDeviceInfo() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    logger.logInfo("Number of CUDA devices: " + std::to_string(deviceCount));
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        std::stringstream ss;
        ss << "Device " << i << ": " << prop.name 
           << " (Compute " << prop.major << "." << prop.minor << ")";
        logger.logInfo(ss.str());
    }
}


int main() {
    logger.logInfo("Starting CUDA error handling demonstration");
    logDeviceInfo();
    
    const int size = 1024;
    const size_t bytes = size * sizeof(float);
    
    float *h_data = nullptr;
    float *d_data = nullptr;
    
    try {
        // Allocate host memory
        h_data = new float[size];
        for (int i = 0; i < size; i++) {
            h_data[i] = static_cast<float>(i);
        }
        logger.logInfo("Host memory allocated successfully");
        
        // Allocate device memory with error checking
        CUDA_CHECK(cudaMalloc(&d_data, bytes));
        logger.logInfo("Device memory allocated: " + std::to_string(bytes) + " bytes");
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
        logger.logInfo("Data copied to device");
        
        // Launch kernel
        dim3 blockSize(256);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
        
        logger.logInfo("Launching kernel with " + std::to_string(gridSize.x) + 
                       " blocks of " + std::to_string(blockSize.x) + " threads");
        
        sampleKernel<<<gridSize, blockSize>>>(d_data, size);
        
        // Check for kernel launch errors
        checkLastCudaError("kernel launch");
        
        // Synchronize and check for execution errors
        CUDA_CHECK(cudaDeviceSynchronize());
        logger.logInfo("Kernel executed successfully");
        
        // Copy result back
        CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
        logger.logInfo("Result copied back to host");
        
        // Verify first few results
        std::stringstream ss;
        ss << "First 5 results: ";
        for (int i = 0; i < 5; i++) {
            ss << h_data[i] << " ";
        }
        logger.logInfo(ss.str());
        
    } catch (const std::exception& e) {
        logger.logError(cudaGetLastError(), "Exception caught: " + std::string(e.what()), 
                       __FILE__, __LINE__);
    }
    
    // Cleanup
    if (d_data) {
        cudaError_t err = cudaFree(d_data);
        if (err != cudaSuccess) {
            logger.logError(err, "Device memory cleanup", __FILE__, __LINE__);
        } else {
            logger.logInfo("Device memory freed");
        }
    }
    
    if (h_data) {
        delete[] h_data;
        logger.logInfo("Host memory freed");
    }

    CUDA_CHECK(cudaDeviceReset());
    logger.logInfo("CUDA device reset");
    return 0;
}