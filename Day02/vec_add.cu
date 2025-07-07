#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
using namespace std;

// Kernel for vector addition
__global__ void vectadd_kernel(int *A, int *B, int *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// GPU-based vector addition
void vectadd_gpu(int *A, int *B, int *C, int n, float &gpu_time_ms) {
    int size = n * sizeof(int);
    int *d_a, *d_b, *d_c;

    // To measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, C, size, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    vectadd_kernel<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);
    cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// CPU-based vector addition
void vectadd_cpu(int *A, int *B, int *C, int n, double &cpu_time_ms) {
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;
    cpu_time_ms = duration.count();
}

int main() {
    int n = 10000000;
    size_t size = n * sizeof(int);

    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);
    int *gpu_sum = (int*)malloc(size);

    // Initialize vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    float gpu_time_ms;
    double cpu_time_ms;

    // CPU Vector Addition
    vectadd_cpu(h_a, h_b, h_c, n, cpu_time_ms);


    // GPU Vector Addition
    vectadd_gpu(h_a, h_b, gpu_sum, n, gpu_time_ms);

    // Verify
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (abs(gpu_sum[i] - h_c[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    // Output
    cout << "GPU Time: " << gpu_time_ms << " ms" << endl;
    cout << "CPU Time: " << cpu_time_ms << " ms" << endl;
    cout << "Result correct: " << (correct ? "Yes" : "No") << endl;

    free(h_a);
    free(h_b);
    free(h_c);
    free(gpu_sum);

    return 0;
}