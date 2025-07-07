#include<cstdio>

// This is a kernel function that runs on the GPU
__global__ void hello_from_gpu() {
    printf("Hello World");
}

int main() {
    hello_from_gpu<<<1,1>>>();

    // Wait for GPU to finish before accessing results
    cudaDeviceSynchronize();

    return 0;
}
