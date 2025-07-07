#include <iostream>
#include <cuda.h>
using namespace std;

__global__ void matMul(int* A, int* B, int* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n * n) {
        int row = idx / n;
        int col = idx % n;
        int sum =0;

        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[idx] = sum;
    }
}

int main() {
    int n;
    cin>>n;
    int size = n * n * sizeof(int);

    int * h_a = (int*)malloc(n * n * sizeof(int*));
    int * h_b = (int*)malloc(n * n * sizeof(int*));
    int * h_c = (int*)malloc(n * n * sizeof(int*));

    for(int i=0;i<n * n;i++){
        h_a[i] = i%100;
        cin>>h_a[i];
    }
    for(int i=0;i<n * n;i++){
        h_b[i] = i%100;
        cin>>h_b[i];
    }

    int *d_a , *d_b, *d_c;
    cudaMalloc((void**)&d_a, n * n * sizeof(int**));
    cudaMalloc((void**)&d_b, n * n * sizeof(int**));
    cudaMalloc((void**)&d_c, n * n * sizeof(int**));

    cudaMemcpy(d_a, h_a, n * n * sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * n * sizeof(int*), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (n * n + threadsPerBlock - 1) / threadsPerBlock;

    matMul<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "Matrix C (Result):\n";
    for (int i = 0; i < n * n; i++) {
        std::cout << h_c[i] << " ";
        if ((i + 1) % n == 0) std::cout << "\n";
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
