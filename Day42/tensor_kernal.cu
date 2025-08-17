#include <iostream>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define MATRIX_SIZE 16

__global__ void tensorCoreGEMM(const half* A, const half* B, float* C) {
    wmma::fragment<wmma::matrix_a, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    wmma::load_matrix_sync(a_frag, A, MATRIX_SIZE);
    wmma::load_matrix_sync(b_frag, B, MATRIX_SIZE);

    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    wmma::store_matrix_sync(C, acc_frag, MATRIX_SIZE, wmma::mem_row_major);
}

int main() {
    const int matrix_elems = MATRIX_SIZE * MATRIX_SIZE;

    half *h_A = new half[matrix_elems];
    half *h_B = new half[matrix_elems];
    float *h_C = new float[matrix_elems];

    for (int i = 0; i < matrix_elems; ++i) {
        h_A[i] = __float2half(1.0f);
        h_B[i] = __float2half(1.0f);
    }

    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, matrix_elems * sizeof(half));
    cudaMalloc(&d_B, matrix_elems * sizeof(half));
    cudaMalloc(&d_C, matrix_elems * sizeof(float));

    cudaMemcpy(d_A, h_A, matrix_elems * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_elems * sizeof(half), cudaMemcpyHostToDevice);

    tensorCoreGEMM<<<1, 32>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, matrix_elems * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result matrix C (first 4 values):\n";
    for (int i = 0; i < 4; ++i)
        std::cout << h_C[i] << " ";
    std::cout << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
