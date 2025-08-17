#include <iostream>
#include <vector>
#include <random>
#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define WMMA_M 16 
#define WMMA_N 16 
#define WMMA_K 16 

#define M_GLOBAL 256
#define N_GLOBAL 256  
#define K_GLOBAL 256

#define WARP_SIZE 32

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

//D = A * B + C
__global__ void wmma_gemm_kernel(half *a, half *b, float *c, float *d, int M, int N, int K) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(acc_frag, 0.0f);
    
    int a_row = warpM * WMMA_M;
    int b_col = warpN * WMMA_N;

    for (int k = 0; k < K; k += WMMA_K) {
        int a_col = k;
        int b_row = k;

        if (a_row < M && a_col < K && b_row < K && b_col < N) {

            wmma::load_matrix_sync(a_frag, a + a_row * K + a_col, K);
            wmma::load_matrix_sync(b_frag, b + b_row * N + b_col, N);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    int c_row = warpM * WMMA_M;
    int c_col = warpN * WMMA_N;
    
    if (c_row < M && c_col < N) {
        wmma::load_matrix_sync(c_frag, c + c_row * N + c_col, N, wmma::mem_row_major);

        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = acc_frag.x[i] + c_frag.x[i];
        }
        wmma::store_matrix_sync(d + c_row * N + c_col, c_frag, N, wmma::mem_row_major);
    }
}

void init_matrices(half *A, half *B, float *C, int M, int N, int K) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < M * K; i++) {
        A[i] = __float2half(dis(gen));
    }

    for (int i = 0; i < K * N; i++) {
        B[i] = __float2half(dis(gen));
    }

    for (int i = 0; i < M * N; i++) {
        C[i] = dis(gen);
    }
}

int main() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (prop.major < 7) {
        std::cerr << "Tensor Cores require compute capability 7.0 or higher" << std::endl;
        return 1;
    }

    const int M = M_GLOBAL;
    const int N = N_GLOBAL;
    const int K = K_GLOBAL;
    
    half *h_A = new half[M * K];
    half *h_B = new half[K * N]; 
    float *h_C = new float[M * N];
    float *h_D = new float[M * N];

    init_matrices(h_A, h_B, h_C, M, N, K);

    half *d_A, *d_B;
    float *d_C, *d_D;
    
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(4 * WARP_SIZE, 4); 
    dim3 gridDim((M + WMMA_M * 4 - 1) / (WMMA_M * 4), (N + WMMA_N * 4 - 1) / (WMMA_N * 4));
    
    std::cout << "Grid: " << gridDim.x << "x" << gridDim.y << ", Block: " << blockDim.x << "x" << blockDim.y << std::endl;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "Launching WMMA kernel..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start));
    
    wmma_gemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, d_D, M, N, K);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaGetLastError());

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    double flops = 2.0 * M * N * K; 
    double tflops = flops / (milliseconds * 1e9);
    
    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_D, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_D;
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    std::cout << "WMMA example completed successfully!" << std::endl;
    
    return 0;
}
