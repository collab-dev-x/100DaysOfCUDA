#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define FILTER_SIZE 3

float h_sobel_h[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
float h_sobel_v[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

__global__ void img_to_patches(float *patches, unsigned char *img, int width, int height) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= height || col >= width) return;
    
    int patch_idx = (row * width + col) * 9;
    
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int y = row + i;
            int x = col + j;
            y = y < 0 ? 0 : y >= height ? height - 1 : y;
            x = x < 0 ? 0 : x >= width ? width - 1 : x;
            patches[patch_idx + (i + 1) * 3 + (j + 1)] = img[y * width + x];
        }
    }
}

__global__ void result_to_img(unsigned char *img, float *result, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;
    img[idx] = min(255.0f, max(0.0f, fabsf(result[idx])));
}

int main() {
    int width, height, channels;
    const char* input_path = "image.jpg";
    const char* h_output_path = "h_edges.png";
    const char* v_output_path = "v_edges.png";

    unsigned char* input_img = stbi_load(input_path, &width, &height, &channels, 1);
    if (!input_img) {
        std::cerr << "Failed to load image!\n";
        return -1;
    }

    int img_size = width * height;
    int patches_size = img_size * 9;

    unsigned char *d_input;
    float *d_patches, *d_h_filter, *d_v_filter, *d_h_result, *d_v_result;
    unsigned char *d_h_output, *d_v_output;

    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_patches, patches_size * sizeof(float));
    cudaMalloc(&d_h_filter, 9 * sizeof(float));
    cudaMalloc(&d_v_filter, 9 * sizeof(float));
    cudaMalloc(&d_h_result, img_size * sizeof(float));
    cudaMalloc(&d_v_result, img_size * sizeof(float));
    cudaMalloc(&d_h_output, img_size);
    cudaMalloc(&d_v_output, img_size);

    cudaMemcpy(d_input, input_img, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h_filter, h_sobel_h, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_filter, h_sobel_v, 9 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    
    img_to_patches<<<gridSize, blockSize>>>(d_patches, d_input, width, height);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f, beta = 0.0f;

    cublasSgemv(handle, CUBLAS_OP_T, 9, img_size, &alpha, d_patches, 9, d_h_filter, 1, &beta, d_h_result, 1);
    cublasSgemv(handle, CUBLAS_OP_T, 9, img_size, &alpha, d_patches, 9, d_v_filter, 1, &beta, d_v_result, 1);

    int linear_blocks = (img_size + 255) / 256;
    result_to_img<<<linear_blocks, 256>>>(d_h_output, d_h_result, img_size);
    result_to_img<<<linear_blocks, 256>>>(d_v_output, d_v_result, img_size);

    unsigned char* output_h_img = new unsigned char[img_size];
    unsigned char* output_v_img = new unsigned char[img_size];

    cudaMemcpy(output_h_img, d_h_output, img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_v_img, d_v_output, img_size, cudaMemcpyDeviceToHost);

    stbi_write_png(h_output_path, width, height, 1, output_h_img, width);
    stbi_write_png(v_output_path, width, height, 1, output_v_img, width);

    stbi_image_free(input_img);
    delete[] output_h_img;
    delete[] output_v_img;
    cudaFree(d_input);
    cudaFree(d_patches);
    cudaFree(d_h_filter);
    cudaFree(d_v_filter);
    cudaFree(d_h_result);
    cudaFree(d_v_result);
    cudaFree(d_h_output);
    cudaFree(d_v_output);
    cublasDestroy(handle);

    return 0;
}