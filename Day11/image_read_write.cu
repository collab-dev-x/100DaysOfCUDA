#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel to separate RGB channels
__global__ void split_channels(unsigned char* img, unsigned char* r, unsigned char* g, unsigned char* b, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        int pixelIdx = y * width + x;
        r[pixelIdx] = img[idx];
        g[pixelIdx] = img[idx + 1];
        b[pixelIdx] = img[idx + 2];
    }
}

// CUDA kernel to merge RGB channels
__global__ void merge_channels(unsigned char* img, unsigned char* r, unsigned char* g, unsigned char* b, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        int pixelIdx = y * width + x;
        img[idx]     = r[pixelIdx];
        img[idx + 1] = g[pixelIdx];
        img[idx + 2] = b[pixelIdx];
    }
}

int main() {
    int width, height, channels;
    const char* input_path = "image.jpg";
    const char* output_path = "output.png";

    // Load image (CPU)
    unsigned char* input_img = stbi_load(input_path, &width, &height, &channels, 3);
    if (!input_img) {
        std::cerr << "Failed to load image!\n";
        return -1;
    }

    size_t img_size = width * height * 3 * sizeof(unsigned char);
    size_t channel_size = width * height * sizeof(unsigned char);

    // Allocate device memory
    unsigned char *d_img, *d_r, *d_g, *d_b;
    cudaMalloc(&d_img, img_size);
    cudaMalloc(&d_r, channel_size);
    cudaMalloc(&d_g, channel_size);
    cudaMalloc(&d_b, channel_size);

    // Copy image to device
    cudaMemcpy(d_img, input_img, img_size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);
    split_channels<<<blocksPerGrid, threadsPerBlock>>>(d_img, d_r, d_g, d_b, width, height);
    cudaDeviceSynchronize();

    // Merge back
    merge_channels<<<blocksPerGrid, threadsPerBlock>>>(d_img, d_r, d_g, d_b, width, height);
    cudaDeviceSynchronize();

    // Copy back to host
    unsigned char* output_img = new unsigned char[width * height * 3];
    cudaMemcpy(output_img, d_img, img_size, cudaMemcpyDeviceToHost);

    // Save output image
    stbi_write_png(output_path, width, height, 3, output_img, width * 3);

    // Free memory
    stbi_image_free(input_img);
    delete[] output_img;
    cudaFree(d_img);
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);

    std::cout << "Image processing complete. Output saved as " << output_path << "\n";
    return 0;
}
