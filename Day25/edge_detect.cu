#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <cuda_runtime.h>

#define FILTER_SIZE 3

__constant__ int h_filter[FILTER_SIZE * FILTER_SIZE] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

__constant__ int v_filter[FILTER_SIZE * FILTER_SIZE] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

__global__ void horizontal_edge_kernel(unsigned char *output, unsigned char *input, int width, int height) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width)
        return;

    int sum = 0;

    for (int i = -FILTER_SIZE / 2; i <= FILTER_SIZE / 2; i++) {
        int y = row + i;
        y = y < 0 ? 0 : y >= height ? height - 1 : y;

        for (int j = -FILTER_SIZE / 2; j <= FILTER_SIZE / 2; j++) {
            int x = col + j;
            x = x < 0 ? 0 : x >= width ? width - 1 : x;

            int pxl_idx = y * width + x;
            int filter_idx = (i + FILTER_SIZE / 2) * FILTER_SIZE + (j + FILTER_SIZE / 2);

            sum += input[pxl_idx] * h_filter[filter_idx];
        }
    }

    int img_idx = row * width + col;
    output[img_idx] = min(255, max(0, abs(sum)));
}

__global__ void vertical_edge_kernel(unsigned char * output, unsigned char * input, int width, int height){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row >= height || col >= width)
        return;

    int sum = 0;
    int img_idx = row * width + col;
    
    for(int i = -FILTER_SIZE/2; i <= FILTER_SIZE/2; i++){
        int y = row + i;
        y = y < 0 ? 0 : y >= height ? height-1 : y;
        for(int j = -FILTER_SIZE/2; j <= FILTER_SIZE/2; j++){
            int x = col + j;
            x = x < 0 ? 0 : x >= width ? width-1 : x;
            int pxl_idx = y * width + x;
            int filter_idx = (i + FILTER_SIZE/2) * FILTER_SIZE + j + FILTER_SIZE/2;

            sum += (input[pxl_idx] * v_filter[filter_idx]);
        }
    }
    output[img_idx] = min(255, max(0, abs(sum)));
}

__global__ void print_filter() {
    printf("Horizontal Filter:\n");
    for (int i = 0; i < FILTER_SIZE; i++) {
        for (int j = 0; j < FILTER_SIZE; j++) {
            int idx = i * FILTER_SIZE + j;
            printf("%2d ", h_filter[idx]);
        }
        printf("\n");
    }
    printf("Vertical Filter:\n");
    for (int i = 0; i < FILTER_SIZE; i++) {
        for (int j = 0; j < FILTER_SIZE; j++) {
            int idx = i * FILTER_SIZE + j;
            printf("%2d ", v_filter[idx]);
        }
        printf("\n");
    }
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

    size_t img_size = width * height * sizeof(unsigned char);

    unsigned char *d_input, *d_h_output, *d_v_output;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_h_output, img_size);
    cudaMalloc(&d_v_output, img_size);

    cudaMemcpy(d_input, input_img, img_size, cudaMemcpyHostToDevice);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,(height + blockSize.y - 1) / blockSize.y);

    print_filter<<<1, 1>>>();
    cudaDeviceSynchronize();

    horizontal_edge_kernel<<<gridSize, blockSize, 0, stream1>>>(d_h_output, d_input, width, height);
    vertical_edge_kernel<<<gridSize, blockSize, 0, stream2>>>(d_v_output, d_input, width, height);

    cudaDeviceSynchronize();

    unsigned char* output_h_img = new unsigned char[width * height];
    unsigned char* output_v_img = new unsigned char[width * height];

    cudaMemcpy(output_h_img, d_h_output, img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_v_img, d_v_output, img_size, cudaMemcpyDeviceToHost);

    stbi_write_png(h_output_path, width, height, 1, output_h_img, width);
    stbi_write_png(v_output_path, width, height, 1, output_v_img, width);

    stbi_image_free(input_img);
    delete[] output_h_img;
    delete[] output_v_img;
    cudaFree(d_input);
    cudaFree(d_h_output);
    cudaFree(d_v_output);

    return 0;
}
