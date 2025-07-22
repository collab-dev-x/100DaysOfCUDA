#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <vector>
#include <cstdio>

#define M_PI 3.14159265358979323846
#define PATH_LEN 100

__constant__ float d_kernel[1024];

void generateGaussianKernel(float* kernel, int size, double sigma = 1.0) {
    double sum = 0.0;
    int half = size / 2;
    double s = 2.0 * sigma * sigma;

    for (int x = -half; x <= half; ++x) {
        for (int y = -half; y <= half; ++y) {
            double r = x * x + y * y;
            double value = exp(-r / s) / (M_PI * s);
            kernel[(x + half) * size + (y + half)] = value;
            sum += value;
        }
    }
    for (int i = 0; i < size * size; ++i) {
        kernel[i] /= sum;
    }
}

__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int half = kernelSize / 2;
    
    for (int c = 0; c < 3; ++c) {
        float sum = 0.0f;
        for (int kx = -half; kx <= half; ++kx) {
            for (int ky = -half; ky <= half; ++ky) {
                int px = x + kx;
                int py = y + ky;
                
                px = max(0, min(px, width - 1));
                py = max(0, min(py, height - 1));
                
                int pixelIdx = (py * width + px) * 3 + c;
                int kernelIdx = (kx + half) * kernelSize + (ky + half);
                
                sum += input[pixelIdx] * d_kernel[kernelIdx];
            }
        }
        int outputIdx = (y * width + x) * 3 + c;
        output[outputIdx] = (unsigned char)min(255.0f, max(0.0f, sum));
    }
}

int main() {

    std::cout<<"Enter the number of images to process\n";
    int n_images;
    std::cin>>n_images;
    std::cin.ignore(); 
    std::vector<char*> input_paths(n_images);
    std::cout<<"Enter filenames\n";
    for (int i = 0; i < n_images; ++i) {
        input_paths[i] = new char[PATH_LEN];
        std::cin.getline(input_paths[i], PATH_LEN);
    }

    std::vector<char*> output_paths(n_images);
    for (int i = 0; i < n_images; ++i) {
        output_paths[i] = new char[PATH_LEN];
        std::sprintf(output_paths[i], "output_image%d.jpg", i + 1);
    }

    std::vector<int>  width(n_images), height(n_images), channels(n_images);

    int kernelSize = 15;  
    double sigma = 2.0; 
    
    float* h_kernel = new float[kernelSize * kernelSize];
    generateGaussianKernel(h_kernel, kernelSize, sigma);
    
    // for (int i = 0; i < kernelSize; ++i) {
    //     for (int j = 0; j < kernelSize; ++j) {
    //         std::cout << h_kernel[i * kernelSize + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    
    size_t kernel_size = kernelSize * kernelSize * sizeof(float);

    cudaMemcpyToSymbol(d_kernel,h_kernel,kernel_size* sizeof(float));

    std::vector<unsigned char*> h_image(n_images);
    std::vector<unsigned char*> d_image(n_images), d_blur_image(n_images);
    std::vector<size_t> image_size(n_images);
    dim3 blockSize(16, 16);
    std::vector<dim3> gridSize(n_images);
    std::vector<cudaStream_t> streams(n_images);
    for (int i = 0; i < n_images; ++i) {
        h_image[i] = stbi_load(input_paths[i], &width[i], &height[i], &channels[i], 3);
        if (!h_image[i]) {
            std::cerr << "Error Loading Image\n";
            return -1;
        }
        image_size[i]= width[i] * height[i] * 3 * sizeof(unsigned char);
        cudaMalloc(&d_image[i], image_size[i]);
        cudaMalloc(&d_blur_image[i], image_size[i]);
        cudaMemcpy(d_image[i], h_image[i], image_size[i], cudaMemcpyHostToDevice);
        gridSize[i] = dim3((width[i] + blockSize.x - 1) / blockSize.x, (height[i] + blockSize.y - 1) / blockSize.y);
        cudaStreamCreate(&streams[i]);
    }

    for (int i = 0; i < n_images; ++i)
        gaussianBlurKernel<<<gridSize[i], blockSize, 0, streams[i]>>>(d_image[i], d_blur_image[i], width[i], height[i], kernelSize);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::vector<unsigned char*> h_blur_image(n_images);
    for (int i = 0; i < n_images; ++i)
        h_blur_image[i] = new unsigned char[width[i] * height[i] * 3];
    cudaDeviceSynchronize();

    for (int i = 0; i < n_images; ++i){
        cudaMemcpy(h_blur_image[i], d_blur_image[i], image_size[i], cudaMemcpyDeviceToHost);
        
        if (stbi_write_png(output_paths[i], width[i], height[i], 3, h_blur_image[i], width[i] * 3)) {
            std::cout << "Blurred image saved to " << output_paths[i] << std::endl;
        } else {
            std::cerr << "Error saving image\n";
        }
    }
    
    delete[] h_kernel;
    cudaFree(d_kernel);
    for (int i = 0; i < n_images; ++i){
        delete[] h_blur_image[i];
        stbi_image_free(h_image[i]);
        cudaFree(d_image[i]);
        cudaFree(d_blur_image[i]);
    }

    
    return 0;
}