#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <cuda_runtime.h>
#include <complex>
#include <vector>

#define WIDTH 20000
#define HEIGHT 20000
#define MAX_ITER 100

struct cuComplex {
    float r; 
    float i; 

    __device__ cuComplex(float a, float b) : r(a), i(b) {}

    __device__ float magnitude2() {
        return r * r + i * i;
    }

    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int mandelbrot(int x, int y) {

    float cx = -2.5f + 3.5f * x / WIDTH;   
    float cy = -1.0f + 2.0f * y / HEIGHT; 
    cuComplex c(cx, cy);
    cuComplex z(0, 0);

    for (int i = 0; i < MAX_ITER; i++) {
        z = z * z + c;
        if (z.magnitude2() > 4.0f)
            return i;
    }
    return MAX_ITER;
}

__global__ void generate_image(unsigned char * image){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row>=HEIGHT || col >= WIDTH)
        return;

    int iter = mandelbrot(col, row);
    int pixelIdx = 3 * ( row* WIDTH + col);
    unsigned char color = (unsigned char)(255 * iter / MAX_ITER);
    image[ pixelIdx + 0] = color;
    image[ pixelIdx + 1] = color;
    image[ pixelIdx + 2] = color;
}

int main() {

    unsigned char *d_image, *image =  new unsigned char [WIDTH * HEIGHT * 3];
    cudaMalloc(&d_image,WIDTH*HEIGHT*3*sizeof(unsigned char));

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

    generate_image<<<gridSize,blockSize>>>(d_image);
    cudaDeviceSynchronize();

    cudaMemcpy(image,d_image,WIDTH*HEIGHT*3*sizeof(unsigned char),cudaMemcpyDeviceToHost);

    if (stbi_write_png("mandelbrot.png", WIDTH, HEIGHT, 3, image, WIDTH * 3))
        std::cout << "Saved 'mandelbrot.png'\n";
    else
        std::cerr << "Failed to save image.\n";

    return 0;
}
