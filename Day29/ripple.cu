#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "gif.h"

#define DIM 512
#define FRAMES 60
#define RIPPLE_SOURCES 1

__global__ void waterRippleKernel(unsigned char *ptr, int ticks, float phase, float amplitude, float frequency) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= DIM || y >= DIM) return;
    
    int offset = x + y * DIM;

    float rippleValue = 0.0f;
    float totalAmplitude = 0.0f;

    float fx = x - DIM * 0.5f;
    float fy = y - DIM * 0.5f;
    float distance = sqrtf(fx * fx + fy * fy);
    
    if (distance > 0.1f) {
        float wave = amplitude * cosf(frequency * distance - phase * ticks);
        wave *= expf(-distance * 0.01f);
        rippleValue += wave;
        totalAmplitude += amplitude;
    }
    
    if (totalAmplitude > 0.0f) {
        rippleValue /= totalAmplitude;
    }

    float baseWave = 0.1f * sinf(x * 0.02f + ticks * 0.1f) * cosf(y * 0.02f + ticks * 0.08f);
    rippleValue += baseWave;

    float intensity = 0.5f + 0.5f * rippleValue;
    intensity = fmaxf(0.0f, fminf(1.0f, intensity));

    unsigned char blue = (unsigned char)(50 + 120 * intensity);
    unsigned char green = (unsigned char)(80 + 100 * intensity);
    unsigned char red = (unsigned char)(20 + 60 * intensity);

    float shimmer = 0.1f * sinf(x * 0.1f + y * 0.1f + ticks * 0.2f);
    blue = (unsigned char)fmaxf(0, fminf(255, blue + shimmer * 30));
    green = (unsigned char)fmaxf(0, fminf(255, green + shimmer * 20));
    red = (unsigned char)fmaxf(0, fminf(255, red + shimmer * 10));
    
    ptr[offset * 4 + 0] = red;
    ptr[offset * 4 + 1] = green;
    ptr[offset * 4 + 2] = blue;
    ptr[offset * 4 + 3] = 255;
}

int main() {

    unsigned char *h_bitmap = new unsigned char[DIM * DIM * 4];

    unsigned char *d_bitmap;
    cudaMalloc(&d_bitmap, DIM * DIM * 4);

    float phase= 0.0f;
    float amplitude= 1.0f;
    float frequency= 0.15f;

    GifWriter gif;
    GifBegin(&gif, "water_ripple.gif", DIM, DIM, 100/6); 

    dim3 blockSize(16, 16);
    dim3 gridSize((DIM + blockSize.x - 1) / blockSize.x, (DIM + blockSize.y - 1) / blockSize.y);
    
    printf("Generating water ripple GIF...\n");

    for (int frame = 0; frame < FRAMES; frame++) {
        
        phase += 0.3f;

        waterRippleKernel<<<gridSize, blockSize>>>(d_bitmap, frame, phase, amplitude, frequency);

        cudaDeviceSynchronize();

        cudaMemcpy(h_bitmap, d_bitmap, DIM * DIM * 4, cudaMemcpyDeviceToHost);

        GifWriteFrame(&gif, h_bitmap, DIM, DIM, 100/6);
        
        printf("Frame %d/%d completed\n", frame + 1, FRAMES);
    }

    GifEnd(&gif);

    cudaFree(d_bitmap);
    delete[] h_bitmap;
    
    printf("Water ripple GIF generated successfully: water_ripple.gif\n");
    
    return 0;
}