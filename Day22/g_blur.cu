#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <cuda_runtime.h>

#define FILTER_SIZE 3

__constant__ int filter[FILTER_SIZE * FILTER_SIZE];

__global__ void blurkernal(unsigned char * output, unsigned char * input, int width, int height){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row>=height || col >= width)
        return;

    int sum[3] = {0,0,0};
    int img_idx = (row * width + col) * 3;
    for(int i=-FILTER_SIZE/2;i<=FILTER_SIZE/2;i++){
        int y = row + i;
        y = y<0?0:y>=height?height-1:y;
        for(int j=-FILTER_SIZE/2;j<=FILTER_SIZE/2;j++){
            int x =col + j;
            x = x<0?0:x>=width?width-1:x;
            int pxl_idx = (y * width + x) *3;
            int filter_idx = (i + FILTER_SIZE/2) * FILTER_SIZE + j + FILTER_SIZE/2;
            sum[0]+=(input[pxl_idx]*filter[filter_idx]);
            sum[1]+=(input[pxl_idx+1]*filter[filter_idx]);
            sum[2]+=(input[pxl_idx+2]*filter[filter_idx]);
        }
    }
    output[img_idx] = (sum[0]/16);
    output[img_idx+1] = (sum[1]/16);
    output[img_idx+2] = (sum[2]/16);
}

__global__ void print_filter(){
    for(int i=-FILTER_SIZE/2;i<=FILTER_SIZE/2;i++){
        for(int j=-FILTER_SIZE/2;j<=FILTER_SIZE/2;j++){
            int filter_idx = (i + FILTER_SIZE/2) * FILTER_SIZE + j + FILTER_SIZE/2;
            printf("%d ",filter[filter_idx]);
        }
    }
    printf("\n");
}

// Convolution
__global__ void average_blur(unsigned char* img, unsigned char* r, unsigned char* g, unsigned char* b, int width, int height){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x < width && y < height){
        int idx = (y*width +x)*3;
        int sum[] = {0,0,0};
        for(int i=-1;i<2;i++){
            int dy = y+i;
            if(dy<0)
                continue;
            for(int j=-1;j<2;j++){
                int dx = x+j;
                if(dx<0)
                    continue;
                sum[0]+=r[dy*width+dx];
                sum[1]+=g[dy*width+dx];
                sum[2]+=b[dy*width+dx];
            }
        }
        img[idx]=sum[0]/9;
        img[idx+1]=sum[1]/9;
        img[idx+2]=sum[2]/9;
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


    unsigned char *d_img, *d_output;
    cudaMalloc(&d_img, img_size);
    cudaMalloc(&d_output, img_size);

    cudaMemcpy(d_img, input_img, img_size, cudaMemcpyHostToDevice);
    
    int gauss_3x3_filter[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    cudaMemcpyToSymbol(filter,gauss_3x3_filter,sizeof(gauss_3x3_filter));

    print_filter<<<1,1>>>();

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);


    blurkernal<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_img, width, height);
    cudaDeviceSynchronize();

    // Copy back to host
    unsigned char* output_img = new unsigned char[width * height * 3];
    cudaMemcpy(output_img, d_output, img_size, cudaMemcpyDeviceToHost);

    // Save output image
    stbi_write_png(output_path, width, height, 3, output_img, width * 3);

    // Free memory
    stbi_image_free(input_img);
    delete[] output_img;
    cudaFree(d_img);
    cudaFree(d_output);

    std::cout << "Image processing complete. Output saved as " << output_path << "\n";
    return 0;
}
