#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void process2DArray(int* array, int row, int column) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < row && y < column) {
        int idx = y * row + x;
        array[idx] = x + y;
    }
}

void print2D(int* arr, int row, int column) {
    for (int y = 0; y < column; y++) {
        for (int x = 0; x < row; x++) {
            cout << arr[y * row + x] << " ";
        }
        cout << "\n";
    }
}


int main() {
    int row=3,column=4;

    int* d_2d;
    int* h_2d = new int[row * column];
    cudaMalloc(&d_2d, row * column * sizeof(int));

    int thread_x = 2, thread_y=2;
    dim3 threadsPerBlock2D(thread_x,thread_y);
    dim3 numBlocks2D((row + thread_x-1) / thread_x, (column + thread_y-1) / thread_y);
    process2DArray<<<numBlocks2D, threadsPerBlock2D>>>(d_2d, row, column);
    cudaMemcpy(h_2d, d_2d, row * column * sizeof(int), cudaMemcpyDeviceToHost);
    print2D(h_2d, row, column);

    cudaFree(d_2d);
    free(h_2d);

    return 0;
}
