#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void process3DArray(int* array, int x, int y, int z) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int column = threadIdx.y + blockIdx.y * blockDim.y;
    int depth = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < row && y < column && z < depth) {
        int idx = depth * x * y + column * x + row;
        array[idx] = x + y + z; // Sample operation
    }
}

void print3D(int* arr, int x, int y, int z) {
    for (int z = 0; z < z; z++) {
        cout << "Slice " << z << ":\n";
        for (int y = 0; y < y; y++) {
            for (int x = 0; x < x; x++) {
                cout << arr[z * x * y + y * x + x] << " ";
            }
            cout << "\n";
        }
    }
}

int main() {
    int x=2,y=3,z=4;
    int* d_3d;
    int* h_3d = new int[x * y * z];
    cudaMalloc(&d_3d, x * y * z * sizeof(int));
    int thread_x = 2, thread_y=2, thread_z=1;
    dim3 threadsPerBlock3D(thread_x, thread_y, thread_z);
    dim3 numBlocks3D((x + thread_x-1) / thread_x, (y + thread_y-1) / thread_y, (z + thread_z-1) / thread_z);
    process3DArray<<<numBlocks3D, threadsPerBlock3D>>>(d_3d, x, y, z);
    cudaMemcpy(h_3d, d_3d, x * y * z * sizeof(int), cudaMemcpyDeviceToHost);
    print3D(h_3d, x, y, z);

    cudaFree(d_3d);
    free(h_3d);

    return 0;
}
