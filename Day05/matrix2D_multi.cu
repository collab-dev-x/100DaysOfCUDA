#include <iostream>
using namespace std;
__global__ void matrix_mul(int *A, int *B, int *C, int row1, int col1, int col2){
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r<row1 && c<col2){
        int sum =0;
        for(int i=0;i<col1;i++){
            sum += A[r * col1 + i] * B[i*col2 +c];
        }
        C[r*col2+c] = sum;
    }
}


int main(){
    int row1=100, col1=50, row2=50, col2=100;

    cin>>row1>>col1>>row2>>col2;

    int * h_a = (int*)malloc(row1 * col1 * sizeof(int*));
    int * h_b = (int*)malloc(row2 * col2 * sizeof(int*));
    int * h_c = (int*)malloc(row1 * col2 * sizeof(int*));

    for(int i=0;i<row1 * col1;i++){
        h_a[i] = i%100;
        cin>>h_a[i];
    }
    for(int i=0;i<row2 * col2;i++){
        h_b[i] = i%100;
        cin>>h_b[i];
    }

    int *d_a , *d_b, *d_c;
    cudaMalloc((void**)&d_a, row1 * col1 * sizeof(int**));
    cudaMalloc((void**)&d_b, row2 * col2 * sizeof(int**));
    cudaMalloc((void**)&d_c, row1 * col2 * sizeof(int**));

    cudaMemcpy(d_a, h_a, row1 * col1 * sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, row2 * col2 * sizeof(int*), cudaMemcpyHostToDevice);
    
    dim3 dimBlock(32, 16);
    dim3 dimGrid((col2 + 32 - 1) / 32, (row1 + 16 - 1) / 16);    
    
    matrix_mul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, row1, col1, col2);

    cudaMemcpy(h_c,d_c,row1*col2*sizeof(int*),cudaMemcpyDeviceToHost);
    

    for(int i=0;i<row1*col2;i++){
        cout<<h_c[i]<<(i%col2==0?" ":"\n");
    }

    
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;

}