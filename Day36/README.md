# Sobel Edge Detection with CUDA and cuBLAS

This document explains the implementation of Sobel edge detection using CUDA and cuBLAS for image processing on the GPU, including the overall workflow, the generation of image patches, and the use of cuBLAS for convolution.

## Overview of the Code
The program applies Sobel edge detection to an input grayscale image to detect edges in both horizontal and vertical directions. It uses CUDA for parallel processing and cuBLAS for efficient matrix operations. The process involves:
1. Converting the image into patches to represent local pixel neighborhoods.
2. Applying Sobel filters (horizontal and vertical) using matrix-vector multiplication.
3. Generating output images with edge intensities.

The code uses the Sobel filters defined as:
- Horizontal filter: `[-1, -2, -1, 0, 0, 0, 1, 2, 1]`
- Vertical filter: `[-1, 0, 1, -2, 0, 2, -1, 0, 1]`

These 3x3 filters detect intensity changes in the horizontal and vertical directions, respectively.

## Workflow
1. **Image Loading**: The input image is loaded as a grayscale image (single channel) to reduce processing complexity.
2. **Patch Generation**: The image is transformed into a set of 3x3 patches, where each patch corresponds to a pixel and its surrounding neighbors.
3. **Convolution with Sobel Filters**: The patches are convolved with the Sobel filters using cuBLAS to compute edge intensities.
4. **Result Processing**: The convolution results are converted to pixel intensities (0–255) and saved as output images for horizontal and vertical edges.
5. **Resource Management**: GPU memory and cuBLAS resources are allocated and freed appropriately.

## Patch Generation
### How Patches Are Generated
The `img_to_patches` CUDA kernel transforms the input image into a set of patches:
```c
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
```
- **Purpose**: For each pixel at position `(row, col)`, a 3x3 patch is extracted, containing the pixel and its eight neighbors.
- **Process**:
  - The kernel is launched with a 2D grid of threads, where each thread corresponds to a pixel in the image.
  - For a pixel at `(row, col)`, the kernel computes the coordinates of neighboring pixels `(row + i, col + j)` for `i, j` in `[-1, 0, 1]`.
  - Boundary conditions are handled by clamping coordinates to the image edges (e.g., `y < 0 ? 0 : y >= height ? height - 1 : y`).
  - The 3x3 patch is flattened into a 9-element vector and stored in the `patches` array at index `(row * width + col) * 9`.
- **Output**: The `patches` array is a matrix of size `9 x (width * height)`, where each column represents a flattened 3x3 patch for a pixel.

### Why Patches Are Generated
- **Convolution as Matrix Operation**: The Sobel filter convolution is equivalent to computing the dot product of the 3x3 filter with a 3x3 patch for each pixel. By organizing the image into a patch matrix, the convolution can be expressed as a matrix-vector multiplication, which is ideal for cuBLAS.
- **Parallelization**: Generating patches allows all pixel neighborhoods to be processed simultaneously on the GPU, leveraging CUDA's parallel architecture.
- **Efficiency**: Storing patches in a matrix format enables the use of optimized linear algebra routines in cuBLAS, reducing the need for custom convolution kernels.

## cuBLAS Integration
The Sobel edge detection is performed using cuBLAS for efficient matrix-vector multiplication.

### 1. **Handle Creation**
A cuBLAS handle is created to manage the library context:
```c
cublasHandle_t handle;
cublasCreate(&handle);
```
This initializes cuBLAS for subsequent operations.

### 2. **Matrix-Vector Multiplication with `cublasSgemv`**
The Sobel filters are applied to the patch matrix using `cublasSgemv`:
```c
const float alpha = 1.0f, beta = 0.0f;
cublasSgemv(handle, CUBLAS_OP_T, 9, img_size, &alpha, d_patches, 9, d_h_filter, 1, &beta, d_h_result, 1);
cublasSgemv(handle, CUBLAS_OP_T, 9, img_size, &alpha, d_patches, 9, d_v_filter, 1, &beta, d_v_result, 1);
```
- **Input Data**:
  - `d_patches`: A matrix of size `9 x (width * height)`, where each column is a flattened 3x3 patch.
  - `d_h_filter` and `d_v_filter`: Vectors of size 9, containing the flattened Sobel horizontal and vertical filter coefficients.
  - `d_h_result` and `d_v_result`: Output vectors of size `width * height`, storing the convolution results.
- **Operation**:
  - `CUBLAS_OP_T`: The patch matrix is transposed to align with the convolution operation, computing the dot product of each patch with the filter.
  - `9`: Number of rows in the patch matrix (size of the flattened 3x3 filter).
  - `img_size`: Number of columns in the patch matrix (total pixels).
  - `alpha = 1.0f`, `beta = 0.0f`: Scaling factors for the matrix-vector product and accumulation.
  - `d_patches`, `9`: The patch matrix and its leading dimension.
  - `d_h_filter`, `d_v_filter`: Input filter vectors.
  - `d_h_result`, `d_v_result`: Output vectors for horizontal and vertical edge detection.
- **Result**: Each element in `d_h_result` and `d_v_result` represents the edge intensity for a pixel after applying the horizontal or vertical Sobel filter.

### 3. **Handle Destruction**
The cuBLAS handle is destroyed to free resources:
```c
cublasDestroy(handle);
```

## Post-Processing
The `result_to_img` kernel converts the floating-point convolution results to pixel intensities:
```c
__global__ void result_to_img(unsigned char *img, float *result, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;
    img[idx] = min(255.0f, max(0.0f, fabsf(result[idx])));
}
```
- The absolute value of the convolution result is clamped to the range `[0, 255]` to produce valid pixel intensities.
- The results are copied to host memory and saved as output images.