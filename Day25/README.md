# Edge Detection

Here I implemented edge detection using CUDA, applying filters to detect horizontal and vertical edges in a grayscale image. It produces two output images highlighting edge details in each direction.

## What is Edge Detection?

Edge detection is a key image processing technique that identifies sharp changes in pixel intensity, corresponding to:

- Object boundaries
- Shape outlines
- Surface texture changes

It enables structural analysis in computer vision applications.

## Why Sobel?

The Sobel filter is preferred over other edge detection methods like Roberts Cross, Prewitt, or Laplacian because it:

- Computes gradient magnitude in both X and Y directions
- Reduces noise sensitivity through smoothing
- Emphasizes central pixels for better edge localization

### Sobel Filter Kernels

The Sobel filter uses two 3×3 convolution kernels:

**Horizontal (h_filter):**

```
[-1 -2 -1]
[ 0  0  0]
[ 1  2  1]
```

**Vertical (v_filter):**

```
[-1  0  1]
[-2  0  2]
[-1  0  1]
```

These kernels are applied via 2D convolution to compute edge strength at each pixel.

## Use Cases

- Object recognition and segmentation
- Barcode or document scanning