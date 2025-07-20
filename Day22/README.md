# Gaussian Blurring

## Overview
Gaussian blurring is a low-pass filtering technique used in image processing to smooth images and reduce noise or detail. It is based on the Gaussian function, which mimics the natural diffusion of light and blur seen in optical systems.


## What Is Gaussian Blur?
Gaussian blur works by convolving an image with a Gaussian kernel, which gives higher weight to pixels near the center and lower weight to pixels farther away. This results in a smoothing effect that maintains structure better than simple averaging.

The 2D Gaussian function is defined as:

 G(i, j) = (1 / (2πσ²)) * exp( - [ (i - c)² + (j - c)² ] / (2σ²) )

Where:
- G(i, j): Kernel value at position (i, j)
- σ: Standard deviation (controls blur strength)
- c: Center index = floor(kernel_size / 2)
- exp(): Exponential function (e^x)


## 3×3 Gaussian Kernel Example (σ ≈ 1.0)
A commonly used normalized 3×3 Gaussian kernel is:

```
1 2 1
2 4 2
1 2 1
```

We need to normalise this with total sum

```
0.0625 0.125 0.0625
0.1250 0.250 0.1250
0.0625 0.125 0.0625
```

This kernel is derived by evaluating the Gaussian function at grid points and normalizing the sum to 1. It gives the most weight to the center pixel (0.25), smoothly tapering off to edges.

## How It Works (Convolution)
For each pixel in the image:
1. Multiply the surrounding pixels by the corresponding kernel weights.
2. Sum the results.
3. Replace the center pixel with this weighted sum.

This operation is repeated for every pixel in the image (with edge handling).

## Notes
- Kernel size must be odd (3×3, 5×5, etc.) to have a center.
- Larger kernels (e.g., 5×5 or 7×7) produce stronger blur.
- Sigma (σ) controls how wide the blur is spread.