# A Generalized Gaussian Blur Program

This applies Gaussian blur to one or more images using CUDA for GPU acceleration.

## Gaussian Blur Parameters

### Sigma (σ)
Sigma controls the strength of the blur:
- **Small Sigma (e.g., 1.0–2.0):** Subtle blur, good for noise reduction. The effect is sharp and concentrated.
- **Large Sigma (e.g., 5.0–10.0):** Strong blur, ideal for depth-of-field effects. The image becomes hazy and soft.

A larger sigma spreads the blur more, blending pixels over a wider area for a smoother, more diffuse effect.

### Kernel Size
Kernel size defines the area around each pixel used for blurring:
- **Small Kernel (e.g., 7x7):** Fast and subtle, only nearby pixels contribute.
- **Large Kernel (e.g., 25x25, 31x31):** Slower but smoother, averaging a wider neighborhood.

We can also calculate kernel size based on sigma with any rule of thumb that suits the person. While prompting chatGPT i was recommended
```
kernelSize = (int)(sigma * 6.0) * + 1
```
This ensures the kernel is large enough (at least 6×σ) and odd-numbered for accuracy.


### Examples
- **Subtle Blur:** σ = 1.2 → Kernel Size = 7
- **Moderate Blur:** σ = 4.0 → Kernel Size = 25
- **Strong Blur:** σ = 10.0 → Kernel Size = 61
