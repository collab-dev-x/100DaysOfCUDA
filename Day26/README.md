# CUDA GIF Animation Generator

This program generates a simple animated GIF using CUDA to parallelize frame generation. Each frame displays a color gradient that shifts over time, creating a smooth animation saved as a `.gif` file.

## 🎯 Goal
Create a 256x256 pixel animated GIF with 60 frames, where each frame features a dynamic color gradient computed in parallel on the GPU using CUDA. The output is a 6-second animation at 10 fps, saved as `animated_cuda.gif`.

## 📂 Files Required
- **animated.cu**: The main CUDA program that orchestrates GIF generation and kernel execution.
- **gif.h**: Header file for the lightweight GIF writing library.
- **gif.c**: Implementation of the GIF writing library.

You must include `gif.h` and link with `gif.c` during compilation.

## 🧱 Key Components
### gif.h and gif.c
A lightweight C library for writing animated GIF files. The main functions used are:
- `GifBegin(GifWriter* writer, const char* filename, int width, int height, int delay)`: Initializes the GIF file.
- `GifWriteFrame(GifWriter* writer, const uint8_t* image, int width, int height, int delay)`: Writes a single frame to the GIF.
- `GifEnd(GifWriter* writer)`: Finalizes and closes the GIF file.

### CUDA Kernel: `generateFrame`
The CUDA kernel `generateFrame` runs on the GPU, with each thread computing the color of a single pixel:
- **Red**: `(x + frame) % 256` — shifts with frame number and x-coordinate.
- **Green**: `(y + frame) % 256` — shifts with frame number and y-coordinate.
- **Blue**: Fixed at `128` for a constant hue.
- **Alpha**: Fixed at `255` for full opacity.

This creates a smooth, dynamic gradient that shifts across frames.

## 🔧 How It Works
### 1. Initialization
- **Resolution**: 256x256 pixels (`width` × `height`).
- **Frames**: 60 frames (`numFrames`).
- **Total Pixels**: `width × height = 65,536`.
- Each pixel uses 4 bytes (RGBA), so total memory per frame is `65,536 × 4 = 262,144` bytes.

### 2. GIF Setup
The GIF is initialized with:
```c
GifBegin(&writer, "animated_cuda.gif", width, height, 10);
```
- **Output**: `animated_cuda.gif`.
- **Delay**: 10 (0.1 seconds per frame, yielding 10 fps).
- **Total Duration**: `60 frames × 0.1 seconds = 6 seconds`.

### 3. Memory Allocation
- **GPU Memory**: Allocate `d_image` on the GPU using `cudaMalloc` for `numPixels × 4` bytes.
- **CPU Memory**: Allocate `h_image` on the CPU to copy frame data from the GPU.

### 4. Frame Generation
For each of the 60 frames:
1. Launch the `generateFrame` kernel with a 2D grid to compute pixel colors in parallel.
2. Copy the frame data from GPU (`d_image`) to CPU (`h_image`) using `cudaMemcpy`.
3. Write the frame to the GIF using `GifWriteFrame`.

### 5. Finalization
Close the GIF file with:
```c
GifEnd(&writer);
```

### 6. Cleanup
Free allocated memory:
- GPU: `cudaFree(d_image)`.
- CPU: `delete[] h_image`.

## 🔁 GIF Frame Timing
- **Delay**: Set to 10 (0.1 seconds per frame) in `GifBegin` and `GifWriteFrame`.
- **Total Duration**: `60 frames × 0.1 seconds = 6 seconds` at 10 fps.

## 📦 Compilation
Compile the program using `nvcc`:
```bash
nvcc animated.cu gif.c -o animated -std=c++11
```
- Ensure `gif.h` and `gif.c` are in the same directory as `animated.cu`.
- The output executable is named `animated`.

## ✅ Summary
| Component          | Role                                          |
|--------------------|-----------------------------------------------|
| `gif.h` / `gif.c` | GIF writing library                           |
| `GifBegin`        | Initializes the GIF file                      |
| `GifWriteFrame`   | Adds a frame to the GIF                       |
| `GifEnd`          | Finalizes the GIF file                        |
| `generateFrame`   | CUDA kernel for parallel pixel color computation |
| `cudaMemcpy`      | Transfers frame data from GPU to CPU          |
| `main()`          | Coordinates memory, kernel execution, and GIF writing |

## 🚀 Extending the Program
To enhance this project, consider:
- Generating fractals (e.g., Mandelbrot or Julia sets).
- Simulating particle systems for dynamic visuals.
- Creating audio-reactive animations by modulating colors based on audio input.

Let me know if you'd like to explore these extensions or need the full `animated.cu` code!