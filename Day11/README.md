# Image Reading and Writing

This project demonstrates how to use CUDA to perform basic image processing on the GPU, specifically loading an image, separating its RGB color channels into three separate data buffers, and merging them back into a single image. It uses `stb_image.h` and `stb_image_write.h` libraries for CPU-based image file handling.

## Core Concepts

- **Image Loading (CPU)**: Use `stb_image.h` to load a JPEG/PNG or other image format from disk into a byte array.
- **Data Transfer (CPU to GPU)**: Copy raw image data from host (CPU) to device (GPU) memory.
- **Parallel Processing (GPU)**: Launch a CUDA kernel (`split_channels`) where each thread processes a pixel, splitting interleaved RGB data into three planar arrays (R, G, B).
- **Data Merging (GPU)**: Launch a second CUDA kernel (`merge_channels`) to combine the three planar arrays back into an interleaved RGB image.
- **Data Transfer (GPU to CPU)**: Copy processed image data from device to host.
- **Image Saving (CPU)**: Use `stb_image_write.h` to save the processed byte array as a PNG file.

## Setup and Installation of STB Libraries

### What are `stb_image.h` and `stb_image_write.h`?

`stb` is a collection of single-file, public-domain C/C++ libraries. They are easy to use, requiring no external linking—just include the headers with specific `#define` directives.

- **`stb_image.h`**: Loads various image formats (JPG, PNG, BMP, TGA, GIF, etc.) into a raw memory buffer.
- **`stb_image_write.h`**: Writes a raw memory buffer to PNG, BMP, TGA, or JPG files.

### How to Download and Use STB

1. **Download the Files**:
   - Visit the [STB GitHub repository](https://github.com/nothings/stb).
   - Download `stb_image.h` and `stb_image_write.h` by right-clicking and selecting "Save Link As...".

2. **Place in Project**:
   - Place `stb_image.h` and `stb_image_write.h` in the same directory as your `.cu` source file for portability.
   - Avoid placing them in the CUDA libraries folder to keep the project self-contained.

   **Project Structure**:
   ```
   your_project_folder/
   ├── code.cu
   ├── stb_image.h
   ├── stb_image_write.h
   ├── image.jpg
   ```

3. **Implement in Code**:
   In your `main.cu` file, add these lines before including the headers (only in one file to avoid linker errors):

   ```cpp
   #define STB_IMAGE_IMPLEMENTATION
   #define STB_IMAGE_WRITE_IMPLEMENTATION
   #include "stb_image.h"
   #include "stb_image_write.h"
   ```

## Code Explanation

### Image Loading with `stbi_load()`

Loads an image from a file into memory.

```cpp
unsigned char* input_img = stbi_load(input_path, &width, &height, &channels, 3);
```

- **Parameters**:
  - `input_path` (const char*): File path (e.g., "image.jpg").
  - `&width` (int*): Stores the image width.
  - `&height` (int*): Stores the image height.
  - `&channels` (int*): Stores the number of channels in the source file (e.g., 3 for RGB, 4 for RGBA).
  - `3` (int desired_channels): Forces the image to 3-channel RGB format, simplifying CUDA processing.
- **Returns**: A pointer (`unsigned char*`) to pixel data in R, G, B, R, G, B, ... order.

### Image Saving with `stbi_write_png()`

Saves a block of pixel data as a PNG file.

```cpp
stbi_write_png(output_path_color, width, height, 3, output_img_color, width * 3);
```

- **Parameters**:
  - `output_path_color` (const char*): Output file path (e.g., "color.png").
  - `width` (int): Image width.
  - `height` (int): Image height.
  - `3` (int comp): Number of channels per pixel (3 for RGB).
  - `output_img_color` (const void*): Pointer to raw pixel data.
  - `width * 3` (int stride_in_bytes): Bytes per row (3 bytes per pixel for RGB).
