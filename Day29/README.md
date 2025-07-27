# Water Ripple

This implementation is my work to implement water ripple as done in book `CUDA by Example` in Chapter 5.

The simulation uses a cosine function to model wave propagation, with parameters for amplitude, frequency, and phase to control the wave behavior. A background wave pattern is also included to add depth to the visual effect.

## Program Structure
The program consists of two main components:
1. **CUDA Kernel (`waterRippleKernel`)**: Computes the pixel values for each frame by calculating the ripple effect at each point on the 2D grid.
2. **Main Function**: Manages memory, orchestrates the frame-by-frame computation, and writes the output to a GIF file.

### Key Parameters
- **DIM**: 512 
- **FRAMES**: 60 
- **phase**: Phase offset for the wave (in radians).
- **amplitude**: Strength of the wave.
- **frequency**: Controls the wavelength of the ripple.

## Ripple Propagation Mechanism
The ripple effect is computed in the `waterRippleKernel` CUDA kernel, which runs on the GPU. Here's how it works:

1. **Wave Calculation**:
   - For each pixel at coordinates `(x, y)`, the kernel calculates the contribution of ripple source.
   - The distance from the pixel to source is computed as `distance = sqrt((x - source.x)^2 + (y - source.y)^2)`.
   - If the distance is greater than 0.1, the wave contribution is calculated using:
     ```math
     wave = amplitude * cos(frequency * distance - phase * ticks) * exp(-distance * 0.01)
     ```
     - `amplitude`: Controls the wave height.
     - `frequency`: Determines the wavelength (higher frequency = shorter wavelength).
     - `phase * ticks`: Animates the wave by shifting the phase over time.
     - `exp(-distance * 0.01)`: Applies exponential damping to reduce amplitude with distance.


2. **Background Wave**:
   - A base wave pattern is added to the ripple value:
     ```math
     baseWave = 0.1 * sin(x * 0.02 + ticks * 0.1) * cos(y * 0.02 + ticks * 0.08)
     ```
     This creates a subtle, dynamic background texture.

3. **Output**:
   - The final RGB values, along with an alpha value of 255 (fully opaque), are written to the pixel buffer at `offset * 4`.

## Main Function Workflow

1. **GIF Setup**:
   - Initializes a GIF writer to create a file named `ripple.gif` with a frame delay of ~1/6 seconds (6 FPS).

2. **Animation Loop**:
   - For each of the 60 frames:
     - Updates the phase of ripple .
     - Launches the CUDA kernel to compute the frame.
     - Copies the resulting pixel data back to the host.
     - Writes the frame to the GIF file.
   - Prints progress for each frame.

4. **Cleanup**:
   - Frees allocated memory and closes the GIF file.

## Output
The program generates a GIF file (`ripple.gif`) that displays an animated water ripple effect.

![Ripple](ripple.gif)