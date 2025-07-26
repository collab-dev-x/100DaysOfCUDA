# Mandelbrot Image


The Mandelbrot set is defined mathematically in the complex plane. For a given complex number **\( c \)**, a point belongs to the Mandelbrot set if the iterative function

![Mandelbrot Set](mandelbrot.png)


starting with \( z_0 = 0 \), remains bounded (i.e., does not tend to infinity) after an infinite number of iterations. In practice, we approximate this by setting a maximum number of iterations (e.g., `MAX_ITER = 100`) and a threshold for boundedness (e.g., magnitude squared greater than 4).

- **Bounded Points**: If the sequence remains bounded after `MAX_ITER` iterations, the point is considered part of the Mandelbrot set and typically colored black.
- **Escaping Points**: If the sequence exceeds the threshold (magnitude squared > 4), the point is not in the set, and the number of iterations before escaping determines the color, creating the intricate boundary patterns.
- **Complex Plane Mapping**:
  - Real axis: \([-2.5, 1.0]\)
  - Imaginary axis: \([-1.0, 1.0]\)
- **Escape Threshold**: A point is considered to escape if \( |z|^2 > 4.0 \).

## Complex Number Structure (`cuComplex`)
   - A CUDA-compatible structure to represent complex numbers with real (`r`) and imaginary (`i`) components.
   - Implements multiplication and addition operations for complex numbers.
   - Includes a `magnitude2` function to compute the squared magnitude (\( r^2 + i^2 \)), used to check if a point escapes the Mandelbrot set.

## Mandelbrot Function (`mandelbrot`)
   - A `__device__` function that runs on the GPU.
   - Takes pixel coordinates (`x`, `y`) and maps them to a complex number \( c \) in the complex plane:
     - Real part: \( c_x = -2.5 + 3.5 * x / WIDTH \)
     - Imaginary part: \( c_y = -1.0 + 2.0 * y / HEIGHT \)
   - Iterates \( z = z^2 + c \) up to `MAX_ITER` times.
   - Returns the number of iterations before the magnitude squared exceeds 4.0 or `MAX_ITER` if the point is bounded.