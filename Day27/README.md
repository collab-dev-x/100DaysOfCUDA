# Julia Set Fractal (CUDA Implementation)

This project generates a visualization of the **Julia Set** fractal using a CUDA-compatible implementation. It computes whether each pixel belongs to the Julia set and produces a grayscale or colored image output.

---

## 🌀 What Is the Julia Set?

The **Julia Set** is a fractal defined by iterating the complex function:

**zₙ₊₁ = zₙ² + c**

Where:
- `z` is a complex number representing a pixel's position in the complex plane.
- `c` is a constant complex number that determines the fractal's shape.
- A point is in the Julia set if the sequence remains **bounded** (does not escape to infinity).
- If the sequence **escapes** (|z| → ∞), the point is outside the set, and the iteration count is used for coloring.

---

## ⚙️ How the `julia()` Function Works

The CUDA `__device__` function `julia(int x, int y)` determines if a pixel at coordinates `(x, y)` belongs to the Julia set. Here's how it operates:

1. **Map Pixel to Complex Plane**:
   - Converts pixel coordinates `(x, y)` to a complex number `z = jx + i·jy` using scaling factors based on image dimensions (`WIDTH`, `HEIGHT`) and a zoom factor (`SCALE`).

2. **Initialize Constant `c`**:
   - A fixed complex number `c` (e.g., `-0.8 + 0.156i`) defines the fractal's shape.

3. **Iterate the Julia Function**:
   - Repeatedly computes `z = z² + c` up to `MAX_ITER` iterations.
   - Checks if `|z|² > 1000` (escape threshold). If true, the point escapes, and the iteration count is returned.

4. **Return Iteration Count**:
   - The count determines the pixel's shade or color (higher count → darker or more intense color).

---

## 💻 How the Code Works

- **Custom `cuComplex` Struct**:
  - A CUDA-compatible struct replaces `std::complex` (which is not supported in `__device__` code).
  - Includes real (`r`) and imaginary (`i`) parts, with operators for addition (`+`), multiplication (`*`), and magnitude calculation (`magnitude2()`).

- **CUDA Kernel**:
  - A CUDA kernel iterates over all pixels in parallel, calling `julia(x, y)` for each.
  - The iteration count returned by `julia()` is used to assign RGB values to the pixel.

- **Output**:
  - Pixel data is written to an image file (e.g., `julia.png`) using a library like `stb_image_write.h`.

---

## 📸 Output

The output image (`julia.png`) visualizes:
- **Dark regions**: Points in the Julia set (bounded, high iteration count).
- **Light/colored regions**: Points outside the set (escaped quickly, low iteration count).

You can customize:
- **`c`**: Change the complex constant to explore different Julia set shapes.
- **`SCALE`**: Adjust zoom level (smaller values zoom in, larger values zoom out).
- **`MAX_ITER`**: Increase for finer detail (at the cost of computation time).

---

## ✨ Example Parameters

```cpp
cuComplex c(-0.8f, 0.156f);  // Defines the Julia set shape
float SCALE = 1.5f;          // Controls zoom level
int MAX_ITER = 200;          // Maximum iterations for detail
```