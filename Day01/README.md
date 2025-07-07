# Day 01

Read initial chapters of Books: 
- Kirk, D. B., & Hwu, W.-m. W. (2017). *Programming Massively Parallel Processors (3rd ed.)*.
- Sanders, J., & Kandrot, E. (2011). *CUDA by Example: An Introduction to General-Purpose GPU Programming*.


## Summary
As CPU performance hit limits due to power and thermal constraints, the computing industry shifted toward parallel architectures. GPUs, designed for high-throughput tasks, enable massive parallelism using thousands of threads. CUDA, introduced by NVIDIA, made GPU programming more accessible by allowing developers to write general-purpose code in C/C++-like syntax.

GPUs differ from CPUs by prioritizing throughput over latency, making them ideal for data-parallel tasks. Efficient GPU programming involves understanding memory hierarchies, minimizing bandwidth bottlenecks, and optimizing common patterns like scans, reductions, and sorting. Real-world applications such as medical imaging, fluid dynamics, and environmental modeling have seen dramatic performance gains using CUDA.

CUDA complements models like OpenMP, OpenACC, and MPI, enabling scalable heterogeneous computing. Parallelism, once niche, is now essential for high-performance software across disciplines.

## CPU vs. GPU Architectures
| Feature           | CPU                           | GPU                             |
|-------------------|-------------------------------|---------------------------------|
| Optimization      | Sequential tasks              | Massively parallel tasks        |
| Logic Complexity  | High                          | Simple                          |
| Cache Size        | Large                         | Small                           |
| Memory Bandwidth  | Moderate                      | High                            |
| Latency/Throughput| Latency-oriented              | Throughput-oriented             |


## Challenges & Optimization
### Common Challenges:
- Parallel algorithm complexity.
- Memory bandwidth limitations.
- Variable performance with different data inputs.
- Difficulty in parallelizing inherently sequential tasks.

### Optimization Strategies:
- Effective utilization of GPU memory (global, shared, local).
- Adoption of parallel patterns: prefix sums, histograms, merge sort, convolution, etc. 

<br />

---

# CUDA Development Setup

## 1. Prerequisites

- **NVIDIA GPU and Drivers**  
  Ensure your system has an NVIDIA GPU and that the appropriate drivers are installed.  
  - Open **Device Manager** → **Display Adapters** to confirm.
  - Run `nvidia-smi` in PowerShell or Command Prompt. If it returns GPU and driver info, you're good to go.

- **C/C++ Compiler**  
  CUDA requires a compatible C/C++ compiler like Microsoft Visual Studio with MSVC.

---

## 2. Install CUDA Toolkit

### Download the Installer

- Go to the [NVIDIA CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads) page.
- Select:
  - **OS**: Windows
  - **Architecture**: x86_64
  - Choose either:
    - **Network Installer** – smaller size, fetches components online
    - **Local Installer** – full offline package

### Run the Installer

- Accept the license agreement.
- Choose **Express (Recommended)** installation to install all core components.
- Restart your system when prompted.

---

## 3. Verify Installation

After reboot, open Command Prompt and run:
```bash
nvcc --version
```

<br />

---
# Running and Testing CUDA Code Without a CUDA-Capable Device

If you don't have a system that supports CUDA programming locally, you can use cloud-based solutions like **Google Colab** or **LeetGPU** to run and test CUDA code effectively.

---

## 🚀 1. Google Colab

## ⚠️ Limitations
- **Unclear GPU access limits** – quota varies with usage and availability.
- **Session duration** – idle disconnects after a few minutes, and max runtime is typically up to 12 hours.
- **GPU quota exhaustion** – may get temporarily locked out of GPU use (can retry later or use a different account).

## ✅ Step-by-Step Instructions

### 1. Enable GPU Runtime
- Go to `Runtime` > `Change runtime type`
- Set **Hardware Accelerator** to `GPU`

### 2. Check CUDA Environment
Run these in a code cell to verify GPU access and CUDA tools:
```bash
!nvidia-smi
!nvcc --version
```

### 3. Write and Save CUDA C++ Code

You can write your CUDA code directly in the notebook in multiple ways:

**Using Python:**
```python
cuda_code = r'''
// Your CUDA C++ Code Here
'''
with open("vector_add.cu", "w") as f:
    f.write(cuda_code)
```

**Or using notebook magic:**
```bash
%%writefile vector_add.cu
// Your CUDA C++ Code Here
```

### Compile CUDA Code

Use nvcc with the proper architecture flag for the available GPU:

```bash
!nvcc -arch=sm_75 -o mycode vector_add.cu    # For Tesla T4
!nvcc -arch=sm_80 -o mycode vector_add.cu    # For A100
```

### Run the Compiled Program
```bash
!./mycode
```

### 🛠️ Diagnostics and Debugging

You can use the following tools and methods to verify that your CUDA code runs correctly in Colab or LeetGPU:

| **Check**             | **Purpose**                              | **Example Result**            |
|-----------------------|-------------------------------------------|-------------------------------|
| `!nvidia-smi`         | Verifies GPU is allocated                 | ✅ Tesla T4 detected           |
| `!nvcc --version`     | Confirms CUDA compiler is available       | ✅ CUDA 11.8 / CUDA 12.2       |
| `cudaGetLastError()`  | Checks for silent kernel launch failures  | ❗ Helpful for debugging        |
| Print from host (CPU) | Confirms GPU output was copied correctly  | ✅ Computed results displayed  |

---

