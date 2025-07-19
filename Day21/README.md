# Performance Monitoring through NVIDIA-SMI and Nsight Compute

# NVIDIA-SMI
The `nvidia-smi` command monitors GPU performance in real-time, outputting key metrics in CSV format every second.

## Command
```bash
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,temperature.gpu,power.draw,memory.used --format=csv -l 1
```

This queries GPU metrics at 1-second intervals and formats the output as CSV.

## Output Fields
Each line in the output includes:

| Field                | Description                              |
|----------------------|------------------------------------------|
| `timestamp`          | Query time                               |
| `name`               | GPU model (e.g., NVIDIA RTX 3080)        |
| `utilization.gpu`    | CUDA core usage (%)                      |
| `utilization.memory` | Memory controller usage (%)              |
| `temperature.gpu`    | Core temperature (°C)                    |
| `power.draw`         | Power consumption (watts)                |
| `memory.used`        | VRAM usage (MiB)                         |

## What the Metrics Mean
| Metric               | Insight                                  | Use Case                              |
|----------------------|------------------------------------------|---------------------------------------|
| `utilization.gpu`    | How busy the GPU cores are               | Check if apps use GPU effectively     |
| `utilization.memory` | Memory access activity                   | Monitor for ML or video encoding      |
| `temperature.gpu`    | GPU thermal state                        | Prevent overheating/throttling        |
| `power.draw`         | Energy usage                             | Optimize power for battery devices    |
| `memory.used`        | VRAM consumption                         | Detect memory leaks or batch issues   |

## Save Output for Analysis
   Log metrics to a file for later review:
   ```bash
   nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,temperature.gpu,power.draw,memory.used --format=csv -l 1 > gpu_log.csv
   ```

## Other Metrics
   Check additional fields like clock speeds or PCIe bandwidth:
   ```bash
   nvidia-smi --help-query-gpu
   ```
   Examples:
   - `clocks.sm`: Streaming multiprocessor clock
   - `fan.speed`: Fan speed (%)
   - `pstate`: Performance state (P0 = max performance)
   - `driver_version`: Driver version
   - `pcie.link.gen.current`, `pcie.link.width.current`: PCIe details

# Nsight Compute
Nsight Compute is an NVIDIA tool for profiling CUDA kernels. It provides detailed performance insights at the kernel level, unlike basic GPU monitoring tools like `nvidia-smi` or NVML, which focus on overall GPU usage (e.g., power, memory, temperature).

Use `ncu` when you need to:
- Identify performance bottlenecks (e.g., memory issues, low occupancy).
- Understand why a kernel is slow.

## How to Use Nsight Compute
Here’s a simple workflow to profile a CUDA application using `ncu`:

### Step 1: Compile Your Code
Compile your CUDA code with debug info (`-lineinfo`) to get source-level insights:
```bash
nvcc -o metrics.exe metrics.cu -lineinfo
```

### Step 2: Run Nsight Compute
Use the command-line interface to profile your application:
```bash
ncu --replay-mode application ./metrics.exe
```
- **`--replay-mode application`**: Replays only the kernel launches, not the entire app state. It’s fast and suitable for most cases.
- **`./metrics.exe`**: Your compiled CUDA program.

### Step 3: Analyze the Output
Nsight Compute generates a detailed report with metrics like:
- **Kernel Name**: Name of the CUDA kernel.
- **Launch Statistics**: Grid/block sizes, registers used, occupancy.
- **Memory Throughput**: Bytes/sec for global, shared, or local memory.
- **Instruction Stats**: Instructions per cycle (IPC).
- **Memory Details**: Cache hit rates, load/store efficiency.
- **SM Utilization**: % of Streaming Multiprocessor (SM) active time, stall reasons.
- **Warp Execution**: Warp occupancy, divergence, stalls.
- **Shared Memory**: Usage, bank conflicts.
- **Branch Efficiency**: % of non-divergent branches.
- **Occupancy**: % of active threads vs maximum.
- **Launch Latency**: Time between kernel enqueue and execution.

### Step 4: Export Results (Optional)
Save the report for later analysis:
```bash
ncu --replay-mode application --export report_metrics ./metrics.exe
```
This creates a file (`report_metrics.ncu-rep`) for further review.

## Why is Replay Important?
Nsight Compute often replays a kernel multiple times to collect all metrics. Here’s why:

- **Hardware Limitation**: GPUs have limited performance counters. Not all metrics (e.g., memory throughput, warp stalls) can be collected in one pass.
- **How Replay Works**: Nsight Compute runs the kernel multiple times, each time collecting a different set of metrics (e.g., first pass for memory stats, second for instruction stats).
- **Replay Modes**:
  - **application**: Replays kernels within the app. Fast, low memory use, ideal for most cases.
  - **kernel**: Captures full GPU state (including memory). Slower, memory-intensive, but precise.
  - **none**: Collects only metrics possible in one pass. Fast but limited.
  - **auto**: Nsight Compute chooses the best mode (may not always work well).

**Why it matters**: Replay ensures complete data collection without missing critical metrics, helping you fully understand kernel performance.

## Understanding Key Metrics
Here’s what key metrics mean and how to use them:

| **Metric**                | **What It Tells You**                              | **What to Do If It’s Bad**                     |
|---------------------------|--------------------------------------------------|-----------------------------------------------|
| **Low IPC**               | Few instructions executed per cycle.             | Optimize memory access or reduce stalls.       |
| **Low Occupancy**         | Few active threads vs maximum.                   | Reduce registers/thread or shared memory use. |
| **High Stall %**          | Kernel waiting (e.g., memory, dependencies).      | Check stall reasons, optimize memory access.   |
| **Low Branch Efficiency**  | Warps taking different paths (divergence).        | Minimize conditional branches in warps.        |
| **Bank Conflicts**        | Shared memory access conflicts.                  | Restructure shared memory layout.              |
| **Low Cache Hit Rate**    | Poor memory access efficiency.                   | Coalesce memory accesses, optimize data layout.|

## When to Use Nsight Compute vs. Other Tools
| **Tool**       | **Detail Level** | **Best For**                              |
|-----------------|------------------|------------------------------------------|
| `nvidia-smi`   | Low              | Monitoring GPU usage (power, temp, memory). |
| NVML           | Low–Medium       | Programmatic GPU stats for runtime logging. |
| `ncu`          | High             | Kernel-level optimization, bottleneck fixes. |

Use `nvidia-smi` or NVML for general monitoring. Switch to `ncu` for deep kernel performance analysis.