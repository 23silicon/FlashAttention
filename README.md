# CUDA FlashAttention Pedagogical Implementation in CUDA C++ (V1)

This repository contains a high-performance CUDA C++ implementation of the **FlashAttention** algorithm (Dao et al.). It demonstrates the progression from a naive memory-bound attention kernel to an optimized, tile-based implementation capable of infinite context scaling on limited hardware.

The project is structured into two main iterations:
* `FlashAttentionFirstAttempt.cu`: Initial FP32 implementation. Functional but limited by bank conflicts and low occupancy. Contain the tiling and online softmax logic at the core of FlashAttention's increased processing capabilities.
* `FlashAttention.cu`: Optimized FP16 implementation featuring strided memory access, increased tile sizes, and reduced instruction overhead. Allows it to outperform naive attention on large datasets, but still far behind Pytorch's Scaled Dot Product Attention (SDPA) due to its industry-grade optimizations. 
* `FlashAttentionBenchmark.ipynb`: Jupyter notebook for benchmarking kernel performance against naive CUDA and PyTorch SDPA.

## Performance Benchmark
*Hardware: NVIDIA T4 Tensor Core GPU | Hidden Dimension d=128 | Precision: FP16*

The optimized kernel (`FlashAttention.cu`) successfully overtakes the Naive baseline at sequence length $N=4096$ and scales linearly to $N=196,608+$ where Naive attention crashes due to Out-Of-Memory (OOM) errors.

| N (Seq Len) | Naive (ms) | Flash (ms) | PyTorch (ms) | Speedup (vs Naive) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1024** | 3.39 | 10.73 | 0.06 | 0.32x | Naive wins (L2 Cache hits) |
| **4096** | 50.20 | 45.70 | 0.61 | **1.10x** | **Crossover Point** (Flash becomes faster) |
| **16384** | 873.05 | 784.98 | 11.35 | **1.11x** | Naive hits HBM bandwidth wall |
| **24576** | 2326.02 | 1559.26 | 25.57 | **1.49x** | Peak relative speedup |
| **32768** | 3958.30 | 2879.11 | 43.32 | **1.37x** | Last successful Naive run |
| **49152** | **OOM** | 6184.66 | 101.62 | **$\infty$** | Naive self-attention can no longer execute |
| **131072** | **OOM** | 43530.52 | 819.68 | **$\infty$** | Scaling to 128k context |
| **196608** | **OOM** | 98195.19 | 2026.16 | **$\infty$** | Scaling to 196k context |

## Optimization Journey

### Iteration 1: `FlashAttentionFirstAttempt.cu`
The initial attempt implemented the core tiling logic and online softmax.
* **Precision:** FP32 (Single Precision).
* **Bottlenecks:**
    * **Bank Conflicts:** The row stride of 128 (512 bytes) caused 32-way shared memory bank conflicts, serializing memory reads.
    * **Low Occupancy:** Small tile sizes ($B_r=64$) resulted in low warp occupancy, preventing the GPU from hiding global memory latency.
    * **Memory Limit:** FP32 data size restricted the maximum tile size per block.

### Iteration 2: `FlashAttention.cu` (Current Optimized Version)
This version addresses the architectural bottlenecks of the first attempt, resulting in a robust speedup.

1.  **Bank Conflict Resolution (Padding):**
    * **The Fix:** Introduced a "dummy" padding of 8 elements to the Shared Memory allocation (`stride = d + 8`).
    * **The Result:** This shifts the memory addresses such that threads in a warp access distinct memory banks simultaneously. This restored full SRAM bandwidth, moving from 32-cycle serial reads to 1-cycle parallel reads.

2.  **Precision Shift (FP32 $\to$ FP16):**
    * Switched from `float` to `half` precision, accelerating computations. This also effectively doubled the available Shared Memory capacity per block (2 bytes vs. 4 per element), allowing for larger tile sizes.

3.  **Increased Occupancy:**
    * Increased Query Tile Size ($B_r$) from 64 to 128.
    * This ensures 4 warps run per block, allowing the CUDA scheduler to execute math instructions on one warp while others stall on global memory loads.

## Comparison to State of the Art (PyTorch SDPA)

While `FlashAttention.cu` beats the Naive implementation, it runs at approximately **1.5%** of the speed of PyTorch's native `scaled_dot_product_attention`.

**Why the massive gap between my kernel and SDPA?**
1.  **Hardware Utilization (The 16x Gap):**
    * **My Kernel:** Uses **CUDA Cores** (Scalar `hadd`/`hmul`). It computes matrix multiplications by iterating through vectors one element at a time.
    * **PyTorch SDPA:** Uses **Tensor Cores** (Matrix `hmma`). These specialized hardware units perform $4 \times 4$ matrix multiplications in a single clock cycle, providing vastly higher theoretical throughput.
2.  **Memory Pipelining:**
    * **My Kernel:** Uses synchronous "Stop-and-Go" loading. (Load Tile $\to$ Wait $\to$ Compute).
    * **PyTorch SDPA:** Uses asynchronous copying (`cp.async`) and multi-stage pipelining. It loads the *next* tile from HBM into registers while simultaneously computing the *current* tile in SRAM.

## Future Optimization Roadmap
To bridge the gap between this implementation and production kernels:

1.  **Tensor Cores (`nvcuda::wmma`):** Rewrite the inner dot-product loops to use Warp Matrix Multiply Accumulate instructions instead of scalar math.
2.  **Software Pipelining:** Implement double-buffering to fetch data for iteration $i+1$ while computing iteration $i$.
3.  **Warp-Level Primitives:** Replace block-level reductions with `__shfl_down_sync` for faster softmax calculation.

## Usage
The kernel is wrapped for Python usage via `torch.utils.cpp_extension`.

```python
import torch
import flash_attn_cuda  # The compiled extension

# Shapes: (N, d) - Must be contiguous and FP16
N, d = 131072, 128
Q = torch.randn(N, d, device='cuda', dtype=torch.float16)
K = torch.randn(N, d, device='cuda', dtype=torch.float16)
V = torch.randn(N, d, device='cuda', dtype=torch.float16)

# Output: (N, d)
output = flash_attn_cuda.run_flash(Q, K, V)
```
