# CUDA FlashAttention Pedagogical Implementation (V1)

This repository contains a custom CUDA C++ inspired implementation of the **FlashAttention** algorithm (Dao et al.), capable of scaling to sequence lengths of $2^{17}$ (131,072) on a single GPU where standard attention implementations fail at $2^{14}$ (16,384).

### Implementation Details
* **Kernel:** CUDA C++ with manual memory management.
* **Precision:** FP32 (Single Precision).
* **Optimization Techniques:**
    * **Tiling:** Loads $Q, K, V$ blocks into Shared Memory (SRAM) to reduce Global Memory (HBM) accesses from $O(N^2)$ to $O(N)$.
    * **Online Softmax:** Computes safe softmax statistics ($m, \ell$) on-the-fly without materializing the full $N \times N$ attention matrix.
    * **Vectorized I/O:** Uses `float4` instructions to load 128 bits per memory transaction, maximizing bandwidth utilization.
    * **Hybrid Loading:** Automatic fallback to scalar loading for dimensions not divisible by 4.

#### Analysis
* **Memory Efficiency:** The FlashAttention kernel successfully trades compute for memory, achieving linear memory scaling. It maintains stability at sequence lengths orders of magnitude larger than the naive baseline.
* **Compute Latency:** At low sequence lengths ($N < 4096$), the FlashAttention kernel performs at approximately **50% of the speed** of the naive kernel.
    * *Reason:* The current implementation utilizes standard CUDA Cores (FFMA instructions). It does not yet utilize **Tensor Cores** (WMMA/MMA instructions), which provide an order-of-magnitude increase in arithmetic throughput for matrix multiplications.
    * *Occupancy:* The naive kernel naturally saturates GPU occupancy with massive grid sizes. The Flash kernel is bounded by Shared Memory capacity per block (48KB/64KB on T4), limiting the number of active warps.

### Usage
The kernel is wrapped for Python usage via `torch.utils.cpp_extension`.

```python
import torch
from flash_attn_v1 import run_flash

# Shapes: (N, d)
Q = torch.randn(131072, 128, device='cuda')
K = torch.randn(131072, 128, device='cuda')
V = torch.randn(131072, 128, device='cuda')

# Runs successfully where standard attention would OOM
output = run_flash(Q, K, V)
```
### Future Optimization Roadmap
To bridge the gap between this V1 implementation and production kernels (PyTorch SDPA/FlashAttention-2):
1.  **Tensor Cores (WMMA):** Replace the inner dot-product loops with Warp Matrix Multiply Accumulate instructions.
2.  **FP16/BF16 Support:** Reduces shared memory footprint by 50%, allowing for larger tile sizes ($B_r, B_c$) and higher occupancy.
3.  **Software Pipelining:** Prefetching the next tile from HBM into registers while computing the current tile to hide memory latency.
