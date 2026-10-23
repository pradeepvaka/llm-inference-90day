#!/usr/bin/env python3
"""Generate notebooks/day45-first-cuda-kernel-rmsnorm.ipynb."""
import json, pathlib

SLUG = "day45-first-cuda-kernel-rmsnorm"
NB_URL = f"https://colab.research.google.com/github/pradeepvaka/llm-inference-90day/blob/master/notebooks/{SLUG}.ipynb"
OUT = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / f"{SLUG}.ipynb"

cells = []
def md(src):
    cells.append({"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.split("\n")]})
def code(src):
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in src.split("\n")]})

md(f"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({NB_URL})

# Day 45 — First CUDA Kernel (RMSNorm)

**Big idea:** RMSNorm is 3 lines of math and one pass over memory. A fused kernel (one block per row, shared-memory + warp-shuffle reduction, one rsqrt, one scaled write) moves ~201 MB for 8,192 rows x 4,096 dims in fp16 -> ~60 us floor on H100; PyTorch's composite (pow -> mean -> rsqrt -> mul) makes ~4 passes and takes ~200+ us. Today you will: (1) read the real CUDA kernel, (2) verify the block-reduction logic on CPU in NumPy, (3) compute traffic floors for 128 / 2,048 / 8,192 rows, (4) benchmark composite vs fused in PyTorch (and compile the .cu with nvcc if a GPU is present), and (5) break coalescing in a strided simulation, predict the slowdown first, then measure it. CPU-OK throughout; T4-GPU for the nvcc path. ~30-60 min.""")

code("""# Cell 1: one-time installs (only cell that installs)
%pip install --quiet numpy matplotlib
print("deps ok")
# Expected: deps ok""")

md("""## Reading — the fused RMSNorm kernel (the artifact of the day)

```cuda
// rmsnorm.cu -- grid = rows, block = 256 threads. One block normalizes one row.
__global__ void rmsnorm_kernel(const half* __restrict__ x,
                               const half* __restrict__ gamma,
                               half* __restrict__ y, int d, float eps) {
    const half* row = x + (long)blockIdx.x * d;
    half* out = y + (long)blockIdx.x * d;
    float thread_sum = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {   // strided chunk: lane t reads base+t -> coalesced
        float v = __half2float(row[i]);
        thread_sum += v * v;
    }
    // warp reduction: 5 shuffle steps -> 1 value per warp
    for (int off = 16; off > 0; off >>= 1)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, off);
    __shared__ float warp_sums[8];
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = thread_sum;
    __syncthreads();
    if (threadIdx.x < 8) {                                 // first warp reduces the 8 warp sums
        float s = warp_sums[threadIdx.x];
        for (int off = 4; off > 0; off >>= 1) s += __shfl_down_sync(0xff, s, off);
        if (threadIdx.x == 0) warp_sums[0] = rsqrtf(s / d + eps);
    }
    __syncthreads();
    float inv_rms = warp_sums[0];                          // broadcast via shared memory
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = __half2float(row[i]) * inv_rms * __half2float(gamma[i]);
        out[i] = __float2half(v);
    }
}
// launch: rmsnorm_kernel<<<rows, 256>>>(x, gamma, y, 4096, 1e-6f);
```

Annotation:
- Mapping: `i = threadIdx.x, threadIdx.x + 256, ...` — consecutive lanes read consecutive addresses -> **coalesced, 1 transaction per warp load**.
- Reduction depth at d=4096, block=256: 16 elements/thread, then 5 shuffle steps + 3 tree steps (8 warps -> 1) = **8 steps**.
- Gamma is 8 KB: after row 0 it lives in L2 — the "3 streams = 201 MB" model is the nominal ceiling; effective traffic is closer to 134 MB.""")

code("""# Step 0: save the kernel source so the notebook owns the artifact
kernel_src = r'''
__global__ void rmsnorm_kernel(const half* __restrict__ x,
                               const half* __restrict__ gamma,
                               half* __restrict__ y, int d, float eps) {
    const half* row = x + (long)blockIdx.x * d;
    half* out = y + (long)blockIdx.x * d;
    float thread_sum = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = __half2float(row[i]);
        thread_sum += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, off);
    __shared__ float warp_sums[8];
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = thread_sum;
    __syncthreads();
    if (threadIdx.x < 8) {
        float s = warp_sums[threadIdx.x];
        for (int off = 4; off > 0; off >>= 1) s += __shfl_down_sync(0xff, s, off);
        if (threadIdx.x == 0) warp_sums[0] = rsqrtf(s / d + eps);
    }
    __syncthreads();
    float inv_rms = warp_sums[0];
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = __half2float(row[i]) * inv_rms * __half2float(gamma[i]);
        out[i] = __float2half(v);
    }
}
'''
open("rmsnorm_kernel.cu.inc", "w").write(kernel_src)
print(f"kernel saved: {len(kernel_src.splitlines())} lines; grid = rows, block = 256")
# Expected: kernel saved: 28 lines; grid = rows, block = 256""")

md("""## Step 1 — Verify the block-reduction logic on CPU (NumPy simulation)

Simulate exactly what the block does: split each row into 256 strided chunks, sum each chunk, tree-reduce the 256 partials in two levels (32-wide warp groups, then across warps), rsqrt, scale. Compare against the direct row-wise formula.""")

code("""import numpy as np

rng = np.random.default_rng(0)
rows, d, block = 64, 4096, 256
x = rng.standard_normal((rows, d), dtype=np.float32)
gamma = rng.standard_normal(d, dtype=np.float32)
eps = 1e-6

def rmsnorm_direct(x, gamma, eps=1e-6):
    ms = np.mean(x * x, axis=1, keepdims=True)
    return x / np.sqrt(ms + eps) * gamma

def rmsnorm_block_sim(x, gamma, block=256, eps=1e-6):
    rows, d = x.shape
    out = np.empty_like(x)
    n_warps = block // 32
    for r in range(rows):
        chunks = np.array([x[r, t::block] ** 2 for t in range(block)])  # (256, 16) per-thread chunk sums
        thread_sums = chunks.sum(axis=1)                                # 256 partials
        warp_sums = thread_sums.reshape(n_warps, 32).sum(axis=1)        # warp shuffle stage -> 8 values
        total = warp_sums.sum()                                         # shared-mem stage -> 1 value
        inv = 1.0 / np.sqrt(total / d + eps)                            # rsqrt
        out[r] = x[r] * inv * gamma
    return out

y_ref = rmsnorm_direct(x, gamma)
y_sim = rmsnorm_block_sim(x, gamma, block)
err = np.max(np.abs(y_ref - y_sim))
print(f"rows={rows}, d={d}, block={block}: per-thread chunk = {d // block} elements")
print(f"reduction: 256 thread sums -> {block // 32} warp sums -> 1 (5 shuffle steps + 3 tree steps)")
print(f"max |direct - block simulation| = {err:.2e}")
assert err < 1e-5
print("reduction logic verified: chunked + tree reduction == direct RMSNorm")
# Expected: per-thread chunk = 16; max error < 1e-5 (typically ~1e-7..1e-6 in float32)""")

md("""## Step 2 — Traffic model: floors by shape and GPU

Three streams (read x, read gamma, write y) at 2 bytes/element. Floor = traffic / bandwidth. Watch the 128-row case: its floor is the same order as one kernel launch (~3-10 us), so small-shape benchmarks measure launch overhead, not bandwidth.""")

code("""d = 4096
for rows in (128, 2048, 8192):
    elems = rows * d
    traffic_mb = 3 * elems * 2 / 1e6
    h100_us = traffic_mb * 1e6 / 3.35e12 * 1e6
    t4_us = traffic_mb * 1e6 / 0.32e12 * 1e6
    print(f"rows {rows:>5}: traffic {traffic_mb:7.1f} MB | H100 floor {h100_us:7.1f} us | T4 floor {t4_us:8.1f} us")
traffic_8192 = 3 * 8192 * d * 2
assert abs(traffic_8192 / 1e6 - 201.3) < 0.5
print("checkpoint: 8192 x 4096 fp16 -> 201 MB, ~60 us at 3.35 TB/s (curriculum number)")
print("effective traffic with gamma in L2: ~134 MB (x read + y write) -> ~40 us floor")
# Expected: 128 rows ~3.1 MB / ~9.8 us on T4; 8192 rows 201.3 MB / 60.1 us H100 / 629 us T4""")

md("""## Step 3 — Benchmark: PyTorch composite vs fused path

Composite = pow -> mean -> rsqrt -> mul as separate ops (~4 HBM passes). Fused = `torch.nn.functional.rms_norm` (a single kernel) when available. On CPU the gap is smaller (cache-resident), but the pass-count ordering holds; on a T4 the gap widens to roughly 2-3x.""")

code("""import time
import torch

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

def rms_composite(x, gamma, eps=1e-6):
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(ms + eps) * gamma

def bench(fn, x, gamma, reps=20):
    for _ in range(3):
        fn(x, gamma)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(x, gamma)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps

has_fused = hasattr(torch.nn.functional, "rms_norm")
for rows in (128, 2048, 8192):
    x = torch.randn(rows, 4096, device=device, dtype=torch.float32)
    g = torch.randn(4096, device=device)
    y_c = rms_composite(x, g)
    t_c = bench(rms_composite, x, g)
    line = f"rows {rows:>5}: composite {t_c*1000:8.2f} ms"
    if has_fused:
        fused = lambda a, b: torch.nn.functional.rms_norm(a, (4096,), weight=b)
        y_f = fused(x, g)
        t_f = bench(fused, x, g)
        ok = torch.allclose(y_c, y_f, atol=1e-4)
        bw = (3 * rows * 4096 * 4) / t_f / 1e12  # float32 in this bench: 4 B/elem
        line += f" | fused {t_f*1000:8.2f} ms | speedup {t_c/t_f:4.1f}x | outputs match: {ok} | fused traffic-rate {bw:.2f} TB/s (fp32 bench)"
    print(line)
print("note: fp32 here for CPU portability; packet numbers are fp16 (2 B/elem). Pass count, not dtype, drives the ratio.")
# Expected: outputs match True; fused faster than composite at 2048/8192 rows (gap grows with size; ~2-3x on GPU)""")

md("""## Step 4 — Compile the real kernel if a GPU + nvcc are present (T4 path)

Guarded: on a CPU runtime this prints the deferred line instead of failing. On Colab, switch to the T4 runtime (Runtime -> Change runtime type -> T4 GPU) and re-run top to bottom.""")

code("""import shutil, subprocess, os
import torch

if torch.cuda.is_available() and shutil.which("nvcc"):
    print("GPU:", torch.cuda.get_device_name(0), "| nvcc:", shutil.which("nvcc"))
    print("compile with: nvcc -O3 -arch=sm_75 rmsnorm.cu -o rmsnorm  (sm_75 = T4)")
    print("then: time rows in {128, 2048, 8192} vs F.rms_norm; achieved TB/s = 201 MB / time at 8192 rows")
else:
    print("no GPU / nvcc in this runtime -- CUDA benchmark DEFERRED to the Day 47 GPU session (per curriculum lab step 4)")
    print("the CPU simulation in Step 1 already verified the reduction logic; nothing is silently skipped")
# Expected: either the GPU/nvcc lines, or the explicit DEFERRED line""")

md("""## Step 5 — Break coalescing: predict first, then measure

In the kernel, lane t reads element base + t (coalesced). The broken mapping makes consecutive lanes read elements a large stride apart. Simulate the access-pattern cost on CPU with NumPy: a contiguous row read vs a stride-32 gather of the same useful bytes, timed per useful byte. **Write your prediction before running:** packet prediction is roughly 1.5-3x slower for the whole kernel on GPU; the CPU gather ratio below is typically larger.""")

code("""import time
import numpy as np

PREDICTION = "fill in before running: packet predicts ~1.5-3x whole-kernel slowdown on GPU"
print("prediction on record:", PREDICTION)

a = np.random.default_rng(1).standard_normal(1 << 25, dtype=np.float32)  # 128 MiB

def best(fn, reps=5):
    b = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); b = min(b, time.perf_counter() - t0)
    return b

t_contig = best(lambda: a.copy())
t_stride = best(lambda: a[::32].copy())
useful_ratio = (a.nbytes / t_contig) / ((a[::32].nbytes) / t_stride)
print(f"contiguous copy : {t_contig*1000:7.2f} ms for {a.nbytes/1e6:.0f} MB useful")
print(f"stride-32 gather: {t_stride*1000:7.2f} ms for {a[::32].nbytes/1e6:.0f} MB useful")
print(f"slowdown per useful byte: {useful_ratio:.1f}x (CPU cache lines; GPU transaction count from Day 44 is the same principle)")
print("fix: restore lane t -> element base + t so a warp load is 1 transaction of 128 B")
# Expected: stride-32 clearly slower per useful byte (machine-dependent ratio; the 32-transaction count is machine-independent)""")

md("""## Step 6 — Occupancy check: the lever that does not bind

H100 per-SM budgets: 228 KB shared memory, 65,536 registers, 2,048 threads. This kernel: ~1 KB shared memory/block, ~32 registers/thread, 256 threads/block.""")

code("""SMEM_KB, REGS, THREADS = 228, 65_536, 2048
BLOCK, K_SMEM_KB, K_REGS = 256, 1, 32
by_smem = SMEM_KB // K_SMEM_KB
by_regs = REGS // (K_REGS * BLOCK)
by_threads = THREADS // BLOCK
resident = min(by_smem, by_regs, by_threads)
print(f"blocks by shared memory: {by_smem}")
print(f"blocks by registers    : {by_regs}")
print(f"blocks by threads      : {by_threads}")
print(f"resident = {resident} blocks = {resident * BLOCK} threads; occupancy = {resident * BLOCK / THREADS * 100:.0f}%")
assert resident == 8
print("bound by registers AND threads (tie); shared memory nowhere close -> occupancy is NOT the bottleneck")
print("if this kernel is slow, look at traffic (Step 2) or the mapping (Step 5), not residency")
# Expected: 228 / 8 / 8 -> 8 blocks, 100% occupancy""")

md("""## The RMSNorm kernel — 5 lines to memorize

1. Decomposition: grid = rows, block = 256 threads — rows are independent, the reduction is the only cooperation, and it needs intra-block sync.
2. Reduction: 16 elements per thread at d=4096, 5 warp-shuffle steps + 3 tree steps = an 8-step reduction replacing 4,095 sequential adds.
3. Traffic: 3 streams x 33.5M elements x 2 B = 201 MB nominal (~134 MB effective, gamma cached in L2) -> ~60 us floor on H100, 50-70 us for a good kernel.
4. Fusion wins on passes, not math: composite ~4 HBM passes ~200+ us vs fused 1 pass — a 2-3x gap that repeats 64 times per forward pass on a 32-layer model.
5. Diagnosis order: mapping (coalescing) first, traffic second, occupancy last — this kernel is at 100% occupancy, so residency is never its excuse.

**Tomorrow (Day 46):** FlashAttention — tiling attention into SRAM blocks plus online softmax, cutting HBM traffic ~9x at 4k context vs the naive N-by-N materialization.""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}}, "nbformat": 4, "nbformat_minor": 4}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
