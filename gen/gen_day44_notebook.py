#!/usr/bin/env python3
"""Generate notebooks/day44-cuda-basics-threads-blocks-memory-coalescing.ipynb."""
import json, pathlib

SLUG = "day44-cuda-basics-threads-blocks-memory-coalescing"
NB_URL = f"https://colab.research.google.com/github/pradeepvaka/llm-inference-90day/blob/master/notebooks/{SLUG}.ipynb"
OUT = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / f"{SLUG}.ipynb"

cells = []
def md(src):
    cells.append({"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.split("\n")]})
def code(src):
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in src.split("\n")]})

md(f"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({NB_URL})

# Day 44 — CUDA Basics: Threads, Blocks, Memory Coalescing

**Big idea:** a warp is 32 threads in SIMT lockstep, and one warp load is one 128-byte transaction if the 32 addresses are contiguous — or up to 32 transactions if they are strided. Performance is address layout, not arithmetic. Today you will: (1) read and annotate a vector-add CUDA kernel (READ-ONLY, no nvcc needed), (2) count warp transactions for strides 1-32 in code, (3) simulate coalescing on CPU with NumPy (contiguous vs stride-32 copy), (4) reproduce the vector-add roofline (768 MB, ~0.23 ms on H100, ~2.4 ms on T4), and (5) compute occupancy limits for a kernel using 48 KB shared memory and 64 registers per thread (answer: 4 blocks/SM, 50%). CPU-OK throughout; ~30-60 min.""")

code("""# Cell 1: one-time installs (only cell that installs)
%pip install --quiet numpy matplotlib
print("deps ok")
# Expected: deps ok""")

md("""## Reading — the vector-add kernel (READ-ONLY)

```cuda
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // thread -> index mapping
    if (i < n) c[i] = a[i] + b[i];                  // 2 loads + 1 store, all coalesced
}
// launch: vec_add<<<262144, 256>>>(a, b, c, 1 << 26);
```

Annotation (work through it before running anything):
- Grid = 262,144 blocks, block = 256 threads -> 67,108,864 threads, one per element.
- Consecutive `threadIdx.x` -> consecutive `i` -> consecutive addresses, so each warp load of `a[i]` and `b[i]` and the store `c[i]` is **1 transaction of 128 B**.
- Blocks are independent (no cross-block communication), so any SM can run any block in any order.
- Bottleneck prediction: traffic 768 MB at 3.35 TB/s = ~0.23 ms, intensity ~0.08 FLOP/byte << ridge 295 -> **bandwidth-bound**.""")

md("""## Step 1 — Count warp transactions by stride

A warp load touches one 128 B cache line per distinct line spanned. For float (4 B) loads with stride s starting at element 0, the 32 lanes touch elements 0, s, 2s, ..., 31s. Count the distinct 128 B lines (32 floats per line).""")

code("""def warp_transactions(stride, warp=32, floats_per_line=32):
    elems = [i * stride for i in range(warp)]
    lines = {e // floats_per_line for e in elems}
    return len(lines)

for s in (1, 2, 4, 8, 16, 32):
    t = warp_transactions(s)
    eff = 128 / (t * 128) * 100
    print(f"stride {s:>2}: {t:>2} transactions, useful-byte efficiency {eff:5.1f}%")
    assert t == min(s, 32)
print("rule: transactions = stride (up to 32); payload is always 128 useful bytes")
# Expected: strides 1,2,4,8,16,32 -> transactions 1,2,4,8,16,32; efficiency 100% down to 3.1%""")

md("""## Step 2 — Simulate coalescing on CPU (NumPy)

Same principle, smaller constant: a contiguous copy streams cache lines at full bandwidth; a stride-32 gather touches a full cache line per useful element. Time both per useful byte. The strided version is slower per useful byte; the ratio is machine-dependent (the curriculum quotes roughly 5-20x on a typical Colab CPU; a shared VM can show less — the transaction count in Step 1 is the machine-independent result).""")

code("""import time
import numpy as np

rng = np.random.default_rng(0)
a = rng.standard_normal(1 << 26, dtype=np.float32)  # 256 MiB
print(f"array: {a.nbytes / 2**20:.0f} MiB, {a.size:,} floats")

def best_time(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best

t_contig = best_time(lambda: a.copy())
useful_contig = a.nbytes
strided = a[::32]  # 2M useful floats, one per 128 B line
t_stride = best_time(lambda: a[::32].copy(), reps=5)
useful_stride = strided.nbytes

bw_c = useful_contig / t_contig / 1e9
bw_s = useful_stride / t_stride / 1e9
print(f"contiguous : {t_contig*1000:7.2f} ms for {useful_contig/1e6:.0f} MB useful -> {bw_c:5.1f} GB/s per useful byte")
print(f"stride-32  : {t_stride*1000:7.2f} ms for {useful_stride/1e6:.0f} MB useful -> {bw_s:5.1f} GB/s per useful byte")
print(f"slowdown per useful byte: {bw_c / bw_s:.1f}x")
# Expected: stride-32 clearly slower per useful byte (curriculum range ~5-20x on Colab CPU; shared VMs vary)""")

md("""## Step 3 — Vector-add roofline

Traffic = 2 reads + 1 write of N floats. Time floor = traffic / bandwidth. Intensity = 1 FLOP (one add) per 12 bytes moved.""")

code("""N = 1 << 26
traffic = 3 * N * 4  # bytes
print(f"N = {N:,} floats; traffic = {traffic/1e6:.0f} MB")
for name, bw_tbs in (("H100", 3.35), ("A100-80GB", 2.04), ("T4", 0.32)):
    t_ms = traffic / (bw_tbs * 1e12) * 1000
    t_stride = t_ms * 32
    print(f"{name:>10}: coalesced floor {t_ms:5.2f} ms | stride-32 floor {t_stride:6.2f} ms")
print(f"blocks at 256 threads: {N // 256:,}; warps: {N // 32:,}")
print(f"intensity: 1 add / 12 bytes = {1/12:.3f} FLOP/byte vs H100 ridge 295 -> bandwidth-bound")
assert abs(traffic / 1e6 - 805) < 1  # 805 MB decimal (768 MiB); ~0.23-0.24 ms either way
# Expected: H100 ~0.24 ms (0.23 ms with the 768 MB rounded figure), T4 ~2.52 ms (2.4 ms rounded); stride-32 ~32x each""")

md("""## Step 4 — Occupancy: three divisions, take the minimum

H100 per-SM budgets: 228 KB shared memory, 65,536 registers, 2,048 threads (64 warps max). Kernel: 256 threads/block, 48 KB shared memory/block, 64 registers/thread.""")

code("""SMEM_KB, REGS, THREADS = 228, 65_536, 2048
BLOCK, K_SMEM_KB, K_REGS = 256, 48, 64

by_smem = SMEM_KB // K_SMEM_KB
by_regs = REGS // (K_REGS * BLOCK)
by_threads = THREADS // BLOCK
resident = min(by_smem, by_regs, by_threads)
warps = resident * BLOCK // 32
print(f"blocks by shared memory: {by_smem}  (228 / 48)")
print(f"blocks by registers    : {by_regs}  (65536 / (64 x 256))")
print(f"blocks by threads      : {by_threads}  (2048 / 256)")
print(f"resident blocks/SM = {resident}; warps/SM = {warps}; occupancy = {warps/64*100:.0f}%")
assert (by_smem, by_regs, by_threads, resident) == (4, 4, 8, 4)
print("bound by: shared memory AND registers (tie); thread count does not bind")
print(f"for 5 blocks: need smem <= {SMEM_KB/5:.1f} KB and regs <= {REGS/(BLOCK*5):.1f} per thread")
# Expected: limits 4, 4, 8 -> 4 blocks, 32 warps, 50% occupancy""")

md("""## Step 5 — Plot: transactions vs stride

The whole day in one chart: payload constant, cost linear in stride until it saturates at 32 transactions.""")

code("""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

strides = [1, 2, 4, 8, 16, 32]
trans = [warp_transactions(s) for s in strides]
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar([str(s) for s in strides], trans)
ax.set_xlabel("stride (elements between consecutive lanes)")
ax.set_ylabel("128 B transactions per warp load")
ax.set_title("Coalescing: same 128 useful bytes, up to 32x the traffic")
for i, v in enumerate(trans):
    ax.text(i, v + 0.6, str(v), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("day44-coalescing.png", dpi=110)
print("saved day44-coalescing.png; bars should read 1, 2, 4, 8, 16, 32")
# Expected: bars 1, 2, 4, 8, 16, 32""")

md("""## The CUDA mental model — 5 lines to memorize

1. Warps/SIMT: 32 threads move in lockstep on one instruction; divergence runs both branches serially.
2. Coalescing: consecutive threads must read consecutive addresses — 32 contiguous floats = 1 transaction of 128 B; stride 32 = 32 transactions for the same payload.
3. SRAM tiling: blocks share up to 228 KB of shared memory per SM on H100 (~20 cycles) vs ~400-600 cycles to HBM — tile data into it and reuse it.
4. Occupancy: resident blocks = min over shared memory, registers, threads; ~10+ warps/SM hides HBM latency, and past bandwidth saturation more warps add nothing.
5. Launch overhead: every kernel launch costs ~3-10 microseconds — trivial for a 0.23 ms vector add, dominant for dozens of tiny decode kernels.

**Tomorrow (Day 45):** your first real CUDA kernel — fused RMSNorm on a T4, benchmarked against PyTorch, with a deliberately broken coalescing variant to measure.""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}}, "nbformat": 4, "nbformat_minor": 4}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
