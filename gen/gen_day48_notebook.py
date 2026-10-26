#!/usr/bin/env python3
"""Generate notebooks/day48-triton-intro.ipynb."""
import json, pathlib

SLUG = "day48-triton-intro"
NB_URL = f"https://colab.research.google.com/github/pradeepvaka/llm-inference-90day/blob/master/notebooks/{SLUG}.ipynb"
OUT = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / f"{SLUG}.ipynb"

cells = []
def md(src):
    cells.append({"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.split("\n")]})
def code(src):
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in src.split("\n")]})

md(f"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({NB_URL})

# Day 48 — Triton Intro

**Big idea:** Triton moves the unit of work from a thread to a tile. You write a Python function per program, build block pointers from `tl.program_id`, load / add / store a whole tile, and let the compiler map tiles to threads and warps. Today: vector add at N=2^26 — price it first (805 MB traffic, 2.52 ms floor on a T4 at 320 GB/s, grid 65,536 at BLOCK=1024), verify it against `torch.add`, benchmark fairly, sweep BLOCK in {{256, 1024, 4096}}, break the mask safely into a padded buffer (expect 448 out-of-range writes at N=1,000,000), then restore it. Triton needs CUDA: GPU steps skip cleanly on CPU, where you still do the model, the reference, and the mask arithmetic. About 30-60 minutes.""")

code("""# Cell 1: one-time installs (only cell that installs)
%pip install --quiet torch
print("deps ok")
# Expected: deps ok (on Colab, torch is preinstalled; triton ships with torch on CUDA builds)""")

md("""## Step 1 — Probe the environment

Triton needs CUDA in practice. Print the device and whether `triton` imports. The result decides which path today is real — record it, do not assume.""")

code("""import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
HAS_TRITON = False
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    try:
        import triton
        import triton.language as tl
        print(f"triton: {triton.__version__}")
        HAS_TRITON = True
    except Exception as e:
        print(f"triton not available (skip, not a failure): {type(e).__name__}: {e}")
else:
    print("device: cpu - Triton kernel steps will skip with a printed reason")
print(f"HAS_TRITON = {HAS_TRITON}")
# Expected: torch version + device printed; triton version on T4, skip reason on CPU""")

md("""## Step 2 — Reference and traffic model (runs everywhere)

Price it before running it: bytes per tensor, total traffic for read-x + read-y + write-out, arithmetic intensity, and the T4 time floor.""")

code("""import torch

def traffic_model(n, dtype_bytes=4, peak_gbs=320.0):
    per_tensor = n * dtype_bytes
    traffic = 3 * per_tensor
    floor_ms = traffic / 1e9 / peak_gbs * 1000
    intensity = n / traffic  # 1 FLOP per element / bytes per element-set
    return per_tensor, traffic, floor_ms, intensity

N = 2 ** 26
per, traffic, floor_ms, inten = traffic_model(N)
print(f"N = {N:,} elements")
print(f"bytes per tensor: {per/1e6:.1f} MB")
print(f"total traffic: {traffic/1e6:.1f} MB")
print(f"arithmetic intensity: {inten:.3f} FLOP/byte")
print(f"T4 floor at 320 GB/s: {floor_ms:.2f} ms")
assert per == 268435456 and traffic == 805306368
assert abs(floor_ms - 2.52) < 0.01

# PyTorch reference on whatever device exists (smaller on CPU)
n_ref = N if device == "cuda" else 1_000_000
x = torch.randn(n_ref, device=device)
y = torch.randn(n_ref, device=device)
ref = torch.add(x, y)
print(f"reference torch.add on {device}: shape {tuple(ref.shape)}, sample {ref[0].item():.4f}")
del x, y, ref
if device == "cuda":
    torch.cuda.empty_cache()
# Expected: 268.4 MB per tensor, 805.3 MB traffic, 0.083 FLOP/byte, 2.52 ms floor""")

md("""## Step 3 — The Triton vector-add kernel

One program per tile: `pid` -> `offs` -> masked `tl.load` -> add -> masked `tl.store`. BLOCK is a compile-time constant (`tl.constexpr`) and must be a power of 2.""")

code("""KERNEL_SRC = '''
import triton
import triton.language as tl

@triton.jit
def vecadd_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
'''
print(KERNEL_SRC)
if HAS_TRITON:
    exec(KERNEL_SRC, globals())
    import math
    def vecadd(x, y, BLOCK=1024):
        out = torch.empty_like(x)
        grid = (math.ceil(x.numel() / BLOCK),)
        vecadd_kernel[grid](x, y, out, x.numel(), BLOCK=BLOCK)
        return out
    print("kernel defined; launcher ready")
else:
    print("SKIPPED: no CUDA / triton - kernel source above is the deliverable to read")
# Expected: kernel source printed; launcher ready on GPU, skip line on CPU""")

md("""## Step 4 — Correctness vs torch.add""")

code("""if HAS_TRITON:
    for n in (1024, 1000000, 999983):
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        out = vecadd(x, y, BLOCK=1024)
        err = (out - torch.add(x, y)).abs().max().item()
        grid = (n + 1023) // 1024
        print(f"n={n:>9,}: grid={grid:>5}  max |triton - torch| = {err:.1e}")
        assert err == 0.0
        del x, y, out
    torch.cuda.empty_cache()
    print("correctness: PASS - exact in fp32, including prime n=999,983 (mask works)")
else:
    print("SKIPPED: no CUDA / triton")
    print("expected on T4: max diff 0.0 for n in {1024, 1000000, 999983}; grid for 1,000,000 is 977")
# Expected: max diff 0.0 on GPU; skip line on CPU""")

md("""## Step 5 — Fair benchmark and BLOCK sweep

Warmup 3, time 20, synchronise, median. Convert median time to achieved GB/s = traffic / time, and to % of T4 peak (320 GB/s).""")

code("""import statistics, time

def bench(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)

if HAS_TRITON:
    N = 2 ** 26
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    traffic_gb = 3 * N * 4 / 1e9
    for BLOCK in (256, 1024, 4096):
        grid = (N + BLOCK - 1) // BLOCK
        ms = bench(lambda: vecadd(x, y, BLOCK=BLOCK))
        gbs = traffic_gb / (ms / 1000)
        print(f"BLOCK={BLOCK:>5}: programs={grid:>7,}  median {ms:.2f} ms  {gbs:.0f} GB/s  {100*gbs/320:.0f}% of T4 peak")
    del x, y
    torch.cuda.empty_cache()
    print("expect: all three near 270-304 GB/s (85-95% of peak), within ~10-20% of each other")
else:
    print("SKIPPED: no CUDA / triton")
    for BLOCK in (256, 1024, 4096):
        print(f"BLOCK={BLOCK:>5}: programs={(2**26 + BLOCK - 1)//BLOCK:>7,}  (model only)")
    print("expected on T4: 65,536 programs at BLOCK=1024; ~2.65-2.96 ms median")
# Expected: sweep table on GPU; model-only grids on CPU""")

md("""## Step 6 — Break the mask, safely

Simulation that runs everywhere: at N=1,000,000 and BLOCK=1024, how many offsets in the last program are out of range? Then, on GPU, the padded-buffer version: write unmasked into a padded output and count changed padding elements — expect 448. Restore the mask and expect 0.""")

code("""n, BLOCK = 1_000_000, 1024
grid = (n + BLOCK - 1) // BLOCK
last_start = (grid - 1) * BLOCK
oor = grid * BLOCK - n
print(f"grid = {grid} programs; last program starts at {last_start:,}")
print(f"out-of-range offsets in last program: {oor}")
assert grid == 977 and oor == 448

if HAS_TRITON:
    import triton
    import triton.language as tl
    @triton.jit
    def vecadd_nomask_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        tl.store(out_ptr + offs, x + y)
    pad = 1024
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    xp = torch.zeros(n + pad, device="cuda"); xp[:n] = x
    yp = torch.zeros(n + pad, device="cuda"); yp[:n] = y
    outp = torch.zeros(n + pad, device="cuda")
    vecadd_nomask_kernel[(grid,)](xp, yp, outp, BLOCK=BLOCK)
    changed = int((outp[n:] != 0).sum().item())
    print(f"unmasked into padded buffer: padding elements changed = {changed} (expect 448; a few may coincidentally sum to 0.0)")
    print("on an unpadded allocation this same bug can segfault or corrupt silently - that is why the mask is not optional")
    del x, y, xp, yp, outp
    torch.cuda.empty_cache()
else:
    print("SKIPPED GPU part: no CUDA / triton - the arithmetic above (448) is the expected count")
# Expected: grid 977, out-of-range 448; GPU padded count about 448""")

md("""## Step 7 — When Triton loses: price a GEMM

A 4096 x 4096 x 4096 GEMM is 137 GFLOP of perfectly regular work. That is the vendor library's home turf. Triton's win is the fused, irregular ops around it.""")

code("""m = k = n_ = 4096
flops = 2 * m * n_ * k
print(f"GEMM {m}x{n_}x{k}: {flops/1e9:.1f} GFLOP")
print(f"at a Triton-typical 85% of cuBLAS on this shape, you give up ~{0.15*flops/1e9:.0f} GFLOP-equivalent of throughput")
print("rule: cuBLAS for large regular GEMMs; Triton for fused / paged / quantised / row-wise ops around them")
print("examples that ship Triton first: paged attention variants, dequantise-then-multiply, fused RMSNorm + cast")
assert abs(flops / 1e9 - 137.4) < 0.1
# Expected: 137.4 GFLOP; the rule printed""")

md("""## Your verdict — fill this in

Replace the placeholders with your own numbers from Steps 1-6, then run the cell.""")

code("""VERDICT = {
    "achieved bandwidth": "fill in: your GB/s and % of peak at BLOCK=1024 from Step 5 (or: skipped on CPU, model floor 2.52 ms)",
    "BLOCK sweep": "fill in: GB/s at 256 / 1024 / 4096 and the spread",
    "mask lesson": "fill in: padded out-of-range count from Step 6 (expect 448 unmasked, 0 masked)",
    "decision rule": "Use the vendor library for large regular GEMMs; use Triton for fused, irregular, paged, quantised, or row-wise ops - most kernels an inference team actually writes.",
}
for k, v in VERDICT.items():
    print(f"{k}: {v}")
print("check every number against your own runs above before calling Day 48 done")
# Expected: four lines printed, numbers copied from your runs""")

md("""## Day 48 — 5 lines to memorize

1. A Triton program handles a whole tile (e.g. 1024 elements); the compiler maps tiles to threads/warps — you never write per-thread indices.
2. Block pointers make coalescing automatic: a block load over consecutive offsets is one contiguous transaction by construction.
3. Vector add at N=2^26 fp32 moves 805 MB; T4 floor is 2.52 ms at 320 GB/s; good Triton lands at 85-95% of peak (2.65-2.96 ms).
4. Every load/store derived from a runtime size carries a mask: at N=1,000,000, BLOCK=1024, the last of 977 programs has 448 out-of-range offsets.
5. Triton vs cuBLAS: cuBLAS for large regular GEMMs; Triton for fused/custom ops around them — 90% of CUDA performance at 10% of the code.

**Tomorrow (Day 49):** Triton RMSNorm + softmax kernels — one program per row, single pass, benchmarked vs PyTorch, target >80% of peak bandwidth.""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}}, "nbformat": 4, "nbformat_minor": 4}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
