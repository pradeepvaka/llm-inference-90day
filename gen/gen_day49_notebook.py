#!/usr/bin/env python3
"""Generate notebooks/day49-triton-rmsnorm-softmax-kernels.ipynb."""
import json, pathlib

SLUG = "day49-triton-rmsnorm-softmax-kernels"
NB_URL = f"https://colab.research.google.com/github/pradeepvaka/llm-inference-90day/blob/master/notebooks/{SLUG}.ipynb"
OUT = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / f"{SLUG}.ipynb"

cells = []
def md(src):
    cells.append({"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.split("\n")]})
def code(src):
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in src.split("\n")]})

md(f"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({NB_URL})

# Day 49 — Triton RMSNorm + Softmax Kernels

**Big idea:** one program per row — masked load once, reduce in registers, broadcast, epilogue, masked store once. RMSNorm reduces a sum of squares; softmax reduces a max then a sum of exponentials (after max-subtraction, or you get NaNs); the fused quantize epilogue reduces a max-abs. Price first: RMSNorm at 8192x4096 fp16 = 201 MB nominal (3 streams), T4 floor 0.63 ms at 320 GB/s, good Triton 0.70-0.80 ms (>75% of peak). Fusing RMSNorm + int8 quantize saves ~134.2 MB (~57%, ~0.42 ms on T4). Triton needs CUDA: GPU steps skip cleanly on CPU, where you still do the references, hand-checks, and traffic model. About 30-60 minutes.""")

code("""# Cell 1: one-time installs (only cell that installs)
%pip install --quiet torch
print("deps ok")
# Expected: deps ok (on Colab, torch is preinstalled; triton ships with torch on CUDA builds)""")

md("""## Step 1 — Probe the environment

Triton needs CUDA in practice. Print the device and whether `triton` imports; record the result, do not assume.""")

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

md("""## Step 2 — References and hand-checks (run everywhere)

RMSNorm on [1, 2, 3, 4] with gain [2, 1, 0.5, 1] must give about [0.730, 0.730, 0.548, 1.461]. Softmax on [3, 2, 1, 0] must give about [0.644, 0.237, 0.087, 0.032], and on [1000, 999] a clean [0.731, 0.269] — while unshifted `exp(1000)` is infinity in fp32. That infinity is why the kernel subtracts the row max.""")

code("""import torch

def rmsnorm_ref(x, w, eps=1e-6):
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps) * w.float()).to(x.dtype)

row = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
gain = torch.tensor([2.0, 1.0, 0.5, 1.0])
out = rmsnorm_ref(row, gain)
print(f"rmsnorm hand-check: {[round(v, 3) for v in out[0].tolist()]}")
assert torch.allclose(out, torch.tensor([[0.730, 0.730, 0.548, 1.461]]), atol=2e-3)

for logits in ([3.0, 2.0, 1.0, 0.0], [1000.0, 999.0]):
    p = torch.softmax(torch.tensor([logits]), dim=-1)
    print(f"softmax {logits}: {[round(v, 3) for v in p[0].tolist()]}")
print(f"exp(1000) in fp32 = {torch.exp(torch.tensor(1000.0)).item()}  (overflow threshold is about 88.7)")
print(f"exp(1000 - 1000) = {torch.exp(torch.tensor(0.0)).item()}  (after max-subtraction: safe)")
assert torch.exp(torch.tensor(1000.0)).isinf()
# Expected: [0.730, 0.730, 0.548, 1.461]; [0.644, 0.237, 0.087, 0.032]; [0.731, 0.269]; inf then 1.0""")

md("""## Step 3 — Traffic model (run everywhere)

Nominal three-stream accounting from Day 45: read x, read gain (per-row nominal), write y. Note the gain is only 8 KB and cache-resident in practice, so effective HBM traffic is closer to 134 MB at the headline shape — the nominal figure is the conservative one used for percent-of-peak comparisons in this course.""")

code("""def rmsnorm_model(rows, d, dtype_bytes=2, peak_gbs=320.0):
    elems = rows * d
    one_stream = elems * dtype_bytes
    traffic = 3 * one_stream
    floor_ms = traffic / 1e9 / peak_gbs * 1000
    return traffic, floor_ms

for rows, d in ((128, 4096), (2048, 4096), (8192, 4096), (8192, 1024)):
    traffic, floor_ms = rmsnorm_model(rows, d)
    print(f"rows={rows:>5}, d={d:>5}: nominal traffic {traffic/1e6:6.1f} MB  T4 floor {floor_ms:.3f} ms")
traffic, floor_ms = rmsnorm_model(8192, 4096)
assert traffic == 201326592 and abs(floor_ms - 0.629) < 0.002

# Softmax fp32 at the same shape: read + write only
sm_traffic = 2 * 8192 * 4096 * 4
print(f"softmax 8192x4096 fp32: traffic {sm_traffic/1e6:.1f} MB  T4 floor {sm_traffic/1e9/320*1000:.2f} ms")
assert sm_traffic == 268435456
# Expected: 201.3 MB / 0.629 ms for RMSNorm headline; 268.4 MB / 0.84 ms for softmax""")

md("""## Step 4 — The Triton RMSNorm kernel

One program per row. `BLOCK_N` is the next power of 2 at or above d, passed as a compile-time constant; the column mask makes any d safe, including d=1000.""")

code("""RMSNORM_SRC = '''
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, row_stride, N, eps, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + pid * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(y_ptr + pid * row_stride + cols, y.to(tl.float16), mask=mask)
'''
print(RMSNORM_SRC)
if HAS_TRITON:
    exec(RMSNORM_SRC, globals())
    import triton as _triton
    def triton_rmsnorm(x, w, eps=1e-6):
        rows, d = x.shape
        y = torch.empty_like(x)
        BLOCK_N = _triton.next_power_of_2(d)
        rmsnorm_kernel[(rows,)](x, w, y, x.stride(0), d, eps, BLOCK_N=BLOCK_N)
        return y
    print("kernel defined; launcher ready")
else:
    print("SKIPPED: no CUDA / triton - kernel source above is the deliverable to read")
# Expected: kernel source printed; launcher ready on GPU, skip line on CPU""")

md("""## Step 5 — RMSNorm correctness, including the hand-check row and a non-power-of-2 d""")

code("""if HAS_TRITON:
    # hand-check row first: a kernel that misses 4 hand numbers does not get benchmarked
    row = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="cuda", dtype=torch.float16)
    gain = torch.tensor([2.0, 1.0, 0.5, 1.0], device="cuda", dtype=torch.float16)
    got = triton_rmsnorm(row, gain)[0].float().cpu()
    print(f"kernel hand-check: {[round(v, 3) for v in got.tolist()]} (expect about [0.730, 0.730, 0.548, 1.461])")
    assert torch.allclose(got, torch.tensor([0.730, 0.730, 0.548, 1.461]), atol=3e-3)
    for rows, d in ((37, 1024), (128, 4096), (64, 1000)):
        x = torch.randn(rows, d, device="cuda", dtype=torch.float16)
        w = torch.randn(d, device="cuda", dtype=torch.float16)
        err = (triton_rmsnorm(x, w).float() - rmsnorm_ref(x, w).float()).abs().max().item()
        print(f"rows={rows:>4}, d={d:>5}: max |triton - ref| = {err:.2e}")
        assert err < 1e-3
    print("correctness: PASS - including non-power-of-2 d=1000 (the mask, not luck)")
else:
    print("SKIPPED: no CUDA / triton")
    print("expected on T4: hand-check [0.730, 0.730, 0.548, 1.461]; max diff < 1e-3 fp16, incl. d=1000")
# Expected: hand-check matches; max diff < 1e-3 on GPU; skip line on CPU""")

md("""## Step 6 — Benchmark RMSNorm fairly, and vs the PyTorch composite

Warmup 3, time 20, synchronise, median. Convert with nominal traffic ( Step 3 ) into GB/s and % of T4 peak (320 GB/s). Expect the % to climb with rows: at 128 rows you are mostly measuring launch overhead.""")

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
    for rows, d in ((128, 4096), (2048, 4096), (8192, 4096), (8192, 1024)):
        x = torch.randn(rows, d, device="cuda", dtype=torch.float16)
        w = torch.randn(d, device="cuda", dtype=torch.float16)
        ms = bench(lambda: triton_rmsnorm(x, w))
        traffic, _ = rmsnorm_model(rows, d)
        gbs = traffic / 1e9 / (ms / 1000)
        print(f"triton rmsnorm rows={rows:>5}, d={d:>5}: median {ms:.3f} ms  {gbs:.0f} GB/s  {100*gbs/320:.0f}% of T4 peak")
        del x, w
    x = torch.randn(8192, 4096, device="cuda", dtype=torch.float16)
    w = torch.randn(4096, device="cuda", dtype=torch.float16)
    def composite(x, w, eps=1e-6):
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + eps) * w.float()).half()
    ms_c = bench(lambda: composite(x, w))
    traffic, _ = rmsnorm_model(8192, 4096)
    print(f"torch composite 8192x4096: median {ms_c:.3f} ms  {traffic/1e9/(ms_c/1000):.0f} GB/s nominal")
    print("expect: triton 8192x4096 about 0.70-0.80 ms and >75% of peak; composite roughly 2 ms or worse (extra HBM passes)")
    del x, w
    torch.cuda.empty_cache()
else:
    print("SKIPPED: no CUDA / triton")
    print("expected on T4: triton 8192x4096 about 0.70-0.80 ms (>75% of peak nominal); 128-row % far lower (launch overhead)")
# Expected: benchmark table on GPU; skip line on CPU""")

md("""## Step 7 — The Triton softmax kernel: same skeleton, max-subtraction built in

Masked load uses `other=-inf` so padded columns can never win the max or add to the sum. The row stays on chip, so this is two reductions over registers and still one pass over HBM. (When a row does not fit on chip, use Day 46's online form instead.)""")

code("""SOFTMAX_SRC = '''
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, y_ptr, row_stride, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + pid * row_stride + cols, mask=mask, other=float("-inf")).to(tl.float32)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    tl.store(y_ptr + pid * row_stride + cols, (e / denom).to(tl.float32), mask=mask)
'''
print(SOFTMAX_SRC)
if HAS_TRITON:
    exec(SOFTMAX_SRC, globals())
    import triton as _triton
    def triton_softmax(x):
        rows, d = x.shape
        y = torch.empty_like(x)
        BLOCK_N = _triton.next_power_of_2(d)
        softmax_kernel[(rows,)](x, y, x.stride(0), d, BLOCK_N=BLOCK_N)
        return y
    print("kernel defined; launcher ready")
else:
    print("SKIPPED: no CUDA / triton - kernel source above is the deliverable to read")
# Expected: kernel source printed; launcher ready on GPU, skip line on CPU""")

md("""## Step 8 — Softmax correctness (including the overflow row) and benchmark""")

code("""if HAS_TRITON:
    for rows, d in ((37, 1024), (128, 4096), (16, 1000)):
        x = torch.randn(rows, d, device="cuda", dtype=torch.float32) * 3
        err = (triton_softmax(x) - torch.softmax(x, dim=-1)).abs().max().item()
        print(f"rows={rows:>4}, d={d:>5}: max |triton - torch| = {err:.2e}")
        assert err < 1e-4
    hot = torch.tensor([[1000.0, 999.0]], device="cuda")
    p = triton_softmax(hot)
    print(f"overflow row [1000, 999]: {[round(v, 3) for v in p[0].tolist()]} (expect [0.731, 0.269], no NaN)")
    assert not torch.isnan(p).any()
    x = torch.randn(8192, 4096, device="cuda", dtype=torch.float32)
    rowsums = triton_softmax(x).sum(dim=-1)
    print(f"row sums: min {rowsums.min().item():.6f}  max {rowsums.max().item():.6f} (expect 1.0 within 1e-5)")
    ms = bench(lambda: triton_softmax(x))
    ms_t = bench(lambda: torch.softmax(x, dim=-1))
    traffic_gb = 2 * 8192 * 4096 * 4 / 1e9
    print(f"triton softmax 8192x4096 fp32: median {ms:.3f} ms  {traffic_gb/(ms/1000):.0f} GB/s")
    print(f"torch.softmax (already fused): median {ms_t:.3f} ms - expect triton near parity to modestly ahead; record honestly")
    del x
    torch.cuda.empty_cache()
else:
    print("SKIPPED: no CUDA / triton")
    print("expected on T4: max diff < 1e-4 fp32; [1000, 999] -> [0.731, 0.269] with no NaN; floor 0.84 ms at 268 MB")
# Expected: correctness PASS + benchmark pair on GPU; skip line on CPU""")

md("""## Step 9 — Fuse the epilogue: RMSNorm + per-row int8 quantize in one kernel

One extra reduction (`max(|y|) / 127` per row), one different store (int8 codes + one fp32 scale per row). Traffic: unfused ~234.9 MB vs fused ~100.7 MB at 8192x4096 — about 134.2 MB (57%, ~0.42 ms on T4) never paid. Accuracy bound: dequantized error per element <= scale/2, plus a small fp16 rounding allowance.""")

code("""FUSED_SRC = '''
import triton
import triton.language as tl

@triton.jit
def rmsnorm_quant_kernel(x_ptr, w_ptr, q_ptr, s_ptr, row_stride, N, eps, BLOCK_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + pid * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    scale = tl.max(tl.abs(y), axis=0) / 127.0
    v = y / scale
    rounded = tl.where(v >= 0, tl.floor(v + 0.5), -tl.floor(-v + 0.5))
    codes = rounded.to(tl.int32)
    codes = tl.minimum(127, tl.maximum(-127, codes))
    tl.store(q_ptr + pid * row_stride + cols, codes.to(tl.int8), mask=mask)
    tl.store(s_ptr + pid, scale)
'''
print(FUSED_SRC)
if HAS_TRITON:
    exec(FUSED_SRC, globals())
    import triton as _triton
    def triton_rmsnorm_quant(x, w, eps=1e-6):
        rows, d = x.shape
        q = torch.empty(rows, d, device=x.device, dtype=torch.int8)
        s = torch.empty(rows, device=x.device, dtype=torch.float32)
        BLOCK_N = _triton.next_power_of_2(d)
        rmsnorm_quant_kernel[(rows,)](x, w, q, s, x.stride(0), d, eps, BLOCK_N=BLOCK_N)
        return q, s
    x = torch.randn(8192, 4096, device="cuda", dtype=torch.float16)
    w = torch.randn(4096, device="cuda", dtype=torch.float16)
    q, s = triton_rmsnorm_quant(x, w)
    deq = q.float() * s.unsqueeze(1)
    ref = rmsnorm_ref(x, w).float()
    err = (deq - ref).abs()
    bound = (s / 2).unsqueeze(1) + 0.01
    print(f"dequant max error: {err.max().item():.4f}; rows within scale/2 (+0.01 allowance): {bool((err <= bound).all())}")
    assert bool((err <= bound).all())
    ms = bench(lambda: triton_rmsnorm_quant(x, w))
    fused_traffic = (8192 * 4096 * 2 + 8192 * 4096 * 1 + 8192 * 4) / 1e6
    unfused_traffic = (2 * 8192 * 4096 * 2 + 8192 * 4096 * 1 + 8192 * 4096 * 2) / 1e6
    print(f"fused kernel median: {ms:.3f} ms; fused traffic about {fused_traffic:.1f} MB vs unfused about {unfused_traffic:.1f} MB")
    print(f"traffic saved: about {unfused_traffic - fused_traffic:.1f} MB (about {(unfused_traffic - fused_traffic)/320*1000:.2f} ms at 320 GB/s)")
    del x, w, q, s, deq, ref
    torch.cuda.empty_cache()
else:
    elems = 8192 * 4096
    fused = (elems * 2 + elems * 1 + 8192 * 4) / 1e6
    unfused = (2 * elems * 2 + elems * 1 + elems * 2) / 1e6
    print("SKIPPED GPU part: no CUDA / triton")
    print(f"model only: fused about {fused:.1f} MB vs unfused about {unfused:.1f} MB; saved about {unfused - fused:.1f} MB")
print("traffic accounting: fused = read fp16 x + write int8 codes + write scales; unfused adds a full fp16 write + read round trip")
# Expected: error within bound on GPU; fused ~100.7 MB vs unfused ~234.9 MB; saved ~134.2 MB""")

md("""## Your verdict — fill this in

Replace the placeholders with your own numbers from Steps 5, 6, 8 and 9, then run the cell.""")

code("""VERDICT = {
    "RMSNorm % of peak": "fill in: median ms, GB/s and % of T4 peak at 8192x4096 from Step 6 (or: skipped on CPU, floor 0.63 ms)",
    "softmax comparison": "fill in: your triton ms vs torch.softmax ms at 8192x4096 fp32 from Step 8 - honestly, even if torch wins",
    "fusion saving": "fill in: MB saved and the error-vs-bound result from Step 9 (expect about 134.2 MB saved, error within scale/2)",
    "the pattern": "Row-wise single-pass: what is the reduction, what is broadcast, what is the epilogue. Only those three change from op to op.",
}
for k, v in VERDICT.items():
    print(f"{k}: {v}")
print("check every number against your own runs above before calling Day 49 done")
# Expected: four lines printed, numbers copied from your runs""")

md("""## Day 49 — 5 lines to memorize

1. One program per row: masked load once, reduce in registers, broadcast, epilogue, masked store once — RMSNorm, softmax and quantize are the same skeleton.
2. RMSNorm at 8192x4096 fp16: 201 MB nominal traffic, T4 floor 0.63 ms; good Triton lands 0.70-0.80 ms (>75% of peak); a multi-kernel composite pays ~4 passes and ~2+ ms.
3. Softmax must subtract the row max: exp overflows fp32 at about 88.7; shifted exponentials are all <= 1 and the probabilities are mathematically identical ([1000, 999] -> [0.731, 0.269], not NaN).
4. Small shapes lie: at 128 rows the floor is ~10 microseconds, the size of one launch — percent of peak climbs with rows because the launch tax amortizes.
5. Fusion is traffic economics: RMSNorm + int8 quantize in one kernel saves ~134.2 MB (~57%, ~0.42 ms on T4) at 8192x4096, with per-element error bounded by scale/2.

**Tomorrow (Day 50):** torch.compile (TorchDynamo + Inductor) and CUDA graphs — let the compiler fuse kernels, and capture the decode loop to kill launch overhead.""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}}, "nbformat": 4, "nbformat_minor": 4}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
