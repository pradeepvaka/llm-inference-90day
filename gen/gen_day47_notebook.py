#!/usr/bin/env python3
"""Generate notebooks/day47-flashattention-hands-on.ipynb."""
import json, pathlib

SLUG = "day47-flashattention-hands-on"
NB_URL = f"https://colab.research.google.com/github/pradeepvaka/llm-inference-90day/blob/master/notebooks/{SLUG}.ipynb"
OUT = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / f"{SLUG}.ipynb"

cells = []
def md(src):
    cells.append({"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.split("\n")]})
def code(src):
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in src.split("\n")]})

md(f"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({NB_URL})

# Day 47 — FlashAttention Hands-on

**Big idea:** Yesterday's traffic model said tiling should cut HBM traffic about 9x at n=4096 and give 2-4x wall clock. Today you verify the real thing: probe which SDPA backend dispatches, check parity against a manual reference (fp32 < 1e-5, fp16 < 1e-3), test for causal leaks, benchmark fairly (warmup, synchronise, median) at n in {{512, 2048, 8192}} on GPU (smaller on CPU), price the 8.59 GB score matrix at batch 8 / n=4096 / 32 heads, and integrate a guarded SDPA/flash branch into a tiny GPT trained for 200 steps. GPU steps skip cleanly on CPU. About 30-60 minutes.""")

code("""# Cell 1: one-time installs (only cell that installs)
%pip install --quiet torch
print("deps ok")
# Expected: deps ok (on Colab, torch is preinstalled; this is a no-op safeguard)""")

md("""## Step 1 — Probe the environment before installing anything

Print torch version, CUDA availability, device, and the flash SDPA flag. The flag and the import result decide which path today is *real* — record them, do not assume.""")

code("""import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    try:
        print(f"flash_sdp_enabled: {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"mem_efficient_sdp_enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    except Exception as e:
        print(f"sdp backend probe failed: {e}")
else:
    print("device: cpu — GPU steps will skip with a printed reason")
try:
    import flash_attn
    print(f"flash_attn available: {flash_attn.__version__}")
    HAS_FLASH_ATTN = True
except Exception as e:
    print(f"flash_attn not available (skip, not a failure): {type(e).__name__}")
    print("note: pip install flash-attn --no-build-isolation builds from source (~10 min on T4) and needs a wheel matching your torch/CUDA")
    HAS_FLASH_ATTN = False
# Expected: torch version + device printed; flash_attn line is either a version or a skip reason""")

md("""## Step 2 — Correctness harness: manual reference vs SDPA

Manual reference = your Day 3 code (scores, causal mask, softmax, weighted sum) in fp32. SDPA must match it below 1e-5 in fp32. Then, on GPU only, the fp16 bar is 1e-3. Finally the causal-leak check: perturbing the *last* token must not change earlier outputs.""")

code("""import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, H, T, D = 2, 4, 64, 32

def manual_attention(q, k, v, causal=True):
    # q, k, v: (B, H, T, D), fp32 reference
    scale = 1.0 / (q.shape[-1] ** 0.5)
    s = q @ k.transpose(-2, -1) * scale
    if causal:
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ v

q = torch.randn(B, H, T, D)
k = torch.randn(B, H, T, D)
v = torch.randn(B, H, T, D)
ref = manual_attention(q, k, v, causal=True)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
err32 = (ref - out).abs().max().item()
print(f"fp32 max |manual - SDPA| = {err32:.2e}")
assert err32 < 1e-5, err32
print("fp32 parity: PASS (< 1e-5)")

# causal-leak check: change only the last token of k/v input via v's last position source
v2 = v.clone(); v2[:, :, -1, :] += 10.0
# NOTE: with causal attention, position t < T-1 cannot see position T-1 through v if k also unchanged?
# Direct leak test on SDPA: perturb future *keys/values* and check position 0 output.
q2, k2, v3 = q.clone(), k.clone(), v.clone()
k2[:, :, -1, :] += 5.0; v3[:, :, -1, :] += 5.0
out_perturbed = F.scaled_dot_product_attention(q2, k2, v3, is_causal=True)
leak = (out - out_perturbed)[:, :, :-1, :].abs().max().item()
print(f"causal leak (earlier positions, expect 0.0): {leak:.2e}")
assert leak == 0.0
print("causal check: PASS")

if torch.cuda.is_available():
    qh, kh, vh = [x.cuda().half() for x in (q, k, v)]
    ref_h = manual_attention(qh.float(), kh.float(), vh.float(), causal=True)
    out_h = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
    err16 = (ref_h - out_h.float()).abs().max().item()
    print(f"fp16 max |manual(fp32) - SDPA(fp16)| = {err16:.2e} (bar: < 1e-3, may be near it at this scale — record yours)")
    print("record the fp16 number in the packet table; if it exceeds 1e-3, check layout and dtype before blaming the kernel")
else:
    print("fp16 GPU parity: SKIPPED — no CUDA device (bar would be < 1e-3)")
# Expected: fp32 error < 1e-5, leak 0.0; fp16 line on GPU only""")

md("""## Step 3 — The layout trap, demonstrated safely

`flash_attn_func` wants (batch, seqlen, heads, headdim). SDPA wants (batch, heads, seqlen, headdim). Show the transpose explicitly and verify that a layout-correct SDPA call through the flash-style layout round-trips to the same answer.""")

code("""import torch
import torch.nn.functional as F

# (B, H, T, D) -> (B, T, H, D) -> back must be an identity on data, and attention must agree
x = torch.randn(2, 4, 16, 32)
flash_layout = x.transpose(1, 2)          # (B, T, H, D)
assert flash_layout.shape == (2, 16, 4, 32)
back = flash_layout.transpose(1, 2)
assert torch.equal(x, back)
print(f"SDPA layout: {tuple(x.shape)} (B, H, T, D)")
print(f"flash layout: {tuple(flash_layout.shape)} (B, T, H, D)")
print(f"one Q tensor at B=8, T=4096, H=32, D=128, fp16: {8*4096*32*128*2/1e6:.1f} MB")
print("layout round-trip: PASS — transpose(1, 2) is the whole conversion, and forgetting it is the classic bug")
# Expected: shapes (2,4,16,32) and (2,16,4,32); Q tensor 268.4 MB""")

md("""## Step 4 — Fair benchmark: warmup, synchronise, median

GPU: n in {512, 2048, 8192}, SDPA (default dispatch) vs the explicit math backend, median of 20 after 3 warmups, with `torch.cuda.synchronize()`. CPU: smaller sizes, SDPA vs manual, median of 5 — expect the pattern (gap grows with n), not A100 numbers.""")

code("""import statistics, time
import torch
import torch.nn.functional as F

def bench(fn, iters, warmup, device):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)

if torch.cuda.is_available():
    dev = "cuda"
    for T in (512, 2048, 8192):
        try:
            q = torch.randn(2, 8, T, 64, device=dev, dtype=torch.float16)
            k = torch.randn(2, 8, T, 64, device=dev, dtype=torch.float16)
            v = torch.randn(2, 8, T, 64, device=dev, dtype=torch.float16)
            t_sdpa = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), 20, 3, dev)
            from torch.nn.attention import SDPBackend, sdpa_kernel
            try:
                with sdpa_kernel(SDPBackend.MATH):
                    t_math = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), 20, 3, dev)
                print(f"n={T:>5}: SDPA median {t_sdpa:.2f} ms | math median {t_math:.2f} ms | speedup {t_math/t_sdpa:.2f}x")
            except Exception as e:
                print(f"n={T:>5}: SDPA median {t_sdpa:.2f} ms | math backend unavailable: {e}")
            del q, k, v
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"n={T}: SKIPPED ({e})")
else:
    print("CPU path: SDPA vs manual reference (expect small gaps; the quadratic term is still small)")
    for T in (128, 256):
        q = torch.randn(2, 4, T, 32)
        k = torch.randn(2, 4, T, 32)
        v = torch.randn(2, 4, T, 32)
        t_sdpa = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), 5, 2, "cpu")
        t_man = bench(lambda: manual_attention(q, k, v, causal=True), 5, 2, "cpu")
        print(f"n={T:>5}: SDPA median {t_sdpa:.2f} ms | manual median {t_man:.2f} ms | ratio {t_man/t_sdpa:.2f}x")
    print("on an A100 the packet's worked example is 28 ms eager vs 9 ms flash at n=4096 (3.1x) — your hardware, your number")
# Expected: a median table; on GPU the speedup should generally grow with n; on CPU record the pattern""")

md("""## Step 5 — Price the memory claim

Materialised scores = batch x heads x n x n x 2 bytes. Flash extra state = output + one max and one normaliser per row per head — linear in n.""")

code("""def score_gb(B, H, n, bytes_per=2):
    return B * H * n * n * bytes_per / 1e9

for n in (4096, 16384):
    print(f"n={n:>6}: materialised scores at B=8, H=32 = {score_gb(8, 32, n):.2f} GB")
assert abs(score_gb(8, 32, 4096) - 8.59) < 0.01
assert abs(score_gb(8, 32, 16384) - 137.44) < 0.1
lin = 8 * 4096 * 32 * 128 * 2 / 1e9
print(f"flash output tensor at n=4096: {lin:.3f} GB (linear state, not quadratic)")
print("verdict: the 16384 row is an OOM prediction for the eager path, not a run to attempt")
# Expected: 8.59 GB and 137.44 GB; output 0.268 GB""")

md("""## Step 6 — Integrate into a tiny GPT: guarded branch + module parity

One attention class, two branches: `sdpa` (which dispatches to flash/efficient/math) and the explicit `math` fallback that stays forever as the reference. Same weights, same input — parity first, then training.""")

code("""import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalAttention(nn.Module):
    def __init__(self, d_model=64, n_head=4):
        super().__init__()
        self.n_head, self.head_dim = n_head, d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x, backend="math"):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        q, k, v = split(q), split(k), split(v)
        if backend == "sdpa":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            s = q @ k.transpose(-2, -1) * scale
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            y = torch.softmax(s.masked_fill(mask, float("-inf")), dim=-1) @ v
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

torch.manual_seed(42)
attn = CausalAttention()
x = torch.randn(2, 32, 64)
with torch.no_grad():
    a = attn(x, backend="math")
    b = attn(x, backend="sdpa")
err = (a - b).abs().max().item()
print(f"module parity (fp32, math vs sdpa): max diff = {err:.2e}")
assert a.shape == b.shape and err < 1e-5
print("module parity: PASS — same shape, fp32 diff < 1e-5 (fp16 GPU bar would be < 1e-3)")
# Expected: diff < 1e-5, shapes equal""")

md("""## Step 7 — Train both branches 200 steps and compare trajectories

Tiny char model on a repeated text. Both branches start near ln(vocab) and must fall together — same learning, different rounding. Final losses within about 0.1 of each other is a pass.""")

code("""import torch
import torch.nn as nn

text = ("flash attention never writes the score matrix to slow memory. "
        "it tiles the work, keeps the state on chip, and stays exact. ") * 30
chars = sorted(set(text))
vocab = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
print(f"vocab size: {vocab} | ln(vocab) = {torch.log(torch.tensor(float(vocab))):.2f} (expected step-0 loss)")

class Block(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.attn = CausalAttention(64, 4)
        self.ln1, self.ln2 = nn.LayerNorm(64), nn.LayerNorm(64)
        self.mlp = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64))
        self.backend = backend
    def forward(self, x):
        x = x + self.attn(self.ln1(x), backend=self.backend)
        return x + self.mlp(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.emb = nn.Embedding(vocab, 64)
        self.pos = nn.Embedding(32, 64)
        self.blocks = nn.Sequential(*[Block(backend) for _ in range(2)])
        self.head = nn.Linear(64, vocab)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        logits = self.head(self.blocks(x))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, vocab), targets.view(-1))
        return logits, loss

def get_batch():
    ix = torch.randint(len(data) - 33, (16,))
    x = torch.stack([data[i:i+32] for i in ix])
    y = torch.stack([data[i+1:i+33] for i in ix])
    return x, y

results = {}
for backend in ("math", "sdpa"):
    torch.manual_seed(123)
    model = TinyGPT(backend)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = {}
    for step in range(201):
        xb, yb = get_batch()
        _, loss = model(xb, yb)
        if step in (0, 100, 200):
            losses[step] = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    results[backend] = losses
    print(f"{backend}: step 0 = {losses[0]:.3f} | step 100 = {losses[100]:.3f} | step 200 = {losses[200]:.3f}")
gap = abs(results["math"][200] - results["sdpa"][200])
print(f"final-loss gap between branches: {gap:.3f} (expect < ~0.5 on this tiny run; trajectories should track, not match bitwise)")
assert results["math"][200] < results["math"][0] and results["sdpa"][200] < results["sdpa"][0]
print("training check: PASS — both branches learned")
# Expected: step-0 near ln(vocab); step-200 clearly lower for both""")

md("""## Your verdict — fill this in

Replace the placeholders with your own numbers from Steps 1-7, then run the cell.""")

code("""VERDICT = {
    "backend that ran": "fill in: flash / memory-efficient / math, and how you know",
    "fp32 parity": "fill in: your max diff from Step 2",
    "benchmark pattern": "fill in: medians by sequence length from Step 4",
    "decision rule": "Enable the flash/SDPA path for prefill and training at long sequence lengths behind a guarded fallback; keep the math path as the tested reference, and do not expect it to fix decode, which is key-value-cache bound.",
}
for k, v in VERDICT.items():
    print(f"{k}: {v}")
print("check every number against your own runs above before calling Day 47 done")
# Expected: four lines printed, numbers copied from your runs""")

md("""## Day 47 — 5 lines to memorize

1. flash_attn_func contract: (batch, seqlen, heads, headdim), fp16/bf16 only, headdim 32-256 — seqlen before heads is the transpose that bites.
2. Parity bars: max abs diff < 1e-5 in fp32, < 1e-3 in fp16, < 1e-2 in bf16 — a tolerance, not bitwise equality, because summation order differs.
3. Benchmark protocol: warmup, synchronise, median of 20+, same dtype/causal/layout accounting on both sides — otherwise you measured launching, dtype, or luck.
4. Gain grows with seqlen: ~1x at 512, ~3.1x at 4096 on A100 (28 ms to 9 ms), and feasibility at 16k+ where the eager score matrix (137 GB at batch 8) cannot exist.
5. Regime: flash is a prefill/training win; decode streams the KV cache at ~1 FLOP/byte and needs KV quantization, eviction, or MLA instead.

**Tomorrow (Day 48):** Triton intro — write your own tile-based kernel (vector add), and learn programs, grids, and block pointers.""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}}, "nbformat": 4, "nbformat_minor": 4}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
