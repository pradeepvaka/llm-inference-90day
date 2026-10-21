#!/usr/bin/env python3
"""Generate notebooks/day43-gpu-architecture-for-inference.ipynb."""
import json, pathlib

SLUG = "day43-gpu-architecture-for-inference"
NB_URL = f"https://colab.research.google.com/github/pradeepvaka/llm-inference-90day/blob/master/notebooks/{SLUG}.ipynb"
OUT = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / f"{SLUG}.ipynb"

cells = []
def md(src):
    cells.append({"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.split("\n")]})
def code(src):
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in src.split("\n")]})

md(f"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({NB_URL})

# Day 43 — GPU Architecture for Inference: SMs, Memory, Tensor Cores

**Big idea:** decode is a logistics problem. HBM is the warehouse (80 GB, 3.35 TB/s on H100, ~400-600 cycles), shared memory / SRAM is the workbench (228 KB per SM, ~20 cycles), registers are your hands. An 8B model streams ~16 GB from the warehouse per token and does ~1 FLOP/byte on it, so tokens/sec is decided by bytes per token, not by the 989 TFLOP/s Tensor Core headline. Today you will: (1) build the GPU spec table and compute ridge points, (2) ground one row with `nvidia-smi` if a T4 is attached, (3) reproduce the per-op decode budget for Llama-3.1-8B on H100 (~4.6-4.8 ms/token), (4) sweep the KV-read term to its crossover with the weight stream (~115k tokens), and (5) classify a mixed op list by binding constraint. CPU-OK throughout; ~30-60 min.""")

code("""# Cell 1: one-time installs (only cell that installs)
%pip install --quiet matplotlib
print("deps ok")
# Expected: deps ok""")

md("""## Step 1 — The spec table and ridge points

Ridge point = peak FLOP/s / bandwidth. Below the ridge an op is bandwidth-bound; above it, compute-bound. B200 figures are approximate headline numbers; memorize the T4 / A100 / H100 rows.""")

code("""GPUS = {
    # name:        SMs, SRAM KB/SM, HBM GB, BW TB/s, fp16 TFLOP/s, FP8 TFLOP/s (None = n/a)
    "T4":          (40,  64,         16,     0.32,    65,          None),
    "A100-40GB":   (108, 164,        40,     1.56,    312,         None),
    "A100-80GB":   (108, 164,        80,     2.04,    312,         None),
    "H100 SXM":    (132, 228,        80,     3.35,    989,         1979),
    "B200":        (148, 228,        192,    8.00,    2250,        4500),
}
print(f"{'GPU':>10} {'SMs':>4} {'SRAM/SM':>8} {'HBM GB':>7} {'TB/s':>5} {'fp16 TF/s':>10} {'ridge':>6} {'FP8 ridge':>10}")
for name, (sms, sram, hbm, bw, tf16, tf8) in GPUS.items():
    ridge = tf16 / bw
    ridge8 = f"{tf8 / bw:>10.0f}" if tf8 else f"{'--':>10}"
    print(f"{name:>10} {sms:>4} {sram:>6}KB {hbm:>7} {bw:>5.2f} {tf16:>10} {ridge:>6.0f} {ridge8}")
h100_ridge = GPUS["H100 SXM"][4] / GPUS["H100 SXM"][3]
assert abs(h100_ridge - 295) < 1
assert abs(GPUS["A100-80GB"][4] / GPUS["A100-80GB"][3] - 153) < 1
print(f"H100 total SRAM = 132 SMs x 228 KB = {132 * 228 / 1024:.1f} MB (vs 50 MB L2, vs 16.06 GB of 8B weights)")
# Expected: ridges T4 ~203, A100-40 ~200, A100-80 ~153, H100 ~295, B200 ~281; H100 FP8 ridge ~591""")

md("""## Step 2 — Ground one row (optional, GPU runtime only)

On a CPU runtime `nvidia-smi` is absent — the guard prints that and the lab continues. On a T4 runtime, compare the reported name/memory with the table's T4 row.""")

code("""import shutil, subprocess
if shutil.which("nvidia-smi"):
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv"], capture_output=True, text=True)
    print(out.stdout)
else:
    print("no nvidia-smi on this runtime (CPU) - table row stands; lab is CPU-OK")
# Expected: 'Tesla T4, 15360 MiB, ...' on a T4 runtime; the CPU message otherwise""")

md("""## Step 3 — Per-op decode budget: Llama-3.1-8B, batch 1, H100

Price every op as bytes / bandwidth. Geometry: 32 layers, hidden 4096, 8 KV heads x 128 head-dim (GQA), MLP intermediate 14336, vocab 128256, fp16 = 2 bytes/param. KV per token = 2 (K,V) x 32 layers x 8 heads x 128 dim x 2 B = 131,072 B.""")

code("""BW = 3.35e12  # H100 HBM3, bytes/s
LAYERS, HIDDEN, KV_HEADS, HEAD_DIM, MLP, VOCAB = 32, 4096, 8, 128, 14336, 128256

def ms(n_bytes):
    return n_bytes / BW * 1000

qkv_params = (HIDDEN * HIDDEN + 2 * HIDDEN * (KV_HEADS * HEAD_DIM)) * LAYERS
o_params = HIDDEN * HIDDEN * LAYERS
mlp_params = 3 * HIDDEN * MLP * LAYERS
head_params = VOCAB * HIDDEN
kv_per_token = 2 * LAYERS * KV_HEADS * HEAD_DIM * 2

rows = [
    ("QKV projections", qkv_params * 2),
    ("Attention KV read (seq 2048)", kv_per_token * 2048),
    ("O projection", o_params * 2),
    ("MLP gate+up+down", mlp_params * 2),
    ("LM head", head_params * 2),
]
total_ms = 0.0
for name, b in rows:
    total_ms += ms(b)
    print(f"{name:>30} {b/1e9:>6.2f} GB  {ms(b):>5.2f} ms")
weight_ms = total_ms - ms(kv_per_token * 2048)
print(f"{'TOTAL':>30} {'':>9} {total_ms:>5.2f} ms/token -> ceiling {1000/total_ms:.0f} tok/s")
print(f"MLP share of weight-stream time: {ms(mlp_params*2)/weight_ms*100:.0f}%")
assert 4.4 < total_ms < 4.7
assert abs(kv_per_token - 131072) < 1
# Expected: QKV 0.48 ms, KV read 0.08 ms, O 0.32 ms, MLP 3.37 ms, head 0.31 ms; total ~4.56 ms; MLP ~75% of weight-stream time""")

md("""## Step 4 — Sweep the KV term to its crossover

The weight stream is fixed per token; the KV read grows with context. Find where KV read time equals weight-stream time: n = weight_bytes / kv_per_token.""")

code("""weight_bytes = (qkv_params + o_params + mlp_params + head_params) * 2
for seq in (512, 2048, 8192, 32768, 131072):
    print(f"seq {seq:>7,}: KV read {kv_per_token * seq / 1e9:>6.2f} GB -> {ms(kv_per_token * seq):>5.2f} ms")
crossover = weight_bytes // kv_per_token
print(f"crossover: KV read == weight stream at ~{crossover:,} tokens of context")
assert 110_000 < crossover < 125_000
# Expected: 0.02 / 0.08 / 0.32 / 1.28 / 5.14 ms; crossover ~114,500 tokens""")

md("""## Step 5 — Classify a mixed op list

Intensity = FLOP / byte streamed. Compare with the H100 ridge (295). Only prefill clears it.""")

code("""OPS = [
    # op, intensity FLOP/byte, note
    ("Prefill GEMM, 2000-token prompt", 2000, "tensor cores at high efficiency"),
    ("Decode GEMM, batch 1", 1, "streams all weights per token"),
    ("Decode GEMM, batch 32", 32, "same weight read, 32 tokens of work"),
    ("Decode attention, 32k context", 2, "KV stream dominates, ~1.28 ms"),
    ("RMSNorm (4096-vector)", 1, "tiny tensor: launch/latency-bound, not BW in practice"),
]
for name, inten, note in OPS:
    bound = "COMPUTE-bound" if inten > h100_ridge else "bandwidth/latency-bound"
    print(f"{name:>34} intensity {inten:>5} FLOP/B vs ridge 295 -> {bound} ({note})")
print("rule: below ridge, the fix is fewer bytes or more tokens per byte - never faster math")
# Expected: only the prefill GEMM classifies as compute-bound""")

md("""## Step 6 — Plot: where the 4.8 ms goes

A bar chart of the Step 3 budget. The MLP bar should visually own the chart — that is the fusion / quantization target list for Days 51-53 in one picture.""")

code("""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = [r[0].split(" (")[0] for r in rows]
vals = [ms(r[1]) for r in rows]
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.barh(names[::-1], vals[::-1])
ax.set_xlabel("ms per decode token (H100, bytes / 3.35 TB/s)")
ax.set_title("Llama-3.1-8B decode budget, batch 1")
for i, v in enumerate(vals[::-1]):
    ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("day43-decode-budget.png", dpi=110)
print("saved day43-decode-budget.png; MLP bar should be ~7x the next bar")
# Expected: MLP ~3.37 ms bar dominates; all others <= 0.48 ms""")

md("""## The 5 bullets to memorize (fill in from your own numbers above)

1. HBM bandwidth decides decode: H100 = 3.35 TB/s; 8B fp16 = ~16 GB/token -> ~4.8 ms/token ceiling at batch 1.
2. SRAM is the workbench: 228 KB per SM on H100, ~20 cycles, vs ~400-600 cycles to HBM — tile into it (Day 47, FlashAttention).
3. L2 is 50 MB: 0.3% of an 8B model's weights — nothing about decode hides in cache.
4. Ridge point on H100 = ~295 FLOP/byte; decode runs at ~1, prefill at ~seq_len. The phase decides the bottleneck.
5. Tensor Cores do dense MMA only: norms, softmax, RoPE, and sampling are CUDA-core, launch/latency-bound work — fuse them, do not tune them.

**Tomorrow (Day 44):** CUDA basics — threads, blocks, warps, coalescing, and occupancy math on paper.""")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}}, "nbformat": 4, "nbformat_minor": 4}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
