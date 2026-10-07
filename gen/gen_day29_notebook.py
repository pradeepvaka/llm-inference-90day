#!/usr/bin/env python3
"""Generate notebooks/day29-quantization-theory.ipynb."""
import json

NB = ("https://colab.research.google.com/github/pradeepvaka/llm-inference-90day/"
      "blob/master/notebooks/day29-quantization-theory.ipynb")

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": [l + "\n" for l in src.split("\n")]})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [l + "\n" for l in src.split("\n")]})

md("""# Day 29 — Quantization Theory: Affine, Symmetric, Granularity

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](%s)

**Big idea:** quantization is lossy compression with a ledger — one scale (and maybe a zero-point)
per bucket that tells you how to turn small ints back into approximate floats. Today you hand-quantize
tensors, compute SQNR, and learn why one outlier destroys per-tensor quantization while per-group survives.

CPU-OK (~40 min). The optional Step 6 loads distilgpt2 (82M params, ~330 MB) — needs ~2 GB RAM.""" % NB)

code("""# Colab: install deps (only cell that installs)
%pip install --quiet matplotlib numpy
print('deps ok')
# Expected: deps ok""")

md("""## Step 1 — The primitives: quantize_affine, quantize_symmetric, dequantize

Formulas: `s = (max-min)/(qmax-qmin)`, `z = round(-min/s) + qmin`, `q = clamp(round(x/s)+z)`,
`x_hat = s*(q-z)`. Symmetric: `s = max|x|/127`, `q = clamp(round(x/s), -128, 127)`.""")

code("""import numpy as np

def quantize_affine(x, qmin=0, qmax=255):
    x = np.asarray(x, dtype=np.float64)
    xmin, xmax = x.min(), x.max()
    s = (xmax - xmin) / (qmax - qmin)
    z = int(round(-xmin / s)) + qmin
    q = np.clip(np.round(x / s) + z, qmin, qmax).astype(np.int64)
    return q, s, z

def quantize_symmetric(x, bits=8):
    x = np.asarray(x, dtype=np.float64)
    amax = np.abs(x).max()
    s = amax / (2 ** (bits - 1) - 1)
    q = np.clip(np.round(x / s), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1).astype(np.int64)
    return q, s, 0

def dequantize(q, s, z):
    return s * (q.astype(np.float64) - z)

def sqnr(x, x_hat):
    x = np.asarray(x, dtype=np.float64)
    return 10 * np.log10(np.mean(x ** 2) / np.mean((x - x_hat) ** 2))

W = np.array([0.1, -2.5, 1.2, 0.05])
q, s, z = quantize_symmetric(W)
x_hat = dequantize(q, s, z)
print(f"s={s:.5f} z={z}")
print("q     =", q.tolist())
print("x_hat =", [f"{v:.4f}" for v in x_hat])
print(f"max abs err={np.abs(W - x_hat).max():.4f}  SQNR={sqnr(W, x_hat):.1f} dB")
# Expected: s=0.01969 z=0, q=[5, -127, 61, 3], x_hat=[0.0984, -2.5000, 1.2008, 0.0591]
# Expected: max abs err=0.0091  SQNR=49.5 dB""")

md("""## Step 2 — Affine on the same tensor: compare the errors

Same four weights, unsigned int8 range 0..255. Expect edges to quantize exactly and max error
~0.0065 — smaller than symmetric, because affine spends its codes on the actual [-2.5, 1.2] span.""")

code("""q, s, z = quantize_affine(W)
x_hat = dequantize(q, s, z)
print(f"s={s:.5f} z={z}")
print("q     =", q.tolist())
print("x_hat =", [f"{v:.4f}" for v in x_hat])
print(f"max abs err={np.abs(W - x_hat).max():.4f}  SQNR={sqnr(W, x_hat):.1f} dB")
# Expected: s=0.01451 z=172, q=[179, 0, 255, 175]
# Expected: max abs err=0.0065, SQNR a bit above 49.5 dB""")

md("""## Step 3 — One outlier wrecks everything (per-tensor)

Add 25.0 to the tensor. Watch the small weights collapse to 1-2 integer codes.""")

code("""W_out = np.array([0.1, -2.5, 1.2, 0.05, 25.0])
q, s, z = quantize_symmetric(W_out)
x_hat = dequantize(q, s, z)
print(f"s={s:.5f}")
for v, qi, h in zip(W_out, q, x_hat):
    rel = abs(v - h) / abs(v) if v != 0 else 0.0
    print(f"x={v:6.2f}  q={qi:5d}  x_hat={h:7.4f}  rel_err={rel:5.1%}")
# Expected: s=0.19685; 0.10 -> q=1 -> 0.1969 (97.0% err); 0.05 -> q=0 -> 0.0000 (100% err)""")

md("""## Step 4 — Granularity showdown: per-tensor vs per-channel vs per-group

Random 4096x4096 fp16 matrix (normal init, like a weight matrix), 1% of entries injected with
20x-magnitude outliers. Quantize three ways (int8 symmetric), report max abs error and SQNR.""")

code("""import time
rng = np.random.default_rng(29)
M = rng.standard_normal((4096, 4096)).astype(np.float32)
n_out = int(M.size * 0.01)
idx = rng.choice(M.size, n_out, replace=False)
OUT_IDX = idx  # shared with Step 4b
Mf = M.flatten()
Mf[idx] = np.sign(Mf[idx]) * 20.0 * np.abs(Mf[idx])
M = Mf.reshape(M.shape)

def q_per_tensor(x):
    s = np.abs(x).max() / 127.0
    return np.clip(np.round(x / s), -128, 127).astype(np.int16) * s

def q_per_channel(x):  # one scale per output channel (rows)
    s = np.abs(x).max(axis=1, keepdims=True) / 127.0
    s = np.maximum(s, 1e-12)
    return np.clip(np.round(x / s), -128, 127).astype(np.int16) * s

def q_per_group(x, g=128):  # groups along input dim (columns)
    xr = x.reshape(x.shape[0], -1, g)
    s = np.abs(xr).max(axis=2, keepdims=True) / 127.0
    s = np.maximum(s, 1e-12)
    xh = np.clip(np.round(xr / s), -128, 127).astype(np.int16) * s
    return xh.reshape(x.shape)

t0 = time.time()
for name, fn in [("per-tensor ", q_per_tensor), ("per-channel", q_per_channel), ("per-group-128", q_per_group)]:
    xh = fn(M.astype(np.float64))
    mae = np.abs(M - xh).max()
    print(f"{name}: max abs err={mae:8.4f}  SQNR={sqnr(M, xh):6.1f} dB")
print(f"elapsed {time.time()-t0:.1f}s")
# Expected (non-outlier weights): per-group cuts mean|err| ~5x vs per-tensor and the
# fraction of weights with |err|>0.1 from ~68% to ~5.5% (~12x); SQNR +12 dB.
# Max abs err is dominated by the outliers themselves, so it barely moves.""")

md("""## Step 4b — Where the win actually lands (non-outlier weights)

Max error is dominated by the outliers themselves (no granularity fixes rounding the outlier).
The per-group win lands on everything else: mean absolute error and the share of weights
badly damaged. Compare on non-outlier entries only.""")

code("""is_out = np.zeros(M.size, dtype=bool)
is_out[OUT_IDX] = True  # same injection mask as Step 4
mask = ~is_out.reshape(M.shape)
for name, fn in [("per-tensor ", q_per_tensor), ("per-channel", q_per_channel), ("per-group-128", q_per_group)]:
    e = np.abs(M[mask] - fn(M.astype(np.float64))[mask])
    print(f"{name}: mean|err|={e.mean():.5f}  frac|err|>0.1={np.mean(e > 0.1):.4f}")
# Expected: per-tensor mean|err|~0.157, frac~0.68; per-group-128 mean|err|~0.030, frac~0.055""")

md("""## Step 5 — Error histograms: see the long tail die

Log-scale histogram of |error| per granularity. Per-tensor shows a long tail from the crushed
weights; per-group-128 nearly erases it.""")

code("""import matplotlib.pyplot as plt

errs = {}
for name, fn in [("per-tensor", q_per_tensor), ("per-channel", q_per_channel), ("per-group-128", q_per_group)]:
    errs[name] = np.abs(M - fn(M.astype(np.float64))).flatten()

plt.figure(figsize=(8, 4))
for name, e in errs.items():
    plt.hist(e[e > 0], bins=100, log=True, alpha=0.5, label=name)
plt.xscale("log"); plt.xlabel("|quantization error|"); plt.ylabel("count (log)")
plt.legend(); plt.title("Error distribution by granularity (outlier-injected matrix)")
plt.tight_layout(); plt.show()
# Expected: per-tensor has a fat right tail; per-group-128's tail is ~10x shorter""")

md("""## Step 6 — Metadata ledger: what the scales cost (optional: real weights)

First the 8B-model arithmetic, then an optional pass over a real distilgpt2 matrix.""")

code("""B = 1e9
for name, groups in [("per-tensor ", 1), ("per-channel", B / 4096), ("per-group-128", B / 128)]:
    bytes_ = groups * 2  # fp16 scales
    print(f"{name}: {groups:12,.0f} scales  {bytes_/1e6:8.1f} MB  ({bytes_/(8e9)*100:.2f}% of int8 weights)")
# Expected: per-tensor 1 scale, 0.0 MB; per-channel ~244k, 0.5 MB; per-group-128 62.5M, 125.0 MB""")

code("""# Optional: quantize one real attention matrix from distilgpt2 (~330 MB download)
try:
    import torch
    from transformers import AutoModelForCausalLM
    tok_w = AutoModelForCausalLM.from_pretrained("distilgpt2", dtype=torch.float32).state_dict()
    Wreal = tok_w["transformer.h.0.attn.c_attn.weight"].numpy()  # 768x2304
    print(f"matrix {Wreal.shape}, {(Wreal.size*4/1e6):.1f} MB fp32")
    for name, fn in [("per-tensor", q_per_tensor), ("per-channel", q_per_channel), ("per-group-128", q_per_group)]:
        xh = fn(Wreal.astype(np.float64))
        print(f"{name}: max abs err={np.abs(Wreal-xh).max():.4f}  SQNR={sqnr(Wreal, xh):.1f} dB")
    scales = (Wreal.size // 128) * 2
    print(f"per-group-128 ledger for this matrix: {scales/1024:.1f} KB")
except Exception as e:
    print("skipped (needs transformers/torch):", e)
# Expected (real weights): per-group-128 best SQNR; ledger a few hundred KB""")

md("""## What to measure — fill in your numbers

| Metric | Your number |
|---|---|
| Symmetric int8 SQNR (4-weight worked tensor) | |
| Per-tensor max abs error after injecting 25.0 outlier | |
| Per-group-128 SQNR gain vs per-tensor (outlier matrix) | |
| int8 weight bytes for a 125M-param model | |
| Per-group-128 scale ledger for that model (fp16) | |

## Checkpoints — answers in the packet

1. **Symmetric int8 quant of [0.1, -2.5, 1.2, 0.05]: scale?** s = 2.5/127 ~ 0.0197.
2. **Why does one outlier wreck per-tensor quant?** It stretches the scale; small values collapse
   to 1-2 distinct integer codes.
3. **Per-group-128 scale overhead on 8B?** ~125 MB fp16 (~0.8% of fp16 weights, ~1.6% of int8).

**Tomorrow (Day 30):** GPTQ — second-order weight quantization. Quantize one column at a time and use
Hessian information to correct the error in the remaining columns before you touch them.""")


path = "/home/hatch/workspace/llm-inference-90day/notebooks/day29-quantization-theory.ipynb"
with open(path, "w") as f:
    json.dump({"nbformat": 4, "nbformat_minor": 5,
               "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                            "accelerator": "GPU"},
               "cells": cells}, f, indent=1, ensure_ascii=False)
print(f"wrote {path}: {len(cells)} cells")
