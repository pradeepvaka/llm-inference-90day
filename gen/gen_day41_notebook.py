#!/usr/bin/env python3
"""Generate notebooks/day41-mla-multi-head-latent-attention.ipynb."""
import json, pathlib

SLUG = "day41-mla-multi-head-latent-attention"
NB_URL = f"https://colab.research.google.com/github/pradeepvaka/llm-inference-90day/blob/master/notebooks/{SLUG}.ipynb"
OUT = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / f"{SLUG}.ipynb"

cells = []
def md(src):
    cells.append({"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.split("\n")]})
def code(src):
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in src.split("\n")]})

md(f"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({NB_URL})

# Day 41 — MLA: DeepSeek's Multi-Head Latent Attention

**Big idea:** cache one shared compressed latent per token per layer (512 elements for DeepSeek-V3) instead of per-head keys and values, and use absorption — folding the key up-projection into the query projection, and the value up-projection into the output projection — so attention scores are computed directly against the cached latents. Position lives in a separate decoupled RoPE key (64 elements), because a position-dependent rotation cannot be absorbed. CPU-only, no model download: you will (1) price V3's cache at 31,232 B/token latent-only in fp8, 35,136 B all-in, vs 124,928 B for a GQA equivalent, (2) build the MLA forward path and verify the identity sanity check, (3) verify absorption on the key and value sides, (4) re-verify RoPE's relative-position identity and show the absorption obstruction, (5) build the Llama / Mistral / V3 comparison table and the 40 GB budget inversion, and (6) write the five-line 'why MLA won' summary. ~30-60 min.""")

code("""# Cell 1: one-time installs (only cell that installs)
%pip install --quiet numpy
print("deps ok")
# Expected: deps ok""")

md("""## Step 1 — Price the cache from geometry

DeepSeek-V3 geometry: 61 layers, 128 heads, head_dim 128, latent d_c = 512, decoupled RoPE key dim 64. Per token per layer: MHA caches 2 x 128 x 128 elements, GQA-8 caches 2 x 8 x 128, MLA caches 512 (latent) or 576 (latent + RoPE key). Multiply by layers and dtype bytes.""")

code("""LAYERS = 61

def elements_per_layer(kind):
    if kind == "MHA":       return 2 * 128 * 128
    if kind == "GQA-8":     return 2 * 8 * 128
    if kind == "MLA latent": return 512
    if kind == "MLA all-in": return 512 + 64
    raise ValueError(kind)

for kind in ("MHA", "GQA-8", "MLA latent", "MLA all-in"):
    el = elements_per_layer(kind)
    per_token_fp8 = el * LAYERS * 1
    per_token_fp16 = el * LAYERS * 2
    at_128k = per_token_fp8 * 131072
    print(f"{kind:>11}: {el:>6} elem/layer | fp8 {per_token_fp8:>9,} B/token | fp16 {per_token_fp16:>9,} B/token | fp8 @131072 tokens = {at_128k/1e9:7.2f} GB")

assert elements_per_layer("MHA") == 32768
assert elements_per_layer("GQA-8") == 2048
assert elements_per_layer("MLA latent") * LAYERS == 31232
assert elements_per_layer("MLA all-in") * LAYERS == 35136
assert elements_per_layer("GQA-8") * LAYERS == 124928
assert elements_per_layer("MHA") * LAYERS == 1998848
print(f"MHA / MLA all-in = {32768/576:.1f}x ; GQA-8 / MLA latent = {2048/512:.1f}x ; GQA-8 / MLA all-in = {2048/576:.2f}x")
print("verified: 31,232 B latent-only, 35,136 B all-in, vs 124,928 B GQA and 1,998,848 B MHA (fp8, per token)")
# Expected: MHA 32,768 elem -> 1,998,848 B/token fp8, 262.0 GB @128k; GQA 16.37 GB; MLA all-in 4.61 GB""")

md("""## Step 2 — MLA forward path: the identity sanity check

MLA attention = down-project h to the latent c, cache c, up-project c to K and V, attend. Sanity case: if the down-projection is the identity and each up-projection equals the standard projection, MLA must reproduce standard attention exactly. Any difference is a bug in your code, not in the idea.""")

code("""import numpy as np

def attention(Q, K, V):
    d = Q.shape[-1]
    s = Q @ K.T / np.sqrt(d)
    mask = np.triu(np.ones_like(s, dtype=bool), k=1)
    s = np.where(mask, -1e30, s)
    s = s - s.max(axis=-1, keepdims=True)
    w = np.exp(s)
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ V, w

rng = np.random.default_rng(0)
n, d_model = 12, 16
X = rng.standard_normal((n, d_model))
Wq, Wk, Wv = [rng.standard_normal((d_model, d_model)) for _ in range(3)]

out_std, _ = attention(X @ Wq.T, X @ Wk.T, X @ Wv.T)

# MLA path: W_down = I (latent dim = d_model), W_upK = Wk, W_upV = Wv
W_down = np.eye(d_model)
C = X @ W_down.T                 # the cached latent, one vector per token
out_mla, _ = attention(X @ Wq.T, C @ Wk.T, C @ Wv.T)

diff = np.abs(out_std - out_mla).max()
print(f"standard vs MLA (identity bottleneck): max output diff = {diff:.2e}")
assert np.allclose(out_std, out_mla, atol=1e-10)
print("verified: MLA with W_down . W_up reproducing the standard projections IS standard attention")
# Expected: max diff 0.00e+00 (or < 1e-12)""")

md("""## Step 3 — Absorption: verify both identities

Key side: q . (W_upK c) == (W_upK^T q) . c — score the absorbed query directly against cached latents, never materialize K. Value side: weighted sum of (W_upV c_i) == W_upV applied to the weighted sum of c_i, so W_upV folds into the output projection. First the packet's hand example (answer must be exactly 7 both ways), then random matrices.""")

code("""# Hand example from the packet
c = np.array([3.0, -1.0, 2.0])
W_upK_hand = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
q_hand = np.array([1.0, 2.0])
K_hand = W_upK_hand @ c
score_mat = q_hand @ K_hand
q_abs_hand = W_upK_hand.T @ q_hand
score_abs = q_abs_hand @ c
print(f"K = {K_hand}, materialized score = {score_mat}, absorbed query = {q_abs_hand}, absorbed score = {score_abs}")
assert score_mat == 7.0 and score_abs == 7.0

# Random-matrix verification, key side and value side
rng = np.random.default_rng(2)
d_h, d_c, seq = 8, 16, 20
W_upK = rng.standard_normal((d_h, d_c))
W_upV = rng.standard_normal((d_h, d_c))
C = rng.standard_normal((seq, d_c))       # cached latents
q1 = rng.standard_normal(d_h)

scores_mat = (C @ W_upK.T) @ q1 / np.sqrt(d_h)
scores_abs = C @ (W_upK.T @ q1) / np.sqrt(d_h)
print(f"key-side absorption: max score diff = {np.abs(scores_mat - scores_abs).max():.2e}")
assert np.allclose(scores_mat, scores_abs, atol=1e-9)

alpha = rng.random(seq); alpha = alpha / alpha.sum()
out_v_mat = alpha @ (C @ W_upV.T)
out_v_abs = (alpha @ C) @ W_upV.T
print(f"value-side absorption: max output diff = {np.abs(out_v_mat - out_v_abs).max():.2e}")
assert np.allclose(out_v_mat, out_v_abs, atol=1e-9)

# Cost contrast (per layer): materialize keys for all cached tokens vs one absorbed query
macs_per_token_keys = 128 * 128 * 512
for ctx in (1024, 8192):
    print(f"context {ctx:>5}: materialize-all-keys = {macs_per_token_keys*ctx/1e9:6.2f} G MACs vs absorbed query once = {macs_per_token_keys/1e6:5.2f} M MACs (ratio {ctx}x)")
print("verified: absorption is exact, and its saving grows linearly with context length")
# Expected: hand scores 7.0 / 7.0; random diffs < 1e-12; ratio equals context length""")

md("""## Step 4 — Decoupled RoPE: the identity holds, and the obstruction is real

RoPE on the decoupled key was never compressed, so the Day 4 relative-position identity must hold exactly: dot(R_m q, R_n k) depends only on n - m. Then show why RoPE cannot live on the compressed path: the would-be absorbed matrix R_p x W_upK changes with the position p, so no single fixed query projection can absorb it.""")

code("""def rope(x, pos, base=10000.0):
    D = x.shape[-1]
    freqs = base ** (-np.arange(0, D, 2) / D)
    ang = pos * freqs
    cos, sin = np.cos(ang), np.sin(ang)
    xe, xo = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = xe * cos - xo * sin
    out[..., 1::2] = xe * sin + xo * cos
    return out

rng = np.random.default_rng(1)
D = 64  # V3's decoupled RoPE key dimension
q = rng.standard_normal(D); k = rng.standard_normal(D)
worst = 0.0
for m, n in [(0, 0), (3, 7), (10, 4), (100, 250)]:
    lhs = rope(q, m) @ rope(k, n)
    rhs = q @ rope(k, n - m)
    worst = max(worst, abs(lhs - rhs))
print(f"RoPE relative-position identity: worst diff over tested (m, n) = {worst:.2e}")
assert worst < 1e-9

def rot_matrix(pos, D, base=10000.0):
    R = np.zeros((D, D))
    for i in range(D):
        e = np.zeros(D); e[i] = 1.0
        R[:, i] = rope(e, pos, base)
    return R

d_h, d_c = 16, 32
W_upK = rng.standard_normal((d_h, d_c))
M0 = rot_matrix(0, d_h) @ W_upK   # absorbed matrix if RoPE were folded in, position 0
M1 = rot_matrix(1, d_h) @ W_upK   # same, position 1
obstruction = np.abs(M1 - M0).max()
print(f"obstruction: max |M(position 1) - M(position 0)| = {obstruction:.3f}  (must be clearly nonzero)")
assert obstruction > 0.1
print("verified: decoupled RoPE keeps exact relative positions; RoPE on the latent path cannot be absorbed")
# Expected: identity diff < 1e-12; obstruction > 0.1 (observed ~2.1 with this seed)""")

md("""## Step 5 — Comparison table + the 40 GB budget inversion

Llama-3.1-8B (GQA), Mistral-7B (GQA + sliding window 4,096), DeepSeek-V3 (MLA). Then invert: how many full-128k sequences fit in a 40 GB KV budget under the GQA equivalent vs under MLA?""")

code("""def per_token_bytes(layers, kv_elem_per_layer, dtype_bytes):
    return layers * kv_elem_per_layer * dtype_bytes

rows = [
    ("Llama-3.1-8B GQA",        32, 2 * 8 * 128),
    ("Mistral-7B GQA (per live token)", 32, 2 * 8 * 128),
    ("DeepSeek-V3 MLA all-in",  61, 512 + 64),
]
for name, layers, elem in rows:
    fp16 = per_token_bytes(layers, elem, 2)
    fp8 = per_token_bytes(layers, elem, 1)
    print(f"{name:>34}: fp16 {fp16:>7,} B/token | fp8 {fp8:>7,} B/token")

llama_fp16 = per_token_bytes(32, 2048, 2)
v3_fp16 = per_token_bytes(61, 576, 2)
assert llama_fp16 == 131072 and v3_fp16 == 70272
print(f"striking fact: V3 (671B) fp16 cache/token {v3_fp16:,} B < Llama-3.1-8B (8B) fp16 {llama_fp16:,} B")

mistral_capped_fp8 = 4096 * per_token_bytes(32, 2048, 1)
print(f"Mistral sliding-window cache per sequence, fp8: {mistral_capped_fp8:,} bytes = {mistral_capped_fp8/1024**2:.0f} MiB, fixed at any length")
assert mistral_capped_fp8 == 268_435_456

budget = 40e9
mla_seq = 35136 * 131072
gqa_seq = 124928 * 131072
print(f"40 GB budget at 128k context: MLA sequences = {int(budget // mla_seq)}, GQA-equivalent sequences = {int(budget // gqa_seq)}")
assert int(budget // mla_seq) == 8 and int(budget // gqa_seq) == 2
print("verified: ~4x the full-context concurrency at a fixed KV budget")
# Expected: Llama fp16 131,072; V3 fp16 70,272; Mistral capped 256 MiB fp8; budget 8 vs 2""")

md("""## Step 6 — Write the 'why MLA won' summary (fill this in before you finish)

Five lines, one fact each, in the next cell:

1. What is cached (one 512-dim latent + one shared 64-dim RoPE key per token per layer).
2. What absorption avoids (materializing per-head keys/values for every cached token, every decode step).
3. What decoupled RoPE protects (exact relative positions, which a position-dependent rotation cannot keep on the absorbed path).
4. The V3 number (35,136 bytes per token in fp8, all-in, across 61 layers).
5. The serving consequence (about 4x the 128k-context concurrency of the GQA equivalent at a fixed KV budget).

Checkpoint answers (from the packet): V3 MLA KV per token in fp8 is about 31 KB latent-only, 35,136 B all-in; absorbed = the K up-projection folded into the query projection (and V up-projection into the output projection); decoupled RoPE exists because RoPE's position-dependent rotation cannot be absorbed into one fixed projection.""")

code("""# Your five lines, as strings -- then run to count them.
lines = [
    "Cached: ",            # TODO: finish in your own words
    "Absorption avoids: ",
    "Decoupled RoPE protects: ",
    "V3 number: ",
    "Serving consequence: ",
]
assert len(lines) == 5
print("\\n".join(lines))
print("notebook complete -- Day 41 done")
# Expected: 5 lines printed; notebook complete""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} with {len(cells)} cells")
