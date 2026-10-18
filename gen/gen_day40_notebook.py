#!/usr/bin/env python3
"""Generate notebooks/day40-sliding-window-attention-attention-sinks.ipynb."""
import json, pathlib

SLUG = "day40-sliding-window-attention-attention-sinks"
NB_URL = f"https://colab.research.google.com/github/pradeepvaka/llm-inference-90day/blob/master/notebooks/{SLUG}.ipynb"
OUT = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / f"{SLUG}.ipynb"

cells = []
def md(src):
    cells.append({"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.split("\n")]})
def code(src):
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in src.split("\n")]})

md(f"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({NB_URL})

# Day 40 — Sliding Window Attention + Attention Sinks (Mistral)

**Big idea:** the window is trained into the architecture. Each layer attends only to the last W tokens, the KV cache is a circular buffer of W slots (position p -> slot p mod W), and stacked layers give an effective reach of about W x layers. CPU-only, no model download: you will (1) price Mistral-7B's 512 MiB rolling cache vs a 16 GiB full cache, (2) build the SWA band mask and verify it matches full attention at length <= W, (3) implement the circular KV buffer, (4) hit the exact receptive-field boundary (W=16, L=4: reach 61), (5) tabulate the 32x FLOPs saving, and (6) write the trained-vs-evicted contrast. ~30-60 min.""")

code("""# Cell 1: one-time installs (only cell that installs)
%pip install --quiet numpy matplotlib
print("deps ok")
# Expected: deps ok""")

md("""## Step 1 — Memory math: rolling cache vs full cache

Mistral-7B geometry: 32 layers, 8 KV heads (GQA), head_dim 128, fp16.
Bytes/token = 2 (K,V) x layers x kv_heads x head_dim x dtype_bytes = 131,072 B = 128 KiB.
Rolling cache replaces sequence length with W = 4,096.""")

code("""def kv_bytes_per_token(layers=32, kv_heads=8, head_dim=128, dtype_bytes=2):
    return 2 * layers * kv_heads * head_dim * dtype_bytes

per_token = kv_bytes_per_token()
W = 4096
print(f"bytes/token = {per_token} = {per_token/1024:.0f} KiB")
rolling = W * per_token
print(f"rolling cache @ W={W}: {rolling} bytes = {rolling/1024**2:.0f} MiB")
for seq in (4096, 32768, 131072):
    full = seq * per_token
    print(f"full cache @ {seq:>6} tokens: {full/1024**3:6.2f} GiB")
full_128k = 131072 * per_token
print(f"ratio at 131072 tokens = {full_128k/rolling:.0f}x")
assert per_token == 131072
assert rolling == 536_870_912
assert full_128k == 17_179_869_184
print("verified: 512 MiB fixed vs 16 GiB full at 128k")
# Expected: 128 KiB/token; rolling 512 MiB; full @131072 = 16.00 GiB; ratio 32x""")

md("""## Step 2 — SWA band mask: identical to full attention at length <= W

Causal mask zeroes the future; the band additionally zeroes anything more than W-1 steps back. For sequences no longer than W the band removes nothing, so outputs must match full attention exactly (to float precision).""")

code("""import numpy as np

def attention(Q, K, V, mask):
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    scores = np.where(mask, scores, -1e30)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ V, w

def causal_mask(n):
    return np.tril(np.ones((n, n), dtype=bool))

def swa_mask(n, W):
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return (j <= i) & (j > i - W)

rng = np.random.default_rng(0)
for n, W in ((8, 8), (12, 5)):
    X = rng.standard_normal((n, 4))
    out_full, _ = attention(X, X, X, causal_mask(n))
    out_swa, w_swa = attention(X, X, X, swa_mask(n, W))
    counts = swa_mask(n, W).sum(axis=-1)
    print(f"n={n} W={W}: max positions attended per row = {counts.max()}, match_full = {np.allclose(out_full, out_swa, atol=1e-6) if n <= W else 'n/a (differs by design)'}")
    if n <= W:
        assert np.allclose(out_full, out_swa, atol=1e-6)
    else:
        assert counts.max() == W
        assert not np.allclose(out_full, out_swa, atol=1e-6)
print("verified: SWA == full attention when n <= W; band caps each row at W positions")
# Expected: n=8 W=8 match_full True; n=12 W=5 max 5 positions/row, outputs differ""")

md("""## Step 3 — Circular KV buffer: position p overwrites slot p mod W

Append-only ring buffer standing in for the per-layer cache. Size must stay exactly W once full; the oldest live position advances as new tokens overwrite old slots.""")

code("""class CircularKV:
    def __init__(self, W):
        self.W = W
        self.slots = [None] * W  # slot -> position stored there
        self.n_appended = 0

    def append(self, position):
        self.slots[position % self.W] = position
        self.n_appended += 1

    def live_positions(self):
        return sorted(p for p in self.slots if p is not None)

buf = CircularKV(W=256)
for p in range(1000):
    buf.append(p)
live = buf.live_positions()
print(f"appended 0..999 into W=256 buffer: stored entries = {len(live)}")
print(f"oldest live position = {live[0]}, newest = {live[-1]}")
print(f"slot for position 1000 = {1000 % 256}")
assert len(live) == 256
assert live[0] == 744 and live[-1] == 999
assert 1000 % 256 == 232
print("verified: size fixed at W; positions 0..743 overwritten; slot = p mod W")
# Expected: 256 entries; oldest live 744; newest 999; slot(1000) = 232""")

md("""## Step 4 — Receptive-field boundary: W=16, L=4, exact reach 61

Propagate boolean reachability layer by layer: after each SWA layer, a position can be influenced by anything within W-1 further back. Exact reach = L x (W-1) + 1 = 61. A key 60 back is reachable; 70 back is not.""")

code("""def reachable_distance(W, L):
    reach = 1  # a position reaches itself
    for _ in range(L):
        reach = reach + (W - 1)
    return reach

W, L = 16, 4
exact = reachable_distance(W, L)
shorthand = W * L
print(f"W={W} L={L}: exact reach = {exact}, shorthand W x L = {shorthand}")
for dist in (60, 70):
    print(f"key {dist} tokens back: reachable = {dist + 1 <= exact}")
assert exact == 61 and shorthand == 64
assert (60 + 1 <= exact) and not (70 + 1 <= exact)
mistral_exact = 32 * (4096 - 1) + 1
print(f"Mistral exact reach = {mistral_exact} (shorthand {32*4096})")
assert mistral_exact == 131041
print("verified: boundary at 61 -- 60 back reachable, 70 back has no information path")
# Expected: exact 61 vs shorthand 64; Mistral exact 131041 vs shorthand 131072""")

md("""## Step 5 — FLOPs table: fixed vs growing

Attention FLOPs per generated token (curriculum compact form): 4 x layers x tokens_attended x head_dim. SWA caps tokens_attended at W; full attention lets it grow with context.""")

code("""def attn_flops(tokens_attended, layers=32, head_dim=128):
    return 4 * layers * tokens_attended * head_dim

swa = attn_flops(4096)
print(f"SWA (W=4096), any context length: {swa/1e6:.1f} MFLOP/token")
for ctx in (4096, 32768, 131072):
    f = attn_flops(ctx)
    print(f"full @ context {ctx:>6}: {f/1e6:8.1f} MFLOP/token  (full/SWA = {f/swa:.0f}x)")
assert swa == 67_108_864
assert attn_flops(131072) == 2_147_483_648
print("verified: SWA flat at ~67 MFLOP; full at 128k is 32x more and still growing")
# Expected: SWA 67.1 MFLOP; full 67.1 / 536.9 / 2147.5 MFLOP""")

md("""## Step 6 — Write the contrast (fill this in before you finish)

In your own words, three bullets as markdown in the next cell:

1. What SWA guarantees (fixed cache, trained use of the window and stacked reach).
2. What Day 39 eviction guarantees (a budget you choose, on a model trained for full attention).
3. One failing task for each: SWA fails on precise recall beyond its stacked reach; eviction fails on mid-context recall in its dropped middle, even at short total lengths.

Checkpoint answers (from the packet): effective field about 131,072 tokens (exact 131,041); rolling KV 512 MiB fixed vs 16 GiB full at 128k; core difference = trained-in architecture vs bolted-on policy.""")

code("""# Your three bullets, as strings -- then run to count them.
bullets = [
    "SWA guarantees: ",   # TODO: finish in your own words
    "Eviction guarantees: ",
    "Failing tasks: ",
]
assert len(bullets) == 3
print("\\n".join(bullets))
print("notebook complete -- Day 40 done")
# Expected: 3 bullets printed; notebook complete""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} with {len(cells)} cells")
