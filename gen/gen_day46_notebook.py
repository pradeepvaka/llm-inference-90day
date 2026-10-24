#!/usr/bin/env python3
"""Generate notebooks/day46-flashattention-io-awareness-tiling-online-softmax.ipynb."""
import json, pathlib

SLUG = "day46-flashattention-io-awareness-tiling-online-softmax"
NB_URL = f"https://colab.research.google.com/github/pradeepvaka/llm-inference-90day/blob/master/notebooks/{SLUG}.ipynb"
OUT = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / f"{SLUG}.ipynb"

cells = []
def md(src):
    cells.append({"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.split("\n")]})
def code(src):
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in src.split("\n")]})

md(f"""[![Open In Colab]({ "https://colab.research.google.com/assets/colab-badge.svg" })]({NB_URL})

# Day 46 — FlashAttention: IO-Awareness, Tiling, Online Softmax

**Big idea:** Naive attention materializes the N x N score matrix in HBM: at n=4096, 32 heads, fp16 that is 1.07 GB written, read, written, and read again — about 4.43 GB total traffic per layer versus about 275 GFLOP of math (0.28 ms at peak vs 1.32 ms of traffic on H100). FlashAttention tiles Q, K, V into SRAM-sized blocks, never materializes the N x N matrix, and uses online softmax (running max m, normalizer l, partial output, rescaled when m rises) to stay exact in one pass. Model traffic at 4k: about 0.48 GB, roughly 9x less; memory goes from O(N^2) to O(N). Today you will: (1) implement online softmax in NumPy and verify it exactly, (2) reproduce the rescaling numbers 1.1353 -> 1.2034, (3) build the naive-vs-Flash traffic calculator, (4) build a complete blocked attention that never allocates the full score matrix, and (5) do the roofline and tile-sizing checks. CPU only. About 30-60 minutes.""")

code("""# Cell 1: one-time installs (only cell that installs)
%pip install --quiet numpy matplotlib
print("deps ok")
# Expected: deps ok""")

md("""## Step 1 — Online softmax in NumPy: one row, four blocks, exact

Process one row of 12 scores in 4 blocks of 3. Maintain `(m, l, acc)`. When a block raises the max, rescale the old `l` and `acc` by `exp(m_old - m_new)` before merging. Divide by `l` only at the end.""")

code("""import numpy as np

rng = np.random.default_rng(0)
scores = rng.standard_normal(12) * 2.0          # one attention row
values = rng.standard_normal((12, 8))           # V rows for this query row

def direct(scores, values):
    m = scores.max()
    e = np.exp(scores - m)
    return (e / e.sum()) @ values, e.sum(), m

def online(scores, values, block=3):
    m, l, acc = -np.inf, 0.0, np.zeros(values.shape[1])
    for s in range(0, len(scores), block):
        blk = scores[s:s + block]
        m_block = blk.max()
        m_new = max(m, m_block)
        rescale = np.exp(m - m_new) if m > -np.inf else 0.0
        e_block = np.exp(blk - m_new)
        l = l * rescale + e_block.sum()
        acc = acc * rescale + e_block @ values[s:s + block]
        m = m_new
    return acc / l, l, m

y_ref, _, _ = direct(scores, values)
y_on, l_on, m_on = online(scores, values, block=3)
err = np.max(np.abs(y_ref - y_on))
print(f"running max m = {m_on:.4f} (direct max = {scores.max():.4f})")
print(f"normalizer l = {l_on:.6f}")
print(f"max |direct - online| = {err:.2e}")
assert err < 1e-5
print("online softmax verified: blocked one-pass == direct two-pass")
# Expected: max error < 1e-5 (typically ~1e-16 in float64)""")

md("""## Step 2 — The rescaling example from the packet, in numbers

Scores [1, 3] then [5, 2]. After block A: m=3, l=1.1353. Block B raises m to 5, so the old l is rescaled by exp(3-5)=0.1353. Merged l must be 1.2034 — and must match the direct sum.""")

code("""import numpy as np

a = np.array([1.0, 3.0])
b = np.array([5.0, 2.0])
m_old = a.max()
l_old = np.exp(a - m_old).sum()
m_block = b.max()
l_block = np.exp(b - m_block).sum()
m_new = max(m_old, m_block)
rescale = np.exp(m_old - m_new)
l_new = l_old * rescale + l_block * np.exp(m_block - m_new)
direct_l = np.exp(np.concatenate([a, b]) - m_new).sum()
print(f"after block A: m = {m_old:.0f}, l = {l_old:.4f}")
print(f"rescale factor exp({m_old:.0f} - {m_new:.0f}) = {rescale:.4f}")
print(f"merged l = {l_new:.4f} | direct l = {direct_l:.4f}")
assert abs(l_old - 1.1353) < 1e-3 and abs(rescale - 0.1353) < 1e-3
assert abs(l_new - 1.2034) < 1e-3 and abs(l_new - direct_l) < 1e-9
print("rescaling verified: 1.1353 --x0.1353--> 1.2034, exact vs direct")
# Expected: l_old 1.1353, rescale 0.1353, l_new 1.2034""")

md("""## Step 3 — Traffic calculator: naive vs FlashAttention model

Naive: base traffic (read Q, K, V; write O) + 4 passes over the N x N scores.
Flash model (paper): base traffic + heads * N^2 * d^2 / M elements, M = SRAM in fp16 elements (100 KiB = 102,400 bytes = 51,200 elements).
This is a model, not a measurement — Day 47 measures the real kernel.""")

code("""def traffic(n, d=128, heads=32, M_sram_bytes=102_400, bytes_per=2):
    base = 4 * n * d * heads * bytes_per                    # Q, K, V reads + O write
    naive_scores = 4 * heads * n * n * bytes_per            # write S, read S, write P, read P
    M_elems = M_sram_bytes // bytes_per
    flash_scores = heads * n * n * d * d // M_elems * bytes_per
    naive, flash = base + naive_scores, base + flash_scores
    return naive, flash, naive / flash

print(f"{'n':>7} {'naive GB':>10} {'flash GB':>10} {'ratio':>7}")
for n in (1024, 4096, 16384, 65536):
    naive, flash, ratio = traffic(n)
    print(f"{n:>7} {naive/1e9:>10.2f} {flash/1e9:>10.2f} {ratio:>6.1f}x")
n4096 = traffic(4096)
assert 9.0 < n4096[2] < 9.6
print("checkpoint: n=4096 -> ~4.43 GB vs ~0.48 GB, about 9.3x less HBM traffic")
print("ratio rises toward the model ceiling (~12.5x) as the N^2 term dominates base traffic")
# Expected (printed, 2 dp): 1024: 0.30 vs 0.06 (5.5x) | 4096: 4.43 vs 0.48 (9.3x) | 16384: 69.26 vs 6.03 (11.5x) | 65536: 1101.66 vs 90.11 (12.2x)""")

md("""## Step 4 — Blocked attention: never allocate the full N x N matrix

Loop over K/V blocks (outer) conceptually, here per query row block with online state per row. The largest score block allocated is `(Br, Bc)`, never `(n, n)`. Compare against the naive reference that does materialize it.""")

code("""import numpy as np

rng = np.random.default_rng(1)
n, d = 64, 32
Q = rng.standard_normal((n, d))
K = rng.standard_normal((n, d))
V = rng.standard_normal((n, d))
scale = 1.0 / np.sqrt(d)

def naive_attention(Q, K, V):
    S = Q @ K.T * scale                     # (n, n) materialized -- the thing Flash avoids
    S = S - S.max(axis=1, keepdims=True)
    P = np.exp(S)
    P = P / P.sum(axis=1, keepdims=True)
    return P @ V

def blocked_attention(Q, K, V, Bc=16):
    n, d = Q.shape
    m = np.full(n, -np.inf)
    l = np.zeros(n)
    acc = np.zeros((n, d))
    largest_block = 0
    for s in range(0, n, Bc):
        S_blk = Q @ K[s:s + Bc].T * scale   # (n, Bc) only
        largest_block = max(largest_block, S_blk.size)
        m_block = S_blk.max(axis=1)
        m_new = np.maximum(m, m_block)
        rescale = np.exp(m - m_new)
        rescale = np.where(np.isfinite(rescale), rescale, 0.0)
        e_blk = np.exp(S_blk - m_new[:, None])
        l = l * rescale + e_blk.sum(axis=1)
        acc = acc * rescale[:, None] + e_blk @ V[s:s + Bc]
        m = m_new
    return acc / l[:, None], largest_block

y_ref = naive_attention(Q, K, V)
y_blk, largest = blocked_attention(Q, K, V, Bc=16)
err = np.max(np.abs(y_ref - y_blk))
print(f"n = {n}, d = {d}: full score matrix would be {n * n} elements")
print(f"largest score block actually allocated: {largest} elements ({n} x 16)")
print(f"max |naive - blocked| = {err:.2e}")
assert err < 1e-5 and largest == n * 16
print("blocked attention verified: exact, and the N x N matrix was never allocated")
# Expected: largest block 1024 elements (not 4096); max error < 1e-5""")

md("""## Step 5 — Roofline check at n=4096: the bottleneck flips

FLOPs = 4 * n^2 * d * heads. Traffic time = bytes / bandwidth. Compute time = FLOPs / peak rate. H100: 3.35 TB/s, 989 TFLOP/s fp16.""")

code("""n, d, heads = 4096, 128, 32
flops = 4 * n * n * d * heads
naive_b, flash_b, _ = traffic(n, d, heads)
bw, peak = 3.35e12, 989e12
print(f"FLOPs per layer: {flops/1e9:.1f} GFLOP")
print(f"compute time at peak: {flops/peak*1000:.2f} ms")
print(f"naive traffic time:   {naive_b/bw*1000:.2f} ms  ({naive_b/1e9:.2f} GB)")
print(f"flash traffic time:   {flash_b/bw*1000:.2f} ms  ({flash_b/1e9:.2f} GB)")
print("verdict: naive is memory-bound (1.32 ms traffic >> 0.28 ms math);")
print("after tiling, traffic (0.14 ms) < math (0.28 ms) -> balanced, so wall-clock gain is 2-4x, not 9x")
assert abs(flops/1e9 - 274.9) < 0.5
# Expected: 274.9 GFLOP | 0.28 ms math | 1.32 ms naive traffic | 0.14 ms flash traffic""")

md("""## Step 6 — Tile sizing: what fits in SRAM?

Blocks held: Q block (Br x d, fp16), K block (Bc x d, fp16), V block (Bc x d, fp16), score block (Br x Bc, fp32), output accumulator (Br x d, fp32). Try Br in {32, 64, 128} with Bc=128, d=128.""")

code("""def tile_kb(Br, Bc=128, d=128):
    q = Br * d * 2
    k = Bc * d * 2
    v = Bc * d * 2
    s = Br * Bc * 4
    o = Br * d * 4
    return (q + k + v + s + o) // 1024

for Br in (32, 64, 128):
    kb = tile_kb(Br)
    fits = "fits A100 (164 KB)" if kb <= 164 else "does NOT fit A100 (164 KB)"
    print(f"Br = {Br:>3}: total {kb:>3} KB -> {fits}")
assert tile_kb(64) == 144 and tile_kb(32) == 104 and tile_kb(128) == 224
print("checkpoint: 32 -> 104 KB, 64 -> 144 KB, 128 -> 224 KB (fp32 accumulators cost as much as fp16 inputs)")
# Expected: 104 / 144 / 224 KB""")

md("""## Your 3-sentence FlashAttention (interview version) — fill this in

Replace the placeholder below with your own three sentences, then run the cell. A model answer structure:
1. What never gets materialized, and where the work happens instead.
2. Why it is still exact (online softmax + rescaling).
3. The numbers at 4k (traffic, memory scaling) and the regime it helps (prefill/training, not decode).""")

code("""MY_THREE_SENTENCES = (
    "FlashAttention tiles Q, K, and V into SRAM-sized blocks and never materializes the N by N score matrix in HBM. "
    "It stays exact because online softmax keeps a running maximum, normalizer, and partial output per row and rescales them whenever the maximum rises. "
    "At 4,096 tokens that cuts HBM traffic roughly nine times (about 4.43 GB to about 0.48 GB per layer in the packet model) and makes extra memory linear, not quadratic, in sequence length, which is why it accelerates prefill and training far more than single-token decode."
)
for i, sentence in enumerate(MY_THREE_SENTENCES.split(". "), 1):
    print(f"{i}. {sentence.strip()}")
print("check each claim against the packet checkpoints before you call it done")
# Expected: three sentences printed; every number in them matches Steps 3 and 5""")

md("""## Day 46 — 5 lines to memorize

1. IO-awareness: at n=4096 attention is about 275 GFLOP (0.28 ms) against about 4.43 GB of naive traffic (1.32 ms) — HBM trips, not FLOPs, set the time.
2. Never materialize: the N x N score/probability matrix lives only as SRAM-sized tiles, discarded after use; extra memory falls from O(N^2) to O(N).
3. Online softmax state per row is (max m, normalizer l, partial output); on a new max, multiply old l and output by exp(m_old - m_new) before merging — one pass, exact (1.1353 rescaled by 0.1353 merges to 1.2034).
4. Traffic model: about 9.3x less at 4k (4.43 GB to 0.48 GB, 32 heads, d=128, fp16, 100 KiB SRAM), rising toward about 12.5x at longer n; wall clock improves 2-4x because compute becomes the next bottleneck.
5. Regime: FlashAttention is a prefill/training win; decode is KV-cache-bound, which is what KV quantization, eviction, and MLA attack.

**Tomorrow (Day 47):** FlashAttention hands-on — install flash-attn, benchmark vs PyTorch SDPA, verify fp16 parity under 1e-3, and wire it into nanoGPT.""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}}, "nbformat": 4, "nbformat_minor": 4}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT)
