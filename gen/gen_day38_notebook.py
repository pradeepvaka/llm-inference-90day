#!/usr/bin/env python3
"""Generate notebooks/day38-medusa-parallel-decoding-variants.ipynb."""
import json, pathlib

SLUG = "day38-medusa-parallel-decoding-variants"
NB_URL = f"https://colab.research.google.com/github/pradeepvaka/llm-inference-90day/blob/master/notebooks/{SLUG}.ipynb"
OUT = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / f"{SLUG}.ipynb"

cells = []
def md(src):
    cells.append({"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.split("\n")]})
def code(src):
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in src.split("\n")]})

md(f"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({NB_URL})

# Day 38 — Medusa & Parallel Decoding Variants

**Big idea:** Medusa removes the draft model entirely — K extra decoding heads on the frozen target guess future tokens in parallel from one hidden state, and one tree-attention verify forward arbitrates among branching candidate futures. Today is paper + mask math (CPU only, no model download): you will (1) build the tree attention mask for K=2, top-2 and check exactly who may attend to whom, (2) scale to K=3, top-2 and count paths vs unique nodes, (3) watch the full cartesian tree explode to 120 nodes at K=4, top-3 and prune it back to a 15-node budget by joint path probability, (4) price a Medusa step from accepted length, and (5) price the heads themselves from hidden size and vocabulary. ~30-60 min.""")

code("""# Cell 1: one-time installs (only cell that installs)
%pip install --quiet numpy matplotlib
print("deps ok")
# Expected: deps ok""")

md("""## Step 1 — Build the tree and its attention mask (K = 2, top-2)

Construction: node 0 is the root (the last verified position; it stands in for the whole shared prefix). Each node at depth d sprouts `topk` children at depth d+1 — the next head's top-k alternatives along that branch. Mask rule: node i may attend to node j exactly when j is i itself or an ancestor of i. Siblings never see each other.""")

code("""import numpy as np

def build_tree(K, topk):
    # nodes: (depth, parent_index, branch_index); node 0 = root
    nodes = [(0, -1, -1)]
    frontier = [0]
    for depth in range(1, K + 1):
        nxt = []
        for parent in frontier:
            for b in range(topk):
                nodes.append((depth, parent, b))
                nxt.append(len(nodes) - 1)
        frontier = nxt
    return nodes

def tree_mask(K, topk):
    nodes = build_tree(K, topk)
    n = len(nodes)
    mask = np.zeros((n, n), dtype=int)
    for i, (depth, parent, b) in enumerate(nodes):
        j = i
        while j != -1:
            mask[i, j] = 1
            j = nodes[j][1]
    return nodes, mask

nodes, mask = tree_mask(K=2, topk=2)
print("nodes (depth, parent, branch):")
for idx, nd in enumerate(nodes):
    print(f"  node {idx}: {nd}")
print("tree attention mask (rows = query node, cols = key node):")
print(mask)
print("row sums:", mask.sum(axis=1).tolist())
leaves = [i for i, (d, p, b) in enumerate(nodes) if d == 2]
print(f"paths (root-to-leaf) = {len(leaves)}   candidate nodes = {len(nodes) - 1}   positions incl. root = {len(nodes)}")
assert mask.shape == (7, 7)
assert mask.sum(axis=1).tolist() == [1, 2, 2, 3, 3, 3, 3]
# no node attends outside its own ancestor chain:
for i, (d, p, b) in enumerate(nodes):
    allowed = set(np.where(mask[i] == 1)[0].tolist())
    chain, j = set(), i
    while j != -1:
        chain.add(j); j = nodes[j][1]
    assert allowed == chain
print("mask verified: every row is exactly the node's ancestor chain + itself")
# Expected: 7x7 mask; row sums [1, 2, 2, 3, 3, 3, 3]
# Expected: paths = 4, candidate nodes = 6, positions incl. root = 7""")

md("""## Step 2 — Scale to K = 3, top-2 and count carefully

Paths multiply (topk^K); unique nodes add (topk + topk^2 + ... + topk^K). Keep the two counts separate — confusing them is the standard Medusa error.""")

code("""nodes3, mask3 = tree_mask(K=3, topk=2)
paths3 = 2 ** 3
cand3 = sum(2 ** i for i in range(1, 4))
print(f"K=3, top-2: paths = {paths3}, candidate nodes = {cand3}, positions incl. root = {len(nodes3)}")
print("mask shape:", mask3.shape)
print(mask3)
depth3_rows = [i for i, (d, p, b) in enumerate(nodes3) if d == 3]
print("depth-3 row sums:", [int(mask3[i].sum()) for i in depth3_rows])
assert mask3.shape == (15, 15)
assert all(int(mask3[i].sum()) == 4 for i in depth3_rows)
print("verified: every depth-3 node attends to root + 2 ancestors + itself = 4 positions")
# Expected: paths = 8, candidate nodes = 14, positions = 15, mask 15x15
# Expected: every depth-3 row sums to 4""")

md("""## Step 3 — The cartesian explosion, and pruning to a budget

K = 4 heads with top-3 alternatives each gives a 120-node full tree — too big to verify every step. Production Medusa ranks root-to-leaf paths by joint probability (product of the per-depth head probabilities along the path) and keeps whole paths greedily until a node budget is spent. The probabilities below are synthetic teaching values, one top-3 list per depth.""")

code("""import itertools

def cartesian_nodes(K, topk):
    return sum(topk ** i for i in range(1, K + 1))

for K, tk in [(2, 2), (3, 2), (4, 3)]:
    print(f"K={K}, top-{tk}: full cartesian candidate nodes = {cartesian_nodes(K, tk)}")
assert cartesian_nodes(4, 3) == 120
print("K=4, top-3 explodes to 120 nodes -> pruning is not optional")

# synthetic per-depth top-3 probabilities for the 4 heads
depth_probs = [
    [0.55, 0.28, 0.17],   # head for t+1
    [0.50, 0.30, 0.20],   # head for t+2
    [0.45, 0.33, 0.22],   # head for t+3
    [0.48, 0.31, 0.21],   # head for t+4
]
K, topk = 4, 3
paths = []
for combo in itertools.product(range(topk), repeat=K):
    p = 1.0
    for d, b in enumerate(combo):
        p *= depth_probs[d][b]
    paths.append((p, combo))
paths.sort(reverse=True)
print(f"total root-to-leaf paths = {len(paths)}; best path prob = {paths[0][0]:.4f}, worst = {paths[-1][0]:.4f}")

BUDGET = 15  # candidate nodes, root excluded
kept_nodes = set()
kept_paths = []
for p, combo in paths:
    prefixes = {combo[:i + 1] for i in range(K)}
    new_nodes = prefixes - kept_nodes
    if len(kept_nodes) + len(new_nodes) <= BUDGET:
        kept_nodes |= prefixes
        kept_paths.append((p, combo))
print(f"budget {BUDGET} candidate nodes -> kept {len(kept_paths)} paths, {len(kept_nodes)} unique nodes")
for p, combo in kept_paths:
    print(f"  path branches {combo}: joint prob {p:.4f}")
# Expected: full cartesian = 120 nodes, 81 paths
# Expected: only a handful of paths kept; kept tree has at most 15 candidate nodes""")

md("""## Step 4 — Price a Medusa step: accepted length to milliseconds per token

70B target: one plain decode forward is about 42 ms (Day 36). A Medusa verify forward over a pruned tree of ~13 positions streams the same weights once, so at batch 1 it costs barely more than a 1-token forward — model it at an overhead factor of 1.15 and be explicit that this is an assumption. Tokens banked per step = accepted length + 1 correction/bonus token.""")

code("""import matplotlib.pyplot as plt

BASE_MS = 42.0
OVERHEAD = 1.15
step_ms = BASE_MS * OVERHEAD
print(f"verify step = {BASE_MS} ms x {OVERHEAD} = {step_ms:.1f} ms")
for acc in (2.2, 3.2, 3.8):
    per_token = step_ms / (acc + 1)
    print(f"accepted {acc}: {per_token:.2f} ms/token -> idealised speedup {BASE_MS / per_token:.2f}x")
accs = [a / 10 for a in range(10, 46)]
plt.plot(accs, [BASE_MS / (step_ms / (a + 1)) for a in accs], marker="o", markersize=3)
plt.axhline(2.2, linestyle="--", label="Medusa-1 measured (paper, >2.2x)")
plt.axhline(2.8, linestyle=":", label="Medusa-2 measured (paper, up to ~2.8x)")
plt.xlabel("average accepted length")
plt.ylabel("idealised speedup at overhead 1.15")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
# Expected: accepted 3.2 -> about 11.50 ms/token, about 3.65x idealised
# Expected: the idealised curve sits ABOVE the paper's measured 2.2-2.8x band;
# the gap is real batch verify cost, bookkeeping, and lower on-workload acceptance""")

md("""## Step 5 — Price the heads: a vocabulary cost, not a hidden-size cost

One Medusa head = one residual block (d x d) + one vocabulary projection (d x vocab). Run it for Vicuna-7B geometry (d = 4096, vocab = 32,000) and for a Llama-3-sized vocabulary (128,256).""")

code("""def head_params(d, vocab, K):
    per_head = d * d + d * vocab
    return per_head, per_head * K

for vocab in (32_000, 128_256):
    per, total = head_params(d=4096, vocab=vocab, K=4)
    print(f"vocab {vocab}: per head {per / 1e6:.1f}M params | K=4 total {total / 1e6:.1f}M params = {total * 2 / 1e9:.2f} GB in fp16")
    print(f"   projection share of one head: {(4096 * vocab) / per:.1%}")
# Expected: vocab 32000 -> per head 147.8M, K=4 total 591.4M params = 1.18 GB fp16
# Expected: vocab 128256 -> K=4 total about 2168.5M params = 4.34 GB fp16
# Expected: the vocab projection is ~89% of a head at 32k vocab, ~97% at 128k""")

md("""## Step 6 — vLLM support shape (read-only) and the decision table

vLLM supports Medusa as a named speculative method, but only for a checkpoint that actually carries trained Medusa heads — there is no Medusa for an arbitrary stock model. Flag spelling has moved between the legacy `--speculative-model` CLI and the newer `--speculative-config` JSON across versions; the cell below prints the *shape*, and you check your version's docs for the exact spelling.""")

code("""speculative_config_shape = {
    "method": "medusa",
    "model": "<checkpoint-with-medusa-heads, paired with its base model>",
    "num_speculative_tokens": 4,   # = number of Medusa heads K
}
print("vLLM speculative config shape for Medusa:")
for k, v in speculative_config_shape.items():
    print(f"  {k}: {v}")
print()
print("Requirements / status notes:")
print("  - the served checkpoint must carry trained Medusa heads; a stock checkpoint has none")
print("  - Medusa-1 heads train on the frozen backbone; Medusa-2 fine-tunes backbone + heads")
print("  - supported in vLLM, but EAGLE-family methods are the actively pushed path in current releases")
print("  - no second KV cache and no second tokenizer: the heads ride on the target itself")
# Expected: config shape printed with method 'medusa' and num_speculative_tokens = K""")

md("""## Decision table — write it from memory, then check

| Approach | Needs training | Extra serving cost | Typical speedup | Best when |
|---|---|---|---|---|
| Vanilla draft model | No | Second model + its KV cache, c ~ 0.1-0.15 | ~2-3x at low batch | Same-family small model already exists |
| EAGLE | Yes, one light layer on features | One layer, c ~ 0.03, no second cache | ~3x+, alpha ~ 0.85-0.95 | One target at volume, one training run affordable |
| Medusa | Yes, K heads (Medusa-2 also backbone) | K head projections, no second cache | 2.2x (M-1) to ~2.3-2.8x (M-2) | No draft model exists; draft-free integration wanted |
| N-gram lookup | No | ~None, c ~ 0 | ~1.05x prose, ~1.37x repetitive | Code / structured / retrieval text — free, try first |

**Checkpoints — answer before you scroll back:**
1. Medusa head k predicts what? -> The token at position t+k, read off the target's hidden state at t.
2. What does the tree attention mask enforce? -> Candidates attend only to their ancestors in the tree plus the shared prefix — never to sibling branches.
3. Medusa vs EAGLE training cost? -> Both light and heads-only at base; Medusa trains K independent heads on the frozen target, EAGLE one draft layer on its features (Medusa-2 excepted: it fine-tunes the backbone too).

Tomorrow (Day 39): KV eviction — attention sinks, heavy hitters, StreamingLLM, and the token-zero ablation.""")

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3"}},
    "nbformat": 4, "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT, "cells:", len(cells))
