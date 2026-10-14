#!/usr/bin/env python3
"""Generate notebooks/day36-speculative-decoding-theory.ipynb."""
import json, pathlib

SLUG = "day36-speculative-decoding-theory"
NB_URL = f"https://colab.research.google.com/github/pradeepvaka/llm-inference-90day/blob/master/notebooks/{SLUG}.ipynb"
OUT = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / f"{SLUG}.ipynb"

cells = []
def md(src):
    cells.append({"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.split("\n")]})
def code(src):
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in src.split("\n")]})

md(f"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({NB_URL})

# Day 36 — Speculative Decoding Theory

**Big idea:** decode is memory-bound, so verifying gamma draft tokens in one target pass costs about the same as verifying one. A cheap draft proposes, the target verifies in parallel, and a corrected rejection-sampling rule makes the output distribution *exactly* the target's — lossless. Today is theory + simulation: no large model download, CPU is enough. ~45 min.""")

code("""# Cell 1: one-time installs (only cell that installs)
%pip install --quiet numpy matplotlib scipy
print("deps ok")
# Expected: deps ok""")

md("""## Step 1 — Expected tokens and speedup formulas

Expected tokens per target pass = (1 - alpha^(gamma+1)) / (1 - alpha). Speedup = expected tokens / (1 + gamma * c), where c = draft cost / target cost per token.""")

code("""def expected_tokens(alpha, gamma):
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)

def speedup(alpha, gamma, c):
    return expected_tokens(alpha, gamma) / (1 + gamma * c)

for a in [0.5, 0.6, 0.7, 0.8, 0.9]:
    print(f"alpha={a:.1f}  gamma=5  expected tokens = {expected_tokens(a, 5):.2f}")
# Expected: 1.97, 2.38, 2.94, 3.69, 4.69""")

md("""## Step 2 — Speedup vs gamma, and the optimal gamma

Higher alpha pushes the optimum right; higher c pushes it left. Curves flatten near the peak, which is why gamma = 5 is a common production default.""")

code("""import matplotlib.pyplot as plt

gammas = list(range(1, 16))
for a in [0.6, 0.8, 0.9]:
    vals = [speedup(a, g, c=0.02) for g in gammas]
    best_g = gammas[vals.index(max(vals))]
    print(f"alpha={a}: speedup(gamma=5)={speedup(a, 5, 0.02):.2f}  best gamma in 1..15 = {best_g} ({max(vals):.2f}x)")
    plt.plot(gammas, vals, marker="o", label=f"alpha={a}")
plt.xlabel("gamma (draft tokens)")
plt.ylabel("speedup vs plain decoding")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
# Expected: alpha=0.9 curve highest throughout, about 4.26x at gamma=5""")

md("""## Step 3 — The procurement example: small draft vs big draft

70B target at 42 ms/token. 1B draft at 0.6 ms/token vs 8B draft at 5.4 ms/token, gamma = 5, alpha = 0.8 for both (isolating cost).""")

code("""target_ms = 42.0
exp_tok = expected_tokens(0.8, 5)
for name, draft_ms in [("1B draft", 0.6), ("8B draft", 5.4)]:
    total_ms = 5 * draft_ms + target_ms
    per_tok = total_ms / exp_tok
    print(f"{name}: {total_ms:.0f} ms for {exp_tok:.2f} tokens = {per_tok:.1f} ms/token -> {target_ms / per_tok:.1f}x speedup")
# Expected: 1B draft: 45 ms -> 12.2 ms/token -> 3.4x speedup
# Expected: 8B draft: 69 ms -> 18.7 ms/token -> 2.2x speedup""")

md("""## Step 4 — Simulator: does the empirical mean match the formula?

Each draft token is accepted independently with probability alpha. A step emits (accepted prefix length + 1), capped at gamma + 1.""")

code("""import numpy as np

rng = np.random.default_rng(0)
alpha, gamma, steps = 0.8, 5, 10_000
emitted = []
for _ in range(steps):
    accepted = 0
    for _ in range(gamma):
        if rng.random() < alpha:
            accepted += 1
        else:
            break
    emitted.append(accepted + 1)  # accepted prefix + correction/bonus token
emp = float(np.mean(emitted))
theory = expected_tokens(alpha, gamma)
print(f"empirical tokens/step = {emp:.2f}   theory = {theory:.2f}   diff = {abs(emp - theory):.2f}")
# Expected: empirical within about 0.05 of 3.69""")

md("""## Step 5 — Corrected rejection sampling on toy distributions

p = [0.5, 0.3, 0.2] (target), q = [0.2, 0.5, 0.3] (draft). Accept x with prob min(1, p(x)/q(x)); on rejection resample from normalized max(0, p - q).""")

code("""import numpy as np

rng = np.random.default_rng(1)
p = np.array([0.5, 0.3, 0.2])
q = np.array([0.2, 0.5, 0.3])
resid = np.maximum(0.0, p - q)
resid = resid / resid.sum()
print("residual resample distribution:", resid)
# Expected: [1. 0. 0.]

def spec_sample():
    while True:
        x = rng.choice(3, p=q)
        if rng.random() < min(1.0, p[x] / q[x]):
            return x
        return rng.choice(3, p=resid)

draws = np.array([spec_sample() for _ in range(50_000)])
freq = np.bincount(draws, minlength=3) / len(draws)
print("empirical frequencies:", np.round(freq, 3), " target p:", p)
# Expected: every frequency within about 0.01 of p""")

md("""## Step 6 — Chi-square: correction vs naive draft sampling

Draft-plus-verify should be indistinguishable from sampling p directly; sampling q naively should be clearly rejected.""")

code("""from scipy.stats import chisquare

n = 50_000
counts_spec = np.bincount(draws, minlength=3)
direct = rng.choice(3, size=n, p=p)
counts_direct = np.bincount(direct, minlength=3)
naive = rng.choice(3, size=n, p=q)
counts_naive = np.bincount(naive, minlength=3)
expected = p * n
print("spec vs p:    chi2 p-value =", round(float(chisquare(counts_spec, expected).pvalue), 4))
print("direct vs p:  chi2 p-value =", round(float(chisquare(counts_direct, expected).pvalue), 4))
print("naive q vs p: chi2 p-value =", float(chisquare(counts_naive, expected).pvalue))
# Expected: spec and direct p-values not tiny (no significant difference);
# Expected: naive q p-value ~0.0 (clearly rejected)""")

md("""## Step 7 — Break-even cost ratio

For alpha = 0.5, gamma = 5, solve expected_tokens = 1 + gamma * c.""")

code("""exp_tok = expected_tokens(0.5, 5)
c_be = (exp_tok - 1) / 5
print(f"expected tokens = {exp_tok:.3f}  break-even c = {c_be:.3f}")
print(f"check speedup at c=0.3: {speedup(0.5, 5, 0.3):.2f}x  (below 1 = speculation loses)")
# Expected: break-even c = 0.194; speedup at c=0.3 about 0.79x""")

md("""## Step 8 — vLLM knobs (for tomorrow)

Read the vLLM speculative-decoding docs, then note the knobs you will set on Day 37:

- `--speculative-model <draft>`: the draft model (same family as the target drafts best).
- `--num-speculative-tokens 5`: this is gamma.
- N-gram drafting: proposes tokens from the prompt/output history with no draft model at all — free, but a weak acceptance rate except on repetitive text.

**Checkpoint answers (from the packet):** alpha=0.8, gamma=5 -> 3.69 tokens per target pass. It is lossless because the corrected acceptance rule makes the output distribution exactly p. The two levers are alpha and c.""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} with {len(cells)} cells")
