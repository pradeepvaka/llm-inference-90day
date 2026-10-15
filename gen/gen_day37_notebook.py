#!/usr/bin/env python3
"""Generate notebooks/day37-speculative-decoding-hands-on.ipynb."""
import json, pathlib

SLUG = "day37-speculative-decoding-hands-on"
NB_URL = f"https://colab.research.google.com/github/pradeepvaka/llm-inference-90day/blob/master/notebooks/{SLUG}.ipynb"
OUT = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / f"{SLUG}.ipynb"

cells = []
def md(src):
    cells.append({"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.split("\n")]})
def code(src):
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in src.split("\n")]})

md(f"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({NB_URL})

# Day 37 — Speculative Decoding Hands-on (vLLM / EAGLE)

**Big idea:** yesterday's formula meets real proposers. You will (1) invert a measured 2.21x speedup to recover the implied acceptance rate alpha, (2) sweep gamma for a draft-model pair vs an EAGLE head and see why their optima differ, (3) build a zero-model n-gram proposer, and (4) run real greedy speculative decoding — DistilGPT-2 drafting for GPT-2, same tokenizer family — and verify the output is token-identical to plain greedy decoding. The vLLM serve commands for the Qwen2.5 0.5B + 1.5B T4 pair are printed read-only in Step 6: the 7B pair needs 15.0 GB of weights alone and does not fit a T4. ~45-60 min.""")

code("""# Cell 1: one-time installs (only cell that installs)
%pip install --quiet numpy matplotlib torch transformers
print("deps ok")
# Expected: deps ok""")

md("""## Step 1 — Invert a measured speedup to recover alpha

Day 36: speedup = E(alpha, gamma) / (1 + gamma * c), with E = (1 - alpha^(gamma+1)) / (1 - alpha). Given a measured speedup and a *measured time ratio* c, solve for alpha by bisection. Reference row: Qwen2.5-7B + 0.5B draft on an A100, gamma = 5, speedup 2.21x, c = 0.14 (time ratio — larger than the 0.07 parameter ratio, because fixed launch overhead dominates a small pass).""")

code("""def expected_tokens(alpha, gamma):
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)

def implied_alpha(measured_speedup, gamma, c):
    target_e = measured_speedup * (1 + gamma * c)
    lo, hi = 0.0, 0.999
    for _ in range(80):
        mid = (lo + hi) / 2
        if expected_tokens(mid, gamma) < target_e:
            lo = mid
        else:
            hi = mid
    return target_e, (lo + hi) / 2

need, alpha = implied_alpha(2.21, gamma=5, c=0.14)
print(f"expected tokens needed = {need:.2f}")
print(f"implied alpha = {alpha:.3f}")
print(f"check: E(0.80, 5) = {expected_tokens(0.80, 5):.2f}   E(0.82, 5) = {expected_tokens(0.82, 5):.2f}")
# Expected: expected tokens needed = 3.76
# Expected: implied alpha about 0.81 (between 0.80 -> 3.69 and 0.82 -> 3.87)""")

md("""## Step 2 — Gamma sweep: draft model vs EAGLE

Draft-model pair: alpha = 0.8, c = 0.14. EAGLE head: alpha = 0.9, c = 0.03 (one layer drafting in feature space, no second model). Higher alpha and lower c both push the optimal gamma to the right.""")

code("""import matplotlib.pyplot as plt

def speedup(alpha, gamma, c):
    return expected_tokens(alpha, gamma) / (1 + gamma * c)

for name, a, c in [("draft model", 0.8, 0.14), ("EAGLE head", 0.9, 0.03)]:
    row = [round(speedup(a, g, c), 2) for g in (3, 5, 7)]
    print(f"{name}: alpha={a} c={c}  speedup gamma 3/5/7 = {row}")
    gammas = list(range(1, 13))
    plt.plot(gammas, [speedup(a, g, c) for g in gammas], marker="o", label=name)
plt.xlabel("gamma (speculative tokens)")
plt.ylabel("idealised speedup")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
# Expected: draft model about [2.08, 2.17, 2.10] -> peaks at gamma=5
# Expected: EAGLE about [3.16, 4.07, 4.71] -> still rising at gamma=7""")

md("""## Step 3 — N-gram proposer: free, and only as good as your text is repetitive

Index a text by its last 3 tokens; propose whatever followed that 3-gram before, and score the proposal against the true continuation. No model, c is about zero — alpha is a property of the text.""")

code("""def ngram_acceptance(tokens, n=3, gamma=5):
    # Streaming, like real prompt lookup: the table only ever contains
    # 3-grams from BEFORE the current position -- no peeking at the future.
    table = {}
    proposed = accepted = 0
    for i in range(len(tokens) - n):
        key = tuple(tokens[i:i + n])
        guess = table.get(key)
        if guess:
            truth = tokens[i + n:i + n + gamma]
            for g_tok, t_tok in zip(guess, truth):
                proposed += 1
                if g_tok == t_tok:
                    accepted += 1
                else:
                    break  # first mismatch ends the accepted prefix
        continuation = tokens[i + n:i + n + gamma]
        if key not in table and continuation:
            table[key] = continuation
    return accepted, proposed

repetitive = ("the server reads the cache and the server writes the log and "
              "the server reads the cache and the server writes the log and "
              "the server reads the cache and the server writes the log and "
              "the server reads the cache and the server writes the log").split()
prose = ("a quiet orchard sloped toward the river where herons nested "
         "among reeds and children skipped flat stones across slow water "
         "while thunder gathered far beyond the blue western hills").split()
for name, toks in [("repetitive", repetitive), ("prose", prose)]:
    acc, prop = ngram_acceptance(toks)
    rate = acc / prop if prop else 0.0
    print(f"{name}: accepted {acc} of {prop} scored draft tokens -> alpha approx {rate:.2f}")
# Expected: repetitive scores clearly higher than prose;
# Expected: prose proposes/scores very few tokens (most 3-grams never repeat)""")

md("""## Step 4 — Real greedy speculative decoding: DistilGPT-2 drafts, GPT-2 verifies

Greedy special case of Day 36's rule: a draft token is accepted iff the target's greedy pick at that position equals it; on the first mismatch, emit the target's pick and stop; if all gamma are accepted, emit one bonus target token. Output must be token-identical to plain GPT-2 greedy decoding. This pair is a teaching pair (not same-family tuned), so expect a modest acceptance rate.""")

code("""import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
tok = AutoTokenizer.from_pretrained("gpt2")
target = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
draft = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device).eval()
print("models loaded: target gpt2, draft distilgpt2")
# Expected: device: cuda on a T4 runtime (cpu also works, slower)
# Expected: models loaded: target gpt2, draft distilgpt2""")

code("""@torch.no_grad()
def greedy_plain(prompt, n_tokens):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    start = time.time()
    for _ in range(n_tokens):
        logits = target(ids).logits[:, -1, :]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return ids, time.time() - start

@torch.no_grad()
def greedy_spec(prompt, n_tokens, gamma=5):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = ids.shape[1]
    proposed = accepted = 0
    start = time.time()
    while ids.shape[1] - prompt_len < n_tokens:
        # draft gamma tokens autoregressively
        draft_ids = ids
        guesses = []
        for _ in range(gamma):
            d_logits = draft(draft_ids).logits[:, -1, :]
            g = torch.argmax(d_logits, dim=-1, keepdim=True)
            guesses.append(g)
            draft_ids = torch.cat([draft_ids, g], dim=1)
        # one target pass verifies all gamma positions at once
        cand = torch.cat(guesses, dim=1)
        full = torch.cat([ids, cand], dim=1)
        t_logits = target(full).logits
        base = ids.shape[1]
        emitted = 0
        for j in range(gamma):
            pick = torch.argmax(t_logits[:, base + j - 1, :], dim=-1, keepdim=True)
            proposed += 1
            if pick.item() == guesses[j].item():
                accepted += 1
                emitted += 1
            else:
                ids = torch.cat([ids, cand[:, :j], pick], dim=1)
                emitted += 1
                break
        else:
            bonus = torch.argmax(t_logits[:, base + gamma - 1, :], dim=-1, keepdim=True)
            ids = torch.cat([ids, cand, bonus], dim=1)
            emitted = gamma + 1
        _ = emitted
    return ids[:, :prompt_len + n_tokens], time.time() - start, accepted, proposed

prompt = "The capital of France is"
plain_ids, plain_s = greedy_plain(prompt, 60)
spec_ids, spec_s, spec_acc, spec_prop = greedy_spec(prompt, 60, gamma=5)
print("identical outputs:", torch.equal(plain_ids, spec_ids))
print("text:", tok.decode(spec_ids[0], skip_special_tokens=True))
print(f"plain: {60 / plain_s:.1f} tok/s   speculative: {60 / spec_s:.1f} tok/s")
print(f"draft tokens accepted {spec_acc} of {spec_prop} proposed -> alpha approx {spec_acc / max(spec_prop, 1):.2f}")
# Expected: identical outputs: True
# Expected: accepted/proposed roughly 0.4-0.7 for this teaching pair
# Expected: modest speedup or near-parity on CPU; clearer win on a T4""")

md("""## Step 5 — Read your own benchmark like Day 36

Whatever Step 4 printed, explain it with the formula before moving on: tokens per target pass from your accepted/proposed ratio, cost from the draft's gamma small passes. If speculative was not faster on CPU, that is the batch-1 / compute-bound caveat showing up on hardware where verification is not free — write down which term in the formula moved.""")

code("""alpha_hat = min(spec_acc / max(spec_prop, 1), 0.99)
e = expected_tokens(alpha_hat, 5)
print(f"your alpha approx {alpha_hat:.2f} -> expected tokens per target pass at gamma=5: {e:.2f}")
print("if your measured speedup is well below this, the gap is c: draft passes")
print("plus one verify pass over gamma extra positions, priced in wall-clock time.")
# Expected: one line with your measured alpha and its expected-tokens value""")

md("""## Step 6 — The vLLM configuration (read-only on this notebook)

On a T4 machine with vLLM installed, the production version of Step 4 is:

```
vllm serve Qwen/Qwen2.5-1.5B-Instruct --speculative-model Qwen/Qwen2.5-0.5B-Instruct --num-speculative-tokens 5
```

Sweep `--num-speculative-tokens` over 3, 5, 7 and benchmark with fixed prompts and 128 output tokens, exactly as in Day 21. Weights check: 1.5B target = 3.0 GB + 0.5B draft = 1.0 GB = 4.0 GB in fp16, leaving about 12 GB of a T4 for KV cache. The 7B pair is 15.0 GB of weights alone — it does not fit. EAGLE needs no second model at all: one trained draft head on the frozen target, drafting in feature space (repo: github.com/SafeAILab/EAGLE — read the README's two-step train-then-serve flow).""")

code("""weights = {"Qwen2.5-1.5B target": 1.5, "Qwen2.5-0.5B draft": 0.5, "Qwen2.5-7B target": 7.0}
for name, b in weights.items():
    print(f"{name}: {b * 2:.1f} GB in fp16")
print(f"T4 pair total: {(1.5 + 0.5) * 2:.1f} GB -> fits a 16 GB T4 with KV headroom")
print(f"7B pair total: {(7.0 + 0.5) * 2:.1f} GB -> does NOT fit a 16 GB T4")
# Expected: T4 pair total: 4.0 GB; 7B pair total: 15.0 GB""")

md("""## Step 7 — Decision table

| Approach | Needs training | Extra serving cost | Typical alpha | Best when |
|---|---|---|---|---|
| Separate draft model | No | Second model + KV cache, c about 0.14 | 0.7-0.85 | Same-family small sibling exists |
| EAGLE head | Yes, one light head | One layer, c about 0.03 | 0.85-0.95 | One target served at volume |
| N-gram lookup | No | About none | Text-dependent | Code, structured output, retrieval quoting |

**Checkpoint answers (from the packet):** the API is unchanged with speculation on. EAGLE drafts in the target's feature space, which is why alpha reaches 0.85-0.95. The measured 2.21x at gamma=5 and c=0.14 inverts to alpha about 0.81.""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} with {len(cells)} cells")
