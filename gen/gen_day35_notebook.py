#!/usr/bin/env python3
"""Generate notebooks/day35-accuracy-eval-after-quantization.ipynb."""
import json, pathlib

SLUG = "day35-accuracy-eval-after-quantization"
NB_URL = f"https://colab.research.google.com/github/pradeepvaka/llm-inference-90day/blob/master/notebooks/{SLUG}.ipynb"
OUT = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / f"{SLUG}.ipynb"

cells = []
def md(src):
    cells.append({"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.split("\n")]})
def code(src):
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in src.split("\n")]})

md(f"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({NB_URL})

# Day 35 — Accuracy Eval After Quantization: Perplexity & Benchmarks

**Big idea:** perplexity is a smoke detector, not a fire inspection. It averages over mostly easy tokens, so it catches a broken quant but can miss reasoning damage. Today you build a correct sliding-window perplexity harness, fake-quantize a small model to INT8 and INT4, re-measure, score a length-normalized multiple-choice probe, and apply a written ship rubric.

T4 recommended for the full runs; CPU works on the shortened slice. ~45 min.""")

code("""# Cell 1: one-time installs (only cell that installs)
%pip install --quiet torch transformers datasets accelerate
print("deps ok")
# Expected: deps ok""")

md("""## Step 1 — Perplexity by hand

Probabilities 0.5, 0.25, 0.4, 0.1. Perplexity = exp(mean negative log-likelihood).""")

code("""import math
probs = [0.5, 0.25, 0.4, 0.1]
nlls = [-math.log(p) for p in probs]
avg_nll = sum(nlls) / len(nlls)
ppl = math.exp(avg_nll)
bits = avg_nll / math.log(2)
print(f"nlls = {[round(x, 3) for x in nlls]}")
print(f"avg NLL = {avg_nll:.3f} nats  perplexity = {ppl:.2f}  bits/token = {bits:.2f}")
print(f"check: 2^bits = {2**bits:.2f}")
# Expected: avg NLL = 1.325 nats  perplexity = 3.76  bits/token = 1.91
# Expected: check: 2^bits = 3.76""")

md("""## Step 2 — Load a small model and WikiText-2 test text

Held-out test split only. Same tokenizer, same text, same window and stride for every model you compare — otherwise the numbers are incomparable.""")

code("""import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
MODEL = "distilgpt2"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).to(device).eval()

try:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\\n\\n".join([t for t in ds["text"] if t.strip()])
    source = "wikitext-2 test"
except Exception as e:
    print("dataset fallback:", e)
    text = ("The history of the Roman Empire covers a long period. "
            "Trade, engineering, and law shaped its cities and provinces. ") * 200
    source = "fallback text"
print(f"source = {source}  chars = {len(text)}")
enc = tok(text, return_tensors="pt")["input_ids"]
print(f"tokens = {enc.shape[1]}")
# Expected: Device: cuda (on T4) or cpu; tokens in the tens of thousands or more""")

md("""## Step 3 — Sliding-window perplexity harness

Window 512, stride 256. Each token is scored exactly once: overlapped tokens are masked with -100 in later windows. Shorten `max_tokens` on CPU.""")

code("""import math
import torch

@torch.no_grad()
def compute_ppl(model, enc, window=512, stride=256, max_tokens=4096):
    ids = enc[:, :max_tokens].to(device)
    seq_len = ids.shape[1]
    total_nll = 0.0
    total_tokens = 0
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + window, seq_len)
        trg_len = end - prev_end
        chunk = ids[:, begin:end]
        targets = chunk.clone()
        targets[:, :-trg_len] = -100
        out = model(chunk, labels=targets)
        # out.loss is mean NLL over the unmasked (target) tokens in this chunk
        total_nll += out.loss.item() * trg_len
        total_tokens += trg_len
        prev_end = end
        if end == seq_len:
            break
    avg_nll = total_nll / total_tokens
    return math.exp(avg_nll), avg_nll, total_tokens

MAX_TOKENS = 4096 if device == "cuda" else 2048
ppl_fp, nll_fp, n_tok = compute_ppl(model, enc, max_tokens=MAX_TOKENS)
print(f"fp16/fp32 baseline: tokens scored = {n_tok}  avg NLL = {nll_fp:.4f}  ppl = {ppl_fp:.2f}")
# Expected: tokens scored = MAX_TOKENS; ppl in the tens for distilgpt2 on a short slice
# The delta between models on this identical slice is the signal, not the absolute.""")

md("""## Step 4 — Fake-quantize weights: INT8, then INT4

Symmetric per-row fake quantization of the attention / MLP weight matrices only (embeddings and the tied head are left alone, as in real weight-only quants). Restore the originals between runs and re-run the *identical* harness.""")

code("""import copy
import torch

TARGET_KEYS = ("c_attn.weight", "c_proj.weight", "c_fc.weight")

def quantize_weights(model, bits):
    backup = {}
    levels = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 2 and any(k in name for k in TARGET_KEYS):
                backup[name] = p.data.clone()
                amax = p.data.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
                scale = amax / levels
                q = torch.clamp(torch.round(p.data / scale), -levels, levels)
                p.data.copy_(q * scale)
    return backup

def restore_weights(model, backup):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name])

results = {"fp": ppl_fp}
for bits in (8, 4):
    backup = quantize_weights(model, bits)
    ppl_q, nll_q, n2 = compute_ppl(model, enc, max_tokens=MAX_TOKENS)
    restore_weights(model, backup)
    results[f"int{bits}"] = ppl_q
    print(f"INT{bits}: ppl = {ppl_q:.2f}  delta = {ppl_q - ppl_fp:+.2f}  "
          f"relative = {(ppl_q - ppl_fp) / ppl_fp * 100:+.1f}%  tokens = {n2}")
# Expected: INT8 delta small; INT4 delta clearly larger than INT8, same token count throughout""")

md("""## Step 5 — Downstream probe: length-normalized multiple choice

The HellaSwag `acc_norm` idea in miniature: score each full completion, divide its log-likelihood by its token count, pick the best. Report accuracy *with its n and standard error*.""")

code("""import math
import torch

ITEMS = [
    ("The capital of France is", [" Paris.", " London.", " Berlin.", " Madrid."], 0),
    ("Water is made of hydrogen and", [" oxygen.", " carbon.", " nitrogen.", " iron."], 0),
    ("Two plus three equals", [" five.", " six.", " four.", " nine."], 0),
    ("The opposite of hot is", [" cold.", " warm.", " bright.", " heavy."], 0),
    ("A triangle has", [" three sides.", " four sides.", " five sides.", " two sides."], 0),
    ("The sun rises in the", [" east.", " west.", " north.", " south."], 0),
    ("Birds use their wings to", [" fly.", " swim.", " dig.", " read."], 0),
    ("Ten minus four equals", [" six.", " five.", " seven.", " fourteen."], 0),
    ("The largest planet in our solar system is", [" Jupiter.", " Mars.", " Venus.", " Mercury."], 0),
    ("Ice is frozen", [" water.", " sand.", " wood.", " metal."], 0),
    ("A week has", [" seven days.", " five days.", " ten days.", " eight days."], 0),
    ("The chemical symbol for gold is", [" Au.", " Ag.", " Fe.", " Pb."], 0),
    ("Plants make food using", [" sunlight.", " darkness.", " plastic.", " stone."], 0),
    ("Twenty divided by four equals", [" five.", " four.", " six.", " ten."], 0),
    ("The first president of the United States was", [" Washington.", " Lincoln.", " Jefferson.", " Adams."], 0),
    ("A spider has", [" eight legs.", " six legs.", " four legs.", " ten legs."], 0),
]

@torch.no_grad()
def score_completion(model, context, choice):
    full = tok(context + choice, return_tensors="pt")["input_ids"].to(device)
    ctx_len = tok(context, return_tensors="pt")["input_ids"].shape[1]
    targets = full.clone()
    targets[:, :ctx_len] = -100
    out = model(full, labels=targets)
    n_choice_tokens = full.shape[1] - ctx_len
    # out.loss is already per-token mean over the choice tokens = length-normalized
    return -out.loss.item(), n_choice_tokens

def run_probe(model):
    correct = 0
    for context, choices, gold in ITEMS:
        scores = [score_completion(model, context, c)[0] for c in choices]
        pred = max(range(len(choices)), key=lambda i: scores[i])
        correct += int(pred == gold)
    acc = correct / len(ITEMS)
    se = math.sqrt(acc * (1 - acc) / len(ITEMS)) if 0 < acc < 1 else math.sqrt(0.25 / len(ITEMS))
    return acc, se

acc_fp, se_fp = run_probe(model)
print(f"probe fp: accuracy = {acc_fp:.2%}  n = {len(ITEMS)}  std err = {se_fp * 100:.1f} points")
backup = quantize_weights(model, 4)
acc_q4, se_q4 = run_probe(model)
restore_weights(model, backup)
print(f"probe INT4: accuracy = {acc_q4:.2%}  n = {len(ITEMS)}  std err = {se_q4 * 100:.1f} points")
print(f"probe delta = {(acc_q4 - acc_fp) * 100:+.1f} points (noisy at this n -- machinery demo, not a ship gate)")
# Expected: n = 16, std err roughly 10+ points -- this probe can only catch catastrophes.
# A real ship decision needs lm-eval-harness with thousands of items (Step 6).""")

md("""## Step 6 — The real harness (optional, T4)

For a ship decision, run lm-evaluation-harness with thousands of items. The command below is shown, not run, to keep this notebook under an hour:

```
pip install lm-eval
lm_eval --model hf --model_args pretrained=distilgpt2 --tasks arc_easy --limit 500
```

Expect a standard error near 1–2 points at that size, versus 10+ points on the Step 5 probe.""")

code("""# Step 6 is a printed reference command (see markdown above) -- nothing to execute.
print("Ship decisions: use lm-eval-harness, n in the hundreds or thousands.")
print("Probes with n=16 teach the machinery; they do not gate a release.")
# Expected: the two lines above""")

md("""## Step 7 — Apply the rubric and print the results table

Course rubric: ship 4-bit iff Δppl < 0.5 AND downstream Δacc < 2 points. Small-model analogue used here (state it, do not hide it): relative Δppl < 5% on this shortened slice, because distilgpt2's absolute perplexity is in the tens, not near 6.""")

code("""print(f"{'format':<10} {'ppl':>8} {'delta':>8} {'rel%':>7}  verdict (ppl criterion only, small-model analogue)")
for key, label in [("int8", "INT8"), ("int4", "INT4")]:
    d = results[key] - results["fp"]
    rel = d / results["fp"] * 100
    verdict = "PASS ppl gate" if rel < 5.0 else "FAIL ppl gate"
    print(f"{label:<10} {results[key]:>8.2f} {d:>+8.2f} {rel:>+6.1f}%  {verdict}")
print()
print(f"downstream probe: fp {acc_fp:.2%} -> INT4 {acc_q4:.2%}  delta {(acc_q4 - acc_fp) * 100:+.1f} points  n={len(ITEMS)}")
print("Full ship verdict also requires the downstream gate (< 2 points) on a large-n harness -- see Step 6.")
# Expected: INT8 passes the ppl gate; INT4 usually fails it on distilgpt2. Either way, the
# table -- format, ppl, delta, task, n, accuracy, delta -- is the row you reuse on Day 42.""")

md("""## What to measure — fill in your numbers

- Hand perplexity (expect 3.76): ___
- Baseline perplexity on your slice, and the exact tokens scored: ___
- INT8 delta and INT4 delta (expect INT4 clearly larger): ___
- Probe accuracy, float vs INT4, with n and standard error: ___
- Ship / no-ship for INT8 and INT4, and which criterion decided each: ___

**Tomorrow:** Day 36 — speculative decoding theory: a draft model proposes, the target verifies, and the acceptance rate sets the speedup.""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} with {len(cells)} cells")
