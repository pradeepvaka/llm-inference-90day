#!/usr/bin/env python3
"""Generate notebooks/day42-quantize-benchmark-report.ipynb."""
import json, pathlib

SLUG = "day42-quantize-benchmark-report"
NB_URL = f"https://colab.research.google.com/github/pradeepvaka/llm-inference-90day/blob/master/notebooks/{SLUG}.ipynb"
OUT = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / f"{SLUG}.ipynb"

cells = []
def md(src):
    cells.append({"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in src.split("\n")]})
def code(src):
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in src.split("\n")]})

md(f"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({NB_URL})

# Day 42 — Mini-Project: Quantize + Benchmark Report

**Big idea:** a benchmark report is a decision document, not a trophy case. Fix one model (Llama-3.1-8B), one hardware target (A100 for server rows; the GGUF row is CPU-lane and marked not comparable), one prompt/data harness, and the Day 35 rubric stated in advance (delta ppl < 0.5 AND downstream drop < 2 pp), plus a 100 tok/s decode SLO. Then: (1) build the master table, (2) apply rubric + SLO in code, (3) compute GPU-hours per million tokens, tok/s per GB, HellaSwag standard error, and KV headroom, (4) draw the two report plots, (5) generate `quant-report.md`, and (6) self-grade Weeks 5–6 checkpoints. CPU is enough for all analysis cells; use a T4 only if you re-measure a row. The shipped RECORDS below are the packet's worked-example dataset — overwrite them with your own re-run numbers from Days 30–34 and the table, plots, and report all update. ~30-60 min for the analysis; the full re-run is the ~3 h project version.""")

code("""# Cell 1: one-time installs (only cell that installs)
%pip install --quiet matplotlib
print("deps ok")
# Expected: deps ok""")

md("""## Step 1 — Freeze the harness, then the master table

Harness (write it before any numbers): model = Llama-3.1-8B (8.03B params); hardware = A100 for server rows, CPU lane separate; perplexity = WikiText-2; downstream = HellaSwag validation (n ≈ 10,042), length-normalized; throughput = decode tok/s, same prompt set for every row; rubric = delta ppl < 0.5 AND downstream drop < 2.0 pp; SLO = >= 100 tok/s on A100. Q4_K_M is a CPU/llama.cpp format (Day 34), so its A100 tok/s is None — not comparable, never copied across lanes.""")

code("""RECORDS = [
    # method, size_gb, wiki_ppl, hellaswag_pct, tok_s_a100 (None = not comparable), load_s
    ("fp16 baseline",          16.06, 6.10, 82.0, 62,   8.0),
    ("GPTQ 4-bit g128",         4.42, 6.48, 80.4, 108,  2.2),
    ("AWQ 4-bit g128",          4.40, 6.38, 80.9, 112,  2.2),
    ("FP8 (sim on A100)",       8.03, 6.13, 81.8, 68,   4.0),
    ("Q4_K_M (GGUF, CPU lane)", 4.93, 6.42, 80.2, None, None),
]
base = RECORDS[0]
print(f"{'method':>26} {'GB':>6} {'ppl':>5} {'dppl':>6} {'HS%':>5} {'dacc':>6} {'tok/s':>6} {'load s':>7}")
for name, gb, ppl, hs, tps, load in RECORDS:
    dppl = ppl - base[2]
    dacc = hs - base[3]
    tps_s = f"{tps:>6}" if tps is not None else "   n/c"
    load_s = f"{load:>7.1f}" if load is not None else "    n/c"
    print(f"{name:>26} {gb:>6.2f} {ppl:>5.2f} {dppl:>+6.2f} {hs:>5.1f} {dacc:>+6.1f} {tps_s} {load_s}")

# Size sanity from first principles (Day 34): 8.03B params
assert abs(8.03 * 2 - 16.06) < 0.01          # fp16
assert abs(8.03 * 1 - 8.03) < 0.01           # FP8
print(f"ideal 4-bit = {8.03*0.5:.2f} GB; AWQ actual 4.40 GB -> shrink {16.06/4.40:.2f}x (group scales are real bytes)")
print(f"load-time check: size / ~2 GB/s -> fp16 {16.06/2:.1f} s, AWQ {4.40/2:.1f} s")
# Expected: table as in the packet; shrink 3.65x; loads 8.0 s vs 2.2 s""")

md("""## Step 2 — Apply the rubric, then the SLO, in code

Order matters: quality gates first (Day 35 rubric, stated in advance), then the serving SLO. FP8 is the instructive row — best quality, yet it fails the SLO on A100 because Ampere has no FP8 tensor cores (Day 33). A verdict is method × hardware, never method alone.""")

code("""DPPL_GATE, DACC_GATE, SLO = 0.5, 2.0, 100
print(f"{'method':>26} {'quality':>9} {'SLO':>9} {'ship on A100?':>14}")
eligible = []
for name, gb, ppl, hs, tps, load in RECORDS[1:]:
    dppl, dacc_drop = ppl - base[2], base[3] - hs
    quality = (dppl < DPPL_GATE) and (dacc_drop < DACC_GATE)
    borderline = quality and (DACC_GATE - dacc_drop) <= 0.3
    slo = (tps is not None) and (tps >= SLO)
    ship = quality and slo
    if ship:
        eligible.append(name)
    tag = "pass*" if borderline else ("pass" if quality else "FAIL")
    print(f"{name:>26} {tag:>9} {('pass' if slo else 'FAIL'):>9} {('YES' if ship else 'no'):>14}")
print("* = borderline: passes with <= 0.3 pp margin on the downstream gate")
print("ship-eligible on A100:", eligible)
assert eligible == ["GPTQ 4-bit g128", "AWQ 4-bit g128"]
# Expected: quality pass for all four rows (Q4_K_M borderline at -1.8 pp); SLO pass only GPTQ/AWQ; fp16 baseline itself is 62 tok/s, under the SLO""")

md("""## Step 3 — Derived numbers: cost, efficiency, significance, capacity

GPU-hours per million tokens = 1e6 / tok/s / 3600. Tok/s per GB exposes the memory-bound decode win. HellaSwag standard error = sqrt(p(1-p)/n). KV headroom: Llama-3.1-8B KV in fp16 = 131,072 B/token (Day 41); AWQ frees 16.06 - 4.40 = 11.66 GB vs fp16.""")

code("""import math

def gpu_hours_per_mtok(tps):
    return 1e6 / tps / 3600

fp16_h, awq_h = gpu_hours_per_mtok(62), gpu_hours_per_mtok(112)
print(f"GPU-hours per 1M tokens: fp16 {fp16_h:.2f}, AWQ {awq_h:.2f}, GPTQ {gpu_hours_per_mtok(108):.2f}")
print(f"AWQ saving vs fp16: {(fp16_h-awq_h)/fp16_h*100:.0f}%")
assert abs(fp16_h - 4.48) < 0.01 and abs(awq_h - 2.48) < 0.01

print(f"tok/s per GB: fp16 {62/16.06:.1f}, AWQ {112/4.40:.1f} -> {112/4.40/(62/16.06):.1f}x")

n, p = 10042, 0.82
se = math.sqrt(p * (1 - p) / n) * 100
print(f"HellaSwag SE at n={n}, p={p}: {se:.2f} pp; AWQ drop 1.1 pp = {1.1/se:.1f} SE; Q4_K_M drop 1.8 pp = {1.8/se:.1f} SE")
assert abs(se - 0.38) < 0.01

kv_bytes_per_token = 131072
headroom_tokens = int((16.06 - 4.40) * 1e9 // kv_bytes_per_token)
print(f"KV headroom from AWQ-freed 11.66 GB: {headroom_tokens:,} tokens (~{headroom_tokens//1000}k, or ~{headroom_tokens//8192} sequences of 8k)")
assert 88000 < headroom_tokens < 90000
print("verified: ~45% cheaper per token, ~6.6x tok/s per GB, ~89k tokens of KV headroom")
# Expected: 4.48 vs 2.48 GPU-h; 3.9 vs 25.5 tok/s/GB (6.6x); SE 0.38 pp; headroom ~88,900 tokens""")

md("""## Step 4 — The two report plots

Plot 1: size vs perplexity (quality-cost frontier, all five rows). Plot 2: tok/s vs perplexity for A100-comparable rows only, with the SLO line (100 tok/s) and the quality-gate line (ppl = 6.10 + 0.5 = 6.60) — the ship region is low-ppl, high-tok/s. FP8 must land left of the SLO line despite the best ppl: that visual is the report's argument.""")

code("""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, gb, ppl, hs, tps, load in RECORDS:
    axes[0].scatter(gb, ppl, s=70)
    axes[0].annotate(name.split(" (")[0], (gb, ppl), textcoords="offset points", xytext=(4, 4), fontsize=7)
axes[0].set_xlabel("size (GB)")
axes[0].set_ylabel("WikiText perplexity")
axes[0].set_title("Size vs perplexity")

for name, gb, ppl, hs, tps, load in RECORDS:
    if tps is None:
        continue
    axes[1].scatter(tps, ppl, s=70)
    axes[1].annotate(name.split(" (")[0], (tps, ppl), textcoords="offset points", xytext=(4, 4), fontsize=7)
axes[1].axvline(100, linestyle="--")
axes[1].axhline(6.60, linestyle="--")
axes[1].set_xlabel("decode tok/s on A100")
axes[1].set_ylabel("WikiText perplexity")
axes[1].set_title("Throughput vs perplexity (ship = right of SLO, below gate)")
fig.tight_layout()
fig.savefig("quant-report-plots.png", dpi=120)
print("saved quant-report-plots.png")
# Expected: AWQ inside the ship region; FP8 left of the SLO line at the lowest ppl""")

md("""## Step 5 — Generate the one-to-two page report

The report: harness in five lines, master table, derived numbers, one recommendation paragraph (ship / fallback / keep-baseline / revisit trigger), and the four caveats — calibration dependence, long context not evaluated, hardware/kernel specificity, single downstream benchmark. A report without caveats is marketing.""")

code("""report = '''# Quantization Report — Llama-3.1-8B (Day 42)

Harness: A100 server rows (GGUF row is CPU-lane, not comparable); WikiText-2 ppl;
HellaSwag validation n=10,042, length-normalized; decode tok/s, same prompt set.
Rubric (Day 35, stated in advance): delta ppl < 0.5 AND downstream drop < 2 pp.
SLO: >= 100 tok/s decode on A100.

| Method | GB | ppl | dPPL | HellaSwag | dAcc | tok/s A100 |
|---|---|---|---|---|---|---|
| fp16 | 16.06 | 6.10 | — | 82.0 | — | 62 |
| GPTQ 4-bit g128 | 4.42 | 6.48 | +0.38 | 80.4 | -1.6 | 108 |
| AWQ 4-bit g128 | 4.40 | 6.38 | +0.28 | 80.9 | -1.1 | 112 |
| FP8 (sim, A100) | 8.03 | 6.13 | +0.03 | 81.8 | -0.2 | 68 |
| Q4_K_M (CPU lane) | 4.93 | 6.42 | +0.32 | 80.2 | -1.8* | n/c |

Derived: AWQ = 2.48 vs 4.48 GPU-hours per 1M tokens (~45% cheaper);
25.5 vs 3.9 tok/s per GB; frees 11.66 GB -> ~88,900 tokens of KV headroom.
HellaSwag SE ~0.38 pp, so *Q4_K_M's -1.8 pp is real and borderline vs the 2.0 gate.

Recommendation: Ship AWQ 4-bit on the A100 fleet — passes both rubric gates and
the SLO at 1.8x fp16 throughput. GPTQ is the fallback (throughput wash, tie broken
on quality). Keep fp16 for the eval-gated offline batch pipeline. Q4_K_M is approved
for the laptop/local lane only. Revisit FP8 when H100s arrive (projected ~2x fp16
on Hopper, Day 33); on Ampere it misses the SLO despite the best quality.

Caveats: (1) GPTQ/AWQ deltas depend on the calibration set; production domain shift
can move them. (2) Long context not evaluated — ppl here is short-window WikiText.
(3) Throughput is A100- and kernel-specific (Marlin vs naive dequant differs by
tens of percent). (4) Downstream coverage is one benchmark; Day 35 showed GSM8K
drops can be relatively larger than perplexity suggests.
'''
open("quant-report.md", "w").write(report)
print(report)
print("wrote quant-report.md")
# Expected: the report above; one ship method, one fallback, one keep-baseline use, one revisit trigger""")

md("""## Step 6 — Self-grade Weeks 5–6, then finish

Re-answer cold, no peeking: (a) Day 29 — when is symmetric quantization the wrong choice? (b) Day 31 — what does AWQ's salient-channel scaling protect, and why is its serving speed a wash vs GPTQ? (c) Day 33 — Hopper vs pre-Hopper: which format lane each, and why? (d) Day 35 — state the ship rubric in full. Score yourself in the cell below, redo the misses from the packets, and name the weakest topic — it is your Day 43 warm-up (GPU architecture builds on the bandwidth intuition behind every number today).

Checkpoint answers (from the packet): the two gates are delta ppl < 0.5 and downstream drop < 2 pp; tok/s is reported alongside ppl because the decision is quality per dollar/latency under an SLO, not quality alone; a required caveat is any of calibration dependence, long context not evaluated, hardware/kernel-specific throughput, or single-benchmark downstream coverage.""")

code("""# Your self-grade, as data — then run to summarize.
self_grade = {
    "Day 29 affine vs symmetric": None,   # True/False after re-answering cold
    "Day 31 AWQ salient channels": None,
    "Day 33 FP8 vs INT4 lanes": None,
    "Day 35 rubric in full": None,
}
weakest_topic = ""  # TODO: name it in your own words
answered = [k for k, v in self_grade.items() if v is not None]
print(f"self-grade recorded for {len(answered)} of {len(self_grade)} topics; weakest: {weakest_topic or '(fill in)'}")
print("notebook complete -- Day 42 done: one table, one rubric, one call")
# Expected: summary line printed; notebook complete""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} with {len(cells)} cells")
