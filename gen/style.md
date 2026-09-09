# Packet Style Guide — LLM Inference 90-Day

Every daily packet ships as three artifacts: a PDF (≤ 10 pages), a companion
audio script (8–12 min, single narrator), and a Colab notebook (~30–60 min of
hands-on lab). Write each day in this fixed structure.

## 1. PDF structure (keep total ≤ 10 pages)

1. **Header / objectives** — day number, date, title; 3–5 measurable objectives
   ("By the end you can ...").
2. **Concepts with worked numeric examples** — every concept gets a concrete
   number. Never leave a formula without plugging in real values.
   Example: KV-cache size = `2 × layers × seq_len × hidden × bytes`, then
   GPT-2 (12 layers, 768 hidden, fp32, 1024 tokens) = `2 × 12 × 1024 × 768 × 4`
   = **75.5 MB**.
3. **Lab steps with expected outputs** — mirror the notebook cell-by-cell.
   Show the exact expected output (e.g. `Device: cpu`, `Generated 20 tokens
   in 1.42 s → 14.1 tok/s`) so the learner can self-check.
4. **Checkpoints** — 3–5 questions; answers at the end of the packet.
5. **Readings** — 2–4 links/papers max, with one line on what to focus on.
6. **Tomorrow teaser** — 2 sentences on the next day.

Rules: code blocks in monospace on a light-gray background; tables only for
small comparisons; no page may be more than ~50% code.

## 2. Audio script format

- Single narrator, conversational but precise. Target **1100–1600 words**
  (8–12 minutes at normal pace).
- **Spoken form everywhere**: spell out numbers ("seventy-five point five
  megabytes"), abbreviations ("key-value cache", not "KV cache" on first
  use), and URLs ("github dot com slash ...").
- Open with the day's objectives, teach the one big idea, close with the
  one thing to try in the notebook today.

## 3. Notebook conventions (Colab-ready)

- First markdown cell = **Colab badge**:
  `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<user>/<repo>/blob/main/notebooks/<file>.ipynb)`
- Second cell = `pip install` block (first and only installs; `--quiet`).
- Resource budget: **CPU or a single free T4 only**. No multi-GPU,
  no downloads over ~2 GB (use `gpt2`, `gpt2-medium`, `distilgpt2`).
- Every code cell shows its expected output (comment or sample block).
- Cells must run top-to-bottom with no manual edits. Guard GPU-only code:
  `device = "cuda" if torch.cuda.is_available() else "cpu"`.
- Keep cells small; one idea per cell. ~30–60 minutes of hands-on.
