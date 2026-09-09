# Packet Style Guide — LLM Inference 90-Day

Every daily packet ships as three artifacts: a PDF (≤ 10 pages), a companion
audio script (8–12 min, single narrator), and a Colab notebook (~30–60 min of
hands-on lab). Write each day in this fixed structure.

## 1. PDF structure (keep total ≤ 10 pages)

Fixed section order — every packet follows it so the 90 days read as one book:

0. **Callout box** — one `>` line: `**Time:**` breakdown · `**Prereqs:**` ·
   `**You need:**` (hardware). Never bury logistics in prose.
1. **Today you'll be able to** — 3–5 numbered, measurable exit criteria
   ("Draw ...", "Compute ...", "Explain ... with numbers"). Written as
   abilities, not topics. (Borrowed from the "exit criteria" pattern in the
   most-shared engineering roadmaps.)
2. **The big idea** — the one mental model of the day, in plain language,
   with the infrastructure analogy up front (caches, queues, schedulers).
3. **Key concepts with worked numeric examples** — every concept gets a
   concrete number. Never leave a formula without plugging in real values.
   Example: KV-cache size = `2 × layers × seq_len × hidden × bytes`, then
   GPT-2 (12 layers, 768 hidden, fp32, 1024 tokens) = `2 × 12 × 1024 × 768 × 4`
   = **75.5 MB**.
4. **What to measure** — a small fill-in table (`| Metric | Your number |`).
   The learner records the day's numbers here during the lab. Every packet
   has one; the numbers are the point.
5. **Lab steps with expected outputs** — mirror the notebook cell-by-cell,
   one numbered item per line. Show the exact expected output
   (e.g. `Device: cpu`, `Generated 20 tokens in 1.42 s → 14.1 tok/s`) so the
   learner can self-check.
6. **Checkpoints** — 3–5 numbered items. Format: bold question line, then the
   answer as an indented continuation line (3 spaces). Keep answers tight.
7. **Readings** — 2–4 links/papers max, with one line on what to focus on.
   Be specific (paper sections, blog subsections) — learners consistently ask
   for curated per-topic resources.
8. **Tomorrow teaser** — 2 sentences on the next day.

Formatting rules (the generator handles the rest, but content must cooperate):

- Tables for all comparisons — never ASCII-art diagrams. Every table gets a
  real header row; add a `Table:` caption line after important tables.
- One list item per line. Never inline "1. ... 2. ..." inside a paragraph.
- Use `*italics*` sparingly; it renders as true italics. Never use `*` for
  multiplication in prose — write `×` or `*` only inside code blocks.
- Code blocks: keep lines short (wrap naturally); the generator wraps the
  rest without splitting words.
- The packet title is supplied via `--title`; do NOT repeat it as an `#`
  heading in the content.
- No page may be more than ~50% code.

### Research notes (Sept 2026 pass over social + web roadmaps)

Surveyed viral LLM-inference study plans (Instagram/Threads "right order to
learn LLM inference" roadmaps, "5 videos to master LLM inference" interview
prep, DeepLearning.AI's vLLM course, the modern-ai-engineer-roadmap's
Learn/Build/Measure/Exit-Criteria format, and several 90-day AI-engineer
plans). Incorporated:

- **Exit criteria + Measure per day** — the highest-engagement roadmaps all
  pair "learn" with "build/measure" and a checklist for moving on. Our
  "Today you'll be able to" + "What to measure" sections are this pattern.
- **Curriculum spine validated** — the consensus 12-part order (transformer
  fundamentals → decoding → GPU/CUDA → compression → serving engines →
  scheduling → KV cache → metrics → distributed → production) matches our
  90-day arc. We keep serving-before-kernels deliberately: measure on real
  systems first, then go deep (the "hardware learning path" variant).
- **Benchmark-first ethos** — every optimization day must record a
  before/after number and hold an accuracy floor (perplexity / eval score),
  not just report speedups.
- **Project checkpoints** — the serving-project roadmaps (vLLM+FastAPI →
  streaming → RAG → batching → Prometheus/Grafana → quantized serving)
  mirror our lab progression; Day 15's API server must include SSE
  streaming explicitly.

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
