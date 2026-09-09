# LLM Inference 90-Day

A 90-day, hands-on program that takes you from **backend software
infrastructure engineer** to an **expert, hands-on LLM inference engineer**.

Every concept comes with a worked numeric example, every day ends with a
Colab lab you can actually run on the free tier (CPU or a single T4 GPU),
and every week compounds: tokenizers → the decode loop → KV cache →
batching → quantization → serving stacks → profiling → distributed
inference → evals → production hardening.

## How a day works

Each day ships three artifacts:

1. **Packet PDF** (`packets/dayNN.pdf`, ≤ 10 pages) — objectives, concepts
   with worked numeric examples, lab steps with expected outputs,
   checkpoints with answers, readings, and a teaser for tomorrow.
2. **Audio companion** (`audio/`) — 8–12 minutes, single narrator, written
   in spoken form. Listen on a walk, then do the lab.
3. **Colab notebook** (`notebooks/`) — the hands-on part. ~30–60 minutes.

Read the packet, listen to the audio, then run the notebook top to bottom.
That order is deliberate: concepts first, muscle memory second.

## Setup

- **Python 3.10+** locally (for building packets and validating notebooks).
- **Google Colab, free tier** — a T4 GPU helps on some days but every
  notebook runs on CPU too. No paid tier needed.
- Python deps: `pip install -r requirements.txt` (`reportlab` for the
  packet builder, `pypdf` for checks).
- **GitHub**: this repo is meant to be published — fork it, keep your own
  notes and lab results in `labs/`, and let the commit history be your
  90-day log.

### Building a packet PDF

```bash
python3 gen/make_packet.py \
  --input packets/day01.md \
  --output packets/day01.pdf \
  --title "The LLM inference stack: from prompt to tokens per second" \
  --day 1 --date 2026-09-09
```

The builder enforces the 10-page limit automatically (it shrinks fonts,
then trims trailing paragraphs, warning on stdout). Authoring rules live
in [`gen/style.md`](gen/style.md).

## Repo layout

```
llm-inference-90day/
├── README.md            # you are here
├── requirements.txt     # reportlab, pypdf
├── gen/
│   ├── make_packet.py   # markdown-ish → ≤10-page PDF builder
│   └── style.md         # packet / audio / notebook style guide
├── packets/             # daily sources (dayNN.md) and built PDFs
├── notebooks/           # Colab-ready labs, one per day
├── audio/               # companion audio scripts (see style guide)
├── readings/            # reading notes and paper summaries
└── labs/                # your own lab notes and results
```

## Colab badges

Every notebook starts with an "Open in Colab" badge pointing at this repo:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-user>/llm-inference-90day/blob/main/notebooks/day01-llm-inference-stack.ipynb)
```

Replace `<your-user>` with your GitHub username after publishing. In Colab,
`Runtime → Run all` should just work.

## The 90-day arc

| Phase | Days | Focus |
|-------|------|-------|
| 1. Foundations | 1–15 | Tokenizers, the decode loop, KV cache, attention cost models |
| 2. Optimization | 16–35 | Batching, continuous batching, quantization (int8/fp8), KV-cache eviction |
| 3. Serving | 36–60 | vLLM / TGI / TensorRT-LLM, OpenAI-compatible APIs, load testing |
| 4. Scale | 61–80 | Tensor/pipeline parallelism, multi-GPU profiling, cost modeling |
| 5. Production | 81–90 | Evals, guardrails, observability, on-call for inference |

Day 1 starts with the single most important picture in the field: a prompt
goes in, tokens come out, and everything in between is a memory-bandwidth
problem. Happy building.
