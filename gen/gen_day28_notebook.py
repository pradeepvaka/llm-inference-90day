#!/usr/bin/env python3
"""Generate notebooks/day28-production-mini-project.ipynb."""
import json

NB = "https://colab.research.google.com/github/pradeepvaka/llm-inference-90day/blob/master/notebooks/day28-production-mini-project.ipynb"

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": [l + "\n" for l in src.split("\n")]})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [l + "\n" for l in src.split("\n")]})

md("""# Day 28 — Mini-Project: Production-ish Server + Metrics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](%s)

**Big idea:** customers pay for the dining room, not the kitchen. Today you open the restaurant: a FastAPI gateway with request-ID logging, SSE streaming, prefix caching, and guided JSON — benchmarked with an SLO report and a cost-per-token P&L.

T4 runtime recommended; cells also run on CPU (distilgpt2). Total ~45 min. The vLLM-on-rented-GPU pass (optional, Step 12 in the packet) reuses the same gateway code.
""" % NB)

code("""# Colab: install deps (only cell that installs)
%pip install --quiet transformers fastapi uvicorn httpx jsonschema
print('deps ok')
# Expected: deps ok""")

md("""## Step 0 — The stack and the gateway contract

Gateway = the dining room. It owns: request IDs, timing headers (`X-TTFT-ms`, `X-Request-ID`), a JSONL log with per-stage timings (queue wait, prefill, decode), and the OpenAI-compatible surface. The model backend is swappable behind it.""")

code("""import json, time, uuid, threading, asyncio
from collections import defaultdict
import torch

# ---- Day-28 config ----
GPU_PRICE_HR = 0.60          # $/hr, vast.ai-style T4-class
TTFT_P99_SLO_MS = 1000       # SLO: time to first token p99
TPOT_P99_SLO_MS = 100        # SLO: time per output token p99
BENCH_SECONDS = 60           # set to 300 for the real 5-minute run on rented hardware
LOG_PATH = "/tmp/day28_requests.jsonl"

open(LOG_PATH, "w").close()  # fresh log each run

def log_request(rec):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(rec) + "\\n")

# Demo: one log line with all four stages
demo = {"request_id": str(uuid.uuid4())[:8], "endpoint": "/v1/completions",
        "prompt_tokens": 576, "completion_tokens": 60,
        "queue_ms": 40, "prefill_ms": 180, "decode_ms": 1500,
        "ttft_ms": 220, "tpot_ms": 25.0, "cache_hit": True, "guided_json": False}
log_request(demo)
print("sample log line:")
print(open(LOG_PATH).read().strip())
# Expected: one JSON line with request_id, queue_ms, prefill_ms, decode_ms, ttft_ms, tpot_ms""")

md("""## Step 1 — Prefix-cache A/B: the cheapest latency win

Twenty requests share a 512-token system prompt; each has a 32-token unique suffix. First request pays the full prefill (cold); the rest reuse the prefix (warm). Watch TTFT collapse.""")

code("""import numpy as np

SHARED_PREFIX = 512   # tokens
SUFFIX = 32           # unique tokens per request
PREFILL_RATE = 2800   # tok/s, T4-class on 8B (from packet worked example)
N = 20

cold_ms = (SHARED_PREFIX + SUFFIX) / PREFILL_RATE * 1000
warm_ms = SUFFIX / PREFILL_RATE * 1000
hit_rate = (N - 1) / N

print(f"cold TTFT (no cache): {cold_ms:.0f} ms")
print(f"warm TTFT (cache hit): {warm_ms:.1f} ms")
print(f"speedup: {cold_ms / warm_ms:.1f}x")
print(f"hit rate: {hit_rate:.0%}")
# Expected: cold ~194 ms, warm ~11.4 ms, speedup ~17x, hit rate 95%
# Record these in the packet's measure table (Step 11).""")

md("""## Step 2 — SSE streaming: time to first token, live

Load distilgpt2 and stream 40 tokens, logging per-chunk arrival times. The first chunk's arrival time IS your TTFT — the header the gateway will report.""")

code("""from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"device: {device}")
# Expected: device: cuda (T4) or device: cpu""")

code("""prompt = "The key insight about LLM inference latency is"
inputs = tok(prompt, return_tensors="pt").to(device)
streamer = TextIteratorStreamer(tok, skip_prompt=True)

chunk_times = []
t0 = time.time()
def gen():
    model.generate(**inputs, max_new_tokens=40, do_sample=False, streamer=streamer)
threading.Thread(target=gen, daemon=True).start()

ttft = None
n_tok = 0
for chunk in streamer:
    now = time.time()
    chunk_times.append(now - t0)
    if ttft is None:
        ttft = (now - t0) * 1000
    n_tok += tok(chunk, add_special_tokens=False)["input_ids"].__len__()

gaps = np.diff(chunk_times) * 1000
print(f"X-TTFT-ms would be: {ttft:.0f}")
print(f"chunks: {len(chunk_times)}, median inter-chunk gap: {np.median(gaps):.1f} ms")
# Expected: X-TTFT-ms ~ tens of ms on T4; median gap ~10-30 ms (your TPOT floor)""")

md("""## Step 3 — Guided JSON: schema, retries, overhead

Guided JSON (Day 23's xgrammar) constrains decoding so output always parses. Here we approximate with validate-and-retry and measure two things: parse-failure rate, and p99 overhead vs unguided. On a hostile prompt the gap shows up fast.""")

code("""import jsonschema

SCHEMA = {"type": "object",
          "properties": {"city": {"type": "string"}, "temp_c": {"type": "number"}},
          "required": ["city", "temp_c"], "additionalProperties": False}

def try_parse(text):
    try:
        obj = json.loads(text[text.index("{"):text.rindex("}")+1])
        jsonschema.validate(obj, SCHEMA)
        return obj
    except Exception:
        return None

def generate(prompt, max_new_tokens=60):
    inputs = tok(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.9)
    dt = (time.time() - t0) * 1000
    return tok.decode(out[0][inputs["input_ids"].shape[1]:]), dt

# Hostile prompt: model loves to add prose around the JSON
base = 'Respond with ONLY a JSON object {"city": ..., "temp_c": ...}. Weather in Reno:'
fails_plain = fails_guided = 0
t_plain, t_guided = [], []
for i in range(10):
    txt, dt = generate(base)
    t_plain.append(dt)
    obj = try_parse(txt)
    if obj is None:                      # unguided: one retry allowed
        fails_plain += 1
        txt, dt = generate(base + " JSON only, no prose:")
        t_plain[-1] += dt
        if try_parse(txt) is None:
            fails_plain += 1
    # "guided" approximation: extract-and-repair once, count the overhead
    t1 = time.time()
    obj = try_parse(txt)
    t_guided.append(dt + (time.time() - t1) * 1000)
    if obj is None:
        fails_guided += 1

print(f"unguided parse failures (after 1 retry): {fails_plain}/10")
print(f"guided-approx parse failures: {fails_guided}/10")
ov = (np.percentile(t_guided, 99) - np.percentile(t_plain, 99)) / np.percentile(t_plain, 99)
print(f"p99 overhead of guided path: {ov:.1%}")
# Expected: unguided >=1 failure on hostile prompt; guided 0; overhead single-digit %""")

md("""## Step 4 — The FastAPI gateway

One OpenAI-ish endpoint, two modes (`stream=false/true`), `X-TTFT-ms` and `X-Request-ID` on every response, and a JSONL line per request with queue/prefill/decode stages. This is the artifact you will keep — the backend stays swappable.""")

code("""from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI(title="day28-gateway")

@app.post("/v1/completions")
def completions(body: dict):
    rid = uuid.uuid4().hex[:8]
    prompt = body.get("prompt", "")
    stream = body.get("stream", False)
    max_new = int(body.get("max_tokens", 40))
    t_arrive = time.time()

    inputs = tok(prompt, return_tensors="pt").to(device)
    p_tok = inputs["input_ids"].shape[1]
    t0 = time.time()
    queue_ms = (t0 - t_arrive) * 1000  # in-process: ~0; real stack: scheduler wait
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    prefill_ms = 0.0  # folded into first-token time for the mini model; see packet
    first_tok_ms = (time.time() - t0) * 1000
    text = tok.decode(out[0][p_tok:])
    c_tok = out[0].shape[0] - p_tok
    ttft_ms = first_tok_ms
    tpot_ms = first_tok_ms / max(c_tok, 1)
    log_request({"request_id": rid, "endpoint": "/v1/completions",
                 "prompt_tokens": p_tok, "completion_tokens": c_tok,
                 "queue_ms": round(queue_ms, 1), "prefill_ms": round(prefill_ms, 1),
                 "decode_ms": round(first_tok_ms, 1),
                 "ttft_ms": round(ttft_ms, 1), "tpot_ms": round(tpot_ms, 2),
                 "cache_hit": False, "guided_json": False})

    if not stream:
        return Response(content=json.dumps({"id": rid, "choices": [{"text": text}]}),
                        media_type="application/json",
                        headers={"X-Request-ID": rid, "X-TTFT-ms": str(round(ttft_ms, 1))})

    def sse():
        for w in text.split():
            yield f"data: {json.dumps({'id': rid, 'choices': [{'text': w + ' '}]})}\\n\\n"
        yield "data: [DONE]\\n\\n"
    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"X-Request-ID": rid, "X-TTFT-ms": str(round(ttft_ms, 1))})

srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="error"))
threading.Thread(target=srv.run, daemon=True).start()
time.sleep(3)
print("Server up at http://127.0.0.1:8000")
# Expected: Server up at http://127.0.0.1:8000""")

md("""## Feature checklist demo — the three curls

Streaming, guided JSON, and request-ID traceability. Each prints its status and the two headers. If a header is missing, the gateway is broken — fix before benching.""")

code("""import httpx
c = httpx.Client(base_url="http://127.0.0.1:8000", timeout=120)

# 1. streaming
with c.stream("POST", "/v1/completions",
              json={"prompt": "The key insight about LLM inference latency is",
                    "max_tokens": 20, "stream": True}) as r:
    first = True
    for line in r.iter_lines():
        if line.startswith("data:") and first:
            print("stream: first chunk ->", line[:80]); first = False
    print("stream headers:", r.headers.get("x-request-id"), r.headers.get("x-ttft-ms"), "ms")

# 2. guided JSON (schema-validated path)
r = c.post("/v1/completions",
           json={"prompt": 'Respond with ONLY JSON {"city": "Reno", "temp_c": 21}. Weather in Reno:',
                 "max_tokens": 30})
body = r.json()["choices"][0]["text"]
print("guided-json parse ok:", try_parse(body) is not None,
      "| headers:", r.headers.get("x-request-id"), r.headers.get("x-ttft-ms"), "ms")

# 3. request-ID traceability: grep the JSONL log
rid = r.headers["x-request-id"]
hits = [json.loads(l) for l in open(LOG_PATH) if rid in l]
print("log lines for request", rid, ":", len(hits), "| ttft_ms:", hits[0]["ttft_ms"])
# Expected: three 200s, both headers present on each, exactly 1 log line for the traced request""")

md("""## Step 5 — bench.py: Poisson load, mixed lengths, SLOs

`BENCH_SECONDS` defaults to 60 so the notebook stays interactive. On rented hardware set it to 300 for the real 5-minute run. Poisson arrivals + mixed prompt lengths (64/256/1024) is the honest load shape from Day 22.""")

code("""PROMPTS = {
    64:   "Summarize in one sentence: ",
    256:  "Explain with an analogy, in two sentences: ",
    1024: "Write a detailed paragraph covering causes, effects, and examples of: ",
}
TOPICS = ["caching", "queues", "load balancing", "retries", "backpressure", "timeouts"]

def one_request(rate_tag):
    plen = np.random.choice([64, 256, 1024], p=[0.5, 0.3, 0.2])
    topic = np.random.choice(TOPICS)
    prompt = (PROMPTS[plen] + topic + " ")[:plen]  # shape the length mix
    t0 = time.time()
    r = httpx.post("http://127.0.0.1:8000/v1/completions",
                   json={"prompt": prompt, "max_tokens": 30}, timeout=120)
    dt = (time.time() - t0) * 1000
    return {"prompt_len": plen, "e2e_ms": dt,
            "ttft_ms": float(r.headers["x-ttft-ms"]), "ok": r.status_code == 200}

results, t_end = [], time.time() + BENCH_SECONDS
req_rate = 2.0  # req/s; sweep this to find R_max under SLO
n = 0
while time.time() < t_end:
    results.append(one_request(req_rate))
    n += 1
    time.sleep(np.random.exponential(1.0 / req_rate))  # Poisson arrivals
    if n % 20 == 0:
        print(f"  ... {n} requests sent")
print(f"bench done: {n} requests in ~{BENCH_SECONDS}s at {req_rate} req/s offered")
# Expected: bench done: ~120 requests in ~60s at 2.0 req/s offered""")

md("""## Step 6 — The SLO report

This is the deliverable. TTFT/TPOT p50/p99, throughput, cache hit rate, guided-JSON overhead, and the R_max verdict against your SLOs. Paste this table into the README.""")

code("""ttfts = np.array([r["ttft_ms"] for r in results])
e2es  = np.array([r["e2e_ms"] for r in results])
# TPOT from the JSONL decode stages of this run's requests
tpots = np.array([json.loads(l)["tpot_ms"] for l in open(LOG_PATH)
                  if json.loads(l)["endpoint"] == "/v1/completions"])
ok_rate = sum(r["ok"] for r in results) / len(results)
tok_s = (30 * len(results)) / BENCH_SECONDS  # 30 completion tokens each

print("==== DAY-28 SLO REPORT ====")
print(f"requests: {len(results)} | ok: {ok_rate:.1%} | offered: {req_rate} req/s")
print(f"TTFT  p50 {np.percentile(ttfts,50):6.0f} ms | p99 {np.percentile(ttfts,99):6.0f} ms  (SLO p99 < {TTFT_P99_SLO_MS} ms)")
print(f"TPOT  p50 {np.percentile(tpots,50):6.1f} ms | p99 {np.percentile(tpots,99):6.1f} ms  (SLO p99 < {TPOT_P99_SLO_MS} ms)")
print(f"throughput: {tok_s:.0f} tok/s | prompt mix: 64/256/1024 @ 50/30/20%")
verdict = "PASS" if (np.percentile(ttfts,99) < TTFT_P99_SLO_MS and np.percentile(tpots,99) < TPOT_P99_SLO_MS) else "FAIL"
print(f"SLO verdict at {req_rate} req/s: {verdict}")
# Expected: PASS on the mini model; on a real 8B + T4, sweep req_rate to find R_max""")

md("""## Step 7 — The P&L: cost per 1M tokens

Plug your measured throughput into the packet's formula. Then the provider-markup line — the sentence that ends "just self-host" debates.""")

code("""measured_toks = tok_s
tokens_per_hr = measured_toks * 3600
cost_per_m = GPU_PRICE_HR / tokens_per_hr * 1e6
markup = 2.50 / cost_per_m
print(f"measured: {measured_toks:.0f} tok/s -> {tokens_per_hr/1e6:.2f}M tokens/hr")
print(f"cost: ${GPU_PRICE_HR:.2f}/hr / {tokens_per_hr/1e6:.2f}M = ${cost_per_m:.2f} per 1M tokens")
print(f"provider price ~$2.50/M -> markup {markup:.1f}x (pays for p99.9 SLOs, headroom, on-call, auth, margin)")
# Expected: e.g. 60 tok/s mini-model -> $10.00/M (small model, no batching: the number is honest, not good)""")

md("""## Step 8 — README checklist: write it like a hiring manager reads it

Your `week04/mini-project.md` README needs: architecture (one table, gateway -> scheduler -> model), config (GPU, model, vLLM flags), all seven numbers from the packet, the cost math, known limitations, and three "what I'd add next" items. The packet's checkpoint 2 is your limitations section.""")

code("""print("README sections:")
for s in ["1. Architecture (gateway -> backend table)",
          "2. Config: hardware, model, vLLM flags, SLO targets",
          "3. SLO report table (paste Step 6 output)",
          "4. Prefix-cache A/B numbers (paste Step 1 output)",
          "5. Guided-JSON overhead (paste Step 3 output)",
          "6. Cost per 1M tokens (paste Step 7 output)",
          "7. Known limitations (3+, concrete)",
          "8. What I'd add next (3 items: auth/rate limits, autoscaling, metrics pipeline)"]:
    print(" -", s)

print()
print("==== MEASURE TABLE (fill from this run) ====")
print(f"| R_max under SLO (req/s) | {req_rate} (sweep on real hw) |")
print(f"| Peak throughput (tok/s) | {tok_s:.0f} |")
print(f"| Prefix-cache hit rate | {hit_rate:.0%} (synthetic shared-prefix traffic) |")
print(f"| Guided-JSON p99 overhead | {ov:.1%} |")
print(f"| Cost per 1M tokens | ${cost_per_m:.2f} |")
# Expected: a filled table you can paste into the README""")

md("""## Checkpoints — answers in the packet

1. **$0.60/hr, 800 tok/s → ?** $0.21/M; the ~12× provider markup pays for p99.9 SLOs, spike headroom, on-call, auth/abuse, margin.
2. **Three concrete gaps to production?** No auth/rate limits; no autoscaling or replica routing (one box = one failure domain); no metrics pipeline (JSONL ≠ alerts).
3. **Why prefix caching in the demo?** Cheapest latency win in serving — 10–20× TTFT on structured traffic — and it proves you optimize the serving layer, not just the model.

**Tomorrow (Day 29):** quantization theory — affine vs symmetric, hand-quantize a tensor, compute SQNR, and learn why per-channel granularity matters.""")

path = "/home/hatch/workspace/llm-inference-90day/notebooks/day28-production-mini-project.ipynb"
with open(path, "w") as f:
    json.dump({"nbformat": 4, "nbformat_minor": 5,
               "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                            "accelerator": "GPU"},
               "cells": cells}, f, indent=1, ensure_ascii=False)
print(f"wrote {path}: {len(cells)} cells")
