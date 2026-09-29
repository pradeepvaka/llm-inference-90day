"""bench.py -- minimal LLM serving benchmark harness (Day 21).

Streams SSE completions, records per-token timestamps, computes
TTFT / TPOT / ITL-p99 / E2E per request, writes JSONL, prints p50/p99.
"""
import argparse, concurrent.futures, json, statistics, sys, time
import requests

def percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def one_request(url, prompt_len, out_len, timeout=120):
    prompt = "token " * prompt_len
    t0 = time.time()
    token_times, n_tok = [], 0
    r = requests.post(url, json={"prompt": prompt, "max_tokens": out_len,
                                "stream": True}, stream=True, timeout=timeout)
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        token_times.append(time.time())
        n_tok += 1
    t_end = time.time()
    gaps = [b - a for a, b in zip(token_times, token_times[1:])]
    return {
        "ttft_ms": (token_times[0] - t0) * 1000 if token_times else None,
        "tpot_ms": statistics.fmean(gaps) * 1000 if gaps else None,
        "itl_p99_ms": percentile(gaps, 99) * 1000 if gaps else None,
        "e2e_ms": (t_end - t0) * 1000,
        "n_tokens": n_tok,
        "prompt_len": prompt_len, "out_len": out_len,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n-requests", type=int, default=20)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--out-len", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--jsonl", default="bench.jsonl")
    a = ap.parse_args()

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = [ex.submit(one_request, a.url, a.prompt_len, a.out_len)
                for _ in range(a.n_requests)]
        # warmup request excluded from stats (kernel autotune / cache warmup)
        warm = futs.pop(0).result()
        recs = [f.result() for f in futs]

    wall = max(r["e2e_ms"] for r in recs) / 1000  # approx wall for concurrency>=1
    with open(a.jsonl, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    tok = sum(r["n_tokens"] for r in recs)
    print(f"requests={len(recs)} (+1 warmup)  total_tokens={tok}")
    for k in ("ttft_ms", "tpot_ms", "itl_p99_ms", "e2e_ms"):
        xs = [r[k] for r in recs if r[k] is not None]
        print(f"{k:10s} p50={percentile(xs,50):8.1f}  p99={percentile(xs,99):8.1f}")
    print(f"throughput ~= {tok / wall:.0f} tok/s (output tokens / max e2e)")

if __name__ == "__main__":
    main()