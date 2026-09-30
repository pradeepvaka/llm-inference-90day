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

def mixed_lengths(rng):
    """Day 22 mixed workload: 80/20 prompts, 70/30 outputs."""
    pl = int(rng.uniform(128, 512) if rng.random() < 0.8 else rng.uniform(1000, 2000))
    ol = int(rng.uniform(20, 80) if rng.random() < 0.7 else rng.uniform(100, 300))
    return pl, ol


def scheduled_request(url, start_at, prompt_len, out_len, timeout=180):
    """Sleep until the scheduled arrival instant, then run one request."""
    delay = start_at - time.time()
    if delay > 0:
        time.sleep(delay)
    return one_request(url, prompt_len, out_len, timeout)


def poisson_main(a):
    """Day 22: Poisson arrivals at --rate req/s for --duration s, mixed lengths."""
    import random
    rng = random.Random(22)
    sched, t = [], 0.0
    while t < a.duration:
        t += rng.expovariate(a.rate)
        if t < a.duration:
            pl, ol = mixed_lengths(rng) if a.mixed else (a.prompt_len, a.out_len)
            sched.append((t, pl, ol))
    print(f"poisson: rate={a.rate}/s duration={a.duration}s -> {len(sched)} arrivals")
    t_start = time.time()
    starts = [t_start + dt for dt, _, _ in sched]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(32, int(a.rate * 4))) as ex:
        futs = [ex.submit(scheduled_request, a.url, st, pl, ol)
                for st, (dt, pl, ol) in zip(starts, sched)]
        recs = [f.result() for f in futs]
    wall = time.time() - t_start
    # warmup: drop the first arrivals (JIT/cache warmup)
    n_warm = max(1, len(recs) // 20)
    recs = recs[n_warm:]
    with open(a.jsonl, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    tok = sum(r["n_tokens"] for r in recs)
    ok = [r for r in recs
          if r["ttft_ms"] is not None and r["itl_p99_ms"] is not None
          and r["ttft_ms"] < 1000 and r["itl_p99_ms"] < 100]
    good_tok = sum(r["n_tokens"] for r in ok)
    print(f"requests={len(recs)} (+{n_warm} warmup)  total_tokens={tok}  wall_s={wall:.1f}")
    for k in ("ttft_ms", "tpot_ms", "itl_p99_ms", "e2e_ms"):
        xs = [r[k] for r in recs if r[k] is not None]
        print(f"{k:10s} p50={percentile(xs,50):8.1f}  p99={percentile(xs,99):8.1f}")
    print(f"SLO attainment (TTFT<1s & ITL p99<100ms): {len(ok)}/{len(recs)} "
          f"= {100*len(ok)/max(1,len(recs)):.1f}%")
    print(f"throughput ~= {tok / wall:.0f} tok/s   "
          f"goodput ~= {good_tok / wall:.0f} tok/s (output tokens / wall clock)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n-requests", type=int, default=20)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--out-len", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--jsonl", default="bench.jsonl")
    # Day 22: realistic load
    ap.add_argument("--arrival", choices=["fixed", "poisson"], default="fixed")
    ap.add_argument("--rate", type=float, default=2.0, help="req/s for poisson arrival")
    ap.add_argument("--duration", type=float, default=60.0, help="s for poisson arrival")
    ap.add_argument("--mixed", action="store_true",
                    help="Day-22 mixed prompt/output length distributions")
    a = ap.parse_args()

    if a.arrival == "poisson":
        return poisson_main(a)

    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = [ex.submit(one_request, a.url, a.prompt_len, a.out_len)
                for _ in range(a.n_requests)]
        # warmup request excluded from stats (kernel autotune / cache warmup)
        warm = futs.pop(0).result()
        recs = [f.result() for f in futs]
    wall = time.time() - t_start  # true wall clock: submit of first -> completion of last
    with open(a.jsonl, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    tok = sum(r["n_tokens"] for r in recs)
    print(f"requests={len(recs)} (+1 warmup)  total_tokens={tok}  wall_s={wall:.1f}")
    for k in ("ttft_ms", "tpot_ms", "itl_p99_ms", "e2e_ms"):
        xs = [r[k] for r in recs if r[k] is not None]
        print(f"{k:10s} p50={percentile(xs,50):8.1f}  p99={percentile(xs,99):8.1f}")
    print(f"throughput ~= {tok / wall:.0f} tok/s (output tokens / wall clock)")

if __name__ == "__main__":
    main()
