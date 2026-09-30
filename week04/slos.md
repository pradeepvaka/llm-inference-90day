# SLOs -- mock 70B-class server (Day 22 load test, seed 7)
## SLOs
- TTFT p99 < 1.0 s, TPOT p99 < 100 ms, availability 99.9% (43.2 min/mo budget)
## R_max
- 3 req/s at 100% attainment (offered 5.8 Erlangs, rho 0.49 on 12 slots)
- goodput at R_max: 267 tok/s
## What breaks first
- Queue wait (TTFT p99) under Poisson bursts: rho=0.65 already bites (R=4: 92.8%)
- KV memory binds before compute when the budget is tight: at R=3, shrinking
  KV 8 GB -> 2 GB drops attainment 100% -> 85.8% with 301 newest-first preemptions
- Raw throughput is a liar past the knee: R=8 does 772 tok/s, goodput 0
## Notes
- Simulator: event-driven, Poisson arrivals, mixed lengths, 12 slots,
  prefill 3000 tok/s, TPOT 20 ms base, KV 320 KB/tok, swap-resume preemption.
- Re-run on a T4 with vLLM: python week03/bench.py --arrival poisson --rate R ...
