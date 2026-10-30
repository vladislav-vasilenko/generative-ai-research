# Inference optimization report

Status: **MEASURED — see scope and repeat counts**.

Model: `Qwen/Qwen2.5-0.5B-Instruct`; FP16; CUDA_VISIBLE_DEVICES=0; TP=1.

Requests per cell: 64; planned independent repeats: 3.

Synthetic closed-loop chat load, 2048-token inputs and forced 256-token outputs. Identical model snapshot and workload files are reused by both engines. Environments are separate; this compares packaged serving stacks, including their PyTorch/tokenizer versions.

Servers restart for every arm/repeat. Startup, semantic smoke and warmup are excluded. Tables show the median across per-repeat metrics, not a pooled percentile. Tail percentiles are descriptive, without a production p99 or production-scale claim.

Warmup: 16 separate 2048-input/256-output requests per point. CUDA graphs are disabled.

Focused validation: only shared-prefix 2048→256 at concurrency=8 is measured. Warmup matches these lengths and covers two full concurrency waves. Warmup IDs occur early in the prompt and differ from measured prompts, leaving the measured shared context unprimed apart from the common chat header. Both engines run sequentially on one physical GPU. 64 requests give eight concurrency waves; no production p99 claim is made.

| Engine | Arm | Workload | C | Repeats | Output tok/s | TTFT p50 ms | TPOT p50 ms | E2E p95 ms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sglang | baseline | shared | 8 | 3 | 95.06 | 9053.85 | 49.08 | 21699.98 |
| sglang | prefix | shared | 8 | 3 | 232.39 | 808.74 | 28.24 | 13581.61 |
| vllm | baseline | shared | 8 | 3 | 211.24 | 1106.05 | 33.69 | 10373.35 |
| vllm | prefix | shared | 8 | 3 | 291.85 | 101.74 | 26.98 | 7270.35 |

## Paired ablations

Relative changes use matched repeats; a negative latency change means lower latency.

- `vllm` baseline → prefix, shared C=8: output throughput +38.2%, E2E p95 -29.2%, TTFT p50 -90.8%; 3 paired repeats.
- `sglang` baseline → prefix, shared C=8: output throughput +145.1%, E2E p95 -37.6%, TTFT p50 -91.1%; 3 paired repeats.

## Measurement boundaries

Client TTFT = first non-empty text SSE delta. TPOT is a client estimate from first/last text event and actual server completion_tokens. Token ITL is unavailable over text-only SSE; raw inter-chunk intervals are stored separately. Errors contribute to wall time and error rate, but not successful token throughput. SLO goodput uses the fixed campaign thresholds.

VRAM is sampled per physical GPU by nvidia-smi; allocated memory includes reserved engine pools. RAM RSS sums may double-count shared pages. Raw Prometheus snapshots are included; no peak KV occupancy is inferred from endpoint snapshots.

Correctness gates cover semantic smoke, API/stream completion and actual prompt/output token counts. These synthetic load tests do not establish general answer quality or lossless decoded-output parity.

Continuous batching remains active. Concurrency and heterogeneous lengths test scheduler behavior; no scheduling-policy speedup is claimed. Speculative decoding and TP=2 are NOT EXECUTED in this first campaign. BF16/FP8 are not tested on T4. Diffusion/video and external workers are separate tracks.

Recorded failures in selected measurement campaigns: 0. See `errors.json`, per-engine `install.log`, `freeze.txt`, `environment.json`, and per-arm `server.log`.
