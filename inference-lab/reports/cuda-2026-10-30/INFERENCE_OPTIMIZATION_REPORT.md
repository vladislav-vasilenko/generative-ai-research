# Inference optimization report

Status: **MEASURED — see scope and repeat counts**.

Model: `Qwen/Qwen2.5-0.5B-Instruct`; FP16; CUDA_VISIBLE_DEVICES=0; TP=1.

Requests per cell: 32; planned independent repeats: 3.

Synthetic closed-loop chat load, exact templated inputs 128/512/2048 tokens and forced outputs 64/128/256. Identical model snapshot and workload files are reused by both engines. Environments are separate; this compares packaged serving stacks, including their PyTorch/tokenizer versions.

Servers restart for every arm/repeat. Startup, semantic smoke and warmup are excluded. Tables show the median across per-repeat metrics, not a pooled percentile. Tail percentiles are descriptive, without a production p99 or production-scale claim.

Warmup: 4 separate 512-input/64-output requests per point. CUDA graphs are disabled.

The pilot does not establish exhaustive shape/JIT warmup; results are exploratory eager serving measurements.

| Engine | Arm | Workload | C | Repeats | Output tok/s | TTFT p50 ms | TPOT p50 ms | E2E p95 ms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sglang | baseline | long | 1 | 3 | 29.71 | 1828.60 | 26.65 | 8703.58 |
| sglang | baseline | long | 2 | 3 | 48.48 | 2733.47 | 31.23 | 10801.37 |
| sglang | baseline | long | 4 | 3 | 71.95 | 5436.25 | 34.46 | 14339.41 |
| sglang | baseline | long | 8 | 3 | 95.25 | 9043.48 | 48.83 | 21574.84 |
| sglang | baseline | long | 16 | 3 | 113.45 | 16264.05 | 77.79 | 36128.02 |
| sglang | baseline | mixed | 8 | 3 | 98.26 | 3650.07 | 56.25 | 20535.85 |
| sglang | baseline | mixed | 16 | 3 | 118.41 | 4576.00 | 84.77 | 34533.25 |
| sglang | baseline | shared | 8 | 3 | 95.20 | 9043.96 | 48.78 | 21563.84 |
| sglang | baseline | short | 1 | 3 | 37.21 | 60.75 | 26.23 | 1762.85 |
| sglang | baseline | short | 2 | 3 | 71.92 | 78.40 | 26.95 | 1826.54 |
| sglang | baseline | short | 4 | 3 | 138.88 | 95.84 | 27.89 | 1912.98 |
| sglang | baseline | short | 8 | 3 | 272.69 | 155.53 | 27.32 | 1892.87 |
| sglang | baseline | short | 16 | 3 | 494.11 | 289.84 | 28.35 | 2106.94 |
| sglang | baseline | standard | 1 | 3 | 36.26 | 160.83 | 26.50 | 3585.31 |
| sglang | baseline | standard | 2 | 3 | 68.38 | 227.27 | 27.67 | 3820.36 |
| sglang | baseline | standard | 4 | 3 | 126.86 | 576.82 | 27.68 | 4103.56 |
| sglang | baseline | standard | 8 | 3 | 221.26 | 1126.63 | 27.61 | 4676.93 |
| sglang | baseline | standard | 16 | 3 | 355.97 | 1527.76 | 33.85 | 5758.01 |
| sglang | chunked4096 | mixed | 8 | 3 | 98.27 | 3245.84 | 50.40 | 20630.17 |
| sglang | chunked4096 | mixed | 16 | 3 | 119.12 | 6120.74 | 80.60 | 34198.90 |
| sglang | chunked512 | mixed | 8 | 3 | 103.58 | 1642.39 | 60.26 | 19352.01 |
| sglang | chunked512 | mixed | 16 | 3 | 127.02 | 4077.68 | 86.55 | 31768.23 |
| sglang | prefix | shared | 8 | 3 | 219.98 | 805.71 | 28.15 | 13392.33 |
| vllm | baseline | long | 1 | 3 | 37.38 | 443.20 | 25.06 | 6936.82 |
| vllm | baseline | long | 2 | 3 | 70.54 | 579.92 | 26.17 | 7378.35 |
| vllm | baseline | long | 4 | 3 | 127.67 | 896.78 | 28.01 | 8126.28 |
| vllm | baseline | long | 8 | 3 | 218.52 | 1069.58 | 32.43 | 10594.56 |
| vllm | baseline | long | 16 | 3 | 332.52 | 1080.32 | 43.49 | 16240.13 |
| vllm | baseline | mixed | 8 | 3 | 192.48 | 383.65 | 30.88 | 8971.51 |
| vllm | baseline | mixed | 16 | 3 | 274.98 | 919.66 | 36.89 | 12187.39 |
| vllm | baseline | shared | 8 | 3 | 220.95 | 1063.98 | 32.14 | 10502.73 |
| vllm | baseline | short | 1 | 3 | 39.33 | 54.81 | 24.93 | 1675.96 |
| vllm | baseline | short | 2 | 3 | 77.15 | 70.02 | 25.17 | 1715.52 |
| vllm | baseline | short | 4 | 3 | 151.50 | 83.70 | 25.60 | 1717.26 |
| vllm | baseline | short | 8 | 3 | 299.34 | 92.48 | 25.68 | 1720.02 |
| vllm | baseline | short | 16 | 3 | 570.30 | 138.50 | 26.02 | 1806.19 |
| vllm | baseline | standard | 1 | 3 | 39.32 | 83.89 | 24.93 | 3339.44 |
| vllm | baseline | standard | 2 | 3 | 76.58 | 90.66 | 25.54 | 3372.61 |
| vllm | baseline | standard | 4 | 3 | 148.44 | 193.72 | 25.70 | 3508.51 |
| vllm | baseline | standard | 8 | 3 | 282.04 | 321.79 | 25.91 | 3691.21 |
| vllm | baseline | standard | 16 | 3 | 522.19 | 420.33 | 27.49 | 3986.81 |
| vllm | chunked4096 | mixed | 8 | 3 | 191.24 | 381.72 | 30.95 | 9003.31 |
| vllm | chunked4096 | mixed | 16 | 3 | 273.52 | 1365.47 | 32.22 | 11850.18 |
| vllm | chunked512 | mixed | 8 | 3 | 196.55 | 417.38 | 30.30 | 8857.58 |
| vllm | chunked512 | mixed | 16 | 3 | 286.31 | 889.44 | 36.50 | 11555.85 |
| vllm | prefix | shared | 8 | 3 | 299.48 | 109.75 | 26.03 | 7080.66 |

## Paired ablations

Relative changes use matched repeats; a negative latency change means lower latency.

- `vllm` baseline → chunked4096, mixed C=8: output throughput -1.4%, E2E p95 +1.3%, TTFT p50 -1.8%; 3 paired repeats.
- `vllm` chunked4096 → chunked512, mixed C=8: output throughput +1.9%, E2E p95 -1.6%, TTFT p50 +11.4%; 3 paired repeats.
- `vllm` baseline → prefix, shared C=8: output throughput +35.4%, E2E p95 -32.6%, TTFT p50 -89.7%; 3 paired repeats.
- `sglang` baseline → chunked4096, mixed C=8: output throughput -0.0%, E2E p95 -0.2%, TTFT p50 -11.1%; 3 paired repeats.
- `sglang` chunked4096 → chunked512, mixed C=8: output throughput +5.5%, E2E p95 -6.5%, TTFT p50 -49.4%; 3 paired repeats.
- `sglang` baseline → prefix, shared C=8: output throughput +133.6%, E2E p95 -38.6%, TTFT p50 -91.1%; 3 paired repeats.

## Measurement boundaries

Client TTFT = first non-empty text SSE delta. TPOT is a client estimate from first/last text event and actual server completion_tokens. Token ITL is unavailable over text-only SSE; raw inter-chunk intervals are stored separately. Errors contribute to wall time and error rate, but not successful token throughput. SLO goodput uses the fixed campaign thresholds.

VRAM is sampled per physical GPU by nvidia-smi; allocated memory includes reserved engine pools. RAM RSS sums may double-count shared pages. Raw Prometheus snapshots are included; no peak KV occupancy is inferred from endpoint snapshots.

Correctness gates cover semantic smoke, API/stream completion and actual prompt/output token counts. These synthetic load tests do not establish general answer quality or lossless decoded-output parity.

Continuous batching remains active. Concurrency and heterogeneous lengths test scheduler behavior; no scheduling-policy speedup is claimed. Speculative decoding and TP=2 are NOT EXECUTED in this first campaign. BF16/FP8 are not tested on T4. Diffusion/video and external workers are separate tracks.

Recorded failures in selected measurement campaigns: 0. See `errors.json`, per-engine `install.log`, `freeze.txt`, `environment.json`, and per-arm `server.log`.
Earlier failed attempts retained in provenance: 12. These are not successful timed requests.

## Separate-session provenance

vLLM and SGLang were measured in separate Kaggle sessions. Model/weight hashes, tokenizer/chat template, workloads, transport code and campaign parameters were validated before merging. Within-engine ablations use matched independent repeats. Cross-engine comparisons describe complete stacks on separate T4 allocations, not paired runs on one physical GPU. Original sources/configurations and earlier failures are retained under provenance/. SGLang's JIT link fix exposes the existing driver library through a scratch link, without changing CUDA kernels.
