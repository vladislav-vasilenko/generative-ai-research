# Scheduling / heterogeneous load

`scheduling.csv` separates short and long requests. Homogeneous short latency uses the same request IDs as the short half of the mixed workload, with the same 128-token input and 64-token output. This measures the effect of competing long requests under closed-loop load. It does not isolate scheduler CPU overhead from prefill GPU work or prove a specific head-of-line mechanism. No scheduling policy is changed. Each short bucket has only 16 requests per pilot repeat; tail estimates are exploratory.

## Paired chunking effects on short-request TTFT

- vllm, baseline → chunked4096, C=8: short-request TTFT p50 -0.4%, 3 paired repeats.
- vllm, baseline → chunked4096, C=16: short-request TTFT p50 +29.7%, 3 paired repeats.
- vllm, chunked4096 → chunked512, C=8: short-request TTFT p50 -0.1%, 3 paired repeats.
- vllm, chunked4096 → chunked512, C=16: short-request TTFT p50 -21.6%, 3 paired repeats.
- sglang, baseline → chunked4096, C=8: short-request TTFT p50 +0.7%, 3 paired repeats.
- sglang, baseline → chunked4096, C=16: short-request TTFT p50 -9.7%, 3 paired repeats.
- sglang, chunked4096 → chunked512, C=8: short-request TTFT p50 -94.5%, 3 paired repeats.
- sglang, chunked4096 → chunked512, C=16: short-request TTFT p50 -45.9%, 3 paired repeats.
