"""Closed-loop OpenAI chat benchmark; stdlib transport, no inference dependencies.

Client TTFT is the first non-empty text delta. SSE chunk gaps are NOT token ITL.
Only server-reported usage is used for token throughput. No retry of timed requests.
"""
import argparse
import concurrent.futures
import http.client
import json
import math
import queue
import random
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def percentile(values, fraction):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def connection_for(base_url):
    url = urllib.parse.urlsplit(base_url)
    if url.scheme not in ("http", "https") or not url.hostname:
        raise ValueError("Expected an http(s) base URL")
    cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    return cls(url.hostname, url.port, timeout=180), url.path.rstrip("/")


def sse_events(response):
    """Handle multi-line SSE data and CRLF; HTTP framing is handled by http.client."""
    parts = []
    while True:
        line = response.readline()
        if not line:
            if parts:
                yield "\n".join(parts)
            return
        line = line.decode("utf-8").rstrip("\r\n")
        if not line:
            if parts:
                yield "\n".join(parts)
                parts = []
        elif line.startswith("data:"):
            parts.append(line[5:].lstrip(" "))


def perform_request(connection, api_path, item, model, *, stream=True, ignore_eos=True):
    payload = dict(model=model, messages=item["messages"], max_tokens=item["output_tokens"],
                   temperature=0, top_p=1, presence_penalty=0, frequency_penalty=0,
                   seed=42, stream=stream, ignore_eos=ignore_eos)
    if stream:
        payload["stream_options"] = {"include_usage": True}
    started = time.perf_counter()
    row = dict(id=item["id"], workload=item.get("workload"), success=False, text="",
               expected_prompt_tokens=item.get("prompt_tokens"),
               requested_output_tokens=item["output_tokens"], text_event_offsets_s=[],
               usage=None, finish_reason=None, error=None, itl_ms=None)
    try:
        connection.request("POST", api_path + "/chat/completions",
                           body=json.dumps(payload).encode(),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read(4096).decode(errors='replace')}")
        if stream:
            done = False
            for data in sse_events(response):
                if data.strip() == "[DONE]":
                    done = True
                    # Drain the HTTP body before reusing the persistent connection.
                    response.read()
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if event.get("usage"):
                    row["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    text_delta = delta.get("content")
                    if text_delta:
                        row["text"] += text_delta
                        row["text_event_offsets_s"].append(time.perf_counter() - started)
                    if choice.get("finish_reason"):
                        row["finish_reason"] = choice["finish_reason"]
            if not done:
                raise RuntimeError("Truncated SSE: missing [DONE]")
        else:
            event = json.loads(response.read())
            row["usage"] = event.get("usage")
            row["text"] = event["choices"][0]["message"].get("content") or ""
            row["finish_reason"] = event["choices"][0].get("finish_reason")
        if not row["text"].strip():
            raise RuntimeError("No generated text")
        usage = row["usage"] or {}
        for key in ("prompt_tokens", "completion_tokens"):
            if not isinstance(usage.get(key), int) or usage[key] <= 0:
                raise RuntimeError(f"Missing/invalid actual usage.{key}; no token estimate substituted")
        expected = item.get("prompt_tokens")
        if expected is not None and usage["prompt_tokens"] != expected:
            raise RuntimeError(f"Chat-template/tokenizer mismatch: expected {expected}, got {usage['prompt_tokens']}")
        if ignore_eos and usage["completion_tokens"] != item["output_tokens"]:
            raise RuntimeError("Forced-length request returned a different token count")
        row["success"] = True
    except (OSError, ValueError, KeyError, IndexError, RuntimeError, http.client.HTTPException) as error:
        row["error"] = f"{type(error).__name__}: {error}"
        connection.close()
    row["e2e_ms"] = (time.perf_counter() - started) * 1000
    offsets = row["text_event_offsets_s"]
    row["ttft_ms"] = offsets[0] * 1000 if offsets else None
    count = (row["usage"] or {}).get("completion_tokens", 0)
    row["tpot_ms"] = ((offsets[-1] - offsets[0]) * 1000 / (count - 1)
                      if count > 1 and len(offsets) > 1 else None)
    row["inter_chunk_ms"] = [(right - left) * 1000 for left, right in zip(offsets, offsets[1:])]
    row["text_chunks"] = len(offsets)
    return row


def smoke_server(base_url, model, output):
    with urllib.request.urlopen(base_url + "/models", timeout=10) as response:
        available = json.load(response)
    if model not in [item["id"] for item in available.get("data", [])]:
        raise RuntimeError("Served model name missing from /v1/models")
    item = dict(id="semantic-smoke", messages=[{"role": "user", "content":
                "What is the capital of France? Answer with the city name only."}], output_tokens=24)
    rows = []
    for stream in (False, True):
        connection, path = connection_for(base_url)
        try:
            row = perform_request(connection, path, item, model, stream=stream, ignore_eos=False)
            row["streaming"] = stream
            rows.append(row)
        finally:
            connection.close()
    write_json(output, rows)
    if not all(row["success"] and "paris" in row["text"].lower() for row in rows):
        raise RuntimeError(f"Semantic or streaming smoke failed; inspect {output}")


def request_batch(base_url, model, items, concurrency):
    pending = queue.Queue()
    for item in items:
        pending.put(item)
    rows = []
    lock = threading.Lock()

    def worker():
        connection, path = connection_for(base_url)
        try:
            while True:
                try:
                    item = pending.get_nowait()
                except queue.Empty:
                    return
                row = perform_request(connection, path, item, model)
                with lock:
                    rows.append(row)
        finally:
            connection.close()

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker) for _ in range(concurrency)]
        for future in futures:
            future.result()
    return rows, time.perf_counter() - started


def summarize_rows(rows, wall_seconds, slo):
    successful = [row for row in rows if row["success"]]
    prompt_tokens = sum(row["usage"]["prompt_tokens"] for row in successful)
    output_tokens = sum(row["usage"]["completion_tokens"] for row in successful)
    good = [row for row in successful if row["ttft_ms"] is not None
            and row["tpot_ms"] is not None and row["ttft_ms"] <= slo["ttft_ms"]
            and row["tpot_ms"] <= slo["tpot_ms"] and row["e2e_ms"] <= slo["e2e_ms"]]
    summary = dict(attempted=len(rows), successful=len(successful), wall_seconds=wall_seconds,
                   error_rate=(len(rows) - len(successful)) / len(rows) if rows else None,
                   request_throughput=len(successful) / wall_seconds,
                   output_tokens_per_second=output_tokens / wall_seconds,
                   total_tokens_per_second=(prompt_tokens + output_tokens) / wall_seconds,
                   goodput_requests_per_second=len(good) / wall_seconds,
                   actual_prompt_tokens=prompt_tokens, actual_output_tokens=output_tokens,
                   p99_small_sample=len(successful) < 100,
                   token_itl_available=False)
    for key in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = [row[key] for row in successful]
        summary[key] = {f"p{p}": percentile(values, p / 100) for p in (50, 95, 99)}
    return summary


def resource_sample(server_pid=None):
    sample = dict(monotonic_s=time.perf_counter(), gpu=[])
    if shutil.which("nvidia-smi"):
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                fields = [part.strip() for part in line.split(",")]
                if len(fields) == 6:
                    sample["gpu"].append(dict(zip(
                        ("index", "name", "memory_used_mib", "memory_total_mib", "utilization_pct", "power_w"), fields)))
    try:
        import psutil
        sample["node_ram_used_bytes"] = psutil.virtual_memory().used
        if server_pid:
            process = psutil.Process(server_pid)
            processes = [process] + process.children(recursive=True)
            # RSS sum double-counts shared pages; it is not unique physical memory.
            sample["server_tree_rss_bytes"] = sum(p.memory_info().rss for p in processes if p.is_running())
    except Exception as error:
        # Monitoring may lose a process during shutdown; timed HTTP results remain valid.
        sample["ram_error"] = type(error).__name__
    return sample


def metrics_snapshot(base_url, path):
    url = urllib.parse.urlsplit(base_url)
    try:
        with urllib.request.urlopen(f"{url.scheme}://{url.netloc}/metrics", timeout=5) as response:
            Path(path).write_bytes(response.read())
    except OSError as error:
        Path(path).write_text(f"# unavailable: {type(error).__name__}\n")


def run_benchmark(base_url, model, items, concurrency, output, *, labels=None,
                  warmup=None, seed=42, server_pid=None,
                  slo=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not items or concurrency < 1:
        raise ValueError("Non-empty workload and positive concurrency required")
    slo = slo or dict(ttft_ms=1000, tpot_ms=100, e2e_ms=30000)
    write_json(output / "config.json", dict(base_url=base_url, model=model, concurrency=concurrency,
               seed=seed, labels=labels, slo=slo, load_model="closed_loop", streaming=True,
               generation=dict(temperature=0, top_p=1, seed=42, ignore_eos=True)))
    if warmup:
        warm_rows, _ = request_batch(base_url, model, warmup, min(concurrency, len(warmup)))
        write_json(output / "warmup.json", warm_rows)
        if not all(row["success"] for row in warm_rows):
            raise RuntimeError(f"Warmup failed; inspect {output / 'warmup.json'}")
    metrics_snapshot(base_url, output / "metrics-before.prom")
    resources = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                resources.append(resource_sample(server_pid))
            except (OSError, subprocess.TimeoutExpired) as error:
                resources.append(dict(error=str(error)))
            stop.wait(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    ordered = list(items)
    random.Random(seed).shuffle(ordered)
    thread.start()
    try:
        rows, wall = request_batch(base_url, model, ordered, concurrency)
    finally:
        stop.set()
        thread.join(timeout=6)
    (output / "requests.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    (output / "resources.jsonl").write_text("".join(json.dumps(row) + "\n" for row in resources))
    metrics_snapshot(base_url, output / "metrics-after.prom")
    summary = summarize_rows(rows, wall, slo)
    summary.update(labels or {})
    summary.update(concurrency=concurrency, timestamp_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    items = [json.loads(line) for line in args.workload.read_text().splitlines() if line]
    result = run_benchmark(args.base_url, args.model, items, args.concurrency, args.output)
    print(json.dumps(result, indent=2))
