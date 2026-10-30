"""Transport and metric correctness only. These fixtures are NOT inference results."""
import copy
import http.server
import io
import json
import sys
import threading
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "benchmarks"))
sys.path.insert(0, str(BASE / "scripts"))
from harness import (connection_for, perform_request, request_batch, run_benchmark,
                     smoke_server, sse_events, summarize_rows)
from run_kaggle import DEFAULT_CONFIG, server_command
from summarize import build_report, build_scheduling_report


@pytest.fixture
def server():
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        mode = "ok"

        def log_message(self, *args):
            pass

        def send(self, body, content_type="application/json", status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send(json.dumps({"data": [{"id": "test"}]}).encode())

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.mode == "http_error":
                self.send(b'{"error":"OOM"}', status=503)
                return
            usage = dict(prompt_tokens=128 if self.mode != "mismatch" else 127, completion_tokens=3)
            if not payload["stream"]:
                self.send(json.dumps(dict(usage=usage, choices=[dict(message=dict(content="Paris"), finish_reason="stop")])).encode())
                return
            events = [dict(choices=[dict(delta=dict(role="assistant"))]),
                      dict(choices=[dict(delta=dict(content="Paris is "))]),
                      dict(choices=[dict(delta=dict(content="a city"), finish_reason="stop")])]
            if self.mode != "missing_usage":
                events.append(dict(choices=[], usage=usage))
            body = "".join("data: " + json.dumps(event) + "\r\n\r\n" for event in events)
            if self.mode != "truncated":
                body += "data: [DONE]\r\n\r\n"
            self.send(body.encode(), "text/event-stream")

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", Handler
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=2)


def workload(index=0):
    return dict(id=str(index), messages=[dict(role="user", content="test")],
                prompt_tokens=128, output_tokens=3, workload="fixture")


def test_stream_usage_and_chunk_token_distinction(server):
    url, _ = server
    connection, path = connection_for(url)
    try:
        for _ in range(2):
            row = perform_request(connection, path, workload(), "test")
            assert row["success"]
            assert row["usage"]["completion_tokens"] == 3
            assert row["text_chunks"] == 2
            assert len(row["inter_chunk_ms"]) == 1
            assert row["itl_ms"] is None
            assert row["ttft_ms"] <= row["e2e_ms"]
    finally:
        connection.close()


@pytest.mark.parametrize("mode,fragment", [("missing_usage", "usage"), ("truncated", "Truncated"),
                                          ("mismatch", "mismatch"), ("http_error", "503")])
def test_invalid_responses_are_errors(server, mode, fragment):
    url, handler = server
    handler.mode = mode
    connection, path = connection_for(url)
    row = perform_request(connection, path, workload(), "test")
    connection.close()
    assert not row["success"]
    assert fragment in row["error"]


def test_concurrency_does_not_duplicate_or_drop_requests(server):
    url, _ = server
    rows, seconds = request_batch(url, "test", [workload(i) for i in range(19)], 4)
    assert len(rows) == 19
    assert len({row["id"] for row in rows}) == 19
    assert all(row["success"] for row in rows)
    summary = summarize_rows(rows, seconds, dict(ttft_ms=1000, tpot_ms=1000, e2e_ms=1000))
    assert summary["actual_output_tokens"] == 57
    assert summary["total_tokens_per_second"] == pytest.approx((128 + 3) * 19 / seconds)


def test_errors_remain_in_denominator_and_single_token_tpot_is_unknown():
    ok = dict(success=True, usage=dict(prompt_tokens=10, completion_tokens=1),
              ttft_ms=10, tpot_ms=None, e2e_ms=15)
    failed = dict(success=False, usage=None, ttft_ms=None, tpot_ms=None, e2e_ms=100)
    summary = summarize_rows([ok, failed], 2, dict(ttft_ms=1000, tpot_ms=100, e2e_ms=1000))
    assert summary["output_tokens_per_second"] == 0.5
    assert summary["error_rate"] == 0.5
    assert summary["tpot_ms"]["p50"] is None
    assert summary["goodput_requests_per_second"] == 0


def test_multiline_sse():
    events = list(sse_events(io.BytesIO(b': heartbeat\r\ndata: {"x":\r\ndata: 1}\r\n\r\ndata: [DONE]\n\n')))
    assert json.loads(events[0]) == {"x": 1}
    assert events[1] == "[DONE]"


def test_smoke_and_artifact_roundtrip(server, tmp_path, monkeypatch):
    url, _ = server
    smoke_server(url, "test", tmp_path / "smoke.json")
    monkeypatch.setattr("harness.resource_sample", lambda pid: {"gpu": []})
    summary = run_benchmark(url, "test", [workload(i) for i in range(4)], 2, tmp_path / "run",
                            warmup=[workload(100)])
    assert summary["successful"] == 4
    assert len((tmp_path / "run/requests.jsonl").read_text().splitlines()) == 4


def test_empty_report_does_not_invent_recruiter_results(tmp_path):
    result = build_report(tmp_path)
    answer = (tmp_path / "RECRUITER_ANSWER_RU.md").read_text()
    assert result["measured_cells"] == 0
    assert "пока не готов" in answer


def test_vllm_one_factor_ablation_arguments():
    baseline = server_command("vllm", "python", "model", DEFAULT_CONFIG, "baseline", 8000)
    chunked = server_command("vllm", "python", "model", DEFAULT_CONFIG, "chunked4096", 8000)
    assert set(baseline) - set(chunked) == {"--no-enable-chunked-prefill"}
    assert set(chunked) - set(baseline) == {"--enable-chunked-prefill"}
    prefix = server_command("vllm", "python", "model", DEFAULT_CONFIG, "prefix", 8000)
    assert set(prefix) - set(baseline) == {"--enable-prefix-caching"}


def test_recruiter_answer_requires_full_baseline_and_uses_paired_changes(tmp_path, monkeypatch):
    """Synthetic summaries exercise report math; never used as inference evidence."""
    monkeypatch.setattr("summarize.plot_results", lambda *args: None)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["concurrency"] = [1]
    (tmp_path / "campaign.json").write_text(json.dumps(config))
    for engine in ("vllm", "sglang"):
        cases = [("baseline", kind, 1, 1) for kind in ("short", "standard", "long")]
        cases += [("baseline", "mixed", 8, 1), ("baseline", "shared", 8, 1),
                  ("chunked4096", "mixed", 8, 1.1), ("chunked512", "mixed", 8, 0.99),
                  ("prefix", "shared", 8, 1.5)]
        for arm, kind, concurrency, multiplier in cases:
            for repeat, baseline_tps in enumerate((100, 200, 300)):
                path = tmp_path / engine / f"r{repeat}-{arm}" / f"{kind}-c{concurrency}" / "summary.json"
                path.parent.mkdir(parents=True)
                row = dict(engine=engine, arm=arm, workload=kind, concurrency=concurrency, repeat=repeat,
                           error_rate=0, output_tokens_per_second=baseline_tps * multiplier,
                           ttft_ms=dict(p50=100 / multiplier), tpot_ms=dict(p50=10 / multiplier),
                           e2e_ms=dict(p95=1000 / multiplier))
                path.write_text(json.dumps(row))
    result = build_report(tmp_path)
    assert len(result["comparisons"]) == 6
    for comparison in result["comparisons"]:
        expected = {"chunked4096": 10, "chunked512": -10, "prefix": 50}[comparison["arm"]]
        assert comparison["output_tps_change_pct"] == pytest.approx(expected)
        assert comparison["paired_repeats"] == 3
    answer = (tmp_path / "RECRUITER_ANSWER_RU.md").read_text()
    assert "-10.0%" in answer and "+50.0%" in answer
    (tmp_path / "sglang/r2-baseline/standard-c1/summary.json").unlink()
    build_report(tmp_path)
    assert "пока не готов" in (tmp_path / "RECRUITER_ANSWER_RU.md").read_text()


def test_mixed_latency_compares_matching_short_ids_not_long_or_unmatched_requests(tmp_path):
    for repeat in range(3):
        for arm, kind, short_times in (("baseline", "short", [10, 30]),
                                        ("baseline", "mixed", [20, 40]),
                                        ("chunked4096", "mixed", [10, 20]),
                                        ("chunked512", "mixed", [20, 28])):
            cell = tmp_path / "vllm" / f"r{repeat}-{arm}" / f"{kind}-c8"
            cell.mkdir(parents=True)
            summary = dict(engine="vllm", arm=arm, repeat=repeat, workload=kind, concurrency=8, error_rate=0)
            (cell / "summary.json").write_text(json.dumps(summary))
            rows = []
            for i in range(4):
                short = i % 2 == 0
                # Unmatched homogeneous IDs have enormous latency and must be excluded.
                latency = short_times[i // 2] if short else 1000
                rows.append(dict(id=f"{kind}-{i:06d}", expected_prompt_tokens=128 if short or kind == "short" else 2048,
                                 ttft_ms=latency, tpot_ms=1, e2e_ms=latency + 10))
            (cell / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    comparisons = build_scheduling_report(tmp_path)
    assert len(comparisons) == 2
    assert comparisons[0]["ttft_p50_change_pct"] == pytest.approx(-50)
    assert comparisons[1]["ttft_p50_change_pct"] == pytest.approx(60)
    import csv
    buckets = list(csv.DictReader((tmp_path / "scheduling.csv").open()))
    homogeneous = [r for r in buckets if r["case"] == "homogeneous-short"]
    assert len(homogeneous) == 3
    assert all(int(r["n"]) == 2 and float(r["ttft_ms_p50"]) == 20 for r in homogeneous)


def test_focused_validation_requires_both_complete_cache_pairs(tmp_path, monkeypatch):
    """Report guard only: these artificial summaries are not inference measurements."""
    monkeypatch.setattr("summarize.plot_prefix_validation", lambda *args: None)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(requests=64, concurrency=[8], arms=["baseline", "prefix"],
                  measurement_scope="prefix_validation",
                  cells={"baseline": [["shared", 8]], "prefix": [["shared", 8]]},
                  warmup=dict(requests=16, input_tokens=2048, output_tokens=256))
    (tmp_path / "campaign.json").write_text(json.dumps(config))
    for engine in ("vllm", "sglang"):
        for repeat in range(3):
            for arm in config["arms"]:
                path = tmp_path / engine / f"r{repeat}-{arm}" / "shared-c8/summary.json"
                path.parent.mkdir(parents=True)
                multiplier = 1 if arm == "baseline" else 1.5
                path.write_text(json.dumps(dict(engine=engine, arm=arm, repeat=repeat,
                    workload="shared", concurrency=8, error_rate=0,
                    output_tokens_per_second=100 * multiplier,
                    ttft_ms=dict(p50=100 / multiplier), tpot_ms=dict(p50=10 / multiplier),
                    e2e_ms=dict(p95=1000 / multiplier))))
    result = build_report(tmp_path)
    assert result["measured_cells"] == 12 and len(result["comparisons"]) == 2
    answer = (tmp_path / "RECRUITER_ANSWER_RU.md").read_text()
    assert "по 64 запроса" in answer and "+50.0%" in answer and "chunked prefill" not in answer
    (tmp_path / "sglang/r2-prefix/shared-c8/summary.json").unlink()
    build_report(tmp_path)
    assert "пока не готов" in (tmp_path / "RECRUITER_ANSWER_RU.md").read_text()
