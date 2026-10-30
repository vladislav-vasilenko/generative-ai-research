"""Sequential CUDA campaign. Engine installations never modify the notebook kernel."""
import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


DEFAULT_CONFIG = dict(model="Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775",
                      engines={"vllm": "0.31.0", "sglang": "0.5.22"},
                      requests=32, repeats=3, concurrency=[1, 2, 4, 8, 16], seed=42,
                      tensor_parallel=1, gpu_devices="0", memory_fraction=0.70,
                      max_model_len=4096, max_sequences=32,
                      arms=["baseline", "chunked4096", "chunked512", "prefix"],
                      slo=dict(ttft_ms=1000, tpot_ms=100, e2e_ms=30000))


def server_command(engine, python, model_path, config, arm, port):
    chunked = arm in ("chunked4096", "chunked512")
    prefix = arm == "prefix"
    if engine == "vllm":
        return [str(python), "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(model_path), "--served-model-name", "qwen-lab",
                "--host", "127.0.0.1", "--port", str(port), "--dtype", "float16",
                "--max-model-len", str(config["max_model_len"]),
                "--max-num-seqs", str(config["max_sequences"]),
                "--gpu-memory-utilization", str(config["memory_fraction"]),
                "--tensor-parallel-size", str(config["tensor_parallel"]),
                "--seed", str(config["seed"]), "--generation-config", "vllm",
                "--attention-backend", "TRITON_ATTN", "--enforce-eager",
                "--enable-prefix-caching" if prefix else "--no-enable-prefix-caching",
                "--enable-chunked-prefill" if chunked else "--no-enable-chunked-prefill",
                "--max-num-batched-tokens", "512" if arm == "chunked512" else "4096"]
    if engine == "sglang":
        argv = [str(python), "-m", "sglang.launch_server", "--model-path", str(model_path),
                "--served-model-name", "qwen-lab", "--host", "127.0.0.1", "--port", str(port),
                "--dtype", "float16", "--context-length", str(config["max_model_len"]),
                "--tp-size", str(config["tensor_parallel"]),
                "--mem-fraction-static", str(config["memory_fraction"]),
                "--max-running-requests", str(config["max_sequences"]),
                "--attention-backend", "triton", "--sampling-backend", "pytorch",
                "--disable-cuda-graph", "--random-seed", str(config["seed"]),
                "--schedule-policy", "fcfs", "--chunked-prefill-size",
                ("512" if arm == "chunked512" else "4096") if chunked else "-1",
                "--max-prefill-tokens", "4096", "--enable-metrics"]
        if not prefix:
            argv.append("--disable-radix-cache")
        if config.get("sglang_max_total_tokens"):
            argv += ["--max-total-tokens", str(config["sglang_max_total_tokens"])]
        return argv
    raise ValueError(f"Unknown engine: {engine}")


def campaign_cells(arm, config):
    if config.get("smoke_only"):
        return [("short", 1)]
    if config.get("cells"):
        return [(kind, int(concurrency)) for kind, concurrency in config["cells"][arm]]
    if arm == "baseline":
        return [(kind, c) for kind in ("short", "standard", "long") for c in config["concurrency"]] + [
            ("mixed", 8), ("mixed", 16), ("shared", 8)]
    if arm in ("chunked4096", "chunked512"):
        return [("mixed", 8), ("mixed", 16)]
    if arm == "prefix":
        return [("shared", 8)]
    raise ValueError(arm)


def stop_server(process):
    if process.poll() is not None:
        return
    # Only the process group created by this campaign is stopped.
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def wait_for_server(process, base_url, timeout=600):
    deadline = time.monotonic() + timeout
    origin = base_url.rsplit("/v1", 1)[0]
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with code {process.returncode}; inspect server.log")
        try:
            with urllib.request.urlopen(origin + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError("Server readiness timeout; inspect server.log")


def run_logged(argv, log, env=None, timeout=1800):
    with Path(log).open("w") as handle:
        result = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT,
                                env=env, timeout=timeout)
    if result.returncode:
        tail = Path(log).read_text(errors="replace")[-3000:]
        raise RuntimeError(f"Command failed ({result.returncode}); {log}\n{tail}")


def cuda_driver_linker_path(scratch, env):
    """Expose the real driver to JIT linkers without changing system CUDA libraries."""
    candidates = []
    ldconfig = shutil.which("ldconfig")
    if ldconfig:
        listing = subprocess.run([ldconfig, "-p"], capture_output=True, text=True, timeout=10)
        for line in listing.stdout.splitlines():
            if "libcuda.so.1 " in line and "=>" in line:
                candidates.append(Path(line.rsplit("=>", 1)[1].strip()))
    candidates += [Path(path) for path in (
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/local/nvidia/lib64/libcuda.so.1",
        "/usr/lib64/libcuda.so.1")]
    driver = next((path.resolve() for path in candidates if path.is_file()), None)
    if driver is None:
        raise RuntimeError("Cannot locate the real libcuda.so.1 driver for the JIT linker")
    link_dir = Path(scratch) / "cuda-driver-link"
    link_dir.mkdir(parents=True, exist_ok=True)
    link = link_dir / "libcuda.so"
    link.unlink(missing_ok=True)
    link.symlink_to(driver)
    previous = env.get("LIBRARY_PATH", "")
    env["LIBRARY_PATH"] = str(link_dir) + (os.pathsep + previous if previous else "")
    return dict(driver=str(driver), link=str(link), library_path=env["LIBRARY_PATH"],
                changes_system_libraries=False)


def run_campaign(config, output, benchmark_fn, smoke_fn, report_fn, source_dir):
    output, source_dir = Path(output), Path(source_dir)
    output.mkdir(parents=True, exist_ok=False)
    scratch = Path("/kaggle/temp/inference-lab")
    if not Path("/kaggle/working").is_dir() or not shutil.which("nvidia-smi"):
        raise RuntimeError("This runner requires Kaggle with NVIDIA GPU; no install started")
    scratch.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_XET="1",
               TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES=config["gpu_devices"],
               UV_CACHE_DIR=str(scratch / "uv-cache"))
    # Do not copy credentials or the process environment into any artifact.
    (output / "campaign.json").write_text(json.dumps(config, indent=2) + "\n")
    if config.get("cuda_driver_linker_fix", True):
        linker = cuda_driver_linker_path(scratch, env)
        (output / "cuda-linker.json").write_text(json.dumps(linker, indent=2) + "\n")
        print("JIT linker uses driver:", linker["driver"], flush=True)
    source_copy = output / "source"
    shutil.copytree(source_dir, source_copy, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    errors = []
    uv = shutil.which("uv")
    if not uv:
        bootstrap = scratch / "bootstrap"
        run_logged([sys.executable, "-m", "pip", "install", "--target", str(bootstrap),
                    "uv==0.9.5"], output / "bootstrap.log", env)
        uv = str(bootstrap / "bin/uv")
    engines = {}
    for engine, version in config["engines"].items():
        engine_out = output / engine
        engine_out.mkdir()
        python = scratch / "venvs" / engine / "bin/python"
        try:
            run_logged([uv, "venv", "--python", "3.12", "--seed", str(python.parent.parent)],
                       engine_out / "venv.log", env)
            run_logged([uv, "pip", "install", "--python", str(python), "--prerelease=allow",
                        f"{engine}=={version}"] + config.get("engine_constraints", {}).get(engine, []),
                       engine_out / "install.log", env)
            run_logged([uv, "pip", "freeze", "--python", str(python)], engine_out / "freeze.txt", env)
            run_logged([str(python), str(source_dir / "discover_hardware.py"),
                        "--output", str(engine_out / "environment.json")],
                       engine_out / "environment.log", env)
            engines[engine] = python
            print(f"Installed {engine} {version}", flush=True)
        except Exception as error:
            errors.append(dict(engine=engine, stage="install", error=str(error)))
            (output / "errors.json").write_text(json.dumps(errors, indent=2))
            print(f"{engine} install FAILED; continuing with other engine", flush=True)
    workloads = output / "workloads"
    if not engines:
        report_fn(output)
        return output
    prepare_python = next(iter(engines.values()))
    warmup = config.get("warmup", dict(requests=4, input_tokens=512, output_tokens=64))
    try:
        run_logged([str(prepare_python), str(source_dir / "prepare_workloads.py"),
                    "--model", config["model"], "--revision", config["revision"],
                    "--cache", str(scratch / "models"), "--output", str(workloads),
                    "--requests", str(config["requests"]),
                    "--warmup-requests", str(warmup["requests"]),
                    "--warmup-input-tokens", str(warmup["input_tokens"]),
                    "--warmup-output-tokens", str(warmup["output_tokens"])], output / "prepare.log", env)
    except Exception as error:
        errors.append(dict(stage="prepare", error=str(error)))
        (output / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")
        report_fn(output)
        raise
    manifest = json.loads((workloads / "model.json").read_text())
    data = {p.stem: [json.loads(line) for line in p.read_text().splitlines()]
            for p in workloads.glob("*.jsonl")}
    unavailable = set()
    for repeat in range(config["repeats"]):
        engine_order = list(engines)
        random.Random(config["seed"] + repeat).shuffle(engine_order)
        for engine in engine_order:
            arms = list(config["arms"])
            random.Random(config["seed"] + repeat).shuffle(arms)
            for arm in arms:
                if (engine, arm) in unavailable:
                    continue
                run_dir = output / engine / f"r{repeat}-{arm}"
                run_dir.mkdir()
                argv = server_command(engine, engines[engine], manifest["snapshot_path"], config, arm, 8000)
                launch = dict(argv=argv, cuda_visible_devices=config["gpu_devices"],
                              model_revision=manifest["resolved_revision"], arm=arm, repeat=repeat,
                              dtype="float16", status="STARTING")
                (run_dir / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
                print(f"Starting {engine} / {arm} / repeat {repeat + 1}", flush=True)
                process = None
                try:
                    with (run_dir / "server.log").open("w") as log:
                        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                                   env=env, start_new_session=True)
                        wait_for_server(process, "http://127.0.0.1:8000/v1")
                        smoke_fn("http://127.0.0.1:8000/v1", "qwen-lab", run_dir / "smoke.json")
                        for kind, concurrency in campaign_cells(arm, config):
                            summary = benchmark_fn("http://127.0.0.1:8000/v1", "qwen-lab", data[kind],
                                concurrency, run_dir / f"{kind}-c{concurrency}",
                                labels=dict(engine=engine, arm=arm, repeat=repeat, workload=kind),
                                warmup=data["warmup"], server_pid=process.pid, seed=config["seed"],
                                slo=config["slo"])
                            print(f"  {kind} C={concurrency}: {summary['output_tokens_per_second']:.1f} tok/s, "
                                  f"errors={summary['error_rate']:.1%}", flush=True)
                            if summary["error_rate"]:
                                raise RuntimeError("Benchmark errors: inspect requests.jsonl; comparison invalid")
                    launch["status"] = "MEASURED"
                except Exception as error:
                    launch["status"] = "FAILED"
                    errors.append(dict(engine=engine, arm=arm, repeat=repeat, stage="serve/benchmark", error=str(error)))
                    print(f"FAILED {engine}/{arm}: {error}", flush=True)
                    if (run_dir / "server.log").exists():
                        print((run_dir / "server.log").read_text(errors="replace")[-7000:], flush=True)
                    if process is not None and process.poll() is not None:
                        # A deterministic startup failure is not retried on identical arms.
                        unavailable.add((engine, arm))
                    if config.get("abort_on_server_failure"):
                        raise
                finally:
                    if process is not None:
                        stop_server(process)
                    launch["cgroup_memory_after"] = {}
                    for name in ("memory.events", "memory.peak", "memory.current"):
                        path = Path("/sys/fs/cgroup") / name
                        if path.is_file():
                            launch["cgroup_memory_after"][name] = path.read_text().strip()
                    (run_dir / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
                    (output / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")
                    report_fn(output)
    report_fn(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from harness import run_benchmark, smoke_server
    from summarize import build_report
    config = json.loads(args.config.read_text()) if args.config else DEFAULT_CONFIG
    run_campaign(config, args.output, run_benchmark, smoke_server, build_report, Path(__file__).parent)
