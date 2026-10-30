"""Read-only environment inventory. No installation or model downloads."""
import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def command_output(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def discover_hardware():
    disk = shutil.disk_usage(Path.cwd())
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  system=platform.system(), release=platform.release(),
                  architecture=platform.machine(), python=platform.python_version(),
                  disk_free_gib=round(disk.free / 2**30, 2), packages={})
    result["cpu_count"] = os.cpu_count()
    try:
        result["physical_ram_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        result["physical_ram_bytes"] = None
    if platform.system() == "Linux":
        result["cgroup"] = {}
        for name in ("memory.max", "memory.events", "memory.peak"):
            path = Path("/sys/fs/cgroup") / name
            if path.is_file():
                result["cgroup"][name] = path.read_text().strip()
    for package in ("torch", "mlx", "mlx-lm", "vllm", "vllm-metal", "sglang",
                    "transformers", "psutil"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    if platform.system() == "Darwin":
        # Select only hardware fields; never persist serial numbers or machine UUIDs.
        raw = command_output(["system_profiler", "SPHardwareDataType", "-json"])
        try:
            hardware = json.loads(raw["stdout"])["SPHardwareDataType"][0]
            result["hardware"] = {key: hardware.get(key) for key in
                                  ("chip_type", "physical_memory", "number_processors")}
        except (KeyError, ValueError):
            result["hardware"] = {"error": raw.get("error", raw.get("stderr"))}
        result["display"] = command_output(["system_profiler", "SPDisplaysDataType"])
        result["macos"] = platform.mac_ver()[0]
    if shutil.which("nvidia-smi"):
        result["nvidia"] = command_output([
            "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits"])
        result["topology"] = command_output(["nvidia-smi", "topo", "-m"])
    if result["packages"]["torch"]:
        import torch
        result["torch_devices"] = dict(cuda=torch.cuda.is_available(),
                                       cuda_runtime=torch.version.cuda,
                                       mps_built=torch.backends.mps.is_built(),
                                       mps_available_in_this_process=torch.backends.mps.is_available())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = discover_hardware()
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded)
