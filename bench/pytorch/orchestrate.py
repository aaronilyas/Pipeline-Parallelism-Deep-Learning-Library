"""Measure workload W on the SSH hosts and write bench/runs plus bench/RESULTS.md.

The hub runs this. Each training host runs one process. Device and CPU come
from the remote probe, not from the SSH alias.
"""

from __future__ import annotations

import base64
import hashlib
import json
import platform
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from metrics import (
    LAYERS,
    MEASURED_STEPS,
    WARMUP_STEPS,
    dump,
    equal_layers,
    median,
    proportional_layers,
    samples_per_sec,
)

REPO = Path(__file__).resolve().parents[2]
PY = Path(__file__).resolve().parent
RUNS = REPO / "bench" / "runs"
TORCH_RELEASE = "2.14.1"
SSH_HOSTS = ("macmini", "ubuntu", "windows")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
SCP = ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
# Windows Smart App Control blocks the unsigned torch.dll. The launch path
# for that host is filled in after the native import is tried.
WSL_USER: dict[str, str] = {}
WSL_DISTRO: dict[str, str] = {}


def git_commit() -> str:
    return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()


def source_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(PY.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def detect_ifname(host: str, ssh_system: str, ipv4: str) -> str | None:
    """Interface that owns the host's LAN address. Gloo must bind that one."""
    if not ipv4:
        return None
    if ssh_system == "WindowsNative":
        command = f"(Get-NetIPAddress -IPAddress {ipv4}).InterfaceAlias"
    elif ssh_system == "Darwin":
        command = f"route -n get {ipv4}"
    else:
        command = f"ip -o addr show to {ipv4}"
    result = ssh(host, command, 30)
    if result.returncode != 0:
        return None
    if ssh_system == "Darwin":
        for line in result.stdout.splitlines():
            if "interface:" in line:
                parts = line.split()
                return parts[-1] if parts else None
        return None
    if ssh_system == "WindowsNative":
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return lines[-1] if lines else None
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            return parts[1]
    return None


def ssh(host: str, command: str, timeout: float | None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*SSH, host, command],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def parse_json_stdout(text: str) -> dict:
    if "__REPORT__" in text:
        text = text.split("__REPORT__")[-1]
    start = text.find("{")
    if start < 0:
        raise ValueError(text[-4000:])
    return json.loads(text[start:])


def remote_os(host: str) -> str:
    result = ssh(host, "uname -s", 30)
    name = result.stdout.strip()
    if result.returncode == 0 and name in {"Linux", "Darwin"}:
        return name
    return "Windows"


def remote_mkdir(host: str, system: str) -> None:
    if system == "Windows":
        command = (
            "New-Item -ItemType Directory -Force -Path "
            "$env:USERPROFILE\\ppdl-w-bench\\pytorch,"
            "$env:USERPROFILE\\ppdl-w-bench\\jobs | Out-Null"
        )
    else:
        command = "mkdir -p ~/ppdl-w-bench/pytorch ~/ppdl-w-bench/jobs"
    result = ssh(host, command, 60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)


def sync_sources(host: str, system: str) -> None:
    remote_mkdir(host, system)
    files = [str(path) for path in sorted(PY.glob("*.py"))]
    if system == "Windows":
        destination = f"{host}:ppdl-w-bench/pytorch/"
    else:
        destination = f"{host}:ppdl-w-bench/pytorch/"
    subprocess.check_call([*SCP, *files, destination])


def push_job(host: str, system: str, name: str, payload: dict) -> None:
    local = RUNS / "jobs"
    local.mkdir(parents=True, exist_ok=True)
    path = local / f"{host}-{name}"
    path.write_text(dump(payload), encoding="utf-8")
    if system == "Windows":
        destination = f"{host}:ppdl-w-bench/jobs/{name}"
    else:
        destination = f"{host}:ppdl-w-bench/jobs/{name}"
    subprocess.check_call([*SCP, str(path), destination])


def launch_command(ssh_system: str, inner: str) -> str:
    """What `ssh host` executes. Windows runs `inner` inside WSL."""
    if ssh_system != "Windows":
        return inner
    encoded = base64.b64encode(inner.encode()).decode()
    return (
        "$env:PPDL_CMD='" + encoded + "'; "
        "powershell -NoProfile -ExecutionPolicy Bypass -File "
        "$env:USERPROFILE\\ppdl-w-bench\\wsl_exec.ps1"
    )


def display_command(host: str, ssh_system: str, inner: str) -> str:
    if ssh_system == "Windows":
        distro = WSL_DISTRO.get(host, "wsl")
        return f"ssh {host} wsl -d {distro} -- bash -lc {json.dumps(inner)}"
    return f"ssh {host} {inner}"


def windows_mount(host: str) -> str:
    user = WSL_USER[host]
    return f"/mnt/c/Users/{user}/ppdl-w-bench"


def python_command(ssh_system: str, script: str, job_name: str, host: str = "") -> str:
    if ssh_system == "Windows":
        mount = windows_mount(host)
        return (
            "mkdir -p ~/ppdl-w-bench/pytorch ~/ppdl-w-bench/jobs && "
            f"cp {mount}/pytorch/*.py ~/ppdl-w-bench/pytorch/ && "
            f"cp {mount}/jobs/{job_name} ~/ppdl-w-bench/jobs/{job_name} && "
            f"~/ppdl-w-bench/venv/bin/python ~/ppdl-w-bench/pytorch/{script} "
            f"~/ppdl-w-bench/jobs/{job_name}"
        )
    if ssh_system == "WindowsNative":
        return (
            "& $env:USERPROFILE\\ppdl-w-bench\\venv\\Scripts\\python.exe "
            f"$env:USERPROFILE\\ppdl-w-bench\\pytorch\\{script} "
            f"$env:USERPROFILE\\ppdl-w-bench\\jobs\\{job_name}"
        )
    return (
        "~/ppdl-w-bench/venv/bin/python "
        f"~/ppdl-w-bench/pytorch/{script} "
        f"~/ppdl-w-bench/jobs/{job_name}"
    )


def probe_command(ssh_system: str, host: str) -> str:
    if ssh_system == "Windows":
        mount = windows_mount(host)
        return (
            "mkdir -p ~/ppdl-w-bench/pytorch && "
            f"cp {mount}/pytorch/*.py ~/ppdl-w-bench/pytorch/ && "
            "~/ppdl-w-bench/venv/bin/python ~/ppdl-w-bench/pytorch/probe.py"
        )
    if ssh_system == "WindowsNative":
        return (
            "& $env:USERPROFILE\\ppdl-w-bench\\venv\\Scripts\\python.exe "
            "$env:USERPROFILE\\ppdl-w-bench\\pytorch\\probe.py"
        )
    return "~/ppdl-w-bench/venv/bin/python ~/ppdl-w-bench/pytorch/probe.py"


def ssh_invoke(host: str, ssh_system: str, inner: str, timeout: float | None) -> subprocess.CompletedProcess[str]:
    launched = inner if ssh_system == "WindowsNative" else launch_command(ssh_system, inner)
    return ssh(host, launched, timeout)


def wsl_distro_name(host: str) -> str:
    result = ssh(host, "wsl.exe -l -q", 60)
    text = result.stdout
    if "\x00" in text:
        text = text.encode("utf-8", errors="ignore").decode("utf-16-le", errors="ignore")
    names = []
    for line in text.replace("\x00", "").splitlines():
        name = line.strip().lstrip("*").strip()
        if name:
            names.append(name)
    ubuntu = [name for name in names if name.lower().startswith("ubuntu")]
    if not (ubuntu or names):
        raise RuntimeError(f"no WSL distro on {host}: {result.stderr[-2000:]}")
    return (ubuntu or names)[0]


def prepare_wsl(host: str) -> None:
    remote_mkdir(host, "Windows")
    subprocess.check_call([*SCP, str(PY / "wsl_exec.ps1"), f"{host}:ppdl-w-bench/wsl_exec.ps1"])
    user = ssh(host, "Write-Output $env:USERNAME", 30)
    if user.returncode != 0 or not user.stdout.strip():
        raise RuntimeError(user.stderr or "windows username missing")
    WSL_USER[host] = user.stdout.strip()
    distro = wsl_distro_name(host)
    WSL_DISTRO[host] = distro
    safe = distro.replace("'", "''")
    wrote = ssh(
        host,
        "Set-Content -Path $env:USERPROFILE\\ppdl-w-bench\\wsl-distro.txt "
        f"-Value '{safe}' -Encoding ascii",
        30,
    )
    if wrote.returncode != 0:
        raise RuntimeError(wrote.stderr or "could not write wsl-distro.txt")


def native_torch_text(host: str) -> tuple[bool, str]:
    command = (
        "& $env:USERPROFILE\\ppdl-w-bench\\venv\\Scripts\\python.exe "
        "-c \"import torch; print(torch.__version__)\""
    )
    try:
        result = ssh(host, command, 180)
    except Exception as exc:
        return False, str(exc)
    text = (result.stdout + "\n" + result.stderr).strip()
    return result.returncode == 0 and release_of(result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "") == TORCH_RELEASE, text[-4000:]


def bootstrap(host: str, ssh_system: str) -> None:
    remote_mkdir(host, "Windows" if ssh_system.startswith("Windows") else ssh_system)
    if ssh_system == "Windows":
        subprocess.check_call([*SCP, str(PY / "bootstrap_unix.sh"), f"{host}:ppdl-w-bench/bootstrap_unix.sh"])
        mount = windows_mount(host)
        inner = f"export UV_HTTP_TIMEOUT=600; sh {mount}/bootstrap_unix.sh"
        result = ssh_invoke(host, ssh_system, inner, 3600)
    elif ssh_system == "WindowsNative":
        subprocess.check_call([*SCP, str(PY / "bootstrap_windows.ps1"), f"{host}:ppdl-w-bench/bootstrap_windows.ps1"])
        command = (
            "powershell -NoProfile -ExecutionPolicy Bypass -File "
            "$env:USERPROFILE\\ppdl-w-bench\\bootstrap_windows.ps1"
        )
        result = ssh(host, command, 3600)
    else:
        subprocess.check_call([*SCP, str(PY / "bootstrap_unix.sh"), f"{host}:ppdl-w-bench/bootstrap_unix.sh"])
        result = ssh(host, "export UV_HTTP_TIMEOUT=600; sh ~/ppdl-w-bench/bootstrap_unix.sh", 3600)
    print(f"bootstrap {host}", flush=True)
    log = RUNS / "logs" / f"bootstrap-{host}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"bootstrap {host} failed\n{result.stderr[-4000:]}")


def release_of(version: str | None) -> str:
    if not version:
        return ""
    return version.split("+", 1)[0]


def run_probe(host: str, ssh_system: str) -> dict:
    inner = probe_command(ssh_system, host)
    result = ssh_invoke(host, ssh_system, inner, 180)
    if result.returncode != 0 and not result.stdout.strip():
        raise RuntimeError(result.stderr[-4000:] or "probe failed")
    info = parse_json_stdout(result.stdout)
    info["ssh"] = host
    info["ssh_system"] = ssh_system
    info["system"] = info.get("system") or ssh_system
    return info


def ensure_host(host: str) -> dict:
    ssh_system = remote_os(host)
    native_error = None
    if ssh_system == "Windows":
        ok, text = native_torch_text(host)
        if ok:
            ssh_system = "WindowsNative"
        else:
            native_error = text
            prepare_wsl(host)
    sync_sources(host, "Windows" if ssh_system.startswith("Windows") else ssh_system)
    try:
        info = run_probe(host, ssh_system)
    except Exception as exc:
        info = {"torch_error": str(exc), "best_device": None, "ssh_system": ssh_system}
    if release_of(info.get("torch")) != TORCH_RELEASE or not info.get("best_device"):
        bootstrap(host, ssh_system)
        sync_sources(host, "Windows" if ssh_system.startswith("Windows") else ssh_system)
        info = run_probe(host, ssh_system)
    info["ssh"] = host
    info["ssh_system"] = ssh_system
    info["gloo_ifname"] = detect_ifname(host, ssh_system, str(info.get("ipv4") or ""))
    info["runtime"] = "wsl" if ssh_system == "Windows" else "native"
    info["wsl_distro"] = WSL_DISTRO.get(host)
    info["native_torch_error"] = native_error
    if release_of(str(info.get("torch"))) != TORCH_RELEASE:
        raise RuntimeError(f"{host} torch {info.get('torch')} is not {TORCH_RELEASE}")
    (RUNS / "probes").mkdir(parents=True, exist_ok=True)
    (RUNS / "probes" / f"{host}.json").write_text(dump(info), encoding="utf-8")
    print(
        f"probe {host} device={info.get('best_device')} torch={info.get('torch')} runtime={info['runtime']} ifname={info.get('gloo_ifname')}",
        flush=True,
    )
    return info


def run_one(host: str, ssh_system: str, script: str, job_name: str, payload: dict, timeout: float) -> dict:
    push_job(host, "Windows" if ssh_system.startswith("Windows") else ssh_system, job_name, payload)
    inner = python_command(ssh_system, script, job_name, host)
    shown = display_command(host, ssh_system, inner)
    print(f"start {host} {script} {job_name}", flush=True)
    try:
        result = ssh_invoke(host, ssh_system, inner, timeout)
    except subprocess.TimeoutExpired as exc:
        kill_remote(host, ssh_system)
        return {
            "status": "not run",
            "reason": "remote process timed out",
            "error": str(exc)[-4000:],
            "hosts": [{"hostname": host, "ssh": host}],
            "devices": [],
            "split": None,
            "per_step_times_s": [],
            "samples_per_sec": None,
            "median_step_s": None,
            "bytes_sent_per_boundary": [],
            "command": shown,
        }
    log = RUNS / "logs" / f"{host}-{job_name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout + "\n--- stderr ---\n" + result.stderr, encoding="utf-8")
    if "__REPORT__" not in result.stdout:
        return {
            "status": "not run",
            "reason": "remote process produced no report",
            "error": (result.stderr or result.stdout)[-8000:],
            "returncode": result.returncode,
            "hosts": [{"hostname": host, "ssh": host}],
            "devices": [],
            "split": None,
            "per_step_times_s": [],
            "samples_per_sec": None,
            "median_step_s": None,
            "bytes_sent_per_boundary": [],
            "command": shown,
        }
    report = parse_json_stdout(result.stdout)
    report["returncode"] = result.returncode
    report["command"] = shown
    return report


def kill_remote(host: str, ssh_system: str) -> None:
    if ssh_system == "Windows":
        ssh_invoke(host, ssh_system, "pkill -f ppdl-w-bench/pytorch || true", 30)
        return
    if ssh_system == "WindowsNative":
        command = (
            "Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | "
            "Where-Object { $_.CommandLine -like '*ppdl-w-bench*' } | "
            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
        )
        ssh(host, command, 30)
        return
    ssh(host, "pkill -f ppdl-w-bench/pytorch || true", 30)


def run_group(items: list[tuple[str, str, str]], timeout: float) -> dict[str, tuple[int, str, str]]:
    """items are (host, system, remote command).

    Pipes are drained while the process runs. Waiting until exit before
    reading would stall a rank once the pipe buffer filled.
    """
    procs: list[tuple[str, str, subprocess.Popen[str]]] = []
    for host, system, command in items:
        procs.append(
            (
                host,
                system,
                subprocess.Popen(
                    [*SSH, host, command],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ),
            )
        )

    def wait_one(host: str, system: str, proc: subprocess.Popen[str]) -> tuple[str, tuple[int, str, str]]:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return host, (proc.returncode or 0, stdout, stderr)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            kill_remote(host, system)
            return host, (proc.returncode or -1, stdout, stderr + "\nTIMEOUT")

    finished: dict[str, tuple[int, str, str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(procs))) as pool:
        futures = [pool.submit(wait_one, host, system, proc) for host, system, proc in procs]
        for future in futures:
            host, outcome = future.result()
            finished[host] = outcome
    return finished


def write_run(name: str, report: dict) -> Path:
    path = RUNS / name
    path.write_text(dump(report), encoding="utf-8")
    return path


def reusable_single(host: dict, device: str, commit: str) -> dict | None:
    """A single-host file from this commit, with all 50 steps, is the measurement."""
    path = RUNS / f"pytorch-single-{host['ssh']}-{device}.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    times = report.get("per_step_times_s") or []
    if (
        report.get("status") == "ok"
        and report.get("commit") == commit
        and report.get("measured_steps") == MEASURED_STEPS
        and report.get("devices") == [device]
        and len(times) == MEASURED_STEPS
    ):
        report["ssh"] = host["ssh"]
        print(f"reuse {path.name}", flush=True)
        return report
    return None


def single_runs(hosts: list[dict], commit: str, digest: str, warmup: int, steps: int) -> list[dict]:
    reports = []

    def one(host: dict, device: str, label: str) -> dict:
        payload = {
            "commit": commit,
            "command": f"single_host.py device={device} warmup={warmup} steps={steps}",
            "source_sha256": digest,
            "warmup": warmup,
            "steps": steps,
            "device": device,
        }
        report = run_one(
            host["ssh"],
            host["ssh_system"],
            "single_host.py",
            f"single-{label}.json",
            payload,
            timeout=60 * 60 * 6,
        )
        report["ssh"] = host["ssh"]
        report["inventory"] = {
            "hostname": host.get("hostname"),
            "cpu": host.get("cpu"),
            "gpu": host.get("gpu"),
            "ipv4": host.get("ipv4"),
            "memory_bytes": host.get("memory_bytes"),
            "cuda_capability": host.get("cuda_capability"),
            "runtime": host.get("runtime"),
            "wsl_distro": host.get("wsl_distro"),
            "native_torch_error": host.get("native_torch_error"),
        }
        return report

    futures = []
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        for host in hosts:
            device = str(host["best_device"])
            kept = reusable_single(host, device, commit)
            if kept is not None:
                reports.append(kept)
                continue
            futures.append(pool.submit(one, host, device, f"{host['ssh']}-{device}"))
        for future in futures:
            report = future.result()
            reports.append(report)
            device_name = (report.get("devices") or ["unknown"])[0] or "unknown"
            write_run(f"pytorch-single-{report['ssh']}-{device_name}.json", report)
    return reports


def port_conflict(text: str) -> bool:
    lowered = text.lower()
    return "address already in use" in lowered or "eaddrinuse" in lowered


def gloo_check(hosts: list[dict], port: int) -> dict:
    ordered = sorted(hosts, key=lambda item: str(item["hostname"]))
    master = master_host(ordered)
    errors = []
    reports = []
    for attempt_port in range(port, port + 5):
        items = []
        for rank, host in enumerate(ordered):
            payload = {
                "rank": rank,
                "world": len(ordered),
                "master": master["ipv4"],
                "port": attempt_port,
                "store_master": host["ssh"] == master["ssh"],
                "gloo_ifname": host.get("gloo_ifname") or "",
            }
            name = f"gloo-{attempt_port}.json"
            push_job(host["ssh"], "Windows" if str(host["ssh_system"]).startswith("Windows") else host["ssh_system"], name, payload)
            inner = python_command(host["ssh_system"], "gloo_probe.py", name, host["ssh"])
            items.append((host["ssh"], host["ssh_system"], launch_command(host["ssh_system"], inner)))
        finished = run_group(items, timeout=180)
        ok = True
        attempt_reports = []
        for host in ordered:
            code, stdout, stderr = finished.get(host["ssh"], (1, "", "missing"))
            log = RUNS / "logs" / f"gloo-{host['ssh']}-{attempt_port}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(stdout + "\n--- stderr ---\n" + stderr, encoding="utf-8")
            if "__REPORT__" not in stdout:
                ok = False
                attempt_reports.append({"ssh": host["ssh"], "status": "not run", "error": stderr[-4000:]})
                continue
            report = parse_json_stdout(stdout)
            report["ssh"] = host["ssh"]
            attempt_reports.append(report)
            if report.get("status") != "ok":
                ok = False
        if ok:
            return {"status": "ok", "port": attempt_port, "master": master["ipv4"], "ranks": attempt_reports, "ordered": ordered}
        errors.append({"port": attempt_port, "ranks": attempt_reports})
        blob = json.dumps(attempt_reports)
        if not port_conflict(blob):
            break
    return {
        "status": "not run",
        "reason": "torch.distributed gloo group failed",
        "master": master.get("ipv4"),
        "attempts": errors,
        "ordered": ordered,
    }


def master_host(ordered: list[dict]) -> dict:
    def reachable(host: dict) -> bool:
        return bool(host.get("ipv4"))

    # A WSL guest can share the LAN address, but the store is steadier on the
    # host that is Linux without a guest. CUDA is preferred inside that pool.
    native_linux = [
        host
        for host in ordered
        if host.get("system") == "Linux" and host.get("runtime") != "wsl" and reachable(host)
    ]
    linux = [host for host in ordered if host.get("system") == "Linux" and reachable(host)]
    pool = native_linux or linux or [host for host in ordered if reachable(host)]
    if not pool:
        raise RuntimeError("no host reported an IPv4 address")
    cuda = [host for host in pool if host.get("best_device") == "cuda"]
    return (cuda or pool)[0]


def pipeline_run(
    ordered: list[dict],
    split: list[int],
    policy: str,
    port: int,
    commit: str,
    digest: str,
    warmup: int,
    steps: int,
    timeout: float,
) -> dict:
    master = master_host(ordered)
    items = []
    launches = []
    for rank, host in enumerate(ordered):
        payload = {
            "rank": rank,
            "world": len(ordered),
            "master": master["ipv4"],
            "port": port,
            "layers": split[rank],
            "commit": commit,
            "command": "",
            "source_sha256": digest,
            "split_policy": policy,
            "warmup": warmup,
            "steps": steps,
            "device": host["best_device"],
            "store_master": host["ssh"] == master["ssh"],
            "gloo_ifname": host.get("gloo_ifname") or "",
        }
        name = f"pipeline-{policy}.json"
        inner = python_command(host["ssh_system"], "pipeline_rank.py", name, host["ssh"])
        payload["command"] = display_command(host["ssh"], host["ssh_system"], inner)
        push_job(
            host["ssh"],
            "Windows" if str(host["ssh_system"]).startswith("Windows") else host["ssh_system"],
            name,
            payload,
        )
        launches.append(payload["command"])
        items.append((host["ssh"], host["ssh_system"], launch_command(host["ssh_system"], inner)))
    command = "\n".join(launches)
    finished = run_group(items, timeout=timeout)
    rank_reports = []
    for rank, host in enumerate(ordered):
        code, stdout, stderr = finished.get(host["ssh"], (1, "", "missing"))
        log = RUNS / "logs" / f"pipeline-{policy}-{host['ssh']}.log"
        log.write_text(stdout + "\n--- stderr ---\n" + stderr, encoding="utf-8")
        if "__REPORT__" not in stdout:
            rank_reports.append(
                {
                    "status": "not run",
                    "ssh": host["ssh"],
                    "rank": rank,
                    "error": (stderr or stdout)[-8000:],
                    "per_step_times_s": [],
                    "bytes_sent_to_rank": {},
                }
            )
            continue
        report = parse_json_stdout(stdout)
        report["ssh"] = host["ssh"]
        rank_reports.append(report)
    return merge_pipeline(ordered, split, policy, command, commit, digest, warmup, steps, rank_reports, master)


def merge_pipeline(
    ordered: list[dict],
    split: list[int],
    policy: str,
    command: str,
    commit: str,
    digest: str,
    warmup: int,
    steps: int,
    rank_reports: list[dict],
    master: dict,
) -> dict:
    errors = [report.get("error") for report in rank_reports if report.get("status") != "ok"]
    per_rank_times = [report.get("per_step_times_s") or [] for report in rank_reports]
    merged: dict = {
        "workload": "W",
        "stack": "pytorch",
        "kind": "pipeline",
        "split_policy": policy,
        "status": "not run",
        "reason": None,
        "hosts": [
            {
                "ssh": host["ssh"],
                "hostname": host.get("hostname"),
                "cpu": host.get("cpu"),
                "gpu": host.get("gpu"),
                "ipv4": host.get("ipv4"),
                "memory_bytes": host.get("memory_bytes"),
                "cuda_capability": host.get("cuda_capability"),
                "rank": rank,
                "torch": host.get("torch"),
                "runtime": host.get("runtime"),
                "wsl_distro": host.get("wsl_distro"),
                "native_torch_error": host.get("native_torch_error"),
            }
            for rank, host in enumerate(ordered)
        ],
        "devices": [host.get("best_device") for host in ordered],
        "split": split,
        "versions": {
            "torch_release": TORCH_RELEASE,
            "torch_by_host": {host["ssh"]: host.get("torch") for host in ordered},
            "master": master.get("ipv4"),
            "backend": "gloo",
            "schedule": "torch.distributed.pipelining.Schedule1F1B",
            "amp": False,
            "dtype": "float32",
        },
        "commit": commit,
        "source_sha256": digest,
        "command": command,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "per_rank_step_times_s": per_rank_times,
        "per_step_times_s": [],
        "median_step_s": None,
        "samples_per_sec": None,
        "bytes_sent_per_boundary": [],
        "rank_errors": [error for error in errors if error],
    }
    if errors or any(len(times) != steps for times in per_rank_times):
        merged["reason"] = "one or more ranks did not finish the measured steps"
        return merged
    per_step = [max(times[index] for times in per_rank_times) for index in range(steps)]
    middle = median(per_step)
    merged["status"] = "ok"
    merged["per_step_times_s"] = per_step
    merged["median_step_s"] = middle
    merged["samples_per_sec"] = samples_per_sec(middle)
    boundaries = []
    for left in range(len(ordered) - 1):
        forward = int((rank_reports[left].get("bytes_sent_to_rank") or {}).get(str(left + 1), 0))
        backward = int((rank_reports[left + 1].get("bytes_sent_to_rank") or {}).get(str(left), 0))
        boundaries.append(
            {
                "from_rank": left,
                "to_rank": left + 1,
                "forward_bytes": forward,
                "backward_bytes": backward,
                "total_bytes": forward + backward,
            }
        )
    merged["bytes_sent_per_boundary"] = boundaries
    return merged


def fmt_num(value: object) -> str:
    if value is None:
        return "not run"
    return json.dumps(value)


def results_markdown(
    singles: list[dict],
    pipelines: list[dict],
    gloo: dict,
    linalg: dict | None,
    hub: dict,
    pair_gloos: list[dict] | None = None,
) -> str:
    lines = [
        "# Baseline",
        "",
        "Workload W is a 12-layer MLP, `y = gelu(x @ W + b)`, width 4096, float32.",
        "One step is 8 microbatches of 256 rows, mean-squared error (`torch.nn.MSELoss`),",
        "and SGD at `1e-3`. Samples/sec is `2048 / median(step time)` over 50 steps after 5 warmup steps.",
        "The median of an even count is the average of the two central values.",
        "Pipeline step time is the max across ranks after a gloo barrier.",
        "Byte counts are the totals sent during the measured steps, one record per stage boundary.",
        "AMP is off. TF32 is PyTorch's default, recorded in each JSON and not changed.",
        "Torch release `2.14.1` is installed on every training host.",
        "The GTX 1080 host uses the CUDA 12.6 wheel (`--index-url https://download.pytorch.org/whl/cu126`).",
        "The Mac uses the macOS arm64 wheel of the same release. No number here is estimated.",
        "",
        "## Hub",
        "",
        f"Coordinator `{hub.get('hostname')}` is not a training rank. CPU `{hub.get('cpu')}`. GPU `{hub.get('gpu')}`.",
        "",
        "## Single host",
        "",
        "| ssh | hostname | device | samples/sec | median step s | status | command | commit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for report in singles:
        device = ",".join(report.get("devices") or []) or "none"
        lines.append(
            "| {ssh} | {hostname} | {device} | {samples} | {median} | {status} | `{command}` | `{commit}` |".format(
                ssh=report.get("ssh"),
                hostname=(report.get("inventory") or {}).get("hostname"),
                device=device,
                samples=fmt_num(report.get("samples_per_sec")),
                median=fmt_num(report.get("median_step_s")),
                status=report.get("status"),
                command=report.get("command"),
                commit=report.get("commit"),
            )
        )
    notes = [
        report.get("inventory", {}).get("native_torch_error")
        for report in singles
        if (report.get("inventory") or {}).get("native_torch_error")
    ]
    if notes:
        lines.extend(
            [
                "",
                "Native Windows torch did not import. The recorded error is below.",
                "That host's runs use the WSL distro named in the probe, on the same GPU.",
                "",
                "```",
                str(notes[0])[:4000],
                "```",
            ]
        )
    lines.extend(["", "## Pipeline", ""])
    if gloo.get("status") != "ok":
        lines.append("Gloo did not form a group on the full host set. Attempts:")
        lines.append("")
        lines.append("```")
        lines.append(json.dumps(gloo.get("attempts", gloo.get("reason")), indent=2)[:12000])
        lines.append("```")
        lines.append("")
    lines.append("| policy | hosts | devices | split | samples/sec | median step s | status | commit |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    error_notes = []
    for report in pipelines:
        hosts = ",".join(str(host.get("ssh")) for host in report.get("hosts") or [])
        lines.append(
            "| {policy} | {hosts} | {devices} | `{split}` | {samples} | {median} | {status} | `{commit}` |".format(
                policy=report.get("split_policy"),
                hosts=hosts,
                devices=",".join(str(device) for device in (report.get("devices") or [])),
                split=report.get("split"),
                samples=fmt_num(report.get("samples_per_sec")),
                median=fmt_num(report.get("median_step_s")),
                status=report.get("status"),
                commit=report.get("commit"),
            )
        )
        if report.get("status") != "ok" and report.get("rank_errors"):
            error_notes.append((report.get("split_policy"), hosts, str(report["rank_errors"][0])[:6000]))
    for policy, hosts, error in error_notes:
        lines.extend(["", f"Exact error for `{policy}` on {hosts}:", "", "```", error, "```"])
    if pair_gloos:
        lines.extend(["", "### Subset groups", ""])
        lines.append("These groups were tried after the full set did not produce a finished 50-step pipeline.")
        lines.append("")
        for report in pair_gloos:
            names = report.get("names") or "subset"
            lines.append(f"- `{names}` status `{report.get('status')}` port `{report.get('port')}`")
            if report.get("status") != "ok":
                lines.extend(["", "```", json.dumps(report.get("attempts", report.get("reason")), indent=2)[:8000], "```", ""])
    ok_single = [report for report in singles if report.get("status") == "ok" and report.get("measured_steps") == MEASURED_STEPS]
    ok_pipe = [report for report in pipelines if report.get("status") == "ok" and report.get("measured_steps") == MEASURED_STEPS]
    lines.extend(["", "## Chosen baseline", ""])
    if ok_pipe:
        chosen = max(ok_pipe, key=lambda report: float(report["samples_per_sec"]))
        lines.append(
            f"The baseline is the {chosen['split_policy']} split on "
            f"{[host.get('ssh') for host in chosen['hosts']]} at {fmt_num(chosen['samples_per_sec'])} samples/sec."
        )
        if ok_single:
            best = max(ok_single, key=lambda report: float(report["samples_per_sec"]))
            if float(chosen["samples_per_sec"]) < float(best["samples_per_sec"]):
                lines.append(
                    f"The pipeline baseline is slower than the best single host: "
                    f"{fmt_num(chosen['samples_per_sec'])} samples/sec versus "
                    f"{fmt_num(best['samples_per_sec'])} samples/sec on {best.get('ssh')} "
                    f"({','.join(best.get('devices') or [])})."
                )
            else:
                lines.append(
                    f"The pipeline baseline is at least as fast as the best single host "
                    f"({fmt_num(best['samples_per_sec'])} samples/sec on {best.get('ssh')})."
                )
    elif ok_single:
        best = max(ok_single, key=lambda report: float(report["samples_per_sec"]))
        lines.append(
            f"No pipeline finished. The baseline is the fastest PyTorch configuration that ran: "
            f"{best.get('ssh')} {','.join(best.get('devices') or [])} at {fmt_num(best['samples_per_sec'])} samples/sec."
        )
    else:
        lines.append("No PyTorch configuration finished. There is no samples/sec baseline.")
    lines.extend(["", "## linalg-pipeline", ""])
    if linalg is None:
        lines.append("not run. The bench JSON was not produced.")
    else:
        lines.append(f"Status: {linalg.get('status')}.")
        lines.append("")
        lines.append(linalg.get("reason") or "")
        lines.append("")
        lines.append(f"Command: `{linalg.get('command')}`")
        lines.append(f"Commit: `{linalg.get('commit')}`")
    lines.append("")
    lines.append("Per-step times and byte counts are in `bench/runs/`.")
    lines.append("")
    return "\n".join(lines)


def hub_inventory() -> dict:
    cpu = "unknown"
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor()
    gpu = "none"
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"hostname": socket.gethostname(), "cpu": cpu, "gpu": gpu}


def measure_set(
    ordered: list[dict],
    singles: list[dict],
    port: int,
    commit: str,
    digest: str,
) -> list[dict]:
    pipelines: list[dict] = []
    policies = [("equal", equal_layers(len(ordered)))]
    speeds = []
    speed_ok = True
    by_ssh = {report.get("ssh"): report for report in singles}
    for host in ordered:
        single = by_ssh.get(host["ssh"])
        device = host["best_device"]
        if (
            single
            and single.get("status") == "ok"
            and single.get("devices") == [device]
            and single.get("measured_steps") == MEASURED_STEPS
        ):
            speeds.append(float(single["samples_per_sec"]))
        else:
            speed_ok = False
    suffix = "-".join(str(host["ssh"]) for host in ordered)
    if speed_ok:
        policies.append(("proportional", proportional_layers(speeds)))
    else:
        skipped = {
            "workload": "W",
            "stack": "pytorch",
            "kind": "pipeline",
            "split_policy": "proportional",
            "status": "not run",
            "reason": "a host has no single-host samples/sec on its best device, so a proportional split is not measured",
            "hosts": [{"ssh": host["ssh"], "hostname": host.get("hostname")} for host in ordered],
            "devices": [host.get("best_device") for host in ordered],
            "split": None,
            "commit": commit,
            "command": "not launched",
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": 0,
            "per_step_times_s": [],
            "median_step_s": None,
            "samples_per_sec": None,
            "bytes_sent_per_boundary": [],
            "rank_errors": [],
        }
        write_run(f"pytorch-pipeline-proportional-{suffix}.json", skipped)
        pipelines.append(skipped)
    for index, (policy, split) in enumerate(policies):
        smoke = pipeline_run(ordered, split, policy, port + index, commit, digest, 1, 1, timeout=1800)
        write_run(f"pytorch-smoke-{policy}-{suffix}.json", smoke)
        if smoke.get("status") != "ok":
            smoke["samples_per_sec"] = None
            smoke["median_step_s"] = None
            smoke["per_step_times_s"] = []
            smoke["reason"] = (
                "the 1-step smoke failed, so the 50-step run was not started. "
                f"The smoke log is bench/runs/pytorch-smoke-{policy}-{suffix}.json"
            )
            pipelines.append(smoke)
            write_run(f"pytorch-smoke-{policy}-{suffix}.json", smoke)
            continue
        report = pipeline_run(
            ordered,
            split,
            policy,
            port + 20 + index,
            commit,
            digest,
            WARMUP_STEPS,
            MEASURED_STEPS,
            timeout=60 * 60 * 6,
        )
        pipelines.append(report)
        write_run(f"pytorch-pipeline-{policy}-{suffix}.json", report)
    return pipelines


def finished_pipeline(report: dict) -> bool:
    return report.get("status") == "ok" and report.get("measured_steps") == MEASURED_STEPS


def main() -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "logs").mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    digest = source_sha256()
    hosts = []
    failures = []
    for alias in SSH_HOSTS:
        try:
            hosts.append(ensure_host(alias))
        except Exception as exc:
            failures.append({"ssh": alias, "error": str(exc)})
            print(f"host {alias} failed: {exc}", file=sys.stderr, flush=True)
    (RUNS / "host-failures.json").write_text(dump({"failures": failures}), encoding="utf-8")
    trainable = [host for host in hosts if host.get("best_device") and host.get("ipv4")]
    singles = single_runs(trainable, commit, digest, WARMUP_STEPS, MEASURED_STEPS) if trainable else []
    pipelines: list[dict] = []
    pair_gloos: list[dict] = []
    gloo: dict = {"status": "not run", "reason": "fewer than 2 trainable hosts", "attempts": []}
    if len(trainable) >= 2:
        gloo = gloo_check(trainable, 29631)
        write_run("pytorch-gloo.json", {key: value for key, value in gloo.items() if key != "ordered"})
        if gloo.get("status") == "ok":
            pipelines.extend(measure_set(gloo["ordered"], singles, int(gloo["port"]) + 1, commit, digest))
        if not any(finished_pipeline(report) for report in pipelines):
            from itertools import combinations

            pair_port = 29801

            def pair_priority(pair: tuple[dict, ...]) -> tuple:
                cuda = sum(1 for host in pair if host.get("best_device") == "cuda")
                names = tuple(str(host.get("hostname")) for host in pair)
                return (-cuda, names)

            for pair in sorted(combinations(trainable, 2), key=pair_priority):
                subset = list(pair)
                if len(trainable) == 2:
                    break
                sub_gloo = gloo_check(subset, pair_port)
                names = "-".join(str(host["ssh"]) for host in sorted(subset, key=lambda item: str(item["hostname"])))
                recorded = {key: value for key, value in sub_gloo.items() if key != "ordered"}
                recorded["names"] = names
                pair_gloos.append(recorded)
                write_run(f"pytorch-gloo-{names}.json", recorded)
                pair_port += 40
                if sub_gloo.get("status") == "ok":
                    pipelines.extend(
                        measure_set(sub_gloo["ordered"], singles, int(sub_gloo["port"]) + 1, commit, digest)
                    )
    linalg_path = RUNS / "linalg-pipeline.json"
    linalg = json.loads(linalg_path.read_text(encoding="utf-8")) if linalg_path.exists() else None
    text = results_markdown(singles, pipelines, gloo, linalg, hub_inventory(), pair_gloos)
    (REPO / "bench" / "RESULTS.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
