"""Measure TCP throughput both ways for every pair of training hosts.

The hub runs this. Each direction is one 30 s run of linkperf.py. The
listener is a host that accepts inbound TCP; the other side connects and
either sends or, with ``recv``, receives. Only ubuntu admits an inbound
Python socket: windows (Public profile) and its WSL2 guest drop them, and
macOS's application firewall drops the unsigned venv Python. So for
macmini <-> windows the Mac listens with the system-signed /usr/bin/nc,
piped to linkperf.py, which still does the timing and counting.
Runs go one at a time so no two measurements share the air or the wire.

Writes bench/runs/link-matrix.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = REPO / "bench" / "runs"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
SCP = ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
SECONDS = 30.0
PORT = 5201
WIDTH = 4096
F32_BYTES_PER_SAMPLE = 4 * WIDTH
BF16_BYTES_PER_SAMPLE = 2 * WIDTH
WSL_DISTRO = "Ubuntu-26.04"


def probe(alias: str) -> dict:
    return json.loads((RUNS / "probes" / f"{alias}.json").read_text(encoding="utf-8"))


# endpoint name -> how to run linkperf.py there
ENDPOINTS: dict[str, dict] = {
    "ubuntu": {
        "ssh": "ubuntu",
        "run": "~/ppdl-w-bench/venv/bin/python ~/ppdl-w-bench/net/linkperf.py",
        "runtime": "native",
    },
    "macmini": {
        "ssh": "macmini",
        "run": "~/ppdl-w-bench/venv/bin/python ~/ppdl-w-bench/net/linkperf.py",
        "runtime": "native",
    },
    "windows": {
        "ssh": "windows",
        "run": (
            "& $env:USERPROFILE\\ppdl-w-bench\\venv\\Scripts\\python.exe "
            "$env:USERPROFILE\\ppdl-w-bench\\net\\linkperf.py"
        ),
        "runtime": "native",
    },
    "windows-wsl": {
        "ssh": "windows",
        "run": f"wsl -d {WSL_DISTRO} -e python3 /mnt/c/Users/aaron/ppdl-w-bench/net/linkperf.py",
        "runtime": f"wsl2 {WSL_DISTRO} mirrored",
    },
}

# (listener, connector, listener runs nc). Every pair of the three training
# hosts, plus the WSL2 path the same-OS pipeline uses.
PAIRS = [
    ("ubuntu", "macmini", False),
    ("ubuntu", "windows", False),
    ("macmini", "windows", True),
    ("ubuntu", "windows-wsl", False),
]


def ssh(alias: str, command: str, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*SSH, alias, command], capture_output=True, text=True, timeout=timeout, check=False)


def ship(alias: str) -> None:
    if alias == "windows":
        ssh(alias, "New-Item -ItemType Directory -Force -Path $env:USERPROFILE\\ppdl-w-bench\\net | Out-Null", 60)
    else:
        ssh(alias, "mkdir -p ~/ppdl-w-bench/net", 60)
    subprocess.check_call([*SCP, str(HERE / "linkperf.py"), f"{alias}:ppdl-w-bench/net/linkperf.py"])


MEDIUM_COMMANDS = {
    "ubuntu": (
        "for f in speed duplex; do printf '%s=' $f; cat /sys/class/net/enp3s0/$f; done; "
        "test -d /sys/class/net/enp3s0/wireless && echo wireless=yes || echo wireless=no; "
        "journalctl -k -b --no-pager 2>/dev/null | grep 'enp3s0: Link is Up' | tail -1"
    ),
    "macmini": (
        "route -n get default | grep interface; networksetup -listallhardwareports | grep -A1 'Wi-Fi'; "
        "ifconfig en0 | grep status; "
        "system_profiler SPAirPortDataType | grep -A8 'Current Network Information' | "
        "grep -E 'PHY Mode|Channel|Transmit Rate'"
    ),
    "windows": (
        "Get-NetAdapter | Where-Object Status -eq Up | "
        "Format-Table -AutoSize Name,InterfaceDescription,LinkSpeed,MediaType | Out-String -Width 200; "
        "netsh wlan show interfaces"
    ),
}

MEDIUM_SUMMARY = {
    "ubuntu": "Ethernet, r8169 NIC, negotiated 100 Mb/s full duplex (kernel: 'downshifted')",
    "macmini": "Wi-Fi 802.11ax (en1), 5 GHz ch 149 80 MHz; Ethernet en0 inactive",
    "windows": "Wi-Fi 802.11ax (Intel AX211), 2.4 GHz ch 1; no wired link up",
    "windows-wsl": "WSL2 mirrored networking over the windows 2.4 GHz Wi-Fi adapter",
}


def medium(alias: str) -> str:
    result = ssh(alias, MEDIUM_COMMANDS[alias], 120)
    return (result.stdout + result.stderr).strip()


def run_direction(listener: str, connector: str, mode: str, port: int, via_nc: bool) -> dict:
    """mode 'send': connector -> listener. mode 'recv': listener -> connector."""
    left = ENDPOINTS[listener]
    right = ENDPOINTS[connector]
    host_ip = probe(left["ssh"])["ipv4"]
    if via_nc and mode == "send":
        server_cmd = f"echo __LISTENING__; nc -l {port} | {left['run']} count"
        client_mode = "rawsend"
    elif via_nc:
        server_cmd = f"echo __LISTENING__; {left['run']} emit {int(SECONDS)} | nc -l {port}"
        client_mode = "rawrecv"
    else:
        server_cmd = f"{left['run']} server {port}"
        client_mode = mode
    client_cmd = f"{right['run']} client {host_ip} {port} {client_mode} {int(SECONDS)}"
    started = datetime.now(timezone.utc).isoformat()
    server = subprocess.Popen(
        [*SSH, left["ssh"], server_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert server.stdout is not None
    line = server.stdout.readline()
    if "__LISTENING__" not in line:
        server.kill()
        _, err = server.communicate()
        raise RuntimeError(f"{listener} server did not listen: {line} {err}")
    try:
        client = ssh(right["ssh"], client_cmd, SECONDS + 180)
        server_out, server_err = server.communicate(timeout=180)
    finally:
        if server.poll() is None:
            server.kill()
    text = server_out + "\n" + client.stdout
    links = [json.loads(row[len("__LINK__"):]) for row in text.splitlines() if row.startswith("__LINK__")]
    sender, receiver = (connector, listener) if mode == "send" else (listener, connector)
    record = {
        "from": sender,
        "to": receiver,
        "listener": listener,
        "listener_tool": "nc -l | linkperf.py" if via_nc else "linkperf.py server",
        "mode": mode,
        "duration_s": SECONDS,
        "started_utc": started,
        "commands": [f"ssh {left['ssh']} {server_cmd}", f"ssh {right['ssh']} {client_cmd}"],
        "status": "not run",
        "error": None,
    }
    if len(links) != 1 or not links[0].get("bytes_per_sec"):
        record["error"] = (client.stderr + "\n" + server_err)[-4000:]
        return record
    measured = links[0]
    rate = float(measured["bytes_per_sec"])
    record.update(
        {
            "status": "ok",
            "bytes": measured["bytes"],
            "seconds": measured["seconds"],
            "bytes_per_sec": rate,
            "mbit_per_sec": rate * 8 / 1e6,
            "f32_ceiling_samples_per_sec": rate / F32_BYTES_PER_SAMPLE,
            "bf16_ceiling_samples_per_sec": rate / BF16_BYTES_PER_SAMPLE,
        }
    )
    return record


def main() -> int:
    commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    for alias in ("ubuntu", "macmini", "windows"):
        ship(alias)
    media = {alias: medium(alias) for alias in ("ubuntu", "macmini", "windows")}
    runs = []
    port = PORT
    for listener, connector, via_nc in PAIRS:
        for mode in ("send", "recv"):
            record = run_direction(listener, connector, mode, port, via_nc)
            port += 1
            print(
                f"{record['from']} -> {record['to']}: {record['status']} "
                f"{record.get('mbit_per_sec')} Mbit/s",
                flush=True,
            )
            runs.append(record)
            time.sleep(2)
    out = {
        "tool": "bench/net/linkperf.py (iperf3-equivalent TCP stream, receiver-side count)",
        "command": "python3 bench/net/link_matrix.py",
        "hub": "omarchy",
        "commit": commit,
        "duration_s": SECONDS,
        "f32_bytes_per_sample_per_direction": F32_BYTES_PER_SAMPLE,
        "bf16_bytes_per_sample_per_direction": BF16_BYTES_PER_SAMPLE,
        "endpoints": {
            name: {
                "ssh": spec["ssh"],
                "ipv4": probe(spec["ssh"])["ipv4"],
                "runtime": spec["runtime"],
                "medium": MEDIUM_SUMMARY[name],
            }
            for name, spec in ENDPOINTS.items()
        },
        "medium_raw": media,
        "runs": runs,
    }
    path = RUNS / "link-matrix.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0 if all(run["status"] == "ok" for run in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
