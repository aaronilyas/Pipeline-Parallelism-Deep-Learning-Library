"""Render bench/RESULTS.md from the JSON files in bench/runs.

Nothing here measures. Every number printed comes from a run file, and the
file is named next to it.

    python3 bench/pytorch/results.py
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from pathlib import Path

from metrics import MEASURED_STEPS

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "bench" / "runs"
WIN_FACTOR = 1.10
PRIOR_B = 6827.089946088578


def load(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    report["file"] = f"bench/runs/{path.name}"
    return report


def maybe(name: str) -> dict | None:
    path = RUNS / name
    return load(path) if path.exists() else None


def fmt(value: object) -> str:
    if value is None:
        return "not run"
    return json.dumps(value)


def runtime_of(report: dict) -> str:
    inventory = report.get("inventory") or {}
    if inventory.get("runtime") == "wsl":
        return f"wsl2 {inventory.get('wsl_distro')} mirrored"
    return "native"


def host_name(host: dict) -> str:
    suffix = "[wsl2]" if host.get("runtime") == "wsl" else ""
    return f"{host.get('ssh')}{suffix}"


def finished(report: dict) -> bool:
    return report.get("status") == "ok" and report.get("measured_steps") == MEASURED_STEPS


def hub_inventory() -> dict:
    cpu = platform.processor() or "unknown"
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    gpu = "none"
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"hostname": socket.gethostname(), "cpu": cpu, "gpu": gpu}


def gloo_ports(report: dict) -> str:
    if report.get("status") == "ok" and report.get("port") is not None:
        return str(report["port"])
    ports = [str(attempt.get("port")) for attempt in report.get("attempts") or [] if attempt.get("port")]
    return ", ".join(ports) or "none"


def gloo_error(report: dict) -> str:
    """The first rank error that names a cause, else the first error."""
    errors = [
        str(rank.get("error"))
        for attempt in report.get("attempts") or []
        for rank in attempt.get("ranks") or []
        if rank.get("error")
    ]
    for error in errors:
        for marker in ("RuntimeError", "enforce fail", "Error:"):
            if marker in error:
                return error
    return errors[0] if errors else str(report.get("reason"))


def pipeline_rows(reports: list[dict]) -> list[str]:
    lines = [
        "| policy | hosts | devices | split | precision | samples/sec | median step s | status | file | commit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for report in reports:
        lines.append(
            "| {policy} | {hosts} | {devices} | `{split}` | f32 | {samples} | {median} | {status} | `{file}` | `{commit}` |".format(
                policy=report.get("split_policy"),
                hosts=",".join(host_name(host) for host in report.get("hosts") or []),
                devices=",".join(str(device) for device in report.get("devices") or []),
                split=report.get("split"),
                samples=fmt(report.get("samples_per_sec")),
                median=fmt(report.get("median_step_s")),
                status=report.get("status"),
                file=report.get("file"),
                commit=report.get("commit"),
            )
        )
    for report in reports:
        if report.get("status") != "ok":
            error = (report.get("rank_errors") or [report.get("reason")])[0]
            hosts = ",".join(host_name(host) for host in report.get("hosts") or [])
            lines.extend(["", f"Exact error for `{report.get('split_policy')}` on {hosts}:", "", "```", str(error)[:6000], "```"])
    return lines


def header(hub: dict) -> list[str]:
    return [
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
        "The CUDA hosts use the CUDA 12.6 wheel (`--index-url https://download.pytorch.org/whl/cu126`),",
        "natively on ubuntu and windows, and inside WSL2 for the windows[wsl2] rows.",
        "The Mac uses the macOS arm64 wheel of the same release. No number here is estimated.",
        "This file is rendered from `bench/runs/` by `python3 bench/pytorch/results.py`.",
        "",
        "## Hub",
        "",
        f"Coordinator `{hub.get('hostname')}` is not a training rank. CPU `{hub.get('cpu')}`. GPU `{hub.get('gpu')}`.",
    ]


def single_section(singles: list[dict]) -> list[str]:
    lines = [
        "",
        "## Single host",
        "",
        "| ssh | runtime | hostname | device | precision | samples/sec | median step s | status | command | file | commit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for report in singles:
        lines.append(
            "| {ssh} | {runtime} | {hostname} | {device} | f32 | {samples} | {median} | {status} | `{command}` | `{file}` | `{commit}` |".format(
                ssh=report.get("ssh"),
                runtime=runtime_of(report),
                hostname=(report.get("inventory") or {}).get("hostname"),
                device=",".join(report.get("devices") or []) or "none",
                samples=fmt(report.get("samples_per_sec")),
                median=fmt(report.get("median_step_s")),
                status=report.get("status"),
                command=report.get("command"),
                file=report.get("file"),
                commit=report.get("commit"),
            )
        )
    by_runtime = {(report.get("ssh"), runtime_of(report) == "native"): report for report in singles}
    native = by_runtime.get(("windows", True))
    wsl = by_runtime.get(("windows", False))
    if native and wsl and finished(native) and finished(wsl):
        ratio = float(wsl["samples_per_sec"]) / float(native["samples_per_sec"])
        lines.extend(
            [
                "",
                f"windows inside WSL2 runs W at {ratio:.4f} times its native samples/sec, on the same GPU and driver.",
            ]
        )
    return lines


def link_section(links: dict | None, b_value: float) -> list[str]:
    lines = ["", "## Link matrix", ""]
    if links is None:
        return lines + ["not run."]
    win = WIN_FACTOR * b_value
    lines.extend(
        [
            f"Tool: {links['tool']}. Command: `{links['command']}` on the hub, commit `{links['commit']}`.",
            f"Each row is one TCP stream for {fmt(links['duration_s'])} s, one direction at a time, counted at the receiver.",
            "The ceilings are `bytes/s / (4*4096)` for f32 and `bytes/s / (2*4096)` for bf16:",
            "every sample crosses a boundary once each way, so a one-way rate caps samples/sec.",
            "Wi-Fi shares airtime between directions. A pipeline sends both ways at once,",
            "so on a Wi-Fi link these one-way ceilings are upper bounds.",
            "",
            "| from | to | from medium | to medium | seconds | Mbit/s | bytes/s | f32 ceiling samples/s | bf16 ceiling samples/s | status |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    endpoints = links["endpoints"]
    for run in links["runs"]:
        lines.append(
            "| {src} | {dst} | {src_medium} | {dst_medium} | {seconds} | {mbit} | {rate} | {f32} | {bf16} | {status} |".format(
                src=run["from"],
                dst=run["to"],
                src_medium=endpoints[run["from"]]["medium"],
                dst_medium=endpoints[run["to"]]["medium"],
                seconds=fmt(run["duration_s"]),
                mbit=fmt(run.get("mbit_per_sec")),
                rate=fmt(run.get("bytes_per_sec")),
                f32=fmt(run.get("f32_ceiling_samples_per_sec")),
                bf16=fmt(run.get("bf16_ceiling_samples_per_sec")),
                status=run["status"],
            )
        )
    lines.extend(["", f"WIN line = 1.10 x B = {fmt(win)} samples/sec. Per link, the slower direction sets the ceiling:", ""])
    pairs: dict[tuple[str, str], list[dict]] = {}
    for run in links["runs"]:
        pairs.setdefault(tuple(sorted((run["from"], run["to"]))), []).append(run)
    for (left, right), runs in pairs.items():
        ok = [run for run in runs if run["status"] == "ok"]
        if len(ok) != 2:
            lines.append(f"- {left} <-> {right}: not measured both ways.")
            continue
        slow = min(ok, key=lambda run: run["bytes_per_sec"])
        f32 = slow["f32_ceiling_samples_per_sec"]
        bf16 = slow["bf16_ceiling_samples_per_sec"]
        lines.append(
            f"- {left} <-> {right}: f32 ceiling {f32:.1f}, bf16 ceiling {bf16:.1f} samples/sec "
            f"({f32 / win:.3f} and {bf16 / win:.3f} of the WIN line)."
        )
    lines.extend(["", "Raw medium readings per host are in the JSON under `medium_raw`."])
    return lines


def subset_section(subsets: list[dict]) -> list[str]:
    if not subsets:
        return []
    lines = ["", "### Subset groups", "", "These groups were tried after the full set did not produce a finished 50-step pipeline.", ""]
    for report in subsets:
        lines.append(
            f"- `{report.get('names') or 'subset'}` status `{report.get('status')}` port `{gloo_ports(report)}` (`{report['file']}`)"
        )
        if report.get("status") != "ok":
            lines.extend(["", "```", json.dumps(report.get("attempts", report.get("reason")), indent=2)[:8000], "```", ""])
    return lines


def prompt1_pipeline_section(gloo: dict | None, subsets: list[dict], pipelines: list[dict]) -> list[str]:
    lines = ["", "## Pipeline", ""]
    if gloo is not None and gloo.get("status") != "ok":
        lines.extend(
            [
                f"Gloo did not form a group on the full native host set (port `{gloo_ports(gloo)}`, `{gloo['file']}`). Attempts:",
                "",
                "```",
                json.dumps(gloo.get("attempts", gloo.get("reason")), indent=2)[:12000],
                "```",
                "",
            ]
        )
    lines.extend(pipeline_rows(pipelines))
    lines.extend(subset_section(subsets))
    return lines


def wsl_section(summary: dict | None) -> list[str]:
    lines = ["", "## Same-OS pipeline (WSL2)", ""]
    if summary is None:
        return lines + ["not run."]
    pair = maybe(Path(summary["pair_gloo"]).name) if summary.get("pair_gloo") else None
    lines.extend(
        [
            "windows runs torch 2.14.1 + CUDA 12.6 inside its WSL2 distro with mirrored networking,",
            "so both ranks are Linux builds of the same wheel. ubuntu holds the TCPStore.",
            f"Command: `{summary['command']}` on the hub, commit `{summary['commit']}`.",
            "",
        ]
    )
    if pair is not None:
        lines.append(
            f"- gloo `{pair.get('names')}` status `{pair.get('status')}` port `{gloo_ports(pair)}` (`{pair['file']}`)"
        )
        if pair.get("status") != "ok":
            lines.extend(["", "Exact error:", "", "```", gloo_error(pair)[:6000], "```"])
    reports = []
    for entry in summary.get("pipelines") or []:
        path = REPO / entry["file"]
        if path.exists():
            reports.append(load(path))
    wsl_reports = [r for r in reports if all(h.get("ssh") != "macmini" for h in r.get("hosts") or [])]
    mac_reports = [r for r in reports if r not in wsl_reports]
    lines.append("")
    lines.extend(pipeline_rows(wsl_reports))
    for report in wsl_reports:
        if report.get("split_speeds_samples_per_sec"):
            lines.extend(
                [
                    "",
                    f"The proportional split `{report.get('split')}` follows single-host samples/sec "
                    f"`{report['split_speeds_samples_per_sec']}` in rank order "
                    f"({','.join(host_name(h) for h in report.get('hosts') or [])}).",
                ]
            )
        for boundary in report.get("bytes_sent_per_boundary") or []:
            lines.append(
                f"- `{report.get('split_policy')}` boundary {boundary['from_rank']}->{boundary['to_rank']}: "
                f"{boundary['forward_bytes']} B forward, {boundary['backward_bytes']} B backward over {MEASURED_STEPS} steps."
            )
    lines.extend(["", "### macmini", ""])
    for entry in summary.get("macmini_gloo") or []:
        report = maybe(Path(entry["file"]).name)
        if report is None:
            continue
        if report.get("status") == "ok":
            lines.append(f"- `{report.get('names')}`: gloo formed on port `{gloo_ports(report)}` (`{report['file']}`).")
        else:
            lines.extend(
                [
                    f"- `{report.get('names')}`: macmini not formable on port `{gloo_ports(report)}` (`{report['file']}`). Exact error:",
                    "",
                    "```",
                    gloo_error(report)[:4000],
                    "```",
                    "",
                ]
            )
    if mac_reports:
        lines.extend(pipeline_rows(mac_reports))
    return lines


def describe(report: dict) -> str:
    if report.get("kind") == "pipeline":
        hosts = ",".join(host_name(host) for host in report.get("hosts") or [])
        return f"{report.get('split_policy')} split `{report.get('split')}` Schedule1F1B pipeline on {hosts} over gloo"
    return f"{report.get('ssh')} single host ({','.join(report.get('devices') or [])}, {runtime_of(report)})"


def baseline(singles: list[dict], pipelines: list[dict]) -> dict | None:
    candidates = [report for report in [*singles, *pipelines] if finished(report)]
    if not candidates:
        return None
    return max(candidates, key=lambda report: float(report["samples_per_sec"]))


def baseline_section(chosen: dict | None, singles: list[dict], pipelines: list[dict]) -> list[str]:
    lines = ["", "## Baseline B", ""]
    if chosen is None:
        return lines + ["No PyTorch configuration finished. There is no samples/sec baseline."]
    value = float(chosen["samples_per_sec"])
    lines.append(f"Baseline B: {fmt(value)} samples/sec on {describe(chosen)}, `{chosen['file']}`.")
    lines.append("")
    lines.append(
        "B is the largest finished f32 configuration above: every single host and every pipeline that formed. "
        f"WIN means at least {fmt(WIN_FACTOR * value)} samples/sec."
    )
    ok_pipes = [report for report in pipelines if finished(report)]
    best_single = baseline(singles, [])
    if ok_pipes and best_single is not None:
        best_pipe = max(ok_pipes, key=lambda report: float(report["samples_per_sec"]))
        lines.append(
            f"The fastest pipeline ({describe(best_pipe)}) runs at {fmt(best_pipe['samples_per_sec'])}; "
            f"the fastest single host ({describe(best_single)}) at {fmt(best_single['samples_per_sec'])}."
        )
    elif not ok_pipes:
        lines.append("No PyTorch pipeline finished, so B is the fastest single host.")
    if abs(value - PRIOR_B) > 1e-9:
        lines.append(f"The previous B was {fmt(PRIOR_B)} (windows native single host).")
    return lines


def linalg_section(linalg: dict | None) -> list[str]:
    lines = ["", "## linalg-pipeline", ""]
    if linalg is None:
        return lines + ["not run. The bench JSON was not produced."]
    return lines + [
        f"Status: {linalg.get('status')}.",
        "",
        linalg.get("reason") or "",
        "",
        f"Command: `{linalg.get('command')}`",
        f"Commit: `{linalg.get('commit')}`",
    ]


def render() -> str:
    summary = maybe("pytorch-wsl2.json")
    wsl_files = set()
    if summary is not None:
        wsl_files.add(Path(summary.get("pair_gloo") or "").name)
        wsl_files.update(Path(entry["file"]).name for entry in summary.get("macmini_gloo") or [])
        wsl_files.update(Path(entry["file"]).name for entry in summary.get("pipelines") or [])
    singles = [load(path) for path in sorted(RUNS.glob("pytorch-single-*.json"))]
    singles.sort(key=lambda report: (str(report.get("ssh")), runtime_of(report) != "native"))
    gloo = maybe("pytorch-gloo.json")
    subsets = [load(path) for path in sorted(RUNS.glob("pytorch-gloo-*.json")) if path.name not in wsl_files]
    all_pipes = [load(path) for path in sorted(RUNS.glob("pytorch-pipeline-*.json"))]
    prompt1_pipes = [report for report in all_pipes if Path(report["file"]).name not in wsl_files]
    chosen = baseline(singles, all_pipes)
    b_value = float(chosen["samples_per_sec"]) if chosen else PRIOR_B
    lines = header(hub_inventory())
    lines.extend(single_section(singles))
    lines.extend(link_section(maybe("link-matrix.json"), b_value))
    lines.extend(prompt1_pipeline_section(gloo, subsets, prompt1_pipes))
    lines.extend(wsl_section(summary))
    lines.extend(baseline_section(chosen, singles, all_pipes))
    lines.extend(linalg_section(maybe("linalg-pipeline.json")))
    lines.extend(["", "Per-step times and byte counts are in `bench/runs/`.", ""])
    return "\n".join(lines)


def main() -> int:
    text = render()
    (REPO / "bench" / "RESULTS.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
