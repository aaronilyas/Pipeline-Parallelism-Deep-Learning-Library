"""Same-OS PyTorch pipeline: windows inside WSL2 (mirrored) with ubuntu over gloo.

The hub runs this after orchestrate.py. Workload W is unchanged. Steps:

1. windows is probed inside its WSL2 distro, and W runs there single-host.
2. gloo is tried on {windows[wsl2], ubuntu}. If it forms, Schedule1F1B runs
   with the equal split and with layers proportional to single-host speed.
3. macmini is tried in a gloo group with the Linux ranks. It joins a pipeline
   only if gloo forms with it.

Every attempt is a JSON file in bench/runs. bench/runs/pytorch-wsl2.json
lists them. bench/pytorch/results.py renders RESULTS.md from those files.
"""

from __future__ import annotations

import json
import sys

import orchestrate as orch
from metrics import MEASURED_STEPS, WARMUP_STEPS, dump

GLOO_PORT = 30101
PIPELINE_PORT = 30131
MAC_PORT = 30201


def single_for(host: dict) -> dict:
    """The host's own single-host file on its best device."""
    label = host.get("label", host["ssh"])
    path = orch.RUNS / f"pytorch-single-{label}-{host['best_device']}.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report.setdefault("ssh", host["ssh"])
    report.setdefault("inventory", {}).setdefault("label", label)
    report["file"] = f"bench/runs/{path.name}"
    return report


def entry(report: dict) -> dict:
    """The run file measure_set wrote: the 50-step file, else the failed smoke."""
    suffix = "-".join(str(host.get("label") or host["ssh"]) for host in report.get("hosts") or [])
    policy = report.get("split_policy")
    name = f"pytorch-pipeline-{policy}-{suffix}.json"
    if not (orch.RUNS / name).exists():
        name = f"pytorch-smoke-{policy}-{suffix}.json"
    return {"policy": policy, "status": report.get("status"), "file": f"bench/runs/{name}"}


def gloo_attempt(hosts: list[dict], port: int) -> tuple[dict, str]:
    result = orch.gloo_check(hosts, port)
    names = "-".join(str(host.get("label", host["ssh"])) for host in result["ordered"])
    recorded = {key: value for key, value in result.items() if key != "ordered"}
    recorded["names"] = names
    recorded["ports_tried"] = (
        [result["port"]] if result.get("status") == "ok" else [a["port"] for a in result.get("attempts", [])]
    )
    name = f"pytorch-gloo-{names}.json"
    orch.write_run(name, recorded)
    print(f"gloo {names}: {result.get('status')}", flush=True)
    return result, f"bench/runs/{name}"


def main() -> int:
    commit = orch.git_commit()
    digest = orch.source_sha256()
    summary: dict = {
        "commit": commit,
        "command": "python3 bench/pytorch/wsl_pair.py",
        "wsl_single": None,
        "pair_gloo": None,
        "pipelines": [],
        "macmini_gloo": [],
    }
    wsl = orch.ensure_host("windows", force_wsl=True)
    ubuntu = orch.ensure_host("ubuntu")
    if wsl.get("runtime") != "wsl":
        raise RuntimeError("windows did not run inside WSL")
    singles = orch.single_runs([wsl], commit, digest, WARMUP_STEPS, MEASURED_STEPS)
    wsl_single = singles[0]
    summary["wsl_single"] = f"bench/runs/pytorch-single-{wsl['label']}-{wsl['best_device']}.json"
    speeds = [wsl_single, single_for(ubuntu)]

    pair, pair_file = gloo_attempt([wsl, ubuntu], GLOO_PORT)
    summary["pair_gloo"] = pair_file
    if pair.get("status") == "ok":
        for report in orch.measure_set(pair["ordered"], speeds, PIPELINE_PORT, commit, digest):
            summary["pipelines"].append(entry(report))

    macmini = orch.ensure_host("macmini")
    port = MAC_PORT
    for group in ([wsl, ubuntu, macmini], [macmini, ubuntu], [macmini, wsl]):
        result, path = gloo_attempt(group, port)
        port += 40
        summary["macmini_gloo"].append({"file": path, "status": result.get("status")})
        if result.get("status") == "ok":
            mac_speeds = [single_for(host) if host["ssh"] != "windows" else wsl_single for host in group]
            for report in orch.measure_set(result["ordered"], mac_speeds, port, commit, digest):
                    summary["pipelines"].append(entry(report))
            port += 40
    (orch.RUNS / "pytorch-wsl2.json").write_text(dump(summary), encoding="utf-8")
    print(dump(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
