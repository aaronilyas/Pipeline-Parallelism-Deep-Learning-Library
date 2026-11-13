"""One host, one process, workload W. No pipeline and no AMP."""

from __future__ import annotations

import gc
import platform
import socket
import sys
import time
import traceback

import torch

from metrics import (
    LAYERS,
    MEASURED_STEPS,
    WARMUP_STEPS,
    dump,
    median,
    samples_per_sec,
)
from workload import (
    build_mlp,
    candidate_devices,
    describe_runtime,
    full_batch,
    loss_fn,
    optimizer,
    probe_device,
    sync,
)


def run_on(device: torch.device, warmup: int, steps: int) -> dict[str, object]:
    probe_device(device)
    model = build_mlp(LAYERS).to(device)
    opt = optimizer(model)
    criterion = loss_fn()
    inputs, targets = full_batch(device)
    rows = inputs.shape[0] // steps_micro()
    if inputs.shape[0] != rows * steps_micro():
        raise RuntimeError("batch is not divisible by the microbatch count")

    def one_step() -> None:
        opt.zero_grad()
        for index in range(steps_micro()):
            start = index * rows
            stop = start + rows
            prediction = model(inputs[start:stop])
            loss = criterion(prediction, targets[start:stop])
            loss.backward()
        opt.step()

    for _ in range(warmup):
        one_step()
        sync(device)
    times: list[float] = []
    for index in range(steps):
        sync(device)
        started = time.perf_counter()
        one_step()
        sync(device)
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        print(f"single step {index} {elapsed:.4f}s", file=sys.stderr, flush=True)
    middle = median(times)
    return {
        "per_step_times_s": times,
        "median_step_s": middle,
        "samples_per_sec": samples_per_sec(middle),
        "versions": describe_runtime(device),
    }


def steps_micro() -> int:
    from metrics import MICROBATCHES

    return MICROBATCHES


def host_record(device: torch.device) -> dict[str, object]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": device.type,
    }


def main() -> int:
    import json

    with open(sys.argv[1], encoding="utf-8") as handle:
        args_json = json.load(handle)

    class Args:
        commit = args_json["commit"]
        command = args_json["command"]
        source_sha256 = args_json["source_sha256"]
        warmup = int(args_json.get("warmup", WARMUP_STEPS))
        steps = int(args_json.get("steps", MEASURED_STEPS))
        device = args_json.get("device", "")

    args = Args()

    errors: list[str] = []
    chosen: list[torch.device] = []
    if args.device:
        chosen = [torch.device(args.device)]
    else:
        chosen = [device for _, device in candidate_devices()]

    report: dict[str, object] | None = None
    for device in chosen:
        try:
            measured = run_on(device, args.warmup, args.steps)
        except Exception:
            errors.append(f"{device.type}:\n{traceback.format_exc()}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        report = {
            "workload": "W",
            "stack": "pytorch",
            "status": "ok",
            "reason": None,
            "kind": "single",
            "split_policy": None,
            "hosts": [host_record(device)],
            "devices": [device.type],
            "split": [LAYERS],
            "versions": measured["versions"],
            "commit": args.commit,
            "source_sha256": args.source_sha256,
            "command": args.command,
            "warmup_steps": args.warmup,
            "measured_steps": args.steps,
            "per_step_times_s": measured["per_step_times_s"],
            "median_step_s": measured["median_step_s"],
            "samples_per_sec": measured["samples_per_sec"],
            "bytes_sent_per_boundary": [],
            "device_errors": errors,
            "amp": False,
            "dtype": "float32",
        }
        break

    if report is None:
        report = {
            "workload": "W",
            "stack": "pytorch",
            "status": "not run",
            "reason": "no device completed workload W",
            "kind": "single",
            "split_policy": None,
            "hosts": [{"hostname": socket.gethostname(), "platform": platform.platform()}],
            "devices": [],
            "split": None,
            "versions": {"torch": torch.__version__, "torch_cuda": torch.version.cuda},
            "commit": args.commit,
            "source_sha256": args.source_sha256,
            "command": args.command,
            "warmup_steps": args.warmup,
            "measured_steps": args.steps,
            "per_step_times_s": [],
            "median_step_s": None,
            "samples_per_sec": None,
            "bytes_sent_per_boundary": [],
            "device_errors": errors,
            "amp": False,
            "dtype": "float32",
        }
    sys.stdout.write("\n__REPORT__\n")
    sys.stdout.write(dump(report))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
