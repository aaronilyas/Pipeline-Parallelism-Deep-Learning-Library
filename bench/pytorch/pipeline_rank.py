"""One rank of torch.distributed.pipelining Schedule1F1B over gloo.

Bytes are the isend payloads observed by the schedule's P2P helper during
the measured steps. Warmup traffic, including shape metadata, is not counted.
``dist.isend`` is left as the real function so the direct-send path stays
selected.
"""

from __future__ import annotations

import faulthandler
import platform

faulthandler.enable()
import socket
import sys
import time
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, Schedule1F1B
from torch.distributed.pipelining import schedules as pipeline_schedules

from metrics import MEASURED_STEPS, MICROBATCH_ROWS, MICROBATCHES, WIDTH, WARMUP_STEPS, dump
from rendezvous import init_gloo
from workload import build_mlp, describe_runtime, loss_fn, optimizer, probe_device, sync


class SendCounter:
    """Count payload bytes that Schedule1F1B hands to its P2P helper.

    ``P2POp`` rejects any op other than the real ``dist.isend`` / ``dist.irecv``.
    Replacing those names would also make ``_batch_p2p`` miss its direct-isend
    path. The helper is patched instead, and ``dist.isend`` stays the original.
    A multi-group batch is counted on the single-group calls it splits into.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.by_peer: dict[int, int] = {}
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        original = pipeline_schedules._batch_p2p

        def wrapped(p2p_ops, desc=None):  # type: ignore[no-untyped-def]
            if self.enabled and p2p_ops:
                groups = {op.group for op in p2p_ops}
                if len(groups) == 1:
                    for op in p2p_ops:
                        if op.op is dist.isend and op.peer is not None:
                            nbytes = int(op.tensor.numel()) * int(op.tensor.element_size())
                            peer = int(op.peer)
                            self.by_peer[peer] = self.by_peer.get(peer, 0) + nbytes
            return original(p2p_ops, desc=desc)

        pipeline_schedules._batch_p2p = wrapped
        self._installed = True


COUNTER = SendCounter()


def main() -> int:
    import json

    with open(sys.argv[1], encoding="utf-8") as handle:
        args_json = json.load(handle)

    class Args:
        rank = int(args_json["rank"])
        world = int(args_json["world"])
        master = args_json["master"]
        port = int(args_json["port"])
        layers = int(args_json["layers"])
        commit = args_json["commit"]
        command = args_json["command"]
        source_sha256 = args_json["source_sha256"]
        split_policy = args_json["split_policy"]
        warmup = int(args_json.get("warmup", WARMUP_STEPS))
        steps = int(args_json.get("steps", MEASURED_STEPS))
        device = args_json.get("device", "")

    args = Args()

    report: dict[str, object] = {
        "workload": "W",
        "stack": "pytorch",
        "status": "not run",
        "reason": None,
        "kind": "pipeline",
        "split_policy": args.split_policy,
        "rank": args.rank,
        "world": args.world,
        "layers": args.layers,
        "hosts": [{"hostname": socket.gethostname(), "platform": platform.platform()}],
        "devices": [],
        "split": None,
        "versions": {},
        "commit": args.commit,
        "source_sha256": args.source_sha256,
        "command": args.command,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "per_step_times_s": [],
        "median_step_s": None,
        "samples_per_sec": None,
        "bytes_sent_to_rank": {},
        "amp": False,
        "dtype": "float32",
        "error": None,
    }
    try:
        if args.device:
            device = torch.device(args.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        probe_device(device)
        report["devices"] = [device.type]
        report["hosts"][0]["device"] = device.type  # type: ignore[index]
        init_gloo(
            args.master,
            args.port,
            args.rank,
            args.world,
            bool(args_json["store_master"]),
            timedelta(minutes=30),
            str(args_json.get("gloo_ifname") or "") or None,
        )
        COUNTER.install()
        module = build_mlp(args.layers).to(device)
        stage = PipelineStage(module, args.rank, args.world, device)
        schedule = Schedule1F1B(stage, n_microbatches=MICROBATCHES, loss_fn=loss_fn())
        opt = optimizer(module)
        generator = torch.Generator()
        generator.manual_seed(0)
        whole = torch.randn(MICROBATCHES * MICROBATCH_ROWS, WIDTH, generator=generator)
        if args.rank == 0:
            inputs = whole.to(device)
            targets = None
        elif args.rank + 1 == args.world:
            inputs = None
            generator.manual_seed(1)
            targets = torch.randn(MICROBATCHES * MICROBATCH_ROWS, WIDTH, generator=generator).to(device)
        else:
            inputs = None
            targets = None

        def one_step() -> None:
            opt.zero_grad()
            if args.rank == 0:
                schedule.step(inputs)
            elif args.rank + 1 == args.world:
                losses: list[torch.Tensor] = []
                schedule.step(target=targets, losses=losses)
            else:
                schedule.step()
            opt.step()

        for _ in range(args.warmup):
            dist.barrier()
            sync(device)
            one_step()
            sync(device)
        times: list[float] = []
        COUNTER.by_peer.clear()
        COUNTER.enabled = True
        for index in range(args.steps):
            dist.barrier()
            sync(device)
            started = time.perf_counter()
            one_step()
            sync(device)
            elapsed = time.perf_counter() - started
            times.append(elapsed)
            print(f"pipeline rank {args.rank} step {index} {elapsed:.4f}s", file=sys.stderr, flush=True)
        COUNTER.enabled = False
        dist.barrier()
        report["status"] = "ok"
        report["per_step_times_s"] = times
        report["versions"] = describe_runtime(device)
        report["bytes_sent_to_rank"] = {str(peer): nbytes for peer, nbytes in sorted(COUNTER.by_peer.items())}
    except Exception:
        report["error"] = traceback.format_exc()
        report["reason"] = "pipeline rank failed"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    sys.stdout.write("\n__REPORT__\n")
    sys.stdout.write(dump(report))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
