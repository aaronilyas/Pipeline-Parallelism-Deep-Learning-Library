"""Form a gloo group and move one tensor to each neighbor. Prints one JSON report."""

from __future__ import annotations

import faulthandler
import json
import socket
import sys
import traceback
from datetime import timedelta

faulthandler.enable()

import torch
import torch.distributed as dist

from rendezvous import init_gloo


def main() -> int:
    from metrics import dump

    with open(sys.argv[1], encoding="utf-8") as handle:
        job = json.load(handle)
    report: dict[str, object] = {
        "hostname": socket.gethostname(),
        "rank": job["rank"],
        "world": job["world"],
        "status": "not run",
        "error": None,
    }
    try:
        init_gloo(
            str(job["master"]),
            int(job["port"]),
            int(job["rank"]),
            int(job["world"]),
            bool(job["store_master"]),
            timedelta(minutes=5),
            str(job.get("gloo_ifname") or "") or None,
        )
        value = torch.ones(1, dtype=torch.float32)
        dist.all_reduce(value)
        if float(value.item()) != float(job["world"]):
            raise RuntimeError(f"all_reduce sum {value.item()} != {job['world']}")
        world = int(job["world"])
        rank = int(job["rank"])
        if world > 1:
            payload = torch.tensor([float(rank)], dtype=torch.float32)
            received = torch.zeros(1, dtype=torch.float32)
            dst = (rank + 1) % world
            src = (rank - 1) % world
            if rank == 0:
                dist.send(payload, dst=dst)
                dist.recv(received, src=src)
            else:
                dist.recv(received, src=src)
                dist.send(payload, dst=dst)
            expected = float(src)
            if float(received.item()) != expected:
                raise RuntimeError(f"ring payload {received.item()} != {expected}")
        report["status"] = "ok"
        report["all_reduce"] = float(value.item())
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    sys.stdout.write("\n__REPORT__\n")
    sys.stdout.write(dump(report))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
