"""Gloo rendezvous on the IPv4 address chosen by the orchestrator.

The libuv TCPStore maps that address through reverse DNS. On this cluster the
reverse name is not a connectable address (Windows error 10049). The legacy
store connects to the address string it is given. ``GLOO_SOCKET_IFNAME`` is
the LAN interface, so the process group does not advertise docker0 or the
WSL virtual adapter.
"""

from __future__ import annotations

import os
from datetime import timedelta

import torch.distributed as dist


def init_gloo(
    master: str,
    port: int,
    rank: int,
    world: int,
    store_master: bool,
    timeout: timedelta,
    ifname: str | None = None,
) -> None:
    # Gloo otherwise advertises docker0 or the WSL virtual adapter.
    if ifname:
        os.environ["GLOO_SOCKET_IFNAME"] = ifname
    store = dist.TCPStore(
        master,
        int(port),
        world_size=int(world),
        is_master=bool(store_master),
        timeout=timeout,
        wait_for_workers=True,
        use_libuv=False,
    )
    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=int(rank),
        world_size=int(world),
        timeout=timeout,
    )
