"""TCP throughput between two hosts, one direction per run, like iperf3.

Standard library only, so every host runs it with the Python it already has.

    linkperf.py server PORT
    linkperf.py client HOST PORT MODE SECONDS

MODE ``send`` makes the client the sender (iperf3 default). MODE ``recv``
makes the server the sender (iperf3 ``-R``), so a host whose firewall
refuses inbound connections can still be measured in both directions.

The sender writes for SECONDS, then closes. The receiver counts payload bytes
from its first byte to end of stream and prints one JSON line prefixed with
``__LINK__``. Only the receiver's count is the measurement.
"""

from __future__ import annotations

import json
import socket
import struct
import sys
import time

CHUNK = 128 * 1024
HEADER = struct.Struct("!4sd")  # mode, seconds


def send_for(sock: socket.socket, seconds: float) -> int:
    payload = memoryview(bytes(CHUNK))
    sent = 0
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        sock.sendall(payload)
        sent += CHUNK
    sock.shutdown(socket.SHUT_WR)
    return sent


def receive_all(sock: socket.socket) -> dict:
    buffer = bytearray(CHUNK)
    view = memoryview(buffer)
    received = 0
    first = None
    last = None
    while True:
        count = sock.recv_into(view)
        if count == 0:
            break
        now = time.perf_counter()
        if first is None:
            first = now
        last = now
        received += count
    elapsed = (last - first) if first is not None and last is not None else 0.0
    return {
        "bytes": received,
        "seconds": elapsed,
        "bytes_per_sec": received / elapsed if elapsed > 0 else None,
    }


def report(role: str, local: str, peer: str, measured: dict) -> None:
    line = {"role": role, "receiver": local, "sender": peer, "hostname": socket.gethostname(), **measured}
    print("__LINK__" + json.dumps(line, sort_keys=True), flush=True)


def tune(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def server(port: int) -> int:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("0.0.0.0", port))
    listener.listen(1)
    listener.settimeout(120)
    print("__LISTENING__", flush=True)
    conn, address = listener.accept()
    listener.close()
    tune(conn)
    with conn:
        header = b""
        while len(header) < HEADER.size:
            more = conn.recv(HEADER.size - len(header))
            if not more:
                raise RuntimeError("client closed before header")
            header += more
        mode, seconds = HEADER.unpack(header)
        if mode == b"send":
            report("server", conn.getsockname()[0], address[0], receive_all(conn))
        elif mode == b"recv":
            sent = send_for(conn, seconds)
            conn.recv(1)
            print("__SENT__" + json.dumps({"bytes": sent}), flush=True)
        else:
            raise RuntimeError(f"unknown mode {mode!r}")
    return 0


def client(host: str, port: int, mode: str, seconds: float) -> int:
    conn = socket.create_connection((host, port), timeout=60)
    conn.settimeout(seconds + 120)
    tune(conn)
    with conn:
        conn.sendall(HEADER.pack(mode.encode(), seconds))
        if mode == "send":
            sent = send_for(conn, seconds)
            conn.recv(1)
            print("__SENT__" + json.dumps({"bytes": sent}), flush=True)
        elif mode == "recv":
            report("client", conn.getsockname()[0], host, receive_all(conn))
        else:
            raise RuntimeError(f"unknown mode {mode!r}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "server":
        return server(int(argv[2]))
    if len(argv) == 6 and argv[1] == "client":
        return client(argv[2], int(argv[3]), argv[4], float(argv[5]))
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
