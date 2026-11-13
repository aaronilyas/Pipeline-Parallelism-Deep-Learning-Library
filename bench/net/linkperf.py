"""TCP throughput between two hosts, one direction per run, like iperf3.

Standard library only, so every host runs it with the Python it already has.

    linkperf.py server PORT
    linkperf.py client HOST PORT MODE SECONDS
    linkperf.py count            (receiver side of `nc -l PORT | linkperf.py count`)
    linkperf.py emit SECONDS     (sender side of `linkperf.py emit S | nc -l PORT`)

MODE ``send`` makes the client the sender (iperf3 default). MODE ``recv``
makes the server the sender (iperf3 ``-R``), so a host whose firewall
refuses inbound connections can still be measured in both directions.

MODE ``rawsend`` / ``rawrecv`` skip the header, for a listener that is a
plain ``nc -l``: macOS's application firewall admits the system-signed
``/usr/bin/nc`` but not an unsigned Python. ``count`` and ``emit`` are the
Python ends of that pipe, so the timing and counting stay here.

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
    return receive_stream(sock.recv_into)


def receive_stream(read_into) -> dict:  # type: ignore[no-untyped-def]
    buffer = bytearray(CHUNK)
    view = memoryview(buffer)
    received = 0
    first = None
    last = None
    while True:
        count = read_into(view)
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


def emit(seconds: float) -> int:
    """Write zeros to stdout. The clock starts once the pipe and nc buffers
    are full, that is, once a peer is draining them."""
    out = sys.stdout.buffer
    payload = bytes(CHUNK)
    primed = 0
    while primed < 8 * 1024 * 1024:
        out.write(payload)
        primed += CHUNK
    sent = 0
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        out.write(payload)
        sent += CHUNK
    out.flush()
    print("__SENT__" + json.dumps({"bytes": sent + primed}), file=sys.stderr, flush=True)
    return 0


def count() -> int:
    measured = receive_stream(sys.stdin.buffer.readinto)
    report("nc-pipe", socket.gethostname(), "nc", measured)
    return 0


def connect(host: str, port: int) -> socket.socket:
    """Retry while a just-started ``nc -l`` is not yet bound."""
    deadline = time.monotonic() + 30
    while True:
        try:
            return socket.create_connection((host, port), timeout=60)
        except ConnectionRefusedError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


def client(host: str, port: int, mode: str, seconds: float) -> int:
    conn = connect(host, port)
    conn.settimeout(seconds + 120)
    tune(conn)
    with conn:
        if mode == "rawsend":
            sent = send_for(conn, seconds)
            print("__SENT__" + json.dumps({"bytes": sent}), flush=True)
            return 0
        if mode == "rawrecv":
            report("client", conn.getsockname()[0], host, receive_all(conn))
            return 0
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
    if len(argv) == 2 and argv[1] == "count":
        return count()
    if len(argv) == 3 and argv[1] == "emit":
        return emit(float(argv[2]))
    if len(argv) == 6 and argv[1] == "client":
        return client(argv[2], int(argv[3]), argv[4], float(argv[5]))
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
