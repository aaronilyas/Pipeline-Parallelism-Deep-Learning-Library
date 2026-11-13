"""Inventory one host. Prints one JSON object on stdout."""

from __future__ import annotations

import platform
import socket
import subprocess
import sys


def primary_ipv4() -> str | None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.168.1.1", 9))
        return str(sock.getsockname()[0])
    except OSError:
        return None
    finally:
        sock.close()


def cpu_name() -> str:
    if sys.platform == "linux":
        try:
            for line in open("/proc/cpuinfo", encoding="utf-8"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def memory_bytes() -> int | None:
    if sys.platform == "linux":
        try:
            for line in open("/proc/meminfo", encoding="utf-8"):
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) * 1024
        except OSError:
            return None
    if sys.platform == "darwin":
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            return None
    if sys.platform == "win32":
        import ctypes

        class Stat(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = Stat()
        stat.dwLength = ctypes.sizeof(Stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullTotalPhys)
    return None


def main() -> int:
    from metrics import dump

    info: dict[str, object] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": cpu_name(),
        "memory_bytes": memory_bytes(),
        "ipv4": primary_ipv4(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        devices: list[str] = []
        errors: list[str] = []
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = list(torch.cuda.get_device_capability(0))
            info["cuda_arch_list"] = list(torch.cuda.get_arch_list())
            try:
                value = torch.randn(64, 64, device="cuda")
                result = value @ value
                torch.cuda.synchronize()
                if result.shape != (64, 64):
                    raise RuntimeError(f"shape {tuple(result.shape)}")
                devices.append("cuda")
            except Exception as exc:
                errors.append(f"cuda: {exc}")
        if torch.backends.mps.is_available():
            info["gpu"] = info.get("gpu") or "Apple MPS"
            try:
                value = torch.randn(64, 64, device="mps")
                result = value @ value
                torch.mps.synchronize()
                if result.shape != (64, 64):
                    raise RuntimeError(f"shape {tuple(result.shape)}")
                devices.append("mps")
            except Exception as exc:
                errors.append(f"mps: {exc}")
        try:
            value = torch.randn(32, 32)
            result = value @ value
            if result.shape != (32, 32):
                raise RuntimeError(f"shape {tuple(result.shape)}")
            devices.append("cpu")
        except Exception as exc:
            errors.append(f"cpu: {exc}")
        preference = ("cuda", "mps", "cpu")
        best = next((name for name in preference if name in devices), None)
        info["devices_ok"] = devices
        info["best_device"] = best
        info["device_errors"] = errors
    except Exception as exc:
        info["torch_error"] = str(exc)
        info["best_device"] = None
    sys.stdout.write(dump(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
