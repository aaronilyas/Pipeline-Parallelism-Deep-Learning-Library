"""Workload W: 12-layer MLP, y = gelu(x @ W + b), f32, MSE, SGD.

Parameter initialization matches ``nn.Linear`` defaults. AMP is not used.
``torch.backends.cuda.matmul.allow_tf32`` is left at PyTorch's default.
"""

from __future__ import annotations

import math
import sys

import torch
from torch import nn

from metrics import LAYERS, LR, MICROBATCH_ROWS, MICROBATCHES, WIDTH

SAMPLES = MICROBATCH_ROWS * MICROBATCHES


class Layer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, width))
        self.bias = nn.Parameter(torch.empty(width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.shape[1]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x @ self.weight + self.bias)


def build_mlp(n_layers: int) -> nn.Sequential:
    return nn.Sequential(*[Layer(WIDTH) for _ in range(n_layers)]).float()


def full_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs and targets, created once, outside the timed region."""
    generator = torch.Generator()
    generator.manual_seed(0)
    inputs = torch.randn(SAMPLES, WIDTH, generator=generator)
    generator.manual_seed(1)
    targets = torch.randn(SAMPLES, WIDTH, generator=generator)
    return inputs.to(device), targets.to(device)


def loss_fn() -> nn.MSELoss:
    return nn.MSELoss()


def optimizer(module: nn.Module) -> torch.optim.SGD:
    return torch.optim.SGD(module.parameters(), lr=LR)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def candidate_devices() -> list[tuple[str, torch.device]]:
    found: list[tuple[str, torch.device]] = []
    if torch.cuda.is_available():
        found.append(("cuda", torch.device("cuda")))
    if torch.backends.mps.is_available():
        found.append(("mps", torch.device("mps")))
    found.append(("cpu", torch.device("cpu")))
    return found


def probe_device(device: torch.device) -> None:
    value = torch.randn(64, 64, device=device)
    result = value @ value
    sync(device)
    if result.shape != (64, 64):
        raise RuntimeError(f"probe matmul on {device} returned {tuple(result.shape)}")


def describe_runtime(device: torch.device) -> dict[str, object]:
    versions: dict[str, object] = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "dtype": "float32",
        "amp": False,
        "lr": LR,
        "layers": LAYERS,
        "width": WIDTH,
        "microbatch_rows": MICROBATCH_ROWS,
        "microbatches": MICROBATCHES,
        "gelu": "erf",
        "loss": "torch.nn.MSELoss",
        "optimizer": "torch.optim.SGD",
    }
    if device.type == "cuda":
        versions["matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        versions["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        versions["cuda_device"] = torch.cuda.get_device_name(0)
        versions["cuda_capability"] = list(torch.cuda.get_device_capability(0))
        versions["cuda_arch_list"] = list(torch.cuda.get_arch_list())
    else:
        versions["matmul_allow_tf32"] = None
        versions["cudnn_allow_tf32"] = None
    if device.type == "mps":
        versions["mps"] = True
    return versions
