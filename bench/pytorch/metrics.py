"""Workload W sizes and the samples/sec metric. No torch import."""

from __future__ import annotations

import json
import math
from typing import Any

LAYERS = 12
WIDTH = 4096
MICROBATCH_ROWS = 256
MICROBATCHES = 8
SAMPLES_PER_STEP = MICROBATCH_ROWS * MICROBATCHES  # 2048
WARMUP_STEPS = 5
MEASURED_STEPS = 50
LR = 1e-3

assert SAMPLES_PER_STEP == 2048


def median(values: list[float]) -> float:
    """Statistical median. Even counts average the two central values."""
    if not values:
        raise ValueError("median of an empty sample")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return 0.5 * (float(ordered[mid - 1]) + float(ordered[mid]))


def samples_per_sec(median_step_s: float) -> float:
    if not math.isfinite(median_step_s) or median_step_s <= 0:
        raise ValueError("median step time must be finite and positive")
    return SAMPLES_PER_STEP / median_step_s


def equal_layers(stages: int, layers: int = LAYERS) -> list[int]:
    """Contiguous equal split. Earlier ranks receive the remainder."""
    if stages <= 0 or layers <= 0 or stages > layers:
        raise ValueError(f"cannot split {layers} layers across {stages} stages")
    base, extra = divmod(layers, stages)
    return [base + (1 if index < extra else 0) for index in range(stages)]


def proportional_layers(speeds: list[float], layers: int = LAYERS) -> list[int]:
    """Hamilton largest-remainder split, then at least one layer per stage.

    A stage with zero layers is not a stage. When the largest-remainder
    assignment leaves a zero, one layer is moved from the stage that holds
    the most layers.
    """
    if not speeds or any(speed <= 0 or not math.isfinite(speed) for speed in speeds):
        raise ValueError("every speed must be finite and positive")
    if len(speeds) > layers:
        raise ValueError(f"cannot split {layers} layers across {len(speeds)} stages")
    total = math.fsum(speeds)
    raw = [layers * speed / total for speed in speeds]
    floors = [math.floor(value) for value in raw]
    out = [int(value) for value in floors]
    leftover = layers - sum(out)
    order = sorted(
        range(len(speeds)),
        key=lambda index: (raw[index] - floors[index], speeds[index], -index),
        reverse=True,
    )
    for step in range(leftover):
        out[order[step]] += 1
    guard = 0
    while 0 in out:
        guard += 1
        if guard > layers:
            raise ValueError("cannot give every stage a layer")
        empty = out.index(0)
        donor = max(range(len(out)), key=lambda index: (out[index], speeds[index], -index))
        if out[donor] <= 1:
            raise ValueError("cannot give every stage a layer")
        out[donor] -= 1
        out[empty] += 1
    if sum(out) != layers:
        raise ValueError("split does not cover every layer")
    return out


def dump(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def _self_test() -> None:
    assert median([1.0, 3.0, 2.0, 4.0]) == 2.5
    assert median([4.0, 1.0, 3.0]) == 3.0
    assert samples_per_sec(2.0) == 1024.0
    assert equal_layers(3) == [4, 4, 4]
    assert equal_layers(2) == [6, 6]
    assert equal_layers(5) == [3, 3, 2, 2, 2]
    assert proportional_layers([1.0, 1.0, 1.0]) == [4, 4, 4]
    assert proportional_layers([1.0, 1.0, 2.0]) == [3, 3, 6]
    assert proportional_layers([1.0, 2.0, 3.0]) == [2, 4, 6]
    assert proportional_layers([100.0, 1.0, 1.0]) == [10, 1, 1]
    try:
        proportional_layers([1.0, 0.0, 1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("zero speed must be rejected")


if __name__ == "__main__":
    _self_test()
    print("metrics ok")
