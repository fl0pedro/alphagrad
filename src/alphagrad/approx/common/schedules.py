"""Episode-progress → scalar annealing schedules.

Used by every trainer to compute a per-episode temperature (or β, or any
other scalar knob) from a `(progress ∈ [0, 1], init, final, schedule)` tuple.
The progress argument is normally `episode / total_episodes`; passing
exactly 1.0 returns ``final``.

Three shapes are exposed:
  * ``constant``    — returns ``init``.
  * ``linear``      — interpolates ``init → final``.
  * ``cosine``      — half-period cosine, slow at the ends, fast in the middle.

Pure Python (no JAX) — the resolved scalar is fed to the rollout function as
a leaf jnp.array, so a host-side compute is fine.
"""

from __future__ import annotations

import math


SCHEDULES = ("constant", "linear", "cosine")


def schedule_at(progress: float, init: float, final: float, kind: str) -> float:
    """Return the scheduled scalar at ``progress ∈ [0, 1]``.

    Clamping: progress < 0 returns ``init``, progress > 1 returns ``final``.
    """
    if kind == "constant":
        return float(init)
    p = max(0.0, min(1.0, float(progress)))
    if kind == "linear":
        return float(init) + (float(final) - float(init)) * p
    if kind == "cosine":
        return float(final) + 0.5 * (float(init) - float(final)) * (
            1.0 + math.cos(math.pi * p)
        )
    raise ValueError(f"Unknown schedule '{kind}'; expected one of {SCHEDULES}.")
