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

Curriculum LR helper
--------------------
``cosine_warmup_exp_decay_lr`` is a JAX-compatible per-step LR schedule used
by the curriculum scheduler. Each stage gets a fresh cosine warm-up followed
by exponential decay over the stage's ``period`` — heads introduced in the
stage start at the bottom of the warm-up, climb to full LR by the warm-up
end, then anneal toward ``end_mult`` over the rest of the stage. Existing
heads from earlier stages run at a reduced flat multiplier (set by the
curriculum code, not this function).
"""

from __future__ import annotations

import math

import jax.numpy as jnp


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


def cosine_warmup_exp_decay_lr(
    step,
    init_lr: float,
    period: int,
    warmup_steps: int,
    end_mult: float = 0.1,
    decay_strength: float = 4.0,
):
    """Cosine-warmup + exponential-decay LR with the given ``period``.

    Within ``[0, warmup_steps)`` of each period, the multiplier rises from
    ``end_mult`` to ``1.0`` along a half-cosine. Within
    ``[warmup_steps, period)`` it decays exponentially back toward
    ``end_mult`` with rate ``decay_strength / (period - warmup_steps)`` —
    larger ``decay_strength`` makes the tail shorter. ``step`` is reduced
    modulo ``period`` so successive curriculum stages each get a fresh hill;
    the curriculum scheduler resets ``step`` (or simply increments without
    reset and lets the modulo wrap) for each stage transition.

    JAX-friendly: all ops are jnp, ``step`` may be a tracer.
    """
    p = max(int(period), 1)
    w = max(int(warmup_steps), 0)
    end = float(end_mult)
    decay_steps = max(p - w, 1)

    in_period = jnp.mod(jnp.asarray(step, dtype=jnp.float32), float(p))

    if w > 0:
        cos_arg = jnp.clip(in_period / float(w), 0.0, 1.0)
        warmup_mult = end + (1.0 - end) * 0.5 * (1.0 - jnp.cos(jnp.pi * cos_arg))
    else:
        warmup_mult = jnp.ones_like(in_period)

    decay_offset = jnp.maximum(in_period - float(w), 0.0)
    decay_mult = end + (1.0 - end) * jnp.exp(
        -float(decay_strength) * decay_offset / float(decay_steps)
    )

    in_warmup = in_period < float(w)
    mult = jnp.where(in_warmup, warmup_mult, decay_mult)
    return float(init_lr) * mult
