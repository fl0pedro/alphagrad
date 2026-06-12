"""``--anti-degeneracy`` desugaring shared by all three trainers.

Research context
----------------
The flag exists because every PPO / MuZero / GFN run on this codebase
prior to 2026-05-17 found ``best_per_channel/cosine_sim = 1.0`` — i.e.
the policy collapsed to pure vertex elimination, applying no
approximation at all. That's the degenerate solution: it satisfies a
naive "maximise quality" objective trivially while ignoring the cost
channels. The research goal is the opposite — find genuinely
approximate gradient sequences in a useful quality band — so we have
to actively prevent the collapse.

Mechanism
---------
``--anti-degeneracy`` is a high-level knob over the lower-level
``--lagrangian-constraint`` mechanism (``ppo.py:parse_lagrangian_constraints``).
The desugaring rewrites it into one or two
``--lagrangian-constraint`` entries and returns the corridor bounds the
worker uses for ``corridor/{in,below,above}_band_fraction``
instrumentation.

Three modes:

* ``none``           — pass user constraints through unchanged.
* ``delta_ceiling``  — single soft ceiling ``cosine_sim <= 1 - δ``.
  Default δ=0.01 → cap at 0.99. The most permissive anti-degenerate.
* ``corridor``       — both floor (``--cosine-lower-bound``, default
  0.8) and ceiling (``--cosine-upper-bound``, default 0.9). Forces the
  policy into a narrow band; useful when we want to study the cost-side
  Pareto front at a controlled quality level.
"""

from __future__ import annotations


def desugar_anti_degeneracy(
    user_constraints: list[str],
    anti_degeneracy: str,
    delta: float,
    cosine_lower_bound: float,
    cosine_upper_bound: float,
) -> tuple[list[str], float | None, float | None]:
    """Augment ``user_constraints`` based on ``anti_degeneracy``.

    Args:
        user_constraints: list of constraint strings the user passed via
            ``--lagrangian-constraint``.
        anti_degeneracy: one of ``{"none", "delta_ceiling", "corridor"}``.
        delta: δ used by ``delta_ceiling`` (ceiling = ``1 - δ``).
        cosine_lower_bound: floor used by ``corridor`` (≤ 0 disables).
        cosine_upper_bound: ceiling used by ``corridor`` (≥ 1 disables).

    Returns:
        Triple ``(augmented_constraints, corridor_low, corridor_high)``.
        ``corridor_low`` / ``corridor_high`` are floats when the
        instrumentation should be emitted, ``None`` when the
        corresponding bound is omitted (one-sided constraint). For
        ``anti_degeneracy="none"`` both are ``None``.
    """
    out = list(user_constraints)
    if anti_degeneracy == "delta_ceiling":
        ceiling = 1.0 - float(delta)
        out.append(f"cosine_sim<={_format_threshold(ceiling)}")
        return out, None, ceiling
    if anti_degeneracy == "corridor":
        low: float | None = None
        high: float | None = None
        if cosine_lower_bound > 0.0:
            out.append(f"cosine_sim>={_format_threshold(cosine_lower_bound)}")
            low = float(cosine_lower_bound)
        if cosine_upper_bound < 1.0:
            out.append(f"cosine_sim<={_format_threshold(cosine_upper_bound)}")
            high = float(cosine_upper_bound)
        return out, low, high
    return out, None, None


def _format_threshold(x: float) -> str:
    """Format a corridor threshold for the ``--lagrangian-constraint`` wire
    format. ``parse_lagrangian_constraints`` parses with ``float()`` so any
    representable form works, but we keep enough precision to round-trip
    very-small δ values (e.g. δ=1e-4 → ceiling=0.9999) without falling
    into scientific notation. Trailing zeros are trimmed so the common
    case (δ=0.01 → 0.99) renders cleanly.
    """
    s = f"{float(x):.6f}".rstrip("0").rstrip(".")
    return s or "0"
