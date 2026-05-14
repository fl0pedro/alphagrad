"""JAX-free variant presets and curriculum helpers.

Lives outside ``alphagrad.approx.common`` and ``alphagrad.approx.mu0`` so the
Ray-driver entry point in ``mu0_ray.py`` can import variant metadata without
pulling JAX into the driver process. mu0.py re-imports these symbols so
behaviour there is unchanged.

``MAX_RULES_PER_VERTEX`` is duplicated from ``alphagrad.approx.env`` (which
imports JAX). The constant is part of the action-space contract — bumping it
requires changing it in both places.
"""

from __future__ import annotations


MAX_RULES_PER_VERTEX = 16


VARIANT_PRESETS: dict[str, dict] = {
    "custom": {},
    "ve_only": {"pin_rules_to_exact": True},
    "diag_gcd": {"factors": "-1", "max_rules": 1, "pin_rules_to_exact": False},
    "diag_factor": {
        "factors": "2,3,4,8,16",
        "max_rules": 1,
        "pin_rules_to_exact": False,
    },
    "compress": {
        "factors": "-1",
        "max_rules": 1,
        "pin_rules_to_exact": False,
    },
    "full": {
        "factors": "-1,2,3,4,8,16",
        "max_rules": MAX_RULES_PER_VERTEX,
        "pin_rules_to_exact": False,
    },
    # full_curriculum acts like `full` for one-shot training but flips the
    # auto-curriculum bit so main() expands an empty --curriculum into
    # ``diag_gcd:N/3, diag_factor:N/3, full:N/3``. The preset still
    # overwrites factors / max_rules with the final-stage values so the
    # agent is built once with the largest action footprint (the
    # curriculum runner gates op_legality per stage rather than rebuilding
    # the agent — mirrors ppo.py's behaviour).
    "full_curriculum": {
        "factors": "-1,2,3,4,8,16",
        "max_rules": MAX_RULES_PER_VERTEX,
        "pin_rules_to_exact": False,
    },
}


def _apply_variant_preset(args, variant: str | None = None) -> None:
    """In-place apply a ``--variant`` preset to ``args``.

    ``variant`` overrides ``args.variant`` if given (used by the curriculum
    scheduler when stepping through stages).
    """
    name = variant if variant is not None else getattr(args, "variant", "custom")
    if name not in VARIANT_PRESETS:
        raise ValueError(
            f"Unknown --variant '{name}'. Valid: {list(VARIANT_PRESETS)}."
        )
    for k, v in VARIANT_PRESETS[name].items():
        setattr(args, k, v)


def _parse_curriculum(spec: str) -> list[tuple[str, int]]:
    """Parse ``stage1:N1,stage2:N2,...`` into ``[(variant, episodes), ...]``."""
    if not spec.strip():
        return []
    stages: list[tuple[str, int]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Curriculum stage '{chunk}' missing ':'. Expected "
                "`variant:episode_count`."
            )
        name, n_str = chunk.split(":", 1)
        name = name.strip()
        if name not in VARIANT_PRESETS:
            raise ValueError(
                f"Curriculum stage variant '{name}' unknown. Valid: "
                f"{list(VARIANT_PRESETS)}."
            )
        try:
            n_episodes = int(n_str.strip())
        except ValueError as e:
            raise ValueError(
                f"Curriculum stage '{chunk}': episode count '{n_str}' is "
                "not an integer."
            ) from e
        if n_episodes <= 0:
            raise ValueError(
                f"Curriculum stage '{chunk}': episode count must be > 0."
            )
        stages.append((name, n_episodes))
    return stages


def _default_full_curriculum(total_episodes: int) -> list[tuple[str, int]]:
    """Split ``total_episodes`` across diag_gcd → diag_factor → full.

    Each stage gets ``floor(N/3)`` episodes; the final stage absorbs the
    remainder so the sum is exactly ``total_episodes``.
    """
    base = max(total_episodes // 3, 1)
    stages = [
        ("diag_gcd", base),
        ("diag_factor", base),
        ("full", max(total_episodes - 2 * base, 1)),
    ]
    return stages


def _current_stage_at(stages: list[tuple[str, int]], ep: int) -> str:
    cumulative = 0
    for name, n in stages:
        if ep < cumulative + n:
            return name
        cumulative += n
    return stages[-1][0] if stages else ""


def _current_stage_index(stages: list[tuple[str, int]], ep: int) -> int:
    cumulative = 0
    for idx, (_, n) in enumerate(stages):
        if ep < cumulative + n:
            return idx
        cumulative += n
    return len(stages) - 1


# Per-variant op-legality / pin-rules masks. The mu0 search tree doesn't
# emit a separate op-type token (every micro-action collapses to (pair,
# factor) inside a fixed depth budget), so these masks act on the prior
# logits during MCTS instead of a typed action head. ``ve_only`` forces
# every pair-slot to STOP, which is exactly the "no rule" output the
# pinned legacy path produces.
def _pin_rules_for_variant(variant: str) -> bool:
    return variant == "ve_only"
