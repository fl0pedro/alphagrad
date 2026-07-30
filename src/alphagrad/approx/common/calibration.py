"""Pre-training reward-channel calibration (Stage G).

Both the PPO-ray and MuZero-ray drivers run a short un-trained
zero-preference rollout, measure mean `|symlog(reward)|` per channel,
and rescale `reward_weights` by `1/mean_abs` so wide-magnitude
channels (flops ~ 1e10) and narrow-magnitude channels (peak_memory ~
1e6) contribute on a comparable scale.

This module is the single source of truth — previously the logic was
inline in `mu0_ray.py:128-160` and missing entirely from `ppo_ray.py`.

**Relevance after the GDPO refactor.** Under
``--advantage-norm gdpo`` (the new PPO default) per-minibatch
per-channel z-scoring inside ``gdpo_normalise_advantages`` makes
this magnitude rescale redundant: each channel's advantage gets
``(adv - μ_k_batch) / σ_k_batch`` and is then priority-weighted into
the summed advantage. The drivers skip the call in that mode. The
calibration path remains the right thing for ``--advantage-norm
scalar`` (legacy) and for MuZero, both of which still rely on
cross-channel magnitude balance to feed a single scalarised reward
stream into GAE / the per-channel discounted-return helper.

The actor argument is duck-typed: anything that exposes
`reward_vec_means.remote(rng_seed, num_rollouts) -> np.ndarray` and
`set_reward_weights.remote(weights_np) -> None` will work. Both
`PPORayWorker` and `mu0_ray_actors.SPMDActor` satisfy this.

The function MUST be called AFTER the CPU-approx pool's
`compile_approximations` warm-up has resolved. The original mu0_ray
ordering was the inverse (calibration before warm-up), which let the
first ~half of the zero-pref rollouts hit cold-cache compile
timeouts. The sentinel rewards (-1e10) those timeouts produced
poisoned the per-channel `mean_abs(symlog)`, microscopic weights
followed, and the value head exploded on the first real training
batch. The `after_warmup=True` parameter is a guard — pass it
explicitly so the call site documents the ordering.
"""

from __future__ import annotations

import numpy as np

from alphagrad.approx.common.cache import SENTINEL_REWARD_VALUE
from alphagrad.approx.common.reward_scaling import (
    COSINE_SIM_IDX,
    NUM_REWARDS,
    build_reward_weights,
    filter_sentinel_mask,
    symlog_np,
)


def compute_calibration_scaling(
    stats: dict,
    *,
    statistic: str = "iqr",
    floor: float = 1e-3,
) -> np.ndarray:
    """Derive the per-channel weight rescale from the actor's stats dict.

    Args:
        stats: dict returned by ``actor.reward_vec_means`` with the
            extended schema (mean / median / quartiles in raw + symlog
            space).
        statistic: which dispersion statistic to use for the rescale:
            * ``iqr`` (recommended): symlog-space IQR / 1.349 — gives
              the σ-equivalent of a Gaussian. Robust to heavy-tail
              outliers (cold-cache spikes, sentinel-adjacent values).
            * ``mean_abs``: legacy ``|symlog(mean)|`` — normalises by the
              channel's BIAS, not its spread. Kept for A/B regression.
            * ``std``: numpy std of symlog'd samples. Less robust than
              IQR but cheap.
        floor: minimum dispersion below which the scaling caps to
            ``1/floor`` — prevents a degenerate near-zero channel
            (e.g. a sparse-terminal channel with mostly-zero
            intermediates) from acquiring a giant weight.

    Returns the ``(NUM_REWARDS,)`` scaling vector to multiply
    ``build_reward_weights(args)`` by.

    Cosine_sim is hard-pinned to scaling=1.0 — it's already in [0,1] so
    rescaling would distort the quality-vs-cost trade-off. (See the
    ``NO_SYMLOG_REWARD_INDICES`` constant for the canonical "don't
    rescale these" set.)
    """
    if statistic == "mean_abs":
        # Legacy path: |symlog(mean)| as the per-channel scale. Documented
        # to be a BIAS proxy rather than a spread proxy — switch via
        # ``--calibration-statistic`` if you need to reproduce a pre-IQR
        # baseline.
        mean_vec = np.asarray(stats["mean"], dtype=np.float32)
        if (mean_vec == SENTINEL_REWARD_VALUE).any():
            mean_vec = np.where(
                mean_vec == SENTINEL_REWARD_VALUE,
                np.float32(0.0),
                mean_vec,
            )
        dispersion = np.abs(symlog_np(mean_vec))
    elif statistic == "iqr":
        # IQR-on-symlog. Symlog squashes wide-magnitude channels (flops
        # ~ 1e10) into a single-decade range; IQR over the squashed
        # samples is a robust σ-proxy that ignores outliers by
        # construction. Convert IQR → σ via the Gaussian ratio
        # ``σ ≈ IQR / 1.349``.
        q25 = np.asarray(stats["q25_symlog"], dtype=np.float32)
        q75 = np.asarray(stats["q75_symlog"], dtype=np.float32)
        iqr = q75 - q25
        dispersion = iqr / 1.349
    elif statistic == "std":
        # Std-on-symlog. Less robust than IQR (squares pull outliers
        # in) but cheap. Provided for completeness.
        q25 = np.asarray(stats["q25_symlog"], dtype=np.float32)
        q75 = np.asarray(stats["q75_symlog"], dtype=np.float32)
        # Use the IQR → σ conversion as a stable proxy of std when only
        # quartiles are available. (We could ship full samples through
        # Ray for true std but it'd 100×-bloat the payload.)
        dispersion = (q75 - q25) / 1.349
    else:
        raise ValueError(
            f"unknown --calibration-statistic {statistic!r}; "
            f"expected one of: iqr, mean_abs, std"
        )

    # Cosine_sim is bounded → no rescale.
    dispersion = np.array(dispersion, dtype=np.float32, copy=True)
    dispersion[COSINE_SIM_IDX] = 1.0
    return 1.0 / np.maximum(dispersion, floor)


def run_calibration(
    actor,
    args,
    num_rollouts: int,
    *,
    after_warmup: bool,
    statistic: str | None = None,
) -> np.ndarray:
    """Run `num_rollouts` zero-pref rollouts on `actor`, compute a
    per-channel scaling (default: symlog-space IQR), and push the
    rescaled weights back via ``actor.set_reward_weights``.

    Returns the absolute (post-scaling) weight vector for inspection.

    Args:
        actor: anything exposing ``reward_vec_means.remote(rng_seed,
            num_rollouts)`` returning the extended stats dict and
            ``set_reward_weights.remote(weights_np)``.
        args: argparse namespace — uses ``args.seed`` and the user's
            ``--lambda-*`` weights via :func:`build_reward_weights`.
            Optionally reads ``args.calibration_statistic`` when
            ``statistic`` is None.
        num_rollouts: how many zero-pref rollouts to collect. More =
            tighter quartile estimates at linear time cost.
        after_warmup: MUST be True. Pre-warmup calibration is
            sentinel-contaminated (see module docstring).
        statistic: ``iqr`` / ``mean_abs`` / ``std`` — falls back to
            ``args.calibration_statistic`` then to ``iqr``.
    """
    if not after_warmup:
        raise AssertionError(
            "run_calibration must be called AFTER the CPU-approx pool "
            "warm-up has resolved. See module docstring for why."
        )
    if num_rollouts <= 0:
        return np.zeros((NUM_REWARDS,), dtype=np.float32)

    import ray

    chosen_stat = (
        statistic
        if statistic is not None
        else getattr(args, "calibration_statistic", "iqr")
    )

    print(
        f"  [calibration] {num_rollouts} zero-pref rollouts on actor "
        f"(statistic={chosen_stat})..."
    )
    stats = ray.get(
        actor.reward_vec_means.remote(
            rng_seed=int(args.seed) + 7,
            num_rollouts=num_rollouts,
        )
    )

    # Defensive: the actor SHOULD filter sentinels but if a NaN /
    # sentinel value slips through, log a warning. The dispersion
    # computation below handles the zero-fallback via the ``floor``.
    mean_vec = np.asarray(stats.get("mean", np.zeros((NUM_REWARDS,))), dtype=np.float32)
    if (mean_vec == SENTINEL_REWARD_VALUE).any():
        print(
            "  [calibration] WARNING: actor returned a sentinel value in "
            "mean_vec; the dispersion floor will keep weights bounded but "
            "consider raising the warmup budget."
        )

    scaling = compute_calibration_scaling(stats, statistic=chosen_stat)
    base_weights = build_reward_weights(args)
    abs_weights = (base_weights * scaling).astype(np.float32)

    ray.get(actor.set_reward_weights.remote(abs_weights))
    print(
        "  [calibration] applied scaling. "
        f"weights_nonzero={ {n: float(abs_weights[i]) for i, n in enumerate(_NAMES_SHORT) if abs_weights[i] != 0.0} }"
    )
    return abs_weights


_NAMES_SHORT = (
    "mul", "flop", "lat", "mio", "byt", "peak", "cos", "frob",
)
