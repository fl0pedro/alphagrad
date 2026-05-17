"""Reward channel scaling, weighting, and per-channel logging.

Numpy-only — importable by the JAX-free drivers AND by workers that
need to construct weights on the host side before pushing them to JAX.

The 8-channel reward vector layout is the single source of truth for
the codebase. Previously the constants were re-imported from
`alphagrad.approx.env` (which transitively imports JAX) — defining
them here means JAX-free callers don't pay the JAX import cost and
the driver process keeps its no-JAX invariant.

This module also centralises:

* `build_reward_weights(args)` — was duplicated in `ppo_ray_worker.py`
  and `mu0.py` with subtly different fallback behaviour.
* `aggregate_per_channel_stats` — extracted from `mu0_ray_worker`'s
  hand-rolled best-per-channel reporting so PPO can produce the same
  shape of stats dict.
* `format_milestone_line` / `build_wandb_log_dict` — extracted from
  the `mu0_ray.py:280-386` driver loop so PPO can emit identical
  greppable logs.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Channel layout — must mirror alphagrad.approx.env (which imports JAX so we
# can't `from env import REWARD_NAMES` from this JAX-free module).
# ---------------------------------------------------------------------------

REWARD_NAMES: tuple[str, ...] = (
    "muls_adds_fmas",
    "flops",
    "latency_ns",
    "max_io_sum",
    "bytes_accessed",
    "peak_memory",
    "cosine_sim",
    "frob_residual",
)
NUM_REWARDS: int = len(REWARD_NAMES)
REWARD_INDEX: dict[str, int] = {n: i for i, n in enumerate(REWARD_NAMES)}
COSINE_SIM_IDX: int = REWARD_INDEX["cosine_sim"]
FROB_RESIDUAL_IDX: int = REWARD_INDEX["frob_residual"]

# Channels whose values are bounded / quality-signal, NOT raw cost — they
# bypass symlog wherever a symlog transform would otherwise apply (Lagrangian
# thresholds, calibration scaling). Mirrored from mu0.py:155 and
# ppo_ray_worker.py:80.
NO_SYMLOG_REWARD_INDICES: tuple[int, ...] = (COSINE_SIM_IDX,)
NO_SYMLOG_MASK_NP: np.ndarray = np.zeros((NUM_REWARDS,), dtype=bool)
NO_SYMLOG_MASK_NP[list(NO_SYMLOG_REWARD_INDICES)] = True

# Sparse-terminal channels: the env's ``_callback`` produces a
# meaningful value for these ONLY on the terminal elimination step
# (graphax's ``jacve`` returns a zero-norm Jacobian for any partial
# order, so the cosine_sim / frob_residual comparison is mathematically
# trivial mid-rollout). Downstream consumers must mask intermediate
# steps when these channels participate in violation / aggregation
# computations — see :func:`aggregate_per_channel_stats` and the
# PPO Lagrangian mask in ``ppo_ray_worker.py``.
SPARSE_TERMINAL_INDICES: tuple[int, ...] = (COSINE_SIM_IDX, FROB_RESIDUAL_IDX)
SPARSE_TERMINAL_MASK_NP: np.ndarray = np.zeros((NUM_REWARDS,), dtype=bool)
SPARSE_TERMINAL_MASK_NP[list(SPARSE_TERMINAL_INDICES)] = True

# Cost vs quality channels — the first six are negative costs (more negative =
# worse) and the last two are quality signals where higher is better.
COST_REWARD_INDICES: tuple[int, ...] = tuple(
    i for i in range(NUM_REWARDS) if i not in (COSINE_SIM_IDX, FROB_RESIDUAL_IDX)
)


# Numpy / JAX-free arg→channel mappings.
_CMP_TYPE_TO_REWARD: dict[str, str] = {
    "graphax": "muls_adds_fmas",
    "flops": "flops",
    "latency": "latency_ns",
}
_MEM_TYPE_TO_REWARD: dict[str, str] = {
    "graphax": "max_io_sum",
    "bytes_accessed": "bytes_accessed",
    "peak_memory": "peak_memory",
}


# ---------------------------------------------------------------------------
# Symlog
# ---------------------------------------------------------------------------

def symlog_np(x: np.ndarray) -> np.ndarray:
    """`sign(x) * log1p(|x|)` — numpy version of `alphagrad.utils.symlog`.

    Stable for the [-1e10, 1e10] range we care about; large sentinel
    values produce `±log1p(1e10) ≈ ±23`.
    """
    return np.sign(x) * np.log1p(np.abs(x))


# ---------------------------------------------------------------------------
# Reward weight construction
# ---------------------------------------------------------------------------

def build_reward_weights(args) -> np.ndarray:
    """Compose CLI lambda flags into a single (NUM_REWARDS,) scalarising
    weight vector.

    Always returns a non-zero vector — if `--rewards` ends up empty,
    falls back to weighting `muls_adds_fmas` (the original PPO-ray
    fallback at `ppo_ray_worker.py:159-160`).
    """
    w = np.zeros(NUM_REWARDS, dtype=np.float32)
    if "cmp" in args.rewards:
        cmp_name = _CMP_TYPE_TO_REWARD[args.cmp_type]
        w[REWARD_INDEX[cmp_name]] = float(getattr(args, "lambda_cmp", 1.0))
    if "mem" in args.rewards:
        mem_name = _MEM_TYPE_TO_REWARD[args.mem_type]
        w[REWARD_INDEX[mem_name]] = float(getattr(args, "lambda_mem", 1.0))
    if "acc" in args.rewards:
        w[COSINE_SIM_IDX] = 1.0
    lam_frob = float(getattr(args, "lambda_frob", 0.0))
    if lam_frob != 0.0:
        w[FROB_RESIDUAL_IDX] = lam_frob
    if not np.any(w):
        w[REWARD_INDEX["muls_adds_fmas"]] = 1.0
    return w


# ---------------------------------------------------------------------------
# Sentinel filtering
# ---------------------------------------------------------------------------

def parse_reward_conditions(
    specs: Sequence[str] | None,
) -> list[tuple[int, int, str, float]]:
    """Parse ``--reward-condition`` CLI specs into ``(easier_idx,
    harder_idx, op, threshold)`` tuples.

    Spec format: ``<easier>:<harder>>=<threshold>`` or
    ``<easier>:<harder><=<threshold>``. Names resolve through
    :data:`REWARD_INDEX`.

    Semantics: at every transition where ``op(harder_reward, thresh)``
    is False, the easier channel's reward is zeroed for that step.
    Acts as a hard alternative to the Lagrangian penalty for cases
    where the user wants to gate one reward on another instead of
    paying a soft cost. The check happens in the worker, BEFORE the
    per-channel GAE / advantage stack — so the easier channel's
    advantage is recomputed cleanly from the gated reward, not gated
    after the fact.

    Mirrors the validation style of
    :func:`alphagrad.approx.ppo_ray_worker._parse_lagrangian_constraints`
    so the two flags accept structurally similar syntax.
    """
    out: list[tuple[int, int, str, float]] = []
    if not specs:
        return out
    for raw in specs:
        spec = str(raw).strip()
        if not spec:
            continue
        if ":" not in spec:
            raise ValueError(
                f"--reward-condition spec missing ':' between easier and "
                f"harder channels: {spec!r}"
            )
        easier_name, rest = spec.split(":", 1)
        easier_name = easier_name.strip()
        rest = rest.strip()
        if ">=" in rest:
            op = ">="
            harder_name, thresh_str = rest.split(">=", 1)
        elif "<=" in rest:
            op = "<="
            harder_name, thresh_str = rest.split("<=", 1)
        else:
            raise ValueError(
                f"--reward-condition spec missing '>=' or '<=' operator: "
                f"{spec!r}"
            )
        harder_name = harder_name.strip()
        thresh_str = thresh_str.strip()
        if easier_name not in REWARD_INDEX:
            raise ValueError(
                f"unknown easier channel {easier_name!r} in "
                f"--reward-condition {spec!r}; valid names: "
                f"{sorted(REWARD_INDEX)}"
            )
        if harder_name not in REWARD_INDEX:
            raise ValueError(
                f"unknown harder channel {harder_name!r} in "
                f"--reward-condition {spec!r}; valid names: "
                f"{sorted(REWARD_INDEX)}"
            )
        try:
            thresh = float(thresh_str)
        except ValueError as exc:
            raise ValueError(
                f"could not parse threshold {thresh_str!r} in "
                f"--reward-condition {spec!r} as float"
            ) from exc
        out.append((
            REWARD_INDEX[easier_name],
            REWARD_INDEX[harder_name],
            op,
            thresh,
        ))
    return out


def filter_sentinel_mask(reward_vec_np: np.ndarray, sentinel: float) -> np.ndarray:
    """Boolean mask `(*reward_vec_np.shape[:-1],)` — True where the
    transition is NOT a sentinel.

    A transition is sentinel iff ANY cost channel exactly equals
    `sentinel`. We test only cost channels because the env's normal
    path can legitimately produce reward ≈ 0 on quality channels.
    """
    cost_slice = reward_vec_np[..., list(COST_REWARD_INDICES)]
    is_sentinel_any = np.any(cost_slice == sentinel, axis=-1)
    return ~is_sentinel_any


# ---------------------------------------------------------------------------
# Per-channel aggregation (replicates mu0_ray_worker.py:783-816)
# ---------------------------------------------------------------------------

def aggregate_per_channel_stats(
    buf_reward_vec_np: np.ndarray,
    reward_weights_np: np.ndarray,
    *,
    sentinel: float,
    action_seq: Sequence[Any] | None = None,
) -> dict:
    """Compute the per-channel stats dict the driver expects.

    Args:
        buf_reward_vec_np: shape (T, N, NUM_REWARDS) raw rewards per
            timestep per env per channel.
        reward_weights_np: shape (NUM_REWARDS,) the scalarising weights.
        sentinel: value to filter from per-channel means (uses
            `filter_sentinel_mask`).
        action_seq: optional per-env per-step action descriptor; if
            given, `best_per_reward[name]["seq"]` and
            ``best_overall_seq`` carry the action sequence of the env
            that argmaxes that channel / the overall scalar. When None
            the seq entries are empty lists.

    Returns:
        dict with keys:

        * ``per_reward_means``: `{name: float}` — mean over (T, N), with
          sentinel rows masked out.
        * ``best_per_reward``: `{name: {"raw_value", "weighted_value",
          "weighted_total", "env_idx", "seq", "all_raw",
          "all_weighted"}}` — argmax over envs of the per-env
          per-channel sum (only for non-zero weight channels). The
          ``all_raw``/``all_weighted`` dicts give the FULL 8-channel
          tuple for the env that won this channel, so the JSON dump can
          reconstruct ``(a_i, b_i, c_i, r_i)`` per the user request.
        * ``best_overall_rewards``: per-channel raw rewards for the env
          whose scalar (weight-weighted) sum is highest.
        * ``best_overall_weighted``: per-channel weighted contributions
          for that same winning env.
        * ``best_overall_seq``: action sequence of the overall-best env
          (empty list if ``action_seq`` was not provided).
    """
    T, N, R = buf_reward_vec_np.shape
    assert R == NUM_REWARDS, f"expected NUM_REWARDS={NUM_REWARDS}, got {R}"

    valid_mask = filter_sentinel_mask(buf_reward_vec_np, sentinel)  # (T, N)

    # per_reward_means: average each channel over the non-sentinel transitions.
    per_reward_means: dict[str, float] = {}
    flat_mask = valid_mask.reshape(-1)
    flat_rewards = buf_reward_vec_np.reshape(-1, R)
    if flat_mask.any():
        masked_mean = flat_rewards[flat_mask].mean(axis=0)
    else:
        masked_mean = np.zeros((R,), dtype=np.float32)
    for j, name in enumerate(REWARD_NAMES):
        per_reward_means[name] = float(masked_mean[j])

    # Per-env per-channel sum (zero-out sentinel rows so a single timeout
    # doesn't poison the env's running per-channel total).
    masked_rv = np.where(valid_mask[:, :, None], buf_reward_vec_np, 0.0)
    r_per_env = masked_rv.sum(axis=0)  # (N, NUM_REWARDS)
    weighted_per_env = r_per_env * reward_weights_np  # (N, NUM_REWARDS)
    per_env_tot = weighted_per_env.sum(axis=-1)  # (N,)

    def _seq_for(env_idx: int) -> Any:
        if action_seq is None:
            return []
        try:
            return action_seq[env_idx]
        except (IndexError, TypeError):
            return []

    best_overall_env = int(per_env_tot.argmax())
    best_overall_rewards: dict[str, float] = {}
    best_overall_weighted: dict[str, float] = {}
    for j, name in enumerate(REWARD_NAMES):
        best_overall_rewards[name] = float(r_per_env[best_overall_env, j])
        best_overall_weighted[name] = float(weighted_per_env[best_overall_env, j])

    best_per_reward: dict[str, dict] = {}
    for j, name in enumerate(REWARD_NAMES):
        if float(reward_weights_np[j]) == 0.0:
            continue
        bidx = int(r_per_env[:, j].argmax())
        # Full 8-channel snapshot of the env that won this channel —
        # this is what lets the JSON dump record `(a_i, b_i, c_i, r_i)`
        # for every per-channel best, not just the winning channel.
        all_raw = {
            REWARD_NAMES[k]: float(r_per_env[bidx, k]) for k in range(NUM_REWARDS)
        }
        all_weighted = {
            REWARD_NAMES[k]: float(weighted_per_env[bidx, k])
            for k in range(NUM_REWARDS)
        }
        best_per_reward[name] = {
            "raw_value": float(r_per_env[bidx, j]),
            "weighted_value": float(weighted_per_env[bidx, j]),
            "weighted_total": float(per_env_tot[bidx]),
            "env_idx": bidx,
            "seq": _seq_for(bidx),
            "all_raw": all_raw,
            "all_weighted": all_weighted,
        }

    return {
        "per_reward_means": per_reward_means,
        "best_per_reward": best_per_reward,
        "best_overall_rewards": best_overall_rewards,
        "best_overall_weighted": best_overall_weighted,
        "best_overall_env": best_overall_env,
        "best_overall_weighted_total": float(per_env_tot[best_overall_env]),
        "best_overall_seq": _seq_for(best_overall_env),
    }


# ---------------------------------------------------------------------------
# Driver-side state and logging
# ---------------------------------------------------------------------------

def init_running_bests() -> dict:
    """Fresh state for tracking running per-channel bests across episodes.

    Layout (consumed by :func:`update_running_bests`,
    :func:`build_wandb_log_dict`, :func:`dump_best_sequences_json`):

    * ``best_global_*`` — the env that won the SCALAR (weight-weighted)
      sum, across all episodes seen so far. ``best_global_rewards`` /
      ``best_global_weighted_split`` are the full per-channel tuple
      ``(a, b, c, ..., r)`` for that env; ``best_global_seq`` is its
      action sequence.
    * ``best_per_reward[name]`` — the env that won each *individual*
      channel ``name``, across all episodes seen so far. Each entry has
      the same shape as ``aggregate_per_channel_stats`` returns plus an
      ``ep`` field, so the JSON dump can record ``(a_i, b_i, c_i, r_i)``
      and the full sequence for every per-channel best.
    """
    return {
        "best_global_return": -float("inf"),
        "best_global_ep": -1,
        "best_global_seq": None,
        "best_global_rewards": {},
        "best_global_weighted_split": {},
        "best_per_reward": {},
    }


def update_running_bests(state: dict, stats: dict, ep: int) -> dict:
    """Update the running-bests state in-place from a fresh stats dict.

    Returns the same dict for convenience.
    """
    ep_best = stats.get("best_return", -float("inf"))
    if ep_best > state["best_global_return"]:
        state["best_global_return"] = ep_best
        # ``best_seq`` is the worker-side action seq of the overall-best
        # env this episode (MuZero already passes it; PPO does too post
        # the action_seq wiring). Fall back to the seq embedded in
        # ``aggregate_per_channel_stats`` if the worker didn't surface
        # ``best_seq`` directly.
        state["best_global_seq"] = stats.get(
            "best_seq", stats.get("best_overall_seq")
        )
        state["best_global_rewards"] = dict(stats.get("best_overall_rewards", {}))
        state["best_global_weighted_split"] = dict(
            stats.get("best_overall_weighted", {})
        )
        state["best_global_ep"] = ep
    for name, info in stats.get("best_per_reward", {}).items():
        prev = state["best_per_reward"].get(name)
        if prev is None or info["raw_value"] > prev["raw_value"]:
            # Preserve the FULL info dict (incl. ``all_raw`` /
            # ``all_weighted`` / ``seq``) so the JSON dump has the
            # cross-channel tuple for every per-channel best.
            state["best_per_reward"][name] = {**info, "ep": ep}
    return state


# ---------------------------------------------------------------------------
# JSON persistence + wandb sync of best sequences
# ---------------------------------------------------------------------------

def best_sequences_snapshot(state: dict) -> dict:
    """Pure-python snapshot of ``state`` suitable for json.dump / wandb.

    Each per-channel best entry holds ``(a_i, b_i, c_i, r_i)`` style
    cross-channel rewards in ``all_raw`` / ``all_weighted`` plus the
    full action sequence in ``seq``. The overall best (scalar) entry is
    the cross-section of the env that won the scalar weighted sum.
    """
    def _coerce(v: Any) -> Any:
        # Already-python types pass through; numpy scalars are unwrapped
        # so json.dump doesn't choke on `np.float32`/`np.int64`.
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    def _coerce_dict(d: dict) -> dict:
        return {k: _coerce(v) for k, v in d.items()}

    overall = {
        "return": _coerce(state.get("best_global_return", float("nan"))),
        "ep": _coerce(state.get("best_global_ep", -1)),
        "rewards_raw": _coerce_dict(state.get("best_global_rewards", {})),
        "rewards_weighted": _coerce_dict(state.get("best_global_weighted_split", {})),
        "seq": state.get("best_global_seq"),
    }
    per_channel: dict[str, dict] = {}
    for name, info in state.get("best_per_reward", {}).items():
        per_channel[name] = {
            "ep": _coerce(info.get("ep", -1)),
            "env_idx": _coerce(info.get("env_idx", -1)),
            "raw_value": _coerce(info.get("raw_value", float("nan"))),
            "weighted_value": _coerce(info.get("weighted_value", float("nan"))),
            "weighted_total": _coerce(info.get("weighted_total", float("nan"))),
            "all_raw": _coerce_dict(info.get("all_raw", {})),
            "all_weighted": _coerce_dict(info.get("all_weighted", {})),
            "seq": info.get("seq", []),
        }
    return {
        "best_overall": overall,
        "best_per_channel": per_channel,
    }


def dump_best_sequences_json(state: dict, path: str) -> str | None:
    """Write the running bests state to ``path`` as JSON. Atomic via tmp+rename.

    Returns the path on success, ``None`` if the dump failed (caller
    decides whether to log / swallow). Best-effort: we don't want a
    JSON-dump failure to crash a long-running trainer.
    """
    import json
    import os
    import tempfile

    if not path:
        return None
    try:
        snap = best_sequences_snapshot(state)
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=directory, delete=False, suffix=".tmp"
        ) as tmp:
            json.dump(snap, tmp, indent=2, default=str)
            tmp_path = tmp.name
        os.replace(tmp_path, path)
        return path
    except Exception:
        return None


def build_best_sequences_wandb_payload(state: dict, *, ep: int) -> dict:
    """Flat wandb-loggable payload of the running bests state.

    Per-channel:
      * ``best_seq_table/{name}_raw_value`` — channel-best raw value
      * ``best_seq_table/{name}_weighted_total`` — its scalar reward
      * ``best_seq_table/{name}/rewards_raw/{ch}`` — full reward tuple
      * ``best_seq_table/{name}/rewards_weighted/{ch}`` — weighted slice
      * ``best_seq_table/{name}_ep`` — episode the best was set
      * ``best_seq_table/{name}_seq_len`` — length of the action seq

    The full action sequences are NOT logged per-step (would balloon
    history); they're written to JSON instead. The end-of-run summary
    logs them as a wandb Table separately (see
    :func:`build_best_sequences_wandb_table`).
    """
    log: dict[str, Any] = {
        "best_sequences/snapshot_ep": ep,
        "best_sequences/overall_return": state.get("best_global_return"),
        "best_sequences/overall_ep": state.get("best_global_ep"),
    }
    for ch_name, raw in state.get("best_global_rewards", {}).items():
        log[f"best_sequences/overall/rewards_raw/{ch_name}"] = raw
    for ch_name, w in state.get("best_global_weighted_split", {}).items():
        log[f"best_sequences/overall/rewards_weighted/{ch_name}"] = w
    seq = state.get("best_global_seq") or []
    try:
        log["best_sequences/overall/seq_len"] = len(seq)
    except TypeError:
        log["best_sequences/overall/seq_len"] = 0
    for name, info in state.get("best_per_reward", {}).items():
        log[f"best_sequences/per_channel/{name}/raw_value"] = info.get("raw_value")
        log[f"best_sequences/per_channel/{name}/weighted_total"] = info.get(
            "weighted_total"
        )
        log[f"best_sequences/per_channel/{name}/ep"] = info.get("ep", -1)
        seq = info.get("seq") or []
        try:
            log[f"best_sequences/per_channel/{name}/seq_len"] = len(seq)
        except TypeError:
            log[f"best_sequences/per_channel/{name}/seq_len"] = 0
        for ch_name, raw in info.get("all_raw", {}).items():
            log[f"best_sequences/per_channel/{name}/rewards_raw/{ch_name}"] = raw
        for ch_name, w in info.get("all_weighted", {}).items():
            log[f"best_sequences/per_channel/{name}/rewards_weighted/{ch_name}"] = w
    return log


def build_best_sequences_wandb_table(state: dict):
    """Return a `wandb.Table` of one row per best category.

    Each row records: category (per-channel name or "overall"), the
    raw-value for that channel (or scalar return for overall), the
    weighted-total, the episode it was set on, the full reward tuple
    serialised as JSON, and the action sequence serialised as JSON.

    Importing wandb is local so this module stays JAX/wandb-free at
    import time.
    """
    import json
    try:
        import wandb  # type: ignore
    except Exception:
        return None
    cols = [
        "category",
        "raw_value",
        "weighted_total",
        "ep",
        "rewards_raw_json",
        "rewards_weighted_json",
        "seq_json",
        "seq_len",
    ]
    table = wandb.Table(columns=cols)
    overall = {
        "category": "overall",
        "raw_value": state.get("best_global_return"),
        "weighted_total": state.get("best_global_return"),
        "ep": state.get("best_global_ep", -1),
        "rewards_raw_json": json.dumps(
            state.get("best_global_rewards", {}), default=str
        ),
        "rewards_weighted_json": json.dumps(
            state.get("best_global_weighted_split", {}), default=str
        ),
        "seq_json": json.dumps(state.get("best_global_seq"), default=str),
        "seq_len": (
            len(state.get("best_global_seq") or [])
            if state.get("best_global_seq") is not None
            else 0
        ),
    }
    table.add_data(*[overall[c] for c in cols])
    for name, info in state.get("best_per_reward", {}).items():
        seq = info.get("seq") or []
        row = {
            "category": name,
            "raw_value": info.get("raw_value"),
            "weighted_total": info.get("weighted_total"),
            "ep": info.get("ep", -1),
            "rewards_raw_json": json.dumps(info.get("all_raw", {}), default=str),
            "rewards_weighted_json": json.dumps(
                info.get("all_weighted", {}), default=str
            ),
            "seq_json": json.dumps(seq, default=str),
            "seq_len": len(seq) if hasattr(seq, "__len__") else 0,
        }
        table.add_data(*[row[c] for c in cols])
    return table


def format_milestone_line(
    variant: str,
    ep: int,
    total_eps: int,
    stats: dict,
    state: dict,
) -> str:
    """Multi-line greppable milestone log entry — same shape as
    `mu0_ray.py:343-350`."""
    per_ch_str = " ".join(
        f"{n[:4]}={info['raw_value']:+.2g}"
        for n, info in state["best_per_reward"].items()
    )
    mean_str = " ".join(
        f"{n[:4]}={v:+.2g}"
        for n, v in stats.get("per_reward_means", {}).items()
        if v != 0.0
    )
    best_overall_str = " ".join(
        f"{n[:4]}={v:+.2g}"
        for n, v in state["best_global_rewards"].items()
        if v != 0.0
    )
    ep_mean = stats.get("mean_return", float("nan"))
    ent = stats.get("entropy_mean", stats.get("entropy", float("nan")))
    bsize = stats.get("buffer_size", 0)
    tstep = stats.get("train_step", 0)
    return (
        f"  [{variant}] ep={ep + 1:>4}/{total_eps} "
        f"best={state['best_global_return']:+.4g}(ep{state['best_global_ep']}) "
        f"mean={ep_mean:+.4g} "
        f"ent={ent:.3f} buf={bsize} step={tstep}\n"
        f"            best-overall-traj-rewards: {best_overall_str}\n"
        f"            per-channel-best:          {per_ch_str}\n"
        f"            mean-per-channel:          {mean_str}"
    )


def build_wandb_log_dict(stats: dict, state: dict, ep: int) -> dict:
    """Flat key/value dict for `wandb.log` — same shape as
    `mu0_ray.py:353-386`. Per-channel bests get a per-name suffix."""
    ep_best = stats.get("best_return", -float("inf"))
    log_dict: dict[str, Any] = {
        "episode": ep,
        "best_return": state["best_global_return"],
        "best_return_this_ep": ep_best,
        "mean_return": stats.get("mean_return", float("nan")),
        "entropy_mean": stats.get("entropy_mean", stats.get("entropy", float("nan"))),
        "entropy_root": stats.get("entropy_root", float("nan")),
        "policy_loss": stats.get("policy_loss", stats.get("ppo_loss", float("nan"))),
        "value_loss": stats.get("value_loss", float("nan")),
        "reward_loss": stats.get("reward_loss", float("nan")),
        "total_loss": stats.get("total_loss", float("nan")),
        "buffer_size": stats.get("buffer_size", 0),
        "train_step": stats.get("train_step", 0),
        "nan_skip_count": stats.get("nan_skip_count", 0),
    }
    for name, val in stats.get("per_reward_means", {}).items():
        log_dict[f"reward_mean/{name}"] = val
    for name, info in state["best_per_reward"].items():
        log_dict[f"best_per_channel/{name}"] = info["raw_value"]
        log_dict[f"best_per_channel_weighted_total/{name}"] = info["weighted_total"]
    for name, v in state["best_global_rewards"].items():
        log_dict[f"best_overall_reward/{name}"] = v
    for name, v in state["best_global_weighted_split"].items():
        log_dict[f"best_overall_weighted/{name}"] = v
    # Pass through any extra namespaced keys the worker chose to emit
    # (e.g. `lagrangian/multipliers/cosine_sim`).
    for k, v in stats.items():
        if k.startswith("lagrangian/") or k.startswith("reward_mean/"):
            log_dict[k] = v
    return log_dict
