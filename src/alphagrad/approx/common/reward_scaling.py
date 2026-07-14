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
    # index 8: deterministic XLA-analysis peak (temp+output+args). Keep in EXACT
    # sync with env.REWARD_NAMES. peak_memory (5) = real measured peak (GPU);
    # xla_peak_memory (8) = compile-time estimate (reliable on CPU).
    "xla_peak_memory",
    # index 9: B_kstep closed-loop trainability accuracy in [0, 1]. Alternative
    # "acc" reward channel (ALPHAGRAD_ACC_PROXY=bkstep) — see env.REWARD_NAMES.
    "bkstep_acc",
)
NUM_REWARDS: int = len(REWARD_NAMES)
REWARD_INDEX: dict[str, int] = {n: i for i, n in enumerate(REWARD_NAMES)}
COSINE_SIM_IDX: int = REWARD_INDEX["cosine_sim"]
FROB_RESIDUAL_IDX: int = REWARD_INDEX["frob_residual"]
BKSTEP_ACC_IDX: int = REWARD_INDEX["bkstep_acc"]

# Channels whose values are bounded / quality-signal, NOT raw cost — they
# bypass symlog wherever a symlog transform would otherwise apply (gate
# thresholds, calibration scaling). Mirrored from mu0.py:155 and
# ppo_ray_worker.py:80.
NO_SYMLOG_REWARD_INDICES: tuple[int, ...] = (COSINE_SIM_IDX, BKSTEP_ACC_IDX)
NO_SYMLOG_MASK_NP: np.ndarray = np.zeros((NUM_REWARDS,), dtype=bool)
NO_SYMLOG_MASK_NP[list(NO_SYMLOG_REWARD_INDICES)] = True

# Sparse-terminal channels: the env's ``_callback`` produces a
# meaningful value for these ONLY on the terminal elimination step
# (graphax's ``jacve`` returns a zero-norm Jacobian for any partial
# order, so the cosine_sim / frob_residual comparison is mathematically
# trivial mid-rollout). Downstream consumers must mask intermediate
# steps when these channels participate in aggregation computations —
# see :func:`aggregate_per_channel_stats`.
SPARSE_TERMINAL_INDICES: tuple[int, ...] = (
    COSINE_SIM_IDX, FROB_RESIDUAL_IDX, BKSTEP_ACC_IDX,
)
SPARSE_TERMINAL_MASK_NP: np.ndarray = np.zeros((NUM_REWARDS,), dtype=bool)
SPARSE_TERMINAL_MASK_NP[list(SPARSE_TERMINAL_INDICES)] = True

# Cost vs quality channels — the cost channels are stored as negative costs
# (more negative = worse); the quality channels (cosine_sim, frob_residual,
# bkstep_acc) are the positive "higher is better" signals. bkstep_acc must be
# excluded here so it is NOT symlog'd/negated like a cost.
_QUALITY_REWARD_INDICES: tuple[int, ...] = (
    COSINE_SIM_IDX, FROB_RESIDUAL_IDX, BKSTEP_ACC_IDX,
)
COST_REWARD_INDICES: tuple[int, ...] = tuple(
    i for i in range(NUM_REWARDS) if i not in _QUALITY_REWARD_INDICES
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
    "peak_memory": "peak_memory",          # RM-sampled peak (idx 5)
    "xla_peak_memory": "xla_peak_memory",  # deterministic XLA peak (idx 8) —
                                           # preferred CPU memory reward; the RM
                                           # peak is still measured/logged at idx 5
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

    # ALL-CHANNEL PopArt reward (bridge-cse, USER-DIRECTED). When
    # ALPHAGRAD_REWARD_ALL_CHANNELS=1 (or --rewards contains ``all``), put a
    # NON-ZERO weight on EVERY applicable measured channel and let PopArt
    # normalise each ((G_k - mu_k)/sigma_k) before the weighted sum. The env's
    # reward vector already stores costs NEGATED (r = -cost) and the quality
    # channels (cosine_sim, bkstep_acc) POSITIVE, so a single POSITIVE weight
    # per channel gives the correct sign (reward low-cost + high-fidelity).
    # frob_residual is emitted as -residual (negated) so it too takes a
    # positive weight. Uniform weight (default 1.0, ALPHAGRAD_ALL_CHANNEL_W)
    # since PopArt handles the disparate raw scales. Channels that are not
    # actually populated are left at 0 so a dead channel (e.g. latency without
    # --measure-latency, bkstep without ALPHAGRAD_BKSTEP=1) never enters the
    # sum. flops (order-discriminating) + muls_adds are always in the set.
    import os as _osac

    # EXPLICIT CHANNEL-LIST selector (bridge-cse, USER-DIRECTED). When
    # ALPHAGRAD_REWARD_CHANNELS names a comma-separated set of channel keys, put
    # a non-zero weight on EXACTLY those channels and 0 on all others — a CURATED
    # subset (e.g. the 5-channel flops,xla_peak_memory,peak_memory,latency_ns,
    # bkstep_acc set) instead of the noisy all-10.
    #
    # PER-CHANNEL WEIGHTS: each entry may be either ``name`` (weight defaults to
    # ALPHAGRAD_ALL_CHANNEL_W, default 1.0) OR ``name:weight`` (explicit float),
    # so a quality-dominant scheme is expressible directly, e.g.
    #   flops:0.06,latency_ns:0.06,xla_peak_memory:0.06,peak_memory:0.06,bkstep_acc:1.0
    # (bkstep_acc dominates; cost channels a minor nudge).
    #
    # Same sign machinery as the all-channel branch: the env reward vector emits
    # costs NEGATED and bkstep_acc POSITIVE, so a single POSITIVE weight per
    # named channel is correct (reward low-cost + high bkstep). PopArt normalises
    # each named channel. Every name is validated against REWARD_INDEX (env.py
    # REWARD_NAMES) so a typo errors CLEARLY; a non-float weight also errors.
    # PRECEDENCE: REWARD_CHANNELS (explicit list) > REWARD_ALL_CHANNELS=1 >
    # the legacy cmp/mem/acc slots.
    _rc_raw = str(_osac.environ.get("ALPHAGRAD_REWARD_CHANNELS", "") or "").strip()
    if _rc_raw:
        _w_default = float(_osac.environ.get("ALPHAGRAD_ALL_CHANNEL_W", "1.0") or 1.0)
        _pairs: list[tuple[str, float]] = []
        for _entry in _rc_raw.split(","):
            _entry = _entry.strip()
            if not _entry:
                continue
            if ":" in _entry:
                _nm, _wraw = _entry.split(":", 1)
                _nm = _nm.strip()
                try:
                    _wv = float(_wraw.strip())
                except ValueError:
                    raise ValueError(
                        f"ALPHAGRAD_REWARD_CHANNELS entry {_entry!r} has a "
                        f"non-float weight {_wraw!r}"
                    )
            else:
                _nm, _wv = _entry, _w_default
            _pairs.append((_nm, _wv))
        _bad = [nm for nm, _ in _pairs if nm not in REWARD_INDEX]
        if _bad:
            raise ValueError(
                f"ALPHAGRAD_REWARD_CHANNELS names unknown channel(s) {_bad}; "
                f"valid keys: {sorted(REWARD_INDEX)}"
            )
        for _nm, _wv in _pairs:
            w[REWARD_INDEX[_nm]] = _wv
        if not np.any(w):
            w[REWARD_INDEX["muls_adds_fmas"]] = 1.0
        return w

    _all_ch = (
        _osac.environ.get("ALPHAGRAD_REWARD_ALL_CHANNELS", "0") == "1"
        or "all" in getattr(args, "rewards", [])
    )
    if _all_ch:
        _w_all = float(_osac.environ.get("ALPHAGRAD_ALL_CHANNEL_W", "1.0") or 1.0)
        # Cost + quality channels that graphax/XLA measure for the output
        # Jacobian. latency_ns only meaningful with --measure-latency;
        # bkstep_acc only when the closed-loop probe is on.
        _names = [
            "flops", "muls_adds_fmas", "max_io_sum", "bytes_accessed",
            "peak_memory", "xla_peak_memory", "cosine_sim", "frob_residual",
        ]
        if bool(getattr(args, "measure_latency", False)) or \
                getattr(args, "cmp_type", "") == "latency":
            _names.append("latency_ns")
        if _osac.environ.get("ALPHAGRAD_BKSTEP", "0") == "1":
            _names.append("bkstep_acc")
        for _nm in _names:
            w[REWARD_INDEX[_nm]] = _w_all
        if not np.any(w):
            w[REWARD_INDEX["muls_adds_fmas"]] = 1.0
        return w

    if "cmp" in args.rewards:
        cmp_name = _CMP_TYPE_TO_REWARD[args.cmp_type]
        w[REWARD_INDEX[cmp_name]] = float(getattr(args, "lambda_cmp", 1.0))
    if "mem" in args.rewards:
        mem_name = _MEM_TYPE_TO_REWARD[args.mem_type]
        w[REWARD_INDEX[mem_name]] = float(getattr(args, "lambda_mem", 1.0))
    if "acc" in args.rewards:
        # The accuracy/quality channel. Default = cosine_sim (Jacobian fidelity);
        # ALPHAGRAD_ACC_PROXY=bkstep routes the acc weight to the B_kstep
        # closed-loop trainability accuracy channel instead (idx 9). Both are
        # symlog-bypassed [0,1] quality signals, so --lambda-acc weights them on
        # the same footing as the cost channels' symlog magnitude. bkstep-acc is
        # in [0,1] so --lambda-acc 1.0 keeps it commensurate with the small
        # (0.005) symlog cost weights per the additive-reward design.
        import os as _os
        _acc_proxy = _os.environ.get("ALPHAGRAD_ACC_PROXY", "cosine").strip().lower()
        _acc_idx = BKSTEP_ACC_IDX if _acc_proxy == "bkstep" else COSINE_SIM_IDX
        w[_acc_idx] = float(getattr(args, "lambda_acc", 1.0))
    lam_frob = float(getattr(args, "lambda_frob", 0.0))
    if lam_frob != 0.0:
        w[FROB_RESIDUAL_IDX] = lam_frob

    # Capped-cossim GUIDE weight (anti flat-zero-basin). When the acc channel
    # is routed to B_kstep (ALPHAGRAD_ACC_PROXY=bkstep), cosine_sim (idx 6) is
    # free to carry a small additive climb term: reward += lambda_guide *
    # min(cossim, C), with the cap C applied in env._callback via
    # ALPHAGRAD_COSSIM_GUIDE_CAP. Below the trainability edge (bkstep==0, flat)
    # this term is the only gradient, so it pulls the untrained policy up to
    # the edge; above C it is constant, so B_kstep takes over. lambda_guide is
    # read from --lambda-cossim-guide or the env var ALPHAGRAD_LAMBDA_COSSIM_GUIDE.
    import os as _os2
    _lam_guide = float(getattr(
        args, "lambda_cossim_guide",
        _os2.environ.get("ALPHAGRAD_LAMBDA_COSSIM_GUIDE", 0.0) or 0.0,
    ))
    if _lam_guide != 0.0:
        # Additive: if acc is on cosine (default), fold the guide onto the
        # same channel; if acc routed to bkstep, cosine_sim is otherwise 0.
        w[COSINE_SIM_IDX] = float(w[COSINE_SIM_IDX]) + _lam_guide

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
    Acts as a hard gate for cases
    where the user wants to gate one reward on another instead of
    paying a soft cost. The check happens in the worker, BEFORE the
    per-channel GAE / advantage stack — so the easier channel's
    advantage is recomputed cleanly from the gated reward, not gated
    after the fact.

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


def build_terminal_solutions(
    per_env_terminal: np.ndarray,
    per_env_seqs,
    sentinel: float,
) -> list:
    """Pareto-archive input rows: each NON-sentinel env's terminal objective
    vector paired with the action sequence that produced it.

    Sentinel filtering uses :func:`filter_sentinel_mask` — i.e. EXACT float
    equality against ``sentinel`` over the cost channels, NOT a magnitude
    threshold. Legitimate raw cost rewards routinely exceed 1e9 in magnitude
    (muls_adds_fmas ≈ -7e12), so a ``<= -1e9``-style threshold flags every
    env as a sentinel and leaves the archive permanently empty (a bug this
    helper exists to centralise the fix for).

    Args:
        per_env_terminal: ``(N, NUM_REWARDS)`` raw terminal reward vectors.
        per_env_seqs: callable ``idx -> seq`` OR an indexable of length N.
        sentinel: the exact sentinel reward value (cache.SENTINEL_REWARD_VALUE).
    """
    keep = np.atleast_1d(filter_sentinel_mask(per_env_terminal, sentinel))
    # Also drop ALL-ZERO reward vectors. A real terminal always has nonzero
    # deterministic cost channels (muls_adds_fmas / flops > 0 for any real
    # graph), so an all-zero vector is a SENTINEL that was zeroed upstream
    # (run_rollout_and_train zeroes sentinel transitions for GAE stability)
    # BEFORE this exact -1e10 filter ran — so the == sentinel test misses it.
    # Such a point reads (latency 0, mem 0, frob 0) and Pareto-dominates every
    # real solution, collapsing the archive to it. Reject it.
    nonzero = np.any(np.abs(np.asarray(per_env_terminal)) > 1e-9, axis=-1)
    keep = np.asarray(keep) & np.atleast_1d(nonzero)
    get_seq = (
        per_env_seqs if callable(per_env_seqs) else (lambda n: per_env_seqs[n])
    )
    return [
        {
            "obj": [float(v) for v in per_env_terminal[n]],
            "seq": get_seq(n),
        }
        for n in range(per_env_terminal.shape[0])
        if bool(keep[n])
    ]


# ---------------------------------------------------------------------------
# Per-channel aggregation (replicates mu0_ray_worker.py:783-816)
# ---------------------------------------------------------------------------

def aggregate_per_channel_stats(
    buf_reward_vec_np: np.ndarray,
    reward_weights_np: np.ndarray,
    *,
    sentinel: float,
    action_seq: Sequence[Any] | None = None,
    dones_mask: np.ndarray | None = None,
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
        dones_mask: optional shape (T, N) bool mask flagging terminal
            transitions. When supplied the result dict additionally
            carries ``terminal_means`` and ``best_terminal`` keyed by
            channel name. cossim / frob_residual are
            :data:`SPARSE_TERMINAL_INDICES` — their per-step mean over
            all timesteps is heavily diluted by intermediate zero-reward
            steps; ``terminal_means`` reports the honest signal.

    Returns:
        dict with keys:

        * ``per_reward_means``: `{name: float}` — mean over (T, N), with
          sentinel rows masked out. This is the per-STEP mean; for
          sparse-terminal channels (cossim/frob) it is diluted by the
          many intermediate zero-reward steps. Use ``terminal_means``
          instead for those.
        * ``terminal_means``: `{name: float}` — mean over (T, N)
          timesteps where ``dones_mask`` is True. Only populated when
          ``dones_mask`` is supplied. Equals ``per_reward_means`` for
          cost channels when every rollout has the same length; differs
          significantly for sparse-terminal quality channels.
        * ``best_terminal``: `{name: float}` — max over terminal
          timesteps for each channel. Only populated when ``dones_mask``
          is supplied.
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
    # Sparse-terminal channels (cosine_sim, frob_residual) are overwritten
    # below with the terminal-step-only mean so they aren't diluted by the
    # T-1 zero intermediate steps.
    per_reward_means: dict[str, float] = {}
    flat_mask = valid_mask.reshape(-1)
    flat_rewards = buf_reward_vec_np.reshape(-1, R)
    if flat_mask.any():
        masked_mean = flat_rewards[flat_mask].mean(axis=0)
    else:
        masked_mean = np.zeros((R,), dtype=np.float32)
    for j, name in enumerate(REWARD_NAMES):
        per_reward_means[name] = float(masked_mean[j])

    # Terminal-only stats — the honest per-episode signal for the
    # sparse-terminal quality channels (cossim / frob_residual).
    terminal_means: dict[str, float] = {}
    best_terminal: dict[str, float] = {}
    if dones_mask is not None:
        dones_b = np.asarray(dones_mask).astype(bool)
        if dones_b.shape != (T, N):
            raise ValueError(
                f"dones_mask shape {dones_b.shape} != (T, N) = {(T, N)}"
            )
        term_and_valid = dones_b & valid_mask
        flat_term = term_and_valid.reshape(-1)
        if flat_term.any():
            term_rewards = flat_rewards[flat_term]  # (n_term, R)
            term_mean = term_rewards.mean(axis=0)
            term_max = term_rewards.max(axis=0)
        else:
            term_mean = np.zeros((R,), dtype=np.float32)
            term_max = np.zeros((R,), dtype=np.float32)
        for j, name in enumerate(REWARD_NAMES):
            terminal_means[name] = float(term_mean[j])
            best_terminal[name] = float(term_max[j])
        for idx in SPARSE_TERMINAL_INDICES:
            name = REWARD_NAMES[idx]
            per_reward_means[name] = terminal_means[name]

    # Per-env per-channel sum (zero-out sentinel rows so a single timeout
    # doesn't poison the env's running per-channel total).
    masked_rv = np.where(valid_mask[:, :, None], buf_reward_vec_np, 0.0)
    r_per_env = masked_rv.sum(axis=0)  # (N, NUM_REWARDS)
    weighted_per_env = r_per_env * reward_weights_np  # (N, NUM_REWARDS)
    per_env_tot = weighted_per_env.sum(axis=-1)  # (N,)

    # ANTI-DEGEN best_overall guard. An env whose every (valid) transition
    # was a sentinel — i.e. ``valid_mask`` is False for ALL of its
    # timesteps — contributes an ALL-ZERO ``r_per_env`` row and so a
    # ``per_env_tot`` of 0. With cost channels stored negated (valid rules
    # carry NEGATIVE cost contributions), that spurious zero can OUTRANK
    # every genuine rule and be crowned best_overall — exactly the
    # "failed/degenerate measure reads as free perfect" corruption. Mark
    # fully-sentinel envs with ``-inf`` so they can never be the argmax.
    # When the caller stamps degenerate-terminal rows with the sentinel
    # value (the anti-degen path in ppo_ray_worker), those envs become
    # fully-sentinel here and are excluded too. Guard against the
    # all-excluded edge case (keep at least the original argmax).
    env_all_sentinel = ~valid_mask.any(axis=0)  # (N,)
    per_env_tot_for_best = per_env_tot.copy()
    if env_all_sentinel.any() and not env_all_sentinel.all():
        per_env_tot_for_best[env_all_sentinel] = -np.inf

    def _seq_for(env_idx: int) -> Any:
        if action_seq is None:
            return []
        try:
            return action_seq[env_idx]
        except (IndexError, TypeError):
            return []

    best_overall_env = int(per_env_tot_for_best.argmax())
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

    # Per-channel order statistics over this episode's N rollout envs, plus
    # the representative env sequence at each quantile (nearest-rank). Lets
    # downstream analysis pull e.g. the median-cosine order (a non-degenerate
    # approximation) instead of only the raw-return-best (often degenerate /
    # exact). Computed for ALL channels (incl. weight-0 ones like cosine_sim).
    n_env = int(r_per_env.shape[0])
    _QLABELS = (("min", 0.0), ("q1", 0.25), ("median", 0.5), ("q3", 0.75), ("max", 1.0))
    reward_quantiles: dict[str, dict] = {}
    quantile_sequences: dict[str, dict] = {}
    for j, name in enumerate(REWARD_NAMES):
        # Sanitize non-finite values: np.argsort sorts NaN to the end, which
        # would make the 'max' (and possibly median/q3) quantile select a NaN
        # env and report NaN for reward_dist/<chan>/max + the dumped sequence.
        # Treat non-finite as 0 (purely a diagnostic channel).
        col = np.nan_to_num(
            r_per_env[:, j].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0,
        )
        order = np.argsort(col, kind="stable")  # ascending by raw value
        qstat: dict[str, float] = {}
        qseq: dict[str, dict] = {}
        for label, q in _QLABELS:
            env_i = int(order[int(round(q * (n_env - 1)))])
            qstat[label] = float(col[env_i])
            qseq[label] = {
                "env_idx": env_i,
                "raw_value": float(col[env_i]),
                "all_raw": {
                    REWARD_NAMES[k]: float(r_per_env[env_i, k])
                    for k in range(NUM_REWARDS)
                },
                "seq": _seq_for(env_i),
            }
        reward_quantiles[name] = qstat
        quantile_sequences[name] = qseq

    return {
        "per_reward_means": per_reward_means,
        "terminal_means": terminal_means,
        "best_terminal": best_terminal,
        "best_per_reward": best_per_reward,
        "best_overall_rewards": best_overall_rewards,
        "best_overall_weighted": best_overall_weighted,
        "best_overall_env": best_overall_env,
        "best_overall_weighted_total": float(per_env_tot[best_overall_env]),
        "best_overall_seq": _seq_for(best_overall_env),
        "reward_quantiles": reward_quantiles,
        "quantile_sequences": quantile_sequences,
    }


# ---------------------------------------------------------------------------
# Grouping helpers for the unified wandb panel layout
# ---------------------------------------------------------------------------

def _reward_group(name: str) -> str:
    """Return ``"cost"`` or ``"quality"`` per REWARD_NAMES layout.

    Cost channels: muls_adds_fmas, flops, latency_ns, max_io_sum,
    bytes_accessed, peak_memory. Quality channels: cosine_sim,
    frob_residual.
    """
    if REWARD_INDEX[name] in (COSINE_SIM_IDX, FROB_RESIDUAL_IDX):
        return "quality"
    return "cost"


def build_unified_reward_log_dict(
    ch_stats: dict,
    *,
    corridor_low: float | None = None,
    corridor_high: float | None = None,
    terminal_cossims: np.ndarray | None = None,
) -> dict:
    """Build the unified ``reward/{cost,quality}/{per_step,terminal,best_terminal}/<name>``
    key dict that PPO / MuZero / GFN all emit.

    Args:
        ch_stats: the dict returned by :func:`aggregate_per_channel_stats`.
        corridor_low / corridor_high: optional corridor bounds for the
            cosine_sim channel. When both are supplied, also emit
            ``corridor/{in,below,above}_band_fraction``. Pass
            ``corridor_low=None`` (no floor) with a finite ``corridor_high``
            to track only the ceiling.
        terminal_cossims: optional 1-D array of terminal cossim values
            across the batch — required when corridor metrics are
            requested. If None, corridor keys are omitted.

    Returns:
        dict mapping the new key names to scalar floats / bools, ready
        for ``wandb.log``.
    """
    out: dict[str, Any] = {}
    per_step = ch_stats.get("per_reward_means", {})
    terminal = ch_stats.get("terminal_means", {})
    best_terminal = ch_stats.get("best_terminal", {})
    for name in REWARD_NAMES:
        grp = _reward_group(name)
        if name in per_step:
            out[f"reward/{grp}/per_step/{name}"] = float(per_step[name])
        if name in terminal:
            out[f"reward/{grp}/terminal/{name}"] = float(terminal[name])
        if name in best_terminal:
            out[f"reward/{grp}/best_terminal/{name}"] = float(best_terminal[name])
    out["reward/cost/symlog_applied"] = True
    out["reward/quality/symlog_applied"] = False

    if terminal_cossims is not None and len(terminal_cossims) > 0:
        cs = np.asarray(terminal_cossims, dtype=np.float32)
        if corridor_high is not None:
            below_mask = cs < (corridor_low if corridor_low is not None else -np.inf)
            above_mask = cs > corridor_high
            in_mask = ~(below_mask | above_mask)
            out["corridor/in_band_fraction"] = float(in_mask.mean())
            out["corridor/below_band_fraction"] = float(below_mask.mean())
            out["corridor/above_band_fraction"] = float(above_mask.mean())
            if corridor_low is not None:
                out["corridor/low"] = float(corridor_low)
            out["corridor/high"] = float(corridor_high)
    return out


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
    # Latest-episode per-channel quantile sequences (overwritten each step;
    # the numeric distribution time-series lives in wandb under reward_dist/).
    if "quantile_sequences" in stats:
        state["quantile_sequences"] = stats["quantile_sequences"]
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

    The ``seq`` field is normalised through
    :func:`alphagrad.approx.common.seq_replay.to_typed_records` so the
    on-disk format carries typed dicts (``{"vertex": v, "ops":
    [{"op": "Diag", "i": ..., "j": ..., "factor": ...}, ...]}``) rather
    than the legacy int-array-with-sentinels representation. The
    decoder accepts both shapes; new runs emit only the typed form.
    """
    # Late import to avoid a circular dep on alphagrad.approx.env via
    # graphax.sparse.micro_actions (seq_replay imports those at module load).
    from alphagrad.approx.common.seq_replay import to_typed_records

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

    def _seq(raw):
        if raw is None:
            return None
        try:
            return to_typed_records(raw)
        except Exception:
            # Never let a snapshot conversion crash the trainer — fall
            # back to the raw payload so the run still produces a valid
            # (if legacy-format) best_sequences.json.
            return raw

    overall = {
        "return": _coerce(state.get("best_global_return", float("nan"))),
        "ep": _coerce(state.get("best_global_ep", -1)),
        "rewards_raw": _coerce_dict(state.get("best_global_rewards", {})),
        "rewards_weighted": _coerce_dict(state.get("best_global_weighted_split", {})),
        "seq": _seq(state.get("best_global_seq")),
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
            "seq": _seq(info.get("seq", [])),
        }
    # Per-channel quantile sequences (latest episode): min/q1/median/q3/max
    # env order for every channel, typed-converted like the bests above.
    quantile_sequences: dict[str, dict] = {}
    for name, qd in state.get("quantile_sequences", {}).items():
        quantile_sequences[name] = {
            label: {
                "env_idx": _coerce(e.get("env_idx", -1)),
                "raw_value": _coerce(e.get("raw_value", float("nan"))),
                "all_raw": _coerce_dict(e.get("all_raw", {})),
                "seq": _seq(e.get("seq", [])),
            }
            for label, e in qd.items()
        }
    return {
        "best_overall": overall,
        "best_per_channel": per_channel,
        "quantile_sequences": quantile_sequences,
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
        "policy_loss": stats.get("policy_loss", stats.get("ppo_loss", float("nan"))),
        "value_loss": stats.get("value_loss", float("nan")),
        "explained_variance": stats.get("explained_variance", float("nan")),
        "total_loss": stats.get("total_loss", float("nan")),
        "train_step": stats.get("train_step", 0),
        "nan_skip_count": stats.get("nan_skip_count", 0),
    }
    if "entropy_root" in stats:
        log_dict["entropy_root"] = stats["entropy_root"]
    # bridge-cse: the LIVE entropy coefficient (now held constant — anneal
    # disabled). Surfaced so the "entropy coef constant" invariant is
    # verifiable in wandb.
    if "entropy_coef" in stats:
        log_dict["entropy_coef"] = stats["entropy_coef"]
    if "reward_loss" in stats:
        log_dict["reward_loss"] = stats["reward_loss"]
    if "buffer_size" in stats:
        log_dict["buffer_size"] = stats["buffer_size"]
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
    # (e.g. `popart/sigma_cosine_sim`).
    for k, v in stats.items():
        if (
            k.startswith("reward_mean/")
            or k.startswith("reward_dist/")
            # bridge-cse: full-range raw cosine_sim / bkstep telemetry
            # (reward/cosine_sim_raw{,_max,_min}, reward/bkstep_acc_raw).
            or k.startswith("reward/")
            # bridge-cse: ALL 10 measurement channels (incl. zero-weight
            # ones) at their raw terminal-mean — measure/<name>.
            or k.startswith("measure/")
            or k.startswith("bkstep/")
            or k.startswith("decomp/")
            or k.startswith("sentinel/")
            or k.startswith("entropy/")
            or k.startswith("popart/")
            or k.startswith("ppo/")
            or k.startswith("cost_head/")
        ):
            log_dict[k] = v
    return log_dict
