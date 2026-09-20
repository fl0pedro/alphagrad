"""Shared policy/value network construction for every trainer.

WHY THIS EXISTS (2026-08-04 parity audit): ``ppo._build_agent`` was already the
de-facto shared factory — ``alpha0``, ``gdpo``, ``gfn``, ``az_gumbel`` and the
tests all import it — but each caller fed it a DIFFERENT args namespace and only
PPO ran the init pipeline afterwards. ``az_gumbel`` built its namespace from
``ppo.make_argparser().parse_args([])`` (PPO's *bare defaults*) and overrode five
fields, while the PPO launcher overrode five *different* fields on the command
line, so every field neither side set diverged silently. Measured result: PPO
161,916 parameters vs AZ 1,171,693 — a 7.2x gap — plus AZ never calling
``init_linear_weights`` / ``_scale_output_heads``, which is what makes the
initial vertex logits near-uniform. For a Gumbel-AlphaZero root prior that is
not cosmetic: raw Equinox ``Linear`` init leaves non-zero biases and full-scale
weights straight in the logits, so the root prior is biased before training
starts.

The rule this module enforces: the BACKBONE, the VERTEX head and the VALUE heads
are byte-identical across trainers; only the APPROXIMATION heads may differ, and
only because the algorithms genuinely address different action spaces.
"""

from __future__ import annotations

import jax.random as jrand

# ---------------------------------------------------------------- shared arch
# The one config both trainers build from. Owner decision (2026-08-04): unify on
# the PPO campaign's architecture, so the existing PPO baselines stay comparable
# and the 7.2x-larger AZ net (which was an accident, not a choice) shrinks.
POLICY_ARCH_DEFAULTS: dict = {
    "vocab_size": 512,
    "embd_dim": 32,
    "num_heads": 2,
    "num_layers": 3,
    "hidden_dim": 256,
    "op_embd_dim": 8,
    "value_dims": "64,32",
    "set_pointer": True,
    "set_pointer_blocks": 2,
    "head_init_scale": 0.1,
}

# The ONLY fields a trainer may legitimately set differently, because they
# select which approximation-action heads exist. Every one of them must be set
# EXPLICITLY by each trainer — never left to an argparser default, which is
# exactly how the two arms drifted apart.
ALGO_HEAD_FIELDS: tuple[str, ...] = (
    "dynamic_substeps",
    "unified_head",
    "no_approx_head",
    "face_actions",
    "unified_face_head",
    "live_faces",
    "max_substeps",
    "axis_group_embedding",
)


def apply_policy_arch(ns, **overrides):
    """Stamp the shared architecture onto an args namespace, in place.

    ``overrides`` is for the ALGO_HEAD_FIELDS (and nothing else in normal use);
    passing a shared-architecture key is allowed but logged by the parity test
    as a deliberate divergence.
    """
    for k, v in POLICY_ARCH_DEFAULTS.items():
        setattr(ns, k, v)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def derive_agent_keys(seed: int):
    """PPO's key derivation, verbatim (ppo.py:3728-3729 + 4419).

    ``key = PRNGKey(seed); key, _args_key = split(key); agent_key, init_key, _ =
    split(key, 3)``. Reproduced here so a given seed yields the SAME weights in
    every trainer — AZ previously used ``PRNGKey(seed)`` directly, so even at
    matched shapes its weights differed.
    """
    key = jrand.PRNGKey(int(seed))
    key, _args_key = jrand.split(key)
    agent_key, init_key, _key = jrand.split(key, 3)
    return agent_key, init_key


REMOVED_ENV_KNOBS = {
    # env var -> the flag that replaced it (ticket dsnn-3qm.44: args only,
    # no fallback period). A set var is refused at startup, never read.
    "ALPHAGRAD_FACE_NONE_BIAS": "--face-none-bias",
}


def refuse_removed_env_knobs(environ=None) -> None:
    """Fail loudly at startup when a knob that became a flag is still set
    in the environment. Nothing reads these vars any more, so a launcher
    exporting one would run with the knob silently OFF."""
    import os as _os
    env = _os.environ if environ is None else environ
    for var, flag in REMOVED_ENV_KNOBS.items():
        if var in env:
            raise SystemExit(
                f"{var} is set but is no longer read: it was replaced by "
                f"the {flag} flag (ticket dsnn-3qm.44, args only, no "
                f"fallback). Unset it and pass {flag} instead.")


def apply_face_none_bias(agent, bias: float, skip_bias: float | None = None):
    """IDENTITY-INIT for the face head (--face-none-bias B, default 0 =
    off): +B on each slot's OP_NONE logit, -B on SKIP.
    Called by build_and_init_agent AND by ppo.main's inline init path
    (which predates the factory and does not route through it -- the
    v54 PPO arm shipped without the bias until this was split out).

    ``skip_bias`` (--face-skip-bias Bs, default None): when given, the SKIP
    logit gets -Bs instead of -B, independent of the none bias, which keeps
    its +B on the OP_NONE logits regardless. None (default) reproduces the
    old coupled behaviour bit for bit -- SKIP gets -B, same as before this
    parameter existed."""
    _nb = float(bias or 0.0)
    _skb = _nb if skip_bias is None else float(skip_bias)
    _fpp = getattr(agent, "face_path_policy", None)
    if (_nb == 0.0 and _skb == 0.0) or _fpp is None \
            or getattr(_fpp, "head", None) is None:
        return agent
    import equinox as _eqx
    from alphagrad.approx.unified_face_head import (
        FACE_SLOTS as _FS, OP_NONE as _NONE, O_SKIP as _SKIP,
        S_OP as _SOP, slot_base as _slot_base)
    _bias = _fpp.head.proj.layers[-1].bias
    for _s in range(_FS):
        _bias = _bias.at[_slot_base(_s) + _SOP + _NONE].add(_nb)
    _bias = _bias.at[_SKIP].add(-_skb)
    agent = _eqx.tree_at(
        lambda a: a.face_path_policy.head.proj.layers[-1].bias,
        agent, _bias)
    if skip_bias is None:
        print(f"[factory] face-head IDENTITY INIT: OP_NONE bias +{_nb}, "
              f"SKIP bias -{_nb} (trainable; P(approx/face) ~ "
              f"{3 * 2.718 ** (-_nb):.3f})", flush=True)
    else:
        print(f"[factory] face-head IDENTITY INIT: OP_NONE bias +{_nb}, "
              f"SKIP bias -{_skb} (--face-skip-bias, independent of "
              f"OP_NONE; trainable; P(approx/face) ~ "
              f"{3 * 2.718 ** (-_nb):.3f}, P(skip) ~ "
              f"{1.0 / (1.0 + 2.718 ** _skb):.3f})", flush=True)
    return agent


def derive_face_none_bias(F: float, S: int, k: int, a: float) -> float:
    """B = ln(F*S*k/a - k), the none-logit bias whose expected requested
    approximations per plan is ``a`` (--face-init-approx-per-plan), given
    ``F`` live faces, ``S`` slots/face and ``k`` non-none ops/slot. Raises
    when ``a`` makes the log undefined (a <= 0, or F*S*k/a <= k -- the
    argument asks for at least as many approximations as there are ops to
    reject, which no finite bias can produce)."""
    F, a = float(F), float(a)
    if a <= 0.0:
        raise ValueError(
            f"--face-init-approx-per-plan {a!r} must be > 0.")
    x = F * S * k / a - k
    if x <= 0.0:
        raise ValueError(
            f"--face-init-approx-per-plan {a!r} is unreachable at F={F:g} "
            f"faces, S={S} slots, k={k} ops: F*S*k/a - k = {x:g} <= 0, so "
            "ln(x) is undefined. a must be < F*S*k/(k+1) "
            f"(={F * S * k / (k + 1):g} here); a this small asks for more "
            "approximations than the head can even offer.")
    import math
    return math.log(x)


def derive_face_skip_bias(F: float, kappa: float) -> float:
    """Bs = ln(F/kappa - 1), the skip-logit bias whose expected skips per
    plan is ``kappa`` (--face-init-skips-per-plan), given ``F`` live faces.
    Raises when ``kappa`` makes the log undefined (kappa <= 0, or kappa >= F
    -- more skips than there are faces)."""
    F, kappa = float(F), float(kappa)
    if kappa <= 0.0:
        raise ValueError(
            f"--face-init-skips-per-plan {kappa!r} must be > 0.")
    x = F / kappa - 1.0
    if x <= 0.0:
        raise ValueError(
            f"--face-init-skips-per-plan {kappa!r} is unreachable at "
            f"F={F:g} faces: F/kappa - 1 = {x:g} <= 0, so ln(x) is "
            f"undefined. kappa must be < F (={F:g} here).")
    import math
    return math.log(x)


def expected_face_counts(F: float, S: int, k: int, B: float,
                         Bs: float) -> tuple[float, float]:
    """``(E[A], E[K])`` -- expected requested approximations and skips per
    plan under none-bias ``B`` and skip-bias ``Bs``, F live faces, S
    slots/face, k non-none ops/slot. The closed forms
    :func:`derive_face_none_bias` / :func:`derive_face_skip_bias` invert."""
    import math
    e_a = F * S * k / (math.exp(B) + k)
    e_k = F / (1.0 + math.exp(Bs))
    return e_a, e_k


def resolve_face_init_bias(args, F: float | None = None):
    """``(B, Bs)`` derived from --face-init-approx-per-plan /
    --face-init-skips-per-plan, or ``(None, None)`` when neither is set
    (inert). Refuses if --face-none-bias / --face-skip-bias is ALSO set
    (the two ways of setting the bias would conflict), if the argument
    makes the closed form's log undefined, or if ``F`` -- the live-face
    count of the built target -- is not known at this call site.

    F is NOT a graph property computable ahead of time: eliminating a
    vertex rewires its neighbours, so a plan's total live-face count is the
    length of a walk over one ELIMINATION ORDER, not a property of the
    jaxpr alone (tools/faces_per_vertex.py: "the unrolling's length is not
    a property of the graph alone ... vertex k's face count depends on the
    k-1 before it"). ``build_and_init_agent`` / ``ppo.main``'s inline copy
    call this BEFORE any order exists: under --fixed-order free (the
    dsnn-dfw.74 rows this flag exists for) the order is the sequence of
    actions the policy itself samples during rollout, produced by code
    that runs AFTER the agent (hence after this call) and differs per
    episode and per environment -- there is no single F for "the built
    target" to read here. A static order (--fixed-order reverse/markowitz,
    common/order.py) does exist before init, but it is a property of that
    CHOSEN ORDER, not of the target; treating its count as F would pick one
    arbitrary order's number and label it the target's, which is the guess
    this function must refuse rather than make. Callers therefore pass no
    F today, and this refuses unconditionally whenever either flag is set."""
    a = getattr(args, "face_init_approx_per_plan", None)
    kappa = getattr(args, "face_init_skips_per_plan", None)
    if a is None and kappa is None:
        return None, None
    _nb = getattr(args, "face_none_bias", 0.0) or 0.0
    _sb = getattr(args, "face_skip_bias", None)
    if float(_nb) != 0.0 or _sb is not None:
        raise ValueError(
            "--face-init-approx-per-plan/--face-init-skips-per-plan derive "
            "B/Bs themselves; --face-none-bias and/or --face-skip-bias is "
            f"also set (face_none_bias={_nb!r}, face_skip_bias={_sb!r}) "
            "and would conflict. Drop one or the other.")
    if F is None:
        raise ValueError(
            "--face-init-approx-per-plan/--face-init-skips-per-plan need "
            "the live-face count F of the built target, and F is not known "
            "at the point of agent init: it is the length of a walk over "
            "one elimination ORDER (tools/faces_per_vertex.py), not a "
            "property of the jaxpr alone, and under --fixed-order free the "
            "order is produced by the policy's own rollout, after the "
            "agent this flag would initialise already exists. See "
            "resolve_face_init_bias's docstring for the exact order of "
            "construction. Not implemented: refusing rather than guessing "
            "which order's count to call F.")
    from alphagrad.approx.unified_face_head import (
        FACE_SLOTS as _S, NUM_APPROX_OPS as _NOPS)
    _k = _NOPS - 1
    B = None if a is None else derive_face_none_bias(F, _S, _k, a)
    Bs = None if kappa is None else derive_face_skip_bias(F, kappa)
    if B is None:
        B = float(_nb)
    if Bs is None:
        Bs = B
    e_a, e_k = expected_face_counts(F, _S, _k, B, Bs)
    print(f"[factory] face-head init from plan targets: F={F:g} faces, "
          f"--face-init-approx-per-plan={a!r} -> B={B:g}, "
          f"--face-init-skips-per-plan={kappa!r} -> Bs={Bs:g}, "
          f"E[A]={e_a:g}, E[K]={e_k:g}", flush=True)
    return B, Bs


def build_and_init_agent(args, total_v: int, num_factors: int, max_rules: int,
                         *, seed: int | None = None, key=None, init_key=None):
    """``_build_agent`` + orthogonal init + output-head scaling, as one step.

    Pass ``seed`` to get PPO's exact key derivation, or pass ``key``/``init_key``
    explicitly. The three-stage pipeline is what produces near-uniform initial
    vertex logits:
      1. ``_build_agent``            — architecture,
      2. ``init_linear_weights``     — orthogonal(sqrt 2) weights, ZERO biases
                                       (kills the constant per-vertex offset),
      3. ``_scale_output_heads``     — x0.1 on the pointer scoring projection and
                                       zeroing of the five additive context paths.
    Stages 2+3 are dispatched through ``ppo.apply_init_scheme`` so this path
    and ``ppo.main``'s inline copy honour ``--init-scheme`` /
    ``--scale-face-head`` identically; the defaults reproduce the pipeline
    above bit for bit.
    """
    # Imported lazily: ppo.py imports heavy JAX/equinox modules and this module
    # is also imported by trainers that ppo.py itself does not know about.
    from alphagrad.approx.ppo import _build_agent, apply_init_scheme

    if key is None or init_key is None:
        if seed is None:
            raise ValueError(
                "build_and_init_agent needs either seed= or both key= and "
                "init_key=; passing only one of the keys silently skips the "
                "init pipeline that makes the root prior near-uniform."
            )
        key, init_key = derive_agent_keys(seed)

    agent = _build_agent(args, total_v, num_factors, max_rules, key)
    agent = apply_init_scheme(agent, init_key, args)

    _derived_B, _derived_Bs = resolve_face_init_bias(args)
    if _derived_B is None:
        _nb = float(getattr(args, "face_none_bias", 0.0) or 0.0)
        _skip_bias = getattr(args, "face_skip_bias", None)
        _skip_bias = None if _skip_bias is None else float(_skip_bias)
    else:
        _nb, _skip_bias = _derived_B, _derived_Bs
    agent = apply_face_none_bias(agent, _nb, _skip_bias)
    return agent


# ------------------------------------------------------------ shared weights
def channel_weights(args) -> tuple[float, float, float]:
    """``(w_cmp, w_mem, w_acc)`` — the objective every trainer optimizes.

    PPO consumes this as the per-head ``preference`` vector; AZ expands it to
    ``W4 = [-w_cmp, -w_mem, 0, +w_acc]`` (its channel order is
    ``[latency, peak, flops, cosine]`` with costs stored positive, hence the
    signs). Sharing one source stops the two arms optimizing different
    functions, which they did until now: PPO weighted cosine x2, AZ x1.
    """
    rewards = set(getattr(args, "rewards", ("cmp", "mem", "acc")) or ())
    w_cmp = float(getattr(args, "lambda_cmp", 1.0)) if "cmp" in rewards else 0.0
    w_mem = float(getattr(args, "lambda_mem", 1.0)) if "mem" in rewards else 0.0
    w_acc = float(getattr(args, "lambda_acc", 1.0)) if "acc" in rewards else 0.0
    return w_cmp, w_mem, w_acc


def az_w4(args):
    """AZ's 4-vector form of :func:`channel_weights` (flops inert)."""
    import numpy as np
    w_cmp, w_mem, w_acc = channel_weights(args)
    return np.array([-w_cmp, -w_mem, 0.0, +w_acc], dtype=np.float64)
