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
    """
    # Imported lazily: ppo.py imports heavy JAX/equinox modules and this module
    # is also imported by trainers that ppo.py itself does not know about.
    from alphagrad.approx.ppo import _build_agent, _scale_output_heads
    from alphagrad.approx.common.init import init_linear_weights

    if key is None or init_key is None:
        if seed is None:
            raise ValueError(
                "build_and_init_agent needs either seed= or both key= and "
                "init_key=; passing only one of the keys silently skips the "
                "init pipeline that makes the root prior near-uniform."
            )
        key, init_key = derive_agent_keys(seed)

    agent = _build_agent(args, total_v, num_factors, max_rules, key)
    agent = init_linear_weights(agent, init_key)
    agent = _scale_output_heads(agent, float(getattr(args, "head_init_scale", 0.1)))

    # IDENTITY-INIT for the face head (owner 2026-08-09,
    # ALPHAGRAD_FACE_NONE_BIAS, default 0 = off): +B on each slot's
    # OP_NONE logit, -B on the SKIP logit. At B=6 the per-face approx
    # probability is ~2%, so an INIT plan is near-exact -- the loss-drop
    # quality channel warms at ~0.85 instead of the constant 0.0 that
    # froze PopArt on TLM -- while ~5 faces per plan still explore
    # approximations. An INIT, not a mask: every logit stays trainable,
    # nothing about WHICH approximations are good is encoded.
    import os as _os
    _nb = float(_os.environ.get("ALPHAGRAD_FACE_NONE_BIAS", "0") or 0.0)
    _fpp = getattr(agent, "face_path_policy", None)
    if _nb != 0.0 and _fpp is not None \
            and getattr(_fpp, "head", None) is not None:
        import equinox as _eqx
        from alphagrad.approx.unified_face_head import (
            FACE_SLOTS as _FS, OP_NONE as _NONE, O_SKIP as _SKIP,
            S_OP as _SOP, slot_base as _sb)
        _bias = _fpp.head.proj.layers[-1].bias
        for _s in range(_FS):
            _bias = _bias.at[_sb(_s) + _SOP + _NONE].add(_nb)
        _bias = _bias.at[_SKIP].add(-_nb)
        agent = _eqx.tree_at(
            lambda a: a.face_path_policy.head.proj.layers[-1].bias,
            agent, _bias)
        print(f"[factory] face-head IDENTITY INIT: OP_NONE bias +{_nb}, "
              f"SKIP bias -{_nb} (trainable; P(approx/face) ~ "
              f"{3 * 2.718 ** (-_nb):.3f})", flush=True)
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
