#!/usr/bin/env python3
"""Replace the JACOBIAN cosine with the GRADIENT cosine in env.py's slot 6.

WHAT CHANGES

  ALPHAGRAD_QUALITY_METRIC / --quality-metric gains two explicit names:

    grad_cosine   the gradient cosine at INIT weights, averaged over K probe
                  batches of real data (K = ALPHAGRAD_GRAD_COSINE_K, default 1).
    jac_cosine    the legacy Jacobian cosine at the calibration eval samples.

  and `cosine` becomes a DEPRECATED ALIAS that resolves to `grad_cosine` for
  any scalar-loss target and to `jac_cosine` for the analytic AD benchmarks
  (where no loss and no data generator exist, so a gradient cosine is not
  defined).  It warns once, loudly, naming the change.

WHY AN ALIAS AND NOT A RENAME.  `cosine` is named by hundreds of archived
launchers and by every recorded replay command line; making it an error would
break re-running them, and silently leaving it as the Jacobian cosine would
mean the owner's requested swap never reaches the runs that ask for "cosine".
Resolving it to the gradient formulation with a one-shot warning does the swap
where it is wanted and still names the legacy behaviour reachable.

WHAT DOES NOT CHANGE.  `auto` still resolves to `loss_drop` for every
trainable (scalar-loss) target -- loss_drop remains the default quality
channel.  This patch changes what the COSINE family means, which is what the
owner asked for; it does not demote loss_drop.

SINGLE-EXACT-REFERENCE.  grad_cosine scores off ONE exact execution per probe
batch and, at the default K=1, that is one exact execution per terminal plan --
the same count the Jacobian cosine paid.  `_fid_from_quality` stays true for
both cosine variants so the A2 fidelity block never asks for a second exact
reference.

Idempotent.
"""
from __future__ import annotations

import sys

PATH = "src/alphagrad/approx/env.py"


STALE_TABLE = '''#   Jacobian cosine            0.610    0.858     9.70 s     4.24 GB
#   gradient cosine at init    0.737    0.857     0.33 s     ~40 MB
#   loss drop, 200 Adam steps  0.922    0.854     0.22 s     40 MB
'''

REMEASURED = '''#
# THE THREE COST NUMBERS ABOVE ARE STALE. They predate 744fc3d, which made the
# traced target an unconditional SCALAR LOSS: what used to be a per-class
# Jacobian is now the gradient, so nothing materialises a Jacobian any more and
# the cosine got ~3000x cheaper. RE-MEASURED 2026-08-28 on one Blackwell GPU,
# VmappedNeuralNetwork/MNIST, 31 plans spanning cos 0.00 (frozen gradient, the
# k24/f0 analogue) to 1.00, ground truth = final downstream test accuracy at
# 20k steps x 3 seeds through downstream_train.py -- the SAME definition the
# table above used, over a NEWLY GENERATED plan population, because the 139
# archived plans no longer replay (they were recorded against a graph whose
# edges were 4-D; apply_compress rejects them now):
#
#                                    Pearson  Spearman  s/plan   peak MB
#   gradient cosine at init, K=1      0.874     0.805    0.003     295
#   gradient cosine at init, K=4      0.861     0.800    0.012     295
#   gradient cosine at init, K=8      0.856     0.807    0.024     295
#   mean cosine over 200 steps        0.815     0.797    1.191     295
#   last-step cosine, 200 steps       0.761     0.800    1.191     295
#   cosine of summed gradients        0.685     0.748    1.191     295
#   clipped relative Frobenius        0.350     0.413    0.003     295
#   loss drop, 200 Adam steps         0.885     0.765    0.606     295
#   Jacobian cosine (legacy)          0.397     0.498    0.003     443
#
# (peak MB is process peak; ~295 MB of it is the resident model+data baseline,
# so the Jacobian cosine's MARGINAL cost is the ~148 MB of Jacobian it builds
# and every gradient-space metric's marginal cost is ~0.)
#
# WHAT THIS SETTLED, and it answers the owner's 2026-08-07 question directly:
#   * the gradient cosine is best AT INIT and K=1 -- averaging over more probe
#     batches makes it slightly WORSE (0.874 -> 0.856), so K=1 is both the most
#     predictive and the cheapest, and it keeps the channel at EXACTLY ONE
#     exact execution per terminal plan;
#   * NONE of the trajectory formulations pay: aggregated-gradient cosine
#     0.685, last-step 0.761, mean-over-steps 0.815 -- all below the K=1 init
#     cosine and ~400x more expensive;
#   * the legacy JACOBIAN cosine is the worst cosine by a wide margin
#     (0.397/0.498). Replacing it with the gradient cosine is a 2.2x gain in
#     Pearson at IDENTICAL wall cost and 33% less peak memory, which is why
#     "cosine" now resolves to grad_cosine;
#   * CLIPPED RELATIVE FROBENIUS -- the A2 channel whose correlation had never
#     been measured -- is WEAK: 0.350 Pearson / 0.413 Spearman, worse than
#     every cosine variant including the legacy one. It should stay LOGGED and
#     should NOT be given a trained slot on this evidence.
#   * loss_drop remains the `auto` default: its Pearson 0.885 is within noise
#     of the K=1 gradient cosine's 0.874, but note it is WORSE on Spearman
#     (0.765 vs 0.805) while costing 200x more. If the default is ever
#     revisited, the gradient cosine is the cheaper equal -- that call is the
#     owner's and is NOT made here.
'''


# --------------------------------------------------------------------------
# 1. quality_metric(): the new names + the deprecated alias
# --------------------------------------------------------------------------
OLD_DISPATCH = '''    if want in ("cosine", "cos", "cosine_sim"):
        return "cosine"
'''

NEW_DISPATCH = '''    if want in ("grad_cosine", "gradcos", "grad_cos"):
        return "grad_cosine"
    if want in ("jac_cosine", "jacobian_cosine", "jaccos"):
        return "jac_cosine"
    # DEPRECATED NAME (2026-08-28). "cosine" used to mean the JACOBIAN cosine.
    # Post-744fc3d the traced target of every trainable example IS a scalar
    # loss, so the leaves jacve returns are already gradient-shaped and the
    # "Jacobian cosine" label had stopped being true; and the owner's own
    # 2026-08-07 sweep measured the GRADIENT cosine as the better predictor of
    # downstream accuracy (0.737 vs 0.610 Pearson). So "cosine" now resolves to
    # the gradient formulation wherever one is defined, and to the legacy
    # Jacobian cosine only for the analytic AD benchmarks, which have neither a
    # loss nor a data generator. Say so once, loudly -- this MOVES the number
    # every "cosine" run reports.
    if want in ("cosine", "cos", "cosine_sim"):
        _warn_cosine_is_now_grad_cosine()
        return ("grad_cosine"
                if bool(getattr(config, "scalar_target", False))
                else "jac_cosine")
'''

OLD_ERR = '''        raise ValueError(
            f"{_QUALITY_METRIC_ENV} must be one of "
            f"auto/loss_drop/cosine/none, got {want!r}"
        )
    return "loss_drop" if bool(getattr(config, "scalar_target", False)) else "cosine"
'''

NEW_ERR = '''        raise ValueError(
            f"{_QUALITY_METRIC_ENV} must be one of "
            f"auto/loss_drop/grad_cosine/jac_cosine/cosine/none, got {want!r}"
        )
    return ("loss_drop" if bool(getattr(config, "scalar_target", False))
            else "jac_cosine")
'''

WARN_FN = '''
_COSINE_RENAME_WARNED: list[int] = []


def _warn_cosine_is_now_grad_cosine() -> None:
    """One-shot, loud: ``cosine`` no longer means the Jacobian cosine."""
    if _COSINE_RENAME_WARNED:
        return
    _COSINE_RENAME_WARNED.append(1)
    print(
        "[measure] NOTE ALPHAGRAD_QUALITY_METRIC=cosine is DEPRECATED and its "
        "MEANING HAS CHANGED: it now resolves to 'grad_cosine' (the gradient "
        "cosine at init over ALPHAGRAD_GRAD_COSINE_K probe batches) for any "
        "scalar-loss target, and to 'jac_cosine' (the legacy Jacobian cosine "
        "at the calibration samples) only for the analytic AD benchmarks. "
        "Ask for 'jac_cosine' explicitly to reproduce a pre-2026-08-28 run.",
        flush=True,
    )


def _grad_cosine_k() -> int:
    """How many probe batches the gradient cosine averages over.

    K=1 is the default because it is the value that keeps the channel at
    EXACTLY ONE exact execution per terminal plan -- the same reference count
    the Jacobian cosine paid -- and because the bake-off found K>1 buys
    essentially no extra correlation on this target.
    """
    try:
        return max(1, int(os.environ.get("ALPHAGRAD_GRAD_COSINE_K", "1")))
    except ValueError:
        return 1
'''


# --------------------------------------------------------------------------
# 2. the scorer
# --------------------------------------------------------------------------
SCORER = '''
def _grad_cosine_quality(config, compiled_approx, compiled_exact, base_args,
                         device, k_batches: int = 1):
    """THE GRADIENT COSINE: cos(g_approx, g_exact) at the INITIAL weights,
    averaged over ``k_batches`` fixed probe batches of REAL data.

    Returns ``(quality, rel_frobs, cosines)`` or ``None`` when undefined (no
    data generator -- the same fall-back signal ``_loss_drop_quality`` uses).

    This differs from the legacy Jacobian cosine in WHERE it is evaluated, not
    only in WHAT is compared: the legacy channel scored at the calibration eval
    samples, which are synthetic argument draws, while this scores on the same
    real-data probe batches the loss-drop walk uses.  On a scalar-loss target
    the compared leaves are gradient-shaped either way.
    """
    if compiled_exact is None:
        return None
    cos_all: list[float] = []
    frob_all: list[float] = []
    for k in range(max(1, int(k_batches))):
        data = _probe_batch(config, base_args, role="train", index=k)
        if data is None:
            return None
        a = list(base_args)
        for slot in range(min(2, len(data))):
            a[slot] = jax.device_put(jnp.asarray(data[slot]), device)
        try:
            out_a = compiled_approx(*a)
            out_e = compiled_exact(*a)
        except Exception:
            return None
        jac_a = out_a[1] if config.has_aux else out_a
        jac_e = out_e[1] if config.has_aux else out_e
        cos, rel = _quality_metrics(jac_e, jac_a)
        cos_all.append(float(cos))
        frob_all.append(float(rel))
        out_a = out_e = jac_a = jac_e = None
    if not cos_all:
        return None
    return float(np.mean(cos_all)), frob_all, cos_all

'''


# --------------------------------------------------------------------------
# 3. call-site rewiring
# --------------------------------------------------------------------------
EDITS = [
    # _probe_batch gains an index so K distinct batches are drawable.
    (
        'def _probe_batch(config, base_args, role: str = "train",\n'
        '                 episode: int | None = None):',
        'def _probe_batch(config, base_args, role: str = "train",\n'
        '                 episode: int | None = None, index: int = 0):',
    ),
    (
        "    _seed = _walk_seed(role, episode)\n"
        "    _key = (id(config.data_gen), _seed,",
        "    # ``index`` draws a DIFFERENT batch for each K of the gradient\n"
        "    # cosine. index=0 is bit-identical to the pre-index behaviour, so\n"
        "    # the loss-drop walk's batch does not move.\n"
        "    _seed = _walk_seed(role, episode) + 104729 * int(index)\n"
        "    _key = (id(config.data_gen), _seed,",
    ),
    # Site A: compile the exact executable for EITHER cosine variant.
    (
        '    _qmetric = quality_metric(config)\n'
        '    if is_terminal and _qmetric == "cosine":',
        '    _qmetric = quality_metric(config)\n'
        '    if is_terminal and _qmetric in ("jac_cosine", "grad_cosine"):',
    ),
    # Fidelity: both cosine variants produce the residual themselves, so the
    # A2 block must NOT ask for its own exact reference in either case.
    (
        '    _fid_from_quality = (_qmetric == "cosine")',
        '    _fid_from_quality = _qmetric in ("jac_cosine", "grad_cosine")',
    ),
    # The PER-POINT Jacobian cosine loop is now jac_cosine ONLY.
    (
        "            if compiled_exact is not None:\n"
        "                _ex_key = _eval_digest(eval_args_i) if _CACHE_EXACT else None",
        '            if compiled_exact is not None and _qmetric == "jac_cosine":\n'
        "                _ex_key = _eval_digest(eval_args_i) if _CACHE_EXACT else None",
    ),
]

# The new scoring branch, inserted next to the loss-drop one.
OLD_BRANCH = '''        _pf("cb.exec_measure")
        if is_terminal and _qmetric == "loss_drop":'''

NEW_BRANCH = '''        _pf("cb.exec_measure")
        # ---- GRADIENT COSINE --------------------------------------------
        # ONE scoring per PLAN, like the walk: the probe batches are fixed
        # across plans, so repeating over the calibration samples would
        # re-measure the same number. Runs after the cost loop so the
        # timing/peak windows never contain it.
        if is_terminal and _qmetric == "grad_cosine":
            _gc = _grad_cosine_quality(
                config, compiled_approx, compiled_exact, list(args),
                callback_device, _grad_cosine_k())
            if _gc is None:
                if not _WALK_UNDEFINED_WARNED:
                    _WALK_UNDEFINED_WARNED.append(1)
                    print(
                        "[measure] WARNING quality channel: the GRADIENT "
                        "COSINE is UNDEFINED for this configuration (no "
                        "data_gen, or the exact reference failed to build). "
                        "The channel reads 0.0 for every affected plan; ask "
                        "for ALPHAGRAD_QUALITY_METRIC=jac_cosine to score at "
                        "the calibration samples instead.", flush=True)
                cosines.append(0.0)
            else:
                _gc_q, _gc_frobs, _gc_cos = _gc
                cosines.append(_gc_q)
                if _fid_on:
                    _rel_frobs.extend(float(x) for x in _gc_frobs)
                    _cos_logged.extend(float(x) for x in _gc_cos)
        if is_terminal and _qmetric == "loss_drop":'''


def main():
    with open(PATH) as fh:
        src = fh.read()

    if "_grad_cosine_quality" in src:
        print("already patched")
        return 0

    def sub(old, new, what):
        nonlocal src
        if old not in src:
            print(f"ANCHOR NOT FOUND: {what}", file=sys.stderr)
            print(repr(old[:160]), file=sys.stderr)
            raise SystemExit(2)
        if src.count(old) != 1:
            print(f"ANCHOR NOT UNIQUE ({src.count(old)}x): {what}",
                  file=sys.stderr)
            raise SystemExit(2)
        src = src.replace(old, new, 1)

    sub(STALE_TABLE, STALE_TABLE + REMEASURED, "re-measured cost table")
    sub(OLD_DISPATCH, NEW_DISPATCH, "quality_metric dispatch")
    sub(OLD_ERR, NEW_ERR, "quality_metric error + auto fallback")
    sub("_QUALITY_METRIC_ENV = \"ALPHAGRAD_QUALITY_METRIC\"",
        "_QUALITY_METRIC_ENV = \"ALPHAGRAD_QUALITY_METRIC\"\n" + WARN_FN,
        "warn helper + K reader")
    for old, new in EDITS:
        sub(old, new, old.splitlines()[0][:70])
    sub(OLD_BRANCH, NEW_BRANCH, "grad_cosine scoring branch")
    # the scorer itself goes immediately before _loss_drop_quality
    sub("def _loss_drop_quality(", SCORER + "def _loss_drop_quality(",
        "scorer insertion")

    with open(PATH, "w") as fh:
        fh.write(src)
    print(f"patched {PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
