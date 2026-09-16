"""The SHD-shaped temporal SNN targets and the GRADIENT WINDOW that sizes them.

ONE place builds the argument tuple of ``LIF_SNN_SHD`` and ``ADALIF_SNN_SHD``,
because the trainer and every measure actor must build the IDENTICAL graph: the
window length decides how many per-step blocks the elimination graph has, and
the detached pre-window forward decides the carry those blocks start from.

THE GRADIENT WINDOW (CONTEXT.md). ``N`` is the number of time steps whose
computation the gradient sees. The ``T - N`` steps before it run here, in
Python, detached, and reach the traced model only as the carry:

    N = 1    no rollout at all -- spatial credit only
    N = 2    one step back
    N = T    the full sequence (T = 100 bins for SHD)

so the elimination graph is a constant BASE plus ``N`` per-step blocks. It is
an ARGUMENT (``--target-grad-window``), never an environment variable:
``ALPHAGRAD_SNN_STEPS`` and ``ALPHAGRAD_SNN_TRUNC`` are refused by
:func:`refuse_legacy_snn_env`, which every entry point calls at import.

THE NAME ``--target-grad-window``, NOT ``--grad-window``. ``ppo.py`` has
carried a ``--grad-window`` since before this work and it means something
else entirely: how many consecutive ENCODER step deltas the PPO loss re-runs
before scoring an elimination step. The campaign passes ``--grad-window 0`` on
every arm. Two unrelated windows cannot share one flag name, so the target's
window says whose window it is. CONTEXT.md's glossary already draws the same
line ("Not the token Window of the encoder").
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

from alphagrad.approx.common.datasets import (
    SHD_CHANNELS,
    SHD_CLASSES,
    SHD_DEFAULT_BIN_MS,
    SHD_TIME_BINS,
    resolve_shd_bin_ms,
    shd_sample,
    shd_split_size,
    shd_time_bins,
)

#: The registered targets that HAVE time steps, i.e. the ones on which a
#: gradient window and a temporal order constraint are defined. A flag that
#: names either raises on anything else rather than being quietly ignored.
TEMPORAL_TARGETS: frozenset[str] = frozenset({
    "LIF_SNN_SHD", "ADALIF_SNN_SHD", "ADALIF_SNN_SEQ",
})

#: The targets this module builds arguments for (the SHD pair).
SHD_TARGETS: frozenset[str] = frozenset({"LIF_SNN_SHD", "ADALIF_SNN_SHD"})

#: Full sequence length of both SHD targets at the DEFAULT 10 ms bin width.
#: ``--shd-bin-ms`` moves it: T = 1000 / bin_ms, so 100 ms bins give T = 10.
SHD_STEPS = SHD_TIME_BINS

#: The two temporal rules ``--temporal-rule`` offers, and what each one does
#: with the state carried into the gradient window.
#:
#:   ``bptt``  BACKPROPAGATION THROUGH TIME. The carry enters as a CONSTANT.
#:             The gradient is the truncated one: only the steps inside the
#:             window contribute, and the credit that reaches the weights is
#:             purely spatial at window 1. The step copies are eliminated
#:             LATER FIRST, which is what backpropagation through time does.
#:   ``rtrl``  REAL-TIME RECURRENT LEARNING. The carried influence matrix
#:             ``J = d state / d W`` enters as a GIVEN EDGE from the weights to
#:             the carried state, valued by the detached prefix. Eliminating
#:             that vertex multiplies ``J`` by the step's state-to-state
#:             Jacobian, which is one RTRL step, and the gradient is EXACT
#:             through the whole prefix. The step copies are eliminated
#:             EARLIER FIRST, which is what real-time recurrent learning does.
TEMPORAL_RULES: tuple[str, ...] = ("bptt", "rtrl")

#: The ``--fixed-temporal-order`` each rule forces.
TEMPORAL_RULE_ORDER: dict[str, str] = {"bptt": "reverse", "rtrl": "forward"}

#: The env vars this flag replaced, and the message each one dies with.
_REFUSED_ENV = {
    "ALPHAGRAD_SNN_STEPS": (
        "it set the unrolled step count of ADALIF_SNN_SEQ"),
    "ALPHAGRAD_SNN_TRUNC": (
        "it set the truncation window of LIF_SNN_SHD"),
}


def refuse_legacy_snn_env() -> None:
    """RAISE if a process still exports one of the two replaced variables.

    Owner ruling: flags replace environment variables. A variable that is
    merely ignored is worse than one that is read -- a launcher carrying
    ``ALPHAGRAD_SNN_TRUNC=1`` would silently run the FULL sequence and its log
    would say nothing -- so the name fails loudly and says which flag to pass.
    """
    for name, what in _REFUSED_ENV.items():
        if name in os.environ:
            raise RuntimeError(
                f"{name} is no longer read: {what}. The gradient window is an "
                f"ARGUMENT now -- pass --target-grad-window N to the trainer "
                f"(1 = no rollout, spatial credit only; 2 = one step back; "
                f"{SHD_STEPS} = the full SHD sequence) and unset {name}.")


def has_time_steps(example: str | None) -> bool:
    """Does ``example`` carry a time axis the gradient window can size?"""
    return bool(example) and str(example) in TEMPORAL_TARGETS


def resolve_bin_ms(example: str | None, bin_ms, *,
                   flag: str = "--shd-bin-ms") -> int:
    """The SHD bin width for ``example``, or raise.

    A bin width is a property of the SHD RECORDINGS, so it is defined on the
    two SHD targets only. Passing it anywhere else would size nothing, and a
    flag that does nothing is how a launcher comes to claim a bin width it
    never ran.
    """
    if example is not None and str(example) not in SHD_TARGETS:
        if bin_ms is not None:
            raise ValueError(
                f"{flag} {bin_ms} was passed with --example {example}, which "
                f"reads no SHD recording. The bin width is defined only on "
                f"{sorted(SHD_TARGETS)}; drop the flag or change the target.")
        return SHD_DEFAULT_BIN_MS
    return resolve_shd_bin_ms(bin_ms)


def steps_for(example: str | None, bin_ms=None) -> int:
    """``T``: how many time steps ``example`` has at that bin width."""
    if example is not None and str(example) in SHD_TARGETS:
        return shd_time_bins(resolve_bin_ms(example, bin_ms))
    return SHD_STEPS


def resolve_temporal_rule(example: str | None, rule, *,
                          flag: str = "--temporal-rule") -> str | None:
    """The temporal rule for ``example``, or raise.

    ``None`` means "not asked for" and keeps the graph this branch has always
    built -- the truncated one, with the carry entering as a constant and no
    order forced. A VALUE on a target without time steps is a hard error: a
    rule that names how the carry enters has nothing to enter on a target with
    one step copy.
    """
    if rule is None:
        return None
    r = str(rule)
    if r not in TEMPORAL_RULES:
        raise ValueError(
            f"{flag} {r!r} is not one of {list(TEMPORAL_RULES)}")
    if not has_time_steps(example):
        raise ValueError(
            f"{flag} {r} was passed with --example {example}, which has NO "
            f"time steps. A temporal rule says how the state carried BETWEEN "
            f"steps enters the gradient; this target carries none. Drop the "
            f"flag or change the target.")
    if r == "rtrl" and str(example) not in SHD_TARGETS:
        raise ValueError(
            f"{flag} rtrl needs a target with a DETACHED PREFIX to take the "
            f"carried Jacobian from, and --example {example} has none: "
            f"ADALIF_SNN_SEQ unrolls its whole sequence, so its gradient is "
            f"already exact and there is nothing to carry. Use "
            f"{sorted(SHD_TARGETS)}.")
    return r


def resolve_fixed_temporal_order(rule: str | None, fixed: str, *,
                                 flag: str = "--fixed-temporal-order") -> str:
    """The temporal order ``rule`` forces, or raise on a contradiction.

    A rule IS an order across the step copies -- backpropagation through time
    eliminates the later copy first, real-time recurrent learning the earlier
    one -- so the two flags cannot disagree. ``free`` is the unset default and
    is FILLED IN with the matching direction; the opposite direction RAISES
    rather than letting one flag quietly win over the other.
    """
    if rule is None:
        return fixed
    want = TEMPORAL_RULE_ORDER[rule]
    if fixed in ("free", want):
        return want
    raise ValueError(
        f"--temporal-rule {rule} and {flag} {fixed} contradict each other. "
        f"{rule} IS an order across the step copies: it eliminates the "
        f"{'later' if want == 'reverse' else 'earlier'} copy first, which is "
        f"{flag} {want}. Pass {flag} {want}, or drop it and let the rule set "
        f"it, or change the rule.")


def resolve_grad_window(example: str | None, grad_window, *,
                        flag: str = "--target-grad-window",
                        bin_ms=None) -> int:
    """The window ``N`` for ``example``, or raise.

    ``None`` means "not asked for" and resolves to 1 on a temporal target --
    the smallest graph, and the one a run gets when it says nothing. A VALUE on
    a target without time steps is a hard error: it would size nothing, and a
    flag that does nothing is how a launcher comes to claim a window it never
    ran.
    """
    if not has_time_steps(example):
        if grad_window is not None:
            raise ValueError(
                f"{flag} {grad_window} was passed with --example {example}, "
                f"which has NO time steps. The gradient window is defined only "
                f"on {sorted(TEMPORAL_TARGETS)}; drop the flag or change the "
                f"target.")
        return 0
    n = 1 if grad_window is None else int(grad_window)
    if n < 1:
        raise ValueError(f"{flag} must be >= 1, got {n}")
    if str(example) in SHD_TARGETS:
        b = resolve_bin_ms(example, bin_ms)
        t = shd_time_bins(b)
        if n > t:
            raise ValueError(
                f"{flag} {n} exceeds the SHD sequence length {t} "
                f"({t} bins of {b} ms). Pass at most {t}.")
    return n


#: The six ``(state, weight)`` blocks that are STRUCTURALLY zero on a
#: three-layer feed-forward-in-space SNN: ``W2`` and ``W3`` cannot reach layer
#: 1 and ``W3`` cannot reach layer 2. They are not carried, and a probe asserts
#: they really are zero rather than assuming it.
_ZERO_CARRY_BLOCKS: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 2), (3, 1), (3, 2), (4, 2),
)


def _prefix_runner(cell, seq, n_pre: int, params, h: int, n_out: int):
    """A function of ``(W1, W2, W3)`` returning the SIX carried states.

    The states come back in the order the target's signature lists them --
    ``U1, U2, U3`` then the second state of each layer (``a1, a2, a3`` for
    adaptive LIF, ``I1, I2, I3`` for current-based LIF) -- because
    :data:`graphax.examples.neuromorphic.SHD_CARRY_BLOCKS` indexes that order.

    It is the SAME arithmetic the detached pre-window forward runs, written
    once so the carried Jacobian and the carried value can never come from two
    different loops.
    """
    def run(W1, W2, W3):
        U1 = jnp.zeros((h,)); U2 = jnp.zeros((h,)); U3 = jnp.zeros((n_out,))
        y1 = jnp.zeros((h,)); y2 = jnp.zeros((h,)); y3 = jnp.zeros((n_out,))
        for t in range(int(n_pre)):
            i1 = W1 @ seq[t]; U1, y1, s1 = cell(U1, y1, i1, *params)
            i2 = W2 @ s1;     U2, y2, s2 = cell(U2, y2, i2, *params)
            i3 = W3 @ s2;     U3, y3, s3 = cell(U3, y3, i3, *params)
        return U1, U2, U3, y1, y2, y3
    return run


def carried_jacobians(cell, seq, n_pre: int, params, weights, h: int,
                      n_out: int, *, check_zeros: bool = True):
    """The RTRL attachment tuple: three reference weights, then TWELVE blocks.

    Block ``(s, w)`` is ``d state_s / d W_w`` at the step the gradient window
    starts, taken by reverse-mode differentiation of the detached prefix. This
    is what real-time recurrent learning carries: ``G_t = H_t G_{t-1} + F_t``
    accumulated over every step before the window, which is the same number
    reverse mode gets in one sweep and is far cheaper to compute here than to
    recur.

    The three reference weights lead the tuple: they are bit-for-bit copies of
    the weights and sit outside ``argnums``, which is what lets the attachment
    build an exactly zero weight delta without an edge-free vertex. See
    :func:`graphax.examples.neuromorphic.attach_carried_jacobians`.

    ``check_zeros`` asserts the six blocks that CANNOT be non-zero -- the
    upper-triangular ones, where a later layer's weights would have to reach an
    earlier layer's state -- really are zero, so the twelve carried blocks are
    the whole influence matrix and not a silent truncation of it.
    """
    from graphax.examples.neuromorphic import SHD_CARRY_BLOCKS

    run = _prefix_runner(cell, seq, n_pre, params, h, n_out)
    jac = jax.jacrev(run, argnums=(0, 1, 2))(*weights)
    if check_zeros:
        for s_i, w_i in _ZERO_CARRY_BLOCKS:
            nz = float(jnp.max(jnp.abs(jac[s_i][w_i])))
            if nz != 0.0:
                raise ValueError(
                    f"carried block (state {s_i}, weight {w_i}) is not zero "
                    f"(max |.| = {nz}). The three layers are feed-forward in "
                    f"space, so W{w_i + 1} cannot reach that state; a "
                    f"non-zero here means the model changed and "
                    f"SHD_CARRY_BLOCKS no longer lists every block.")
    # THE THREE REFERENCE WEIGHTS COME FIRST. They are bit-for-bit copies of
    # the weights, live outside ``argnums``, and are what makes the attached
    # delta exactly zero without adding an edge-free stop_gradient vertex.
    sg = jax.lax.stop_gradient
    return (tuple(sg(W) for W in weights)
            + tuple(sg(jac[s_i][w_i]) for s_i, w_i in SHD_CARRY_BLOCKS))


def _spike_sequence(key, dataset: str | None, dataset_size: int | None,
                    bin_ms: int | None = None):
    """``(seq [T, 700] float32, target [20] float32)`` for ONE recording.

    ``dataset == "shd"`` draws a real Spiking Heidelberg Digits recording from
    the ``--dataset-size`` subset. Anything else keeps the SYNTHETIC Poisson
    train LIF_SNN_SHD shipped with, so a run that asks for no dataset still
    builds the same SHAPE and every archived LIF_SNN_SHD result reproduces.
    """
    if dataset is not None and dataset not in ("shd", "none"):
        raise ValueError(
            f"--dataset {dataset} cannot feed an SHD target: a spike window is "
            f"(T, {SHD_CHANNELS}) and nothing in {dataset} has that shape. Use "
            f"--dataset shd for the real recordings, or --dataset none for the "
            f"synthetic Poisson train.")
    if dataset == "shd":
        n = shd_split_size(dataset_size, bin_ms=bin_ms)
        if n == 0:
            raise ValueError(
                "the SHD subset is empty -- --dataset-size cut every sample")
        # ONE recording reaches the device. The split stays uint8 on the host:
        # the full train split as float32 is 2.28 GB, and the trainer plus its
        # measure actors would each hold a copy of it.
        idx = int(jax.random.randint(key, (), 0, n))
        seq, tgt = shd_sample(idx, bin_ms=bin_ms)
        return jnp.asarray(seq), jnp.asarray(tgt)
    k = jax.random.split(key, 2)
    seq = jax.random.bernoulli(
        k[0], 0.1, (shd_time_bins(resolve_shd_bin_ms(bin_ms)), SHD_CHANNELS)
    ).astype(jnp.float32)
    tgt = jax.nn.one_hot(
        jax.random.randint(k[1], (), 0, SHD_CLASSES), SHD_CLASSES
    ).astype(jnp.float32)
    return seq, tgt


def _weights(key):
    """W1 (128, 700), W2 (128, 128), W3 (20, 128) at the shipped 6/sqrt(fan-in)
    scale, and the three scalar decays plus the threshold."""
    h, n_out = 128, SHD_CLASSES
    k = jax.random.split(key, 3)
    W1 = jax.random.normal(k[0], (h, SHD_CHANNELS)) * (6.0 / (SHD_CHANNELS ** 0.5))
    W2 = jax.random.normal(k[1], (h, h)) * (6.0 / (h ** 0.5))
    W3 = jax.random.normal(k[2], (n_out, h)) * (6.0 / (h ** 0.5))
    return W1, W2, W3


def lif_shd_args(grad_window: int, *, key=None, dataset: str | None = None,
                 dataset_size: int | None = -1, bin_ms: int | None = None,
                 temporal_rule: str | None = None):
    """``LIF_SNN_SHD`` arguments: 700-128-20, ``T`` bins, window ``N``.

    The first ``T - N`` steps run HERE with :func:`graphax.examples.lif_cb`
    and are stopped out of the gradient, so the traced graph is the base plus
    ``N`` per-step blocks and the carry entering it reflects the whole
    recording. Weights are args 8/9/10.

    ``temporal_rule == "rtrl"`` APPENDS the twelve carried Jacobian blocks, so
    the graph also holds the given edge from the weights to the carried state
    and the gradient is exact through the whole prefix. ``bptt`` and ``None``
    append nothing and the tuple is the one this branch always built.
    """
    from graphax.examples.neuromorphic import lif_cb

    n = int(grad_window)
    key = jax.random.PRNGKey(1) if key is None else key
    k = jax.random.split(key, 2)
    full_seq, S_target = _spike_sequence(k[0], dataset, dataset_size, bin_ms)
    W1, W2, W3 = _weights(k[1])
    h, n_out = 128, SHD_CLASSES
    alpha = jnp.array(0.9); beta = jnp.array(0.8); thresh = jnp.array(0.3)
    params = (alpha, beta, thresh)
    T = int(full_seq.shape[0])
    run = _prefix_runner(lif_cb, full_seq, T - n, params, h, n_out)
    sg = jax.lax.stop_gradient
    U1, U2, U3, I1, I2, I3 = (sg(x) for x in run(W1, W2, W3))
    window = full_seq[T - n:]
    head = (window, S_target, U1, U2, U3, I1, I2, I3,
            W1, W2, W3, alpha, beta, thresh)
    if temporal_rule == "rtrl":
        return head + carried_jacobians(lif_cb, full_seq, T - n, params,
                                        (W1, W2, W3), h, n_out)
    return head


def adalif_shd_args(grad_window: int, *, key=None, dataset: str | None = None,
                    dataset_size: int | None = -1, bin_ms: int | None = None,
                    temporal_rule: str | None = None):
    """``ADALIF_SNN_SHD`` arguments: the LIF twin with the ADAPTATION state.

    Same 700-128-20 shape, same ``T = 1000 / bin_ms`` bins, same
    gradient-window semantics, same weight slots 8/9/10. The carry the detached
    pre-window forward produces is ``(U*, a*)`` and one extra decay ``rho``
    joins the tuple, because the cell is :func:`graphax.examples.ada_lif`.

    ``temporal_rule == "rtrl"`` appends the twelve carried Jacobian blocks;
    see :func:`lif_shd_args`.
    """
    from graphax.examples.neuromorphic import ada_lif

    n = int(grad_window)
    key = jax.random.PRNGKey(1) if key is None else key
    k = jax.random.split(key, 2)
    full_seq, S_target = _spike_sequence(k[0], dataset, dataset_size, bin_ms)
    W1, W2, W3 = _weights(k[1])
    h, n_out = 128, SHD_CLASSES
    alpha = jnp.array(0.9); beta = jnp.array(0.8)
    rho = jnp.array(0.95); thresh = jnp.array(0.3)
    params = (alpha, beta, rho, thresh)
    T = int(full_seq.shape[0])
    run = _prefix_runner(ada_lif, full_seq, T - n, params, h, n_out)
    sg = jax.lax.stop_gradient
    U1, U2, U3, a1, a2, a3 = (sg(x) for x in run(W1, W2, W3))
    window = full_seq[T - n:]
    head = (window, S_target, U1, U2, U3, a1, a2, a3,
            W1, W2, W3, alpha, beta, rho, thresh)
    if temporal_rule == "rtrl":
        return head + carried_jacobians(ada_lif, full_seq, T - n, params,
                                        (W1, W2, W3), h, n_out)
    return head


def shd_args(example: str, grad_window: int, *, key=None,
             dataset: str | None = None, dataset_size: int | None = -1,
             bin_ms: int | None = None, temporal_rule: str | None = None):
    """Dispatch on the target name. Raises on anything not an SHD target."""
    if example == "LIF_SNN_SHD":
        return lif_shd_args(grad_window, key=key, dataset=dataset,
                            dataset_size=dataset_size, bin_ms=bin_ms,
                            temporal_rule=temporal_rule)
    if example == "ADALIF_SNN_SHD":
        return adalif_shd_args(grad_window, key=key, dataset=dataset,
                               dataset_size=dataset_size, bin_ms=bin_ms,
                               temporal_rule=temporal_rule)
    raise ValueError(
        f"{example!r} is not an SHD target; expected one of "
        f"{sorted(SHD_TARGETS)}")
