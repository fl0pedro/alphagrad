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
    SHD_TIME_BINS,
    shd_sample,
    shd_split_size,
)

#: The registered targets that HAVE time steps, i.e. the ones on which a
#: gradient window and a temporal order constraint are defined. A flag that
#: names either raises on anything else rather than being quietly ignored.
TEMPORAL_TARGETS: frozenset[str] = frozenset({
    "LIF_SNN_SHD", "ADALIF_SNN_SHD", "ADALIF_SNN_SEQ",
})

#: The targets this module builds arguments for (the SHD pair).
SHD_TARGETS: frozenset[str] = frozenset({"LIF_SNN_SHD", "ADALIF_SNN_SHD"})

#: Full sequence length of both SHD targets, in 10 ms bins.
SHD_STEPS = SHD_TIME_BINS

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


def resolve_grad_window(example: str | None, grad_window, *,
                        flag: str = "--target-grad-window") -> int:
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
    if str(example) in SHD_TARGETS and n > SHD_STEPS:
        raise ValueError(
            f"{flag} {n} exceeds the SHD sequence length {SHD_STEPS} "
            f"({SHD_TIME_BINS} bins of 10 ms). Pass at most {SHD_STEPS}.")
    return n


def _spike_sequence(key, dataset: str | None, dataset_size: int | None):
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
        n = shd_split_size(dataset_size)
        if n == 0:
            raise ValueError(
                "the SHD subset is empty -- --dataset-size cut every sample")
        # ONE recording reaches the device. The split stays uint8 on the host:
        # the full train split as float32 is 2.28 GB, and the trainer plus its
        # measure actors would each hold a copy of it.
        idx = int(jax.random.randint(key, (), 0, n))
        seq, tgt = shd_sample(idx)
        return jnp.asarray(seq), jnp.asarray(tgt)
    k = jax.random.split(key, 2)
    seq = jax.random.bernoulli(
        k[0], 0.1, (SHD_TIME_BINS, SHD_CHANNELS)).astype(jnp.float32)
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
                 dataset_size: int | None = -1):
    """``LIF_SNN_SHD`` arguments: 700-128-20, ``T = 100`` bins, window ``N``.

    The first ``T - N`` steps run HERE with :func:`graphax.examples.lif_cb`
    and are stopped out of the gradient, so the traced graph is the base plus
    ``N`` per-step blocks and the carry entering it reflects the whole
    recording. Weights are args 8/9/10.
    """
    from graphax.examples.neuromorphic import lif_cb

    n = int(grad_window)
    key = jax.random.PRNGKey(1) if key is None else key
    k = jax.random.split(key, 2)
    full_seq, S_target = _spike_sequence(k[0], dataset, dataset_size)
    W1, W2, W3 = _weights(k[1])
    h, n_out = 128, SHD_CLASSES
    U1 = jnp.zeros((h,)); U2 = jnp.zeros((h,)); U3 = jnp.zeros((n_out,))
    I1 = jnp.zeros((h,)); I2 = jnp.zeros((h,)); I3 = jnp.zeros((n_out,))
    alpha = jnp.array(0.9); beta = jnp.array(0.8); thresh = jnp.array(0.3)
    T = int(full_seq.shape[0])
    for t in range(T - n):          # detached pre-window forward, activations only
        i1 = W1 @ full_seq[t]; U1, I1, s1 = lif_cb(U1, I1, i1, alpha, beta, thresh)
        i2 = W2 @ s1;          U2, I2, s2 = lif_cb(U2, I2, i2, alpha, beta, thresh)
        i3 = W3 @ s2;          U3, I3, s3 = lif_cb(U3, I3, i3, alpha, beta, thresh)
    sg = jax.lax.stop_gradient
    U1, U2, U3 = sg(U1), sg(U2), sg(U3)
    I1, I2, I3 = sg(I1), sg(I2), sg(I3)
    window = full_seq[T - n:]
    return (window, S_target, U1, U2, U3, I1, I2, I3,
            W1, W2, W3, alpha, beta, thresh)


def adalif_shd_args(grad_window: int, *, key=None, dataset: str | None = None,
                    dataset_size: int | None = -1):
    """``ADALIF_SNN_SHD`` arguments: the LIF twin with the ADAPTATION state.

    Same 700-128-20 shape, same ``T = 100`` bins, same gradient-window
    semantics, same weight slots 8/9/10. The carry the detached pre-window
    forward produces is ``(U*, a*)`` and one extra decay ``rho`` joins the
    tuple, because the cell is :func:`graphax.examples.ada_lif`.
    """
    from graphax.examples.neuromorphic import ada_lif

    n = int(grad_window)
    key = jax.random.PRNGKey(1) if key is None else key
    k = jax.random.split(key, 2)
    full_seq, S_target = _spike_sequence(k[0], dataset, dataset_size)
    W1, W2, W3 = _weights(k[1])
    h, n_out = 128, SHD_CLASSES
    U1 = jnp.zeros((h,)); U2 = jnp.zeros((h,)); U3 = jnp.zeros((n_out,))
    a1 = jnp.zeros((h,)); a2 = jnp.zeros((h,)); a3 = jnp.zeros((n_out,))
    alpha = jnp.array(0.9); beta = jnp.array(0.8)
    rho = jnp.array(0.95); thresh = jnp.array(0.3)
    T = int(full_seq.shape[0])
    for t in range(T - n):          # detached pre-window forward, activations only
        i1 = W1 @ full_seq[t]; U1, a1, s1 = ada_lif(U1, a1, i1, alpha, beta, rho, thresh)
        i2 = W2 @ s1;          U2, a2, s2 = ada_lif(U2, a2, i2, alpha, beta, rho, thresh)
        i3 = W3 @ s2;          U3, a3, s3 = ada_lif(U3, a3, i3, alpha, beta, rho, thresh)
    sg = jax.lax.stop_gradient
    U1, U2, U3 = sg(U1), sg(U2), sg(U3)
    a1, a2, a3 = sg(a1), sg(a2), sg(a3)
    window = full_seq[T - n:]
    return (window, S_target, U1, U2, U3, a1, a2, a3,
            W1, W2, W3, alpha, beta, rho, thresh)


def shd_args(example: str, grad_window: int, *, key=None,
             dataset: str | None = None, dataset_size: int | None = -1):
    """Dispatch on the target name. Raises on anything not an SHD target."""
    if example == "LIF_SNN_SHD":
        return lif_shd_args(grad_window, key=key, dataset=dataset,
                            dataset_size=dataset_size)
    if example == "ADALIF_SNN_SHD":
        return adalif_shd_args(grad_window, key=key, dataset=dataset,
                               dataset_size=dataset_size)
    raise ValueError(
        f"{example!r} is not an SHD target; expected one of "
        f"{sorted(SHD_TARGETS)}")
