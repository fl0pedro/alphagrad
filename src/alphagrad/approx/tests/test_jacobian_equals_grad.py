"""THE JACOBIAN IS THE GRADIENT, for EVERY example in the registry.

The owner's requirement, verbatim in substance: *"remove --measure-grad and
automatically just have this be the default. I.e. have the correct Jacobian
=== grad of the scalar loss for each model we use."*

WHAT MAKES THAT TRUE HERE. The registered target IS model + loss:
``examples.get_fn(name)`` returns a callable whose output is a 0-d scalar, so
``jacve`` of the traced graph is the gradient a training run takes. Nothing
wraps it afterwards -- ``grad_target_setup`` is the identity outside the
deprecated ``--seed-vertices`` replay branch -- and no flag selects anything
else. ``--measure-grad`` is an accepted no-op; ``--no-measure-grad``, which
asked for the per-class Jacobian, is a hard error in ``tools/landscape_map``.

THREE THINGS THIS FILE PINS

  1. ``jacve(target) == jax.grad(target)`` leaf by leaf, for every family in
     the registry that has a training loss.
  2. Every such target is 0-d, and the ANALYTIC AD benchmarks (which have no
     training run) are EXEMPT by name rather than by accident -- they are the
     one place where "make them all the same" would have meant inventing an
     objective, so they keep their full-Jacobian target and say so.
  3. ``--measure-grad`` does not move the traced graph, on or off.

TOLERANCE: ``max|jacve - grad| / max|grad| <= 1e-5``, PER LEAF, in float32,
plus the graph-wide relative Frobenius residual ``||j-g|| / ||g|| <= 2e-6``.
Two families need a documented per-leaf relaxation because one of their leaves
has a near-zero gradient, which makes the per-leaf ratio's denominator tiny
while the graph-wide agreement stays at 5e-07 or better -- see
``_LEAF_RTOL``. Both were
already at those values before this workstream; the relaxation records a
float32 accumulation-order fact, not a slackened claim.

MEASURED, float32, CPU, on the args ``get_args`` builds from ``PRNGKey(0)``
(2026-08-28). Per-leaf worst / graph-wide relative Frobenius -- run this file
with ``-s`` and every parametrisation prints its own pair:

    family                  per-leaf   rel-Frobenius
    NeuralNetwork            1.3e-07   1.1e-07
    VmappedNeuralNetwork     1.1e-07   7.1e-08
    TransformerLM            1.1e-06   2.9e-07
    TransformerLM3           1.2e-06   2.7e-07
    Encoder                  1.7e-05   4.9e-07   (relaxed, see _LEAF_RTOL)
    VmappedEncoder           6.2e-07   3.2e-07
    EncoderDecoder           8.2e-05   1.5e-07   (relaxed, see _LEAF_RTOL)
    VmappedEncoderDecoder    5.3e-06   1.9e-07
    ConvNet                  8.1e-07   3.4e-08
    VmappedConvNet           2.5e-06   2.9e-07
    MoE                      2.8e-07   9.8e-08
    VmappedMoE               5.1e-07   2.2e-07
    ViT                      2.9e-07   1.5e-07
    VmappedViT               2.6e-07   1.3e-07
    LIF_SNN                  0.0       0.0
    ADALIF_SNN               0.0       0.0
    ADALIF_SNN_SEQ           0.0       0.0
    LIF_SNN_SHD              5.6e-07   4.1e-08

The seven ANALYTIC targets are not in the table and cannot be: they have no
training loss, so "the gradient" is not defined for them. They are pinned by
test_analytic_targets_are_exempt_from_the_scalar_contract instead.
Perceptron / VmappedPerceptron are unbuildable for a pre-existing reason
recorded in UNTESTABLE below.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
# TLM-SHAPED, not TLM-SIZED: the property is about the graph, and the small
# shapes keep this a unit test. Set before the module-level _tlm_dims() read.
os.environ.setdefault("ALPHAGRAD_TLM_SEQ", "8")
os.environ.setdefault("ALPHAGRAD_TLM_DMODEL", "8")
os.environ.setdefault("ALPHAGRAD_TLM_VOCAB", "16")
# LIF_SNN_SHD unrolls T=100 steps (5603 eqns) unless truncated. The property
# is per-step-shape-independent; 1 step keeps this a unit test. Read at import
# of examples, so it must be set first.
os.environ.setdefault("ALPHAGRAD_SNN_TRUNC", "1")

from types import SimpleNamespace as NS                           # noqa: E402

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import numpy as np                                                # noqa: E402
import pytest                                                     # noqa: E402
from graphax import jacve                                         # noqa: E402

from alphagrad.approx.common.examples import (                    # noqa: E402
    _ANALYTIC_JACOBIAN_TARGETS, base_name, get_args, get_fn, get_raw_fn,
    grad_target_setup, has_scalar_loss, infer_argnums, scalar_loss_fn,
)

RTOL = 1e-5
FROB_RTOL = 2e-6

#: Per-leaf relaxations, with the reason. In both families the offending leaf
#: is a parameter whose gradient is ~0 at the test point, so ``max|grad|`` --
#: the denominator of the per-leaf ratio -- is tiny while the graph-wide
#: residual is 1.5e-07 / 4.9e-07. ``EncoderDecoder`` leaf 0 measures
#: 8.2e-05 and leaf 3
#: 4.4e-05 with every other leaf below 1e-06; ``Encoder`` peaks at 1.7e-05.
#: Both numbers predate this workstream (they are properties of jacve vs
#: XLA's fused reverse-mode accumulation order in float32, not of the loss),
#: and the FROB_RTOL assertion below still applies to them unrelaxed.
_LEAF_RTOL = {"Encoder": 5e-5, "EncoderDecoder": 2e-4}

#: The whole registry, by family. Vmapped variants are separate entries
#: because they are separate graphs with separate reductions.
TRAINABLE = [
    "NeuralNetwork", "VmappedNeuralNetwork",
    "TransformerLM", "TransformerLM3",
    "Encoder", "VmappedEncoder",
    "EncoderDecoder", "VmappedEncoderDecoder",
    "ConvNet", "VmappedConvNet",
    "MoE", "VmappedMoE",
    "ViT", "VmappedViT",
    "LIF_SNN", "ADALIF_SNN", "ADALIF_SNN_SEQ", "LIF_SNN_SHD",
]

ANALYTIC = sorted(_ANALYTIC_JACOBIAN_TARGETS)

#: NOT TESTED, and why. ``Perceptron`` is registered but has been unbuildable
#: since long before this workstream: ``get_args("Perceptron")`` hands
#: ``graphax.examples.Perceptron`` shapes its own body cannot contract
#: (``dot_general requires contracting dimensions to have the same shape, got
#: (4,) and (8,)``). It fails identically at commit 4744979 and on a pristine
#: HEAD, with or without any of this change, so it is a pre-existing registry
#: bug and is recorded here rather than silently omitted.
UNTESTABLE = {
    "Perceptron": "get_args builds shapes graphax.examples.Perceptron cannot "
                  "contract (pre-existing; fails identically at 4744979)",
    "VmappedPerceptron": "same as Perceptron",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tlm_args(example):
    """``TransformerLM`` / ``TransformerLM3`` on SYNTHETIC data.

    ``get_args`` routes those two through ``data_gen``, which loads wikitext2
    off disk. The property under test is a property of the GRAPH, so the test
    builds the same arg layout (x (S, D), y (S, V) one-hot, 7 weights per
    block, Wout (D, V)) from a PRNG instead and stays hermetic.
    """
    n_blk = 3 if example.endswith("3") else 2
    S, D, V = 8, 8, 16
    ks = jax.random.split(jax.random.PRNGKey(0), 7 * n_blk + 4)
    x = jax.random.normal(ks[0], (S, D))
    y = jax.nn.one_hot(jax.random.randint(ks[1], (S,), 0, V), V)

    def w(i, shape):
        return jax.random.normal(ks[i], shape) / jnp.sqrt(jnp.float32(shape[0]))

    ws = []
    for blk in range(n_blk):
        o = 2 + blk * 4
        ws += [w(o, (D, D)), w(o + 1, (D, D)), w(o + 2, (D, D)),
               w(o + 3, (D, D)), jnp.zeros((D,)),
               jnp.ones((D,)), jnp.zeros((D,))]
    ws.append(w(7 * n_blk + 2, (D, V)))
    return [x, y, *ws]


def _args_for(example):
    if base_name(example).startswith("TransformerLM"):
        return _tlm_args(example)
    return get_args(example, jax.random.PRNGKey(0), dataset=None)


def _assert_jacve_is_grad(loss, xs, argnums, label):
    """jacve(loss) == jax.grad(loss), leaf by leaf. Returns (worst, frob)."""
    # The output must be a scalar first, or "Jacobian == gradient" is not even
    # a well-posed claim. This is the same contract env.from_jaxpr checks.
    out_avals = jax.make_jaxpr(loss)(*xs).out_avals
    assert [a.shape for a in out_avals] == [()], (
        f"{label}: traced target is not scalar: "
        f"{[a.shape for a in out_avals]}")

    g = jax.jit(jax.grad(loss, argnums=tuple(argnums)))(*xs)
    j = jax.jit(jacve(loss, order="rev", argnums=tuple(argnums)))(*xs)
    gl = jax.tree_util.tree_leaves(g)
    jl = jax.tree_util.tree_leaves(j)
    assert len(jl) == len(gl), f"{label}: leaf count {len(jl)} != {len(gl)}"

    rtol = _LEAF_RTOL.get(label, RTOL)
    worst = 0.0
    flat_j, flat_g = [], []
    for i, (a, b) in enumerate(zip(jl, gl)):
        # A leading size-1 axis would be acceptable (and is reshaped away);
        # record which we actually got.
        assert a.shape in (b.shape, (1,) + b.shape), (
            f"{label}: leaf {i} jacve shape {a.shape} vs grad {b.shape}")
        a = jnp.reshape(a, b.shape)
        flat_j.append(np.asarray(a).ravel())
        flat_g.append(np.asarray(b).ravel())
        scale = float(jnp.max(jnp.abs(b)))
        rel = float(jnp.max(jnp.abs(a - b))) / (scale if scale else 1.0)
        assert rel <= rtol, (
            f"{label}: leaf {i} (shape {b.shape}) rel err {rel:.3e} > {rtol}")
        worst = max(worst, rel)

    fg = np.concatenate(flat_g)
    frob = float(np.linalg.norm(np.concatenate(flat_j) - fg)
                 / (np.linalg.norm(fg) or 1.0))
    assert frob <= FROB_RTOL, (
        f"{label}: graph-wide relative Frobenius {frob:.3e} > {FROB_RTOL}")
    return worst, frob


# ---------------------------------------------------------------------------
# 1. THE PROPERTY, for every trainable family in the registry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("example", TRAINABLE)
def test_jacve_equals_jax_grad(example):
    loss = get_fn(example)
    xs = _args_for(example)
    worst, frob = _assert_jacve_is_grad(
        loss, xs, infer_argnums(example), example)
    print(f"[{example}] worst leaf rel err {worst:.3e}  rel-frob {frob:.3e}")


@pytest.mark.parametrize("example", sorted(UNTESTABLE))
def test_untestable_families_are_recorded_not_forgotten(example):
    """A family the property CANNOT be checked on is named here with its
    reason, and the reason is re-verified: it must still be broken for the
    stated cause, so that fixing it turns this test red instead of leaving a
    silent gap in the table above."""
    with pytest.raises(Exception) as ei:
        get_args(example, jax.random.PRNGKey(0), dataset=None)
        get_fn(example)(*get_args(example, jax.random.PRNGKey(0)))
    assert "contracting dimensions" in str(ei.value), (
        f"{example} now fails for a DIFFERENT reason than the recorded one "
        f"({UNTESTABLE[example]}): {ei.value}")


# ---------------------------------------------------------------------------
# 2. the registered target IS model + loss
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("example", TRAINABLE)
def test_registered_target_is_a_scalar(example):
    xs = _args_for(example)
    avals = jax.make_jaxpr(get_fn(example))(*xs).out_avals
    assert [a.shape for a in avals] == [()], (
        f"{example}: {[a.shape for a in avals]}")
    assert has_scalar_loss(example)


def test_the_raw_model_is_not_the_target():
    """The premise. ``get_raw_fn`` is the bare per-element model; it is NOT
    scalar, which is exactly why a flag that traced it measured a full
    per-class Jacobian and reported it as a gradient."""
    xs = _args_for("VmappedNeuralNetwork")
    raw = get_raw_fn("VmappedNeuralNetwork")(*xs)
    assert np.ndim(raw) == 2                     # (B, C): batch x class
    assert float(get_fn("VmappedNeuralNetwork")(*xs)) == pytest.approx(
        float(jnp.mean(jnp.sum(raw, axis=-1))), rel=1e-6)


def test_unbatched_nn_sums_the_class_axis():
    """``NeuralNetwork`` returns ``(C,)`` for ONE sample, so its class-sum IS
    the loss: no mean over the class axis (which would optimize NLL/C) and no
    mean around the scalar (which would add two dead vertices)."""
    xs = _args_for("NeuralNetwork")
    raw = get_raw_fn("NeuralNetwork")(*xs)
    assert np.ndim(raw) == 1
    assert float(get_fn("NeuralNetwork")(*xs)) == pytest.approx(
        float(jnp.sum(raw)), rel=1e-6)


def test_tlm_loss_is_the_mean_token_nll():
    xs = _tlm_args("TransformerLM")
    raw = get_raw_fn("TransformerLM")(*xs)
    assert np.ndim(raw) == 1                    # (S,), classes already summed
    assert float(get_fn("TransformerLM")(*xs)) == pytest.approx(
        float(jnp.mean(raw)), rel=1e-6)


def test_encoder_decoder_means_its_position_axis():
    """``EncoderDecoder`` carries a SEQUENCE axis even unbatched -- (S, D),
    not (C,) -- so unlike the one-sample classifiers its loss is
    mean_positions(sum_features), batched or not."""
    xs = _args_for("EncoderDecoder")
    raw = get_raw_fn("EncoderDecoder")(*xs)
    assert np.ndim(raw) == 2
    assert float(get_fn("EncoderDecoder")(*xs)) == pytest.approx(
        float(jnp.mean(jnp.sum(raw, axis=-1))), rel=1e-6)


@pytest.mark.parametrize("example", ["LIF_SNN", "ADALIF_SNN"])
def test_aux_returning_snn_targets(example):
    """``LIF_SNN`` / ``ADALIF_SNN`` return ``(loss, U1..U3, I1..I3 / a1..a3)``.

    The loss is element 0 BY THE MODEL'S SIGNATURE. The old generic wrapper
    inferred it from the pytree and called ``jnp.ndim`` on the whole tuple --
    seven leaves of the same shape reported ndim 2 and the ``jnp.sum(out,
    axis=-1)`` that followed raised ``TypeError: sum requires ndarray or
    scalar arguments, got tuple``, so grad mode on these two targets could
    never run at all.
    """
    xs = _args_for(example)
    raw = get_raw_fn(example)(*xs)
    assert isinstance(raw, tuple) and len(raw) == 7
    assert float(get_fn(example)(*xs)) == pytest.approx(
        float(jnp.mean(raw[0])), rel=1e-6)


@pytest.mark.parametrize("example", ["LIF_SNN_SHD", "ADALIF_SNN_SEQ"])
def test_already_scalar_snn_targets_are_untouched(example):
    """These two reduce INSIDE the model, so the registered target is the
    model itself: not even a ``jnp.mean`` is wrapped around it, because the
    mean of a scalar emits a live reduce_sum over no axes plus a divide by 1
    -- two dead vertices in the pointer's action space."""
    xs = _args_for(example)
    assert jnp.ndim(get_raw_fn(example)(*xs)) == 0
    n_raw = len(jax.make_jaxpr(get_raw_fn(example))(*xs).jaxpr.eqns)
    n_tgt = len(jax.make_jaxpr(get_fn(example))(*xs).jaxpr.eqns)
    assert n_raw == n_tgt, (
        f"{example}: the registered target added {n_tgt - n_raw} equations to "
        "an already-scalar model")


# ---------------------------------------------------------------------------
# 3. the analytic AD benchmarks are EXEMPT, by name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("example", ANALYTIC)
def test_analytic_targets_are_exempt_from_the_scalar_contract(example):
    """No model, no training run, no loss -- so no scalar contract.

    These are AD micro-benchmarks whose measured object IS the full Jacobian,
    which is what they exist to measure and what every archived run of them
    measured. Five of the seven return SEVERAL unrelated outputs, so any
    scalarisation would be an arbitrary adjoint seed dressed up as a training
    loss. The exemption is declared (``_ANALYTIC_JACOBIAN_TARGETS``), not
    inferred, and ``has_scalar_loss`` is the single place that reports it --
    the env's contract check and the quality-channel default both read it.
    """
    assert not has_scalar_loss(example)
    xs = _args_for(example)
    avals = jax.make_jaxpr(get_fn(example))(*xs).out_avals
    assert [a.shape for a in avals] != [()], (
        f"{example} became scalar; the exemption is documented as deliberate")
    # ...and the registered target IS the bare model: nothing was added.
    n_raw = len(jax.make_jaxpr(get_raw_fn(example))(*xs).jaxpr.eqns)
    n_tgt = len(jax.make_jaxpr(get_fn(example))(*xs).jaxpr.eqns)
    assert n_raw == n_tgt


def test_scalar_loss_fn_is_an_assertion_not_a_reduction():
    """It adds no equation, and it names the offender when handed a target
    that is not a scalar loss."""
    xs = _args_for("VmappedNeuralNetwork")
    tgt = get_fn("VmappedNeuralNetwork")
    n_plain = len(jax.make_jaxpr(tgt)(*xs).jaxpr.eqns)
    n_checked = len(jax.make_jaxpr(scalar_loss_fn(tgt, "VmappedNeuralNetwork"))
                    (*xs).jaxpr.eqns)
    assert n_plain == n_checked

    bad = scalar_loss_fn(get_raw_fn("VmappedNeuralNetwork"), "raw-model")
    with pytest.raises(ValueError, match="not a scalar loss"):
        bad(*xs)


def test_an_unregistered_target_is_a_loud_error():
    """Two refusals, both loud, neither guessing.

    A name graphax has never heard of dies in ``get_raw_fn`` -- there is no
    model to resolve. A name graphax DOES export but which declares no
    training loss here dies in ``get_fn``: the point of the registry is that a
    new target states what a training run of it optimizes instead of letting a
    shape heuristic invent a reduction.
    """
    with pytest.raises(ValueError, match="not found in examples"):
        get_fn("NoSuchExample")

    from graphax import examples as _gx
    unregistered = [n for n in ("PropaneCombustion", "HumanHeartDipole",
                                "f", "g")
                    if hasattr(_gx, n)]
    assert unregistered, ("expected at least one graphax example outside this "
                          "registry to pin the second refusal on")
    for n in unregistered:
        assert get_raw_fn(n) is not None
        with pytest.raises(ValueError, match="not registered"):
            get_fn(n)


# ---------------------------------------------------------------------------
# 4. --measure-grad is a no-op
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("example", ["NeuralNetwork", "VmappedNeuralNetwork"])
def test_measure_grad_does_not_move_the_traced_graph(example):
    """The retirement, stated as the property it must preserve. Equality is on
    the PRINTED jaxpr; the two targets may be distinct Python objects."""
    xs = _args_for(example)
    txt = []
    for flag in (False, True):
        t, out_xs, argnums = grad_target_setup(
            NS(measure_grad=flag, seed_vertices=False), get_fn(example), xs,
            example)
        assert tuple(argnums) == tuple(infer_argnums(example))
        assert len(out_xs) == len(xs)
        txt.append(str(jax.make_jaxpr(t)(*out_xs)))
    assert txt[0] == txt[1]


def test_measure_grad_warns_once():
    from alphagrad.approx.common import examples as ex
    ex._MEASURE_GRAD_WARNED.clear()
    assert ex.warn_measure_grad_deprecated(NS(measure_grad=True)) is True
    assert ex._MEASURE_GRAD_WARNED == [1]
    assert ex.warn_measure_grad_deprecated({"measure_grad": True}) is True
    assert ex._MEASURE_GRAD_WARNED == [1]            # still ONE warning
    assert ex.warn_measure_grad_deprecated(NS(measure_grad=False)) is False


# ---------------------------------------------------------------------------
# 5. downstream_train optimizes THE SAME function
# ---------------------------------------------------------------------------

def test_downstream_train_uses_the_same_loss():
    """``downstream_train`` IS the repo's definition of "a real training run",
    so it must not carry a loss of its own."""
    from alphagrad.approx.downstream_train import _make_loss_fns
    example = "VmappedNeuralNetwork"
    raw = get_raw_fn(example)
    xs = _args_for(example)
    loss_scalar, _ = _make_loss_fns(raw, example)
    assert float(loss_scalar(*xs)) == pytest.approx(
        float(get_fn(example)(*xs)), rel=1e-6)
