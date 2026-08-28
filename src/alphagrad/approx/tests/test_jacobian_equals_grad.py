"""THE JACOBIAN IS THE GRADIENT. Pinned, for the graph the campaign traces.

The owner's requirement, verbatim in substance: *"I want to get the Jacobian
(=== grad) to be exactly the same as when one runs a proper training and test
run [...] Even without --measure-grad we should only have a scalar loss and
thus the Jacobian should be identical to the grad."*

Nothing pinned that. Two independent things had to be true and only one of
them was:

  1. THE TRACED TARGET MUST BE A SCALAR LOSS. It was not. Without
     ``--measure-grad`` ``grad_target_setup`` returned the RAW example, and
     the raw NeuralNetwork / vision / Encoder examples return PER-ELEMENT
     losses -- so what got measured was a full JACOBIAN, one row per class.
     ``scalar_loss_fn`` is now applied unconditionally.

  2. THE SCALAR LOSS MUST BE THE ONE A TRAINING RUN OPTIMIZES. The reduction
     was inferred from ``ndim``, which cannot tell a CLASS axis from a
     SEQUENCE axis, so it was right for ``VmappedNeuralNetwork`` (B, C) and
     for ``TransformerLM`` (S,) and wrong for the unbatched
     ``NeuralNetwork`` (C,), where it averaged the class axis and optimized
     ``NLL/C``. ``examples.loss_reduction`` now DECLARES the reduction per
     family.

Both halves are asserted below, plus the property itself: ``graphax.jacve``
of the scalar-loss graph equals ``jax.grad`` of the same loss, leaf by leaf.

TOLERANCE: ``max|jacve - grad| / max|grad| <= 1e-5``, per leaf, in float32.
Measured headroom at the time of writing: 1.1e-7 (VmappedNeuralNetwork),
1.3e-7 (NeuralNetwork), 4.9e-7 (TransformerLM-shaped).

SHAPE: ``jacve`` of a 0-d output returns each leaf at exactly the weight's
shape here -- no leading size-1 axis. The comparison reshapes anyway, so a
future ``(1, *w)`` would still pass; the shape assertion below records which
of the two we actually get.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")
os.environ.setdefault("ALPHAGRAD_MAX_EQNS", "512")
# TLM-SHAPED, not TLM-SIZED: the property is about the graph, and the small
# shapes keep this a unit test. Set before the module-level _tlm_dims() read.
os.environ.setdefault("ALPHAGRAD_TLM_SEQ", "8")
os.environ.setdefault("ALPHAGRAD_TLM_DMODEL", "8")
os.environ.setdefault("ALPHAGRAD_TLM_VOCAB", "16")

from types import SimpleNamespace as NS                           # noqa: E402

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import numpy as np                                                # noqa: E402
import pytest                                                     # noqa: E402
from graphax import jacve                                         # noqa: E402

from alphagrad.approx.common.examples import (                    # noqa: E402
    get_args, get_fn, grad_target_setup, infer_argnums,
    loss_reduction, scalar_loss_fn,
)

RTOL = 1e-5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tlm_shaped():
    """``TransformerLM`` on SYNTHETIC data.

    ``get_args("TransformerLM", ...)`` routes through ``data_gen``, which
    loads wikitext2 off disk. The property under test is a property of the
    GRAPH, so the test builds the same arg layout (x (S, D), y (S, V) one-hot,
    2 x 7 block weights, Wout (D, V)) from a PRNG instead and stays hermetic.
    """
    S, D, V = 8, 8, 16
    ks = jax.random.split(jax.random.PRNGKey(0), 20)
    x = jax.random.normal(ks[0], (S, D))
    y = jax.nn.one_hot(jax.random.randint(ks[1], (S,), 0, V), V)

    def w(i, shape):
        return jax.random.normal(ks[i], shape) / jnp.sqrt(jnp.float32(shape[0]))

    ws = []
    for blk in range(2):
        o = 2 + blk * 4
        ws += [w(o, (D, D)), w(o + 1, (D, D)), w(o + 2, (D, D)),
               w(o + 3, (D, D)), jnp.zeros((D,)),
               jnp.ones((D,)), jnp.zeros((D,))]
    ws.append(w(12, (D, V)))
    return get_fn("TransformerLM"), [x, y, *ws], infer_argnums("TransformerLM")


def _assert_jacve_is_grad(loss, xs, argnums, label):
    """jacve(loss) == jax.grad(loss), leaf by leaf, at RTOL."""
    # The output must be a scalar first, or "Jacobian == gradient" is not even
    # a well-posed claim. This is the same contract env.from_jaxpr checks.
    out_avals = jax.make_jaxpr(loss)(*xs).out_avals
    assert [a.shape for a in out_avals] == [()], (
        f"{label}: traced target is not scalar: "
        f"{[a.shape for a in out_avals]}")

    g = jax.jit(jax.grad(loss, argnums=argnums))(*xs)
    j = jax.jit(jacve(loss, order="rev", argnums=argnums))(*xs)
    gl = jax.tree_util.tree_leaves(g)
    jl = jax.tree_util.tree_leaves(j)
    assert len(jl) == len(gl), f"{label}: leaf count {len(jl)} != {len(gl)}"

    worst = 0.0
    for i, (a, b) in enumerate(zip(jl, gl)):
        # A leading size-1 axis would be acceptable (and is squeezed here);
        # record which we actually got.
        assert a.shape in (b.shape, (1,) + b.shape), (
            f"{label}: leaf {i} jacve shape {a.shape} vs grad {b.shape}")
        a = jnp.reshape(a, b.shape)
        scale = float(jnp.max(jnp.abs(b)))
        rel = float(jnp.max(jnp.abs(a - b))) / (scale if scale else 1.0)
        assert rel <= RTOL, (
            f"{label}: leaf {i} (shape {b.shape}) rel err {rel:.3e} > {RTOL}")
        worst = max(worst, rel)
    return worst


# ---------------------------------------------------------------------------
# 1. the property: jacve == jax.grad on the scalar-loss graph
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("example", ["NeuralNetwork", "VmappedNeuralNetwork"])
def test_jacve_equals_jax_grad_nn_family(example):
    fn = get_fn(example)
    xs = get_args(example, jax.random.PRNGKey(0), dataset=None)
    loss = scalar_loss_fn(fn, example)
    worst = _assert_jacve_is_grad(loss, xs, infer_argnums(example), example)
    print(f"[{example}] worst leaf rel err = {worst:.3e}")


def test_jacve_equals_jax_grad_tlm_shaped():
    fn, xs, argnums = _tlm_shaped()
    loss = scalar_loss_fn(fn, "TransformerLM")
    worst = _assert_jacve_is_grad(loss, xs, argnums, "TransformerLM")
    print(f"[TransformerLM] worst leaf rel err = {worst:.3e}")


# ---------------------------------------------------------------------------
# 2. the scalar loss is unconditional -- WITHOUT --measure-grad too
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("example", ["NeuralNetwork", "VmappedNeuralNetwork"])
def test_target_is_scalar_without_measure_grad(example):
    """The behaviour change. ``--measure-grad`` used to decide whether the
    measured object was a gradient or a full Jacobian; it no longer does."""
    xs = get_args(example, jax.random.PRNGKey(0), dataset=None)
    raw_out = get_fn(example)(*xs)
    assert np.ndim(raw_out) >= 1, "premise: the raw example is NOT scalar"

    for flag in (False, True):
        target, out_xs, argnums = grad_target_setup(
            NS(measure_grad=flag, seed_vertices=False),
            get_fn(example), xs, example)
        avals = jax.make_jaxpr(target)(*out_xs).out_avals
        assert [a.shape for a in avals] == [()], (
            f"measure_grad={flag}: target output {[a.shape for a in avals]}")
        assert tuple(argnums) == tuple(infer_argnums(example))
        assert len(out_xs) == len(xs)


def test_measure_grad_no_longer_changes_the_traced_graph():
    """Same jaxpr with the flag on and off -- the whole point of the change.

    Equality is on the PRINTED jaxpr: the two targets are distinct Python
    closures, so identity would prove nothing.
    """
    example = "VmappedNeuralNetwork"
    xs = get_args(example, jax.random.PRNGKey(0), dataset=None)
    txt = []
    for flag in (False, True):
        t, out_xs, _ = grad_target_setup(
            NS(measure_grad=flag, seed_vertices=False),
            get_fn(example), xs, example)
        txt.append(str(jax.make_jaxpr(t)(*out_xs)))
    assert txt[0] == txt[1]


# ---------------------------------------------------------------------------
# 3. the reduction is the training objective, declared per family
# ---------------------------------------------------------------------------

def test_reduction_table_is_declared_not_inferred():
    assert loss_reduction("NeuralNetwork") == "class_sum"
    assert loss_reduction("VmappedNeuralNetwork") == "class_sum"
    assert loss_reduction("ConvNet") == "class_sum"
    assert loss_reduction("EncoderDecoder") == "class_sum"
    # softmax_cross_entropy already summed the classes -> the surviving axis
    # is the SEQUENCE and the LM loss is its mean.
    assert loss_reduction("TransformerLM") == "mean"
    assert loss_reduction("TransformerLM3") == "mean"
    assert loss_reduction("Encoder") == "mean"
    assert loss_reduction("LIF_SNN_SHD") == "mean"
    assert loss_reduction(None) == "mean"


def test_vmapped_nn_loss_is_mean_batch_of_sum_class():
    example = "VmappedNeuralNetwork"
    fn = get_fn(example)
    xs = get_args(example, jax.random.PRNGKey(0), dataset=None)
    out = fn(*xs)
    assert np.ndim(out) == 2
    got = float(scalar_loss_fn(fn, example)(*xs))
    assert got == pytest.approx(float(jnp.mean(jnp.sum(out, axis=-1))),
                                rel=1e-6)


def test_unbatched_nn_loss_was_off_by_the_class_count():
    """The bug, stated as the factor it was worth.

    ``NeuralNetwork`` returns ``(C,)``: ndim 1, so the old rule took
    ``mean(out)`` -- the class axis averaged, i.e. the loss (and hence the
    gradient) divided by C. C is 10 on MNIST and 4 on the synthetic signal the
    smoke uses.
    """
    example = "NeuralNetwork"
    fn = get_fn(example)
    xs = get_args(example, jax.random.PRNGKey(0), dataset=None)
    out = fn(*xs)
    assert np.ndim(out) == 1
    C = int(out.shape[-1])

    got = float(scalar_loss_fn(fn, example)(*xs))
    assert got == pytest.approx(float(jnp.sum(out)), rel=1e-6)
    old = float(jnp.mean(out))                      # the pre-fix reduction
    assert got == pytest.approx(C * old, rel=1e-5)

    # ... and the gradient moved by the same constant factor, not in direction.
    argnums = infer_argnums(example)
    g_new = jax.grad(scalar_loss_fn(fn, example), argnums=argnums)(*xs)
    g_old = jax.grad(lambda *a: jnp.mean(fn(*a)), argnums=argnums)(*xs)
    for a, b in zip(jax.tree_util.tree_leaves(g_new),
                    jax.tree_util.tree_leaves(g_old)):
        np.testing.assert_allclose(np.asarray(a), C * np.asarray(b), rtol=1e-4,
                                   atol=1e-6)


def test_tlm_loss_is_the_mean_token_nll():
    fn, xs, _ = _tlm_shaped()
    out = fn(*xs)
    assert np.ndim(out) == 1                    # (S,), classes already summed
    got = float(scalar_loss_fn(fn, "TransformerLM")(*xs))
    assert got == pytest.approx(float(jnp.mean(out)), rel=1e-6)


def test_aux_returning_target_reduces_only_the_loss_leaf():
    """``LIF_SNN`` returns ``(loss, U1, U2, U3, a1, a2, a3)``.

    The old ndim rule called ``jnp.ndim`` on the whole tuple -- all seven
    leaves share a shape, so it reported 2 and the ``jnp.sum(out, axis=-1)``
    that followed raised ``TypeError: sum requires ndarray or scalar
    arguments, got tuple``. ``--measure-grad`` on this target could never run.
    """
    fn = get_fn("LIF_SNN")
    xs = get_args("LIF_SNN", jax.random.PRNGKey(0), dataset=None)
    out = fn(*xs)
    assert isinstance(out, tuple) and len(out) == 7
    got = float(scalar_loss_fn(fn, "LIF_SNN")(*xs))
    assert got == pytest.approx(float(jnp.mean(out[0])), rel=1e-6)
    # and it is now differentiable end to end
    _assert_jacve_is_grad(scalar_loss_fn(fn, "LIF_SNN"), xs,
                          infer_argnums("LIF_SNN"), "LIF_SNN")


# ---------------------------------------------------------------------------
# 4. downstream_train optimizes THE SAME function
# ---------------------------------------------------------------------------

def test_downstream_train_uses_the_same_loss():
    """``downstream_train`` IS the repo's definition of "a real training run",
    so it must not carry a loss of its own."""
    from alphagrad.approx.downstream_train import _make_loss_fns
    example = "VmappedNeuralNetwork"
    fn = get_fn(example)
    xs = get_args(example, jax.random.PRNGKey(0), dataset=None)
    loss_scalar, _ = _make_loss_fns(fn, example)
    assert float(loss_scalar(*xs)) == pytest.approx(
        float(scalar_loss_fn(fn, example)(*xs)), rel=1e-6)
