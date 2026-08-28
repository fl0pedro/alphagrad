"""Lookup helpers for the per-example target function, args, and data generator.

All trainers used to copy these three functions verbatim. They now live here so
adding a new example or a new variant (Vmapped, dataset-backed) is a one-place
change.
"""

from __future__ import annotations

import inspect

import os
import jax
import jax.numpy as jnp
import jax.random as jrand
from graphax import examples

from alphagrad.approx.common.datasets import (
    NN_HIDDEN_DIM,
    NN_VMAP_BATCH,
    dataset_dims,
    load_dataset,
    loss_mode,
)


def example_width(default: int | None = None):
    """ALPHAGRAD_EXAMPLE_WIDTH -> common hidden/model width for the synthetic
    examples (MLP / Encoder / EncoderDecoder / LIF_SNN).

    The examples ship at wildly different scales -- MLP 4->8->4, EncoderDecoder
    square dim 4, LIF_SNN 16 -- so comparing architectures across them measures
    SIZE as much as structure. Setting one width makes the comparison
    apples-to-apples. Unset keeps each example's native shape.
    """
    import os as _o
    v = _o.environ.get("ALPHAGRAD_EXAMPLE_WIDTH", "")
    if not v:
        return default
    w = int(v)
    if w < 2:
        raise ValueError(f"ALPHAGRAD_EXAMPLE_WIDTH must be >= 2, got {w}")
    return w


# Which objective the MNIST-style models use. See the patch note: MSE through a
# saturating tanh is the wrong loss for classification and produced both the
# all-zeros attractor and (with +/-1 targets) the asymptote trap.
#   "xent" — softmax cross-entropy (DEFAULT; needs {0,1} targets)
#   "mse"  — the legacy 0.5*(tanh(z)-y)^2, kept so prior runs reproduce
# Measured, jax.grad, 40k steps, seed 250197:
#   MSE {0,1}   final 0.8602  drawdown 0.1642  never reached 0.90
#   MSE +/-0.9  final 0.8565  drawdown 0.0059  never reached 0.90
#   XENT        final 0.9467  drawdown 0.0053  reached 0.90 at step 8000
# xent also closes the graphax-vs-exact gap: graphax_rev 0.9470 vs
# jax.grad 0.9467 (0.0003), where under MSE it trailed by ~0.03.
# Resolved by datasets.loss_mode() -- the single reader of ALPHAGRAD_LOSS.
# This module and datasets.py previously read the var separately with
# different defaults, silently pairing the xent branch with +/-0.9 targets.
_LOSS_MODE = loss_mode()


def _neural_network(x, y, W1, b1, W2, b2):
    a1 = jnp.tanh(x @ W1.T + b1)
    logits = a1 @ W2.T + b2
    if _LOSS_MODE == "xent":
        # log_softmax in its stable form. Per-ELEMENT so the (10,) output
        # convention (and hence the measured Jacobian shape) is unchanged;
        # summing over the last axis gives the usual NLL.
        logp = logits - jax.scipy.special.logsumexp(logits, axis=-1,
                                                    keepdims=True)
        return -(y * logp)
    return 0.5 * (jnp.tanh(logits) - y) ** 2


# graphax core-v2 vision MNIST models: each takes (x_flat784, y_onehot10, *weights)
# and returns a per-element squared error — same convention as _neural_network,
# so they slot into the Vmapped/dataset harness. Weights come from the matching
# graphax initializer.
_VISION_MODELS = {
    "ConvNet": "conv_weights",
    "MoE": "moe_weights",
    "ViT": "vit_weights",
}

# Model dims matched by REVERSE-MODE GRAD FLOPs (not params). Param-matching
# badly mismatched grad COMPUTE — conv/attention reuse each weight over many
# sites, so at ~50k params ConvNet/MoE/ViT had 4.6x / 26x / 28x NN's
# value_and_grad flops. ConvNet then OOM'd the GPU measure actor at 25-51 GiB
# (NOT the base grad — exact value_and_grad exec peak is 0.126 GiB for ALL
# sizes, and deterministic memory_analysis is ~0; the blowup is the approx
# substep jacve + GPU conv-autotuner workspace, which scales with conv size).
# Shrinking to FLOP-match shrinks that workspace. Target = NN(h=63) reverse-
# grad flops 3.23e6 (seeds off): ConvNet Cout=2 -> 1.09x, MoE d=8 -> 1.19x,
# ViT d=8 -> 1.00x (was d=65 = 28x NN: 410 s/it + 137 contraction-mismatch
# sentinels — FLOP-matching it down is the single biggest pace win).
_EQ_NN_HIDDEN = int(__import__("os").environ.get("ALPHAGRAD_NN_HIDDEN", "63"))
_EQ_VISION_KW = {
    "ConvNet": {"Cout": 2},
    "MoE": {"d": 8},
    "ViT": {"d": 8},
}


def _vision_base(fn_str):
    """Return the bare vision-model name (handling the ``Vmapped`` prefix) if
    ``fn_str`` names one of the graphax vision models, else ``None``."""
    base = fn_str[len("Vmapped"):] if fn_str.startswith("Vmapped") else fn_str
    return base if base in _VISION_MODELS else None


def data_gen(fn_str: str, dataset: str | None = None, dataset_size: int | None = -1):
    """Return a `keys -> data` jit-able function used to refresh dataset args.

    Returns `None` if the example does not have a data generator (most analytic
    examples like Helmholtz/Lighthouse/RoeFlux fall back to fixed args at startup).
    """
    if fn_str == "Helmholtz":

        @jax.jit
        def fn(keys):
            x = jrand.uniform(keys[0], (4,))
            return (x / jnp.sum(x) * 0.9,)

        return fn

    if fn_str in _TLM_BLOCKS:
        # Same data path for every depth: the window/embedding are a property
        # of (S, D, V), not of how many encoder blocks consume them.
        from alphagrad.approx.common.datasets import load_wikitext2
        S, D, V = _tlm_dims()
        ids = jnp.asarray(load_wikitext2(V))
        E = _tlm_embedding(D, V)
        n_start = int(ids.shape[0]) - (S + 1)

        @jax.jit
        def fn(keys):
            s = jrand.randint(keys[0], (), 0, n_start)
            win = jax.lax.dynamic_slice(ids, (s,), (S + 1,))
            return E[win[:S]], jax.nn.one_hot(win[1:], V)

        return fn

    # NeuralNetwork and the graphax vision models all consume MNIST as
    # (flat-784 image, onehot-10 label), so they share the dataset sampler.
    if fn_str.endswith("NeuralNetwork") or _vision_base(fn_str) is not None:
        if dataset is not None:
            x_data, y_data = load_dataset(dataset, dataset_size)
            n_samples = int(x_data.shape[0])
            is_vmapped = fn_str.startswith("Vmapped")

            @jax.jit
            def fn(keys):
                if is_vmapped:
                    idx = jrand.randint(keys[0], (NN_VMAP_BATCH,), 0, n_samples)
                else:
                    idx = jrand.randint(keys[0], (), 0, n_samples)
                return x_data[idx], y_data[idx]

            return fn

        # vision models require a dataset (no synthetic fallback)
        if _vision_base(fn_str) is not None:
            return None

        # The synthetic signal below is a fixed 4-feature (r, theta) pair. When
        # ALPHAGRAD_EXAMPLE_WIDTH widens the model, get_args() compiles for the
        # wider input while this generator still produced 4 features, and the
        # AOT-compiled measurement then failed with "Argument 'x' compiled with
        # float32[16,64] and called with float32[16,4]" -- every terminal
        # measurement fell back to the sentinel, so every reward read as zero.
        # Tile the base signal up to the model width instead.
        _w = example_width()

        def _widen(v):
            if not _w or v.shape[-1] == _w:
                return v
            reps = -(-_w // v.shape[-1])          # ceil-div
            return jnp.concatenate([v] * reps, axis=-1)[..., :_w]

        @jax.jit
        def fn(keys):
            shape = (NN_VMAP_BATCH,) if fn_str.startswith("Vmapped") else ()
            r1 = jrand.uniform(keys[0], shape)
            th1 = jrand.uniform(keys[1], shape, minval=-jnp.pi, maxval=jnp.pi)
            r2 = jrand.uniform(keys[2], shape)
            th2 = jrand.uniform(keys[3], shape, minval=-jnp.pi, maxval=jnp.pi)
            x = jnp.stack([r1, th1 / jnp.pi, r2, th2 / jnp.pi], axis=-1)
            y = jnp.stack(
                [
                    r1 * jnp.cos(th1),
                    r1 * jnp.sin(th1),
                    r2 * jnp.cos(th2),
                    r2 * jnp.sin(th2),
                ],
                axis=-1,
            )
            y += 0.05 * jrand.normal(keys[4], y.shape)
            return _widen(x), _widen(y)

        return fn

    if "Encoder" in fn_str or "Decoder" in fn_str:

        @jax.jit
        def fn(keys):
            if fn_str.startswith("Vmapped"):
                shape_x = (16, 4, 4)
                shape_y = (16, 4, 4)
            else:
                shape_x = (4, 4)
                shape_y = (4, 4)
            x = jrand.normal(keys[0], shape_x)
            y_base = jnp.sin(x * jnp.pi) + jnp.cos(x * jnp.pi)
            y = jax.nn.sigmoid(y_base) + 0.05 * jrand.normal(keys[1], shape_y)
            return x, y

        return fn

    return None


def _lif_snn_args():
    """Small LIF_SNN arg tuple (n_in = n_out = h = 16) matching
    LIF_SNN(S_in, S_target, U1, U2, U3, I1, I2, I3, W1, W2, W3, alpha, beta, thresh).
    Weight args W1/W2/W3 are indices 8/9/10 (pass --argnums 8,9,10)."""
    n_in = n_out = h = example_width(16)
    shapes = [
        (n_in,), (n_out,),              # S_in, S_target
        (h,), (h,), (n_out,),           # U1, U2, U3
        (h,), (h,), (n_out,),           # I1, I2, I3
        (h, n_in), (h, h), (n_out, h),  # W1, W2, W3
        (), (), (),                     # alpha, beta, thresh
    ]
    keys = jax.random.split(jax.random.PRNGKey(0), len(shapes))
    return tuple(jax.random.normal(kk, s) for kk, s in zip(keys, shapes))


def _adalif_snn_args():
    """ADALIF_SNN arg tuple, same 3-layer shape as :func:`_lif_snn_args`.

    ADALIF_SNN(S_in, S_target, U1, U2, U3, a1, a2, a3, W1, W2, W3,
               alpha, beta, rho, thresh)

    Adaptive LIF swaps the synaptic-current state (I1..I3) for an adaptation
    state (a1..a3) plus an extra decay ``rho``, so it is one arg longer than
    LIF. Weights stay at indices 8/9/10, so --argnums 8,9,10 is unchanged.
    """
    n_in = n_out = h = example_width(16)
    shapes = [
        (n_in,), (n_out,),              # S_in, S_target
        (h,), (h,), (n_out,),           # U1, U2, U3
        (h,), (h,), (n_out,),           # a1, a2, a3  (adaptation state)
        (h, n_in), (h, h), (n_out, h),  # W1, W2, W3
        (), (), (), (),                 # alpha, beta, rho, thresh
    ]
    keys = jax.random.split(jax.random.PRNGKey(0), len(shapes))
    return tuple(jax.random.normal(kk, s) for kk, s in zip(keys, shapes))


def _adalif_seq_args():
    """ADALIF_SNN_SEQ args. ``ALPHAGRAD_SNN_STEPS`` (default 1) sets N.

    N=1 is the single-step ("one loop") case and N=T the fully-unrolled
    ("multi loop / state") case. Both use the SAME function, so the only thing
    that differs between those two runs is the number of unrolled steps --
    which is the point: ADALIF_SNN is single-timestep and ignores
    ALPHAGRAD_SNN_TRUNC entirely, so configuring a one-loop vs multi-loop pair
    through that variable would have produced two identical runs.
    """
    import os as _o
    n_in = n_out = h = example_width(16)
    steps = int(_o.environ.get("ALPHAGRAD_SNN_STEPS", "1"))
    if steps < 1:
        raise ValueError(f"ALPHAGRAD_SNN_STEPS must be >= 1, got {steps}")
    shapes = [
        (steps, n_in), (n_out,),        # S_in_seq, S_target
        (h,), (h,), (n_out,),           # U1, U2, U3
        (h,), (h,), (n_out,),           # a1, a2, a3
        (h, n_in), (h, h), (n_out, h),  # W1, W2, W3
        (), (), (), (),                 # alpha, beta, rho, thresh
    ]
    keys = jax.random.split(jax.random.PRNGKey(0), len(shapes))
    return tuple(jax.random.normal(kk, sh) for kk, sh in zip(keys, shapes))


def _lif_shd_args():
    """SHD-shaped temporal LIF args with REVERSE-mode truncation baked into the
    graph. n_in=700, hidden=128, n_out=20, T=100 Poisson spike train. The full
    T-step forward runs here (detached) to produce the recurrent carry entering
    the truncation window; only the last N steps (N from ALPHAGRAD_SNN_TRUNC:
    unset=T full BPTT, k>0=window k, 0=online=1) are returned as the differentiable
    window, so the elimination graph = constant base + N*per-step. Weights 8,9,10."""
    import os as _os
    from graphax.examples.neuromorphic import lif_cb as _lif
    n_in, h, n_out, T = 700, 128, 20, 100
    k = jax.random.split(jax.random.PRNGKey(1), 8)
    full_seq = jax.random.bernoulli(k[0], 0.1, (T, n_in)).astype(jnp.float32)
    S_target = jax.nn.one_hot(jax.random.randint(k[1], (), 0, n_out), n_out).astype(jnp.float32)
    U1 = jnp.zeros((h,)); U2 = jnp.zeros((h,)); U3 = jnp.zeros((n_out,))
    I1 = jnp.zeros((h,)); I2 = jnp.zeros((h,)); I3 = jnp.zeros((n_out,))
    W1 = jax.random.normal(k[2], (h, n_in)) * (6.0 / (n_in ** 0.5))
    W2 = jax.random.normal(k[3], (h, h)) * (6.0 / (h ** 0.5))
    W3 = jax.random.normal(k[4], (n_out, h)) * (6.0 / (h ** 0.5))
    alpha = jnp.array(0.9); beta = jnp.array(0.8); thresh = jnp.array(0.3)
    v = _os.environ.get("ALPHAGRAD_SNN_TRUNC", None)
    if v is None or v == "":
        N = T
    elif int(v) <= 0:
        N = 1               # online
    else:
        N = min(int(v), T)
    for t in range(T - N):  # FULL detached pre-window forward (activations only)
        i1 = W1 @ full_seq[t]; U1, I1, s1 = _lif(U1, I1, i1, alpha, beta, thresh)
        i2 = W2 @ s1;          U2, I2, s2 = _lif(U2, I2, i2, alpha, beta, thresh)
        i3 = W3 @ s2;          U3, I3, s3 = _lif(U3, I3, i3, alpha, beta, thresh)
    sg = jax.lax.stop_gradient
    U1, U2, U3 = sg(U1), sg(U2), sg(U3)
    I1, I2, I3 = sg(I1), sg(I2), sg(I3)
    window = full_seq[T - N:]
    return (window, S_target, U1, U2, U3, I1, I2, I3, W1, W2, W3, alpha, beta, thresh)


_BASIC_ARGS = {
    "LIF_SNN": _lif_snn_args(),
    "ADALIF_SNN": _adalif_snn_args(),
    "ADALIF_SNN_SEQ": _adalif_seq_args(),
    "LIF_SNN_SHD": _lif_shd_args(),
    "Simple": (5.0, 7.0),
    "Lighthouse": (0.02,) * 4,
    "Helmholtz": (jnp.array([0.05, 0.15, 0.25, 0.35]),),
    "RobotArm_6DOF": (0.02,) * 6,
    "RoeFlux_1d": (0.01, 0.02, 0.02, 0.01, 0.03, 0.03),
    "RoeFlux_3d": (
        jnp.array([0.1]),
        jnp.array([0.1, 0.2, 0.3]),
        jnp.array([0.5]),
        jnp.array([0.2]),
        jnp.array([0.2, 0.2, 0.4]),
        jnp.array([0.6]),
    ),
    "BlackScholes_Jacobian": (1.0,) * 5,
}


def get_args(fn_str: str, key, dataset: str | None = None):
    """Build the initial argument tuple for the example function `fn_str`."""
    if fn_str.endswith("NeuralNetwork"):
        if dataset is not None:
            in_dim, out_dim = dataset_dims(dataset)
            h = _EQ_NN_HIDDEN   # equalized ~50k params (was NN_HIDDEN_DIM=256)
            shapes = [
                (in_dim,), (out_dim,),
                (h, in_dim), (h,),
                (out_dim, h), (out_dim,),
            ]
        else:
            _w = example_width()
            if _w:
                shapes = [(_w,), (_w,), (_w, _w), (_w,), (_w, _w), (_w,)]
            else:
                shapes = [(4,), (4,), (8, 4), (8,), (4, 8), (4,)]
    elif fn_str in _TLM_BLOCKS:
        n_blk = _TLM_BLOCKS[fn_str]
        S, D, V = _tlm_dims()
        # Slot layout is per-block-strided (7 slots/block, 4 of them used) so
        # block b of the 3-block target draws the SAME key as block b of the
        # 2-block one -- depth is the only difference between them.
        ks = jax.random.split(key, 7 * n_blk + 2)

        def _w(i, shape):
            return jax.random.normal(ks[i], shape) / jnp.sqrt(
                jnp.float32(shape[0]))

        ws = []
        for blk in range(n_blk):
            o = blk * 7
            ws += [_w(o, (D, D)), _w(o + 1, (D, D)), _w(o + 2, (D, D)),
                   _w(o + 3, (D, D)), jnp.zeros((D,)),
                   jnp.ones((D,)), jnp.zeros((D,))]
        ws.append(_w(7 * n_blk, (D, V)))
        gen = data_gen(fn_str, dataset=dataset or "wikitext2")
        x, y = gen(jax.random.split(ks[7 * n_blk + 1], 2))
        return [x, y, *ws]
    elif fn_str.endswith("Perceptron"):
        shapes = [(4,), (4,), (8, 4), (8,), (4, 8), (4,), (8,), (8,)]
    elif "EncoderDecoder" in fn_str:
        _w = example_width(4)
        shapes = [(_w, _w)] * 13 + [(_w,)] * 8
    elif "Encoder" in fn_str:
        _w = example_width(4)
        shapes = [(_w, _w)] * 10 + [(_w,)] * 6
    elif _vision_base(fn_str) is not None:
        # x = flat MNIST (784,), y = onehot (10,); weights from the graphax
        # initializer (correct per-model shapes). Vmapped batches x and y only.
        vbase = _vision_base(fn_str)
        bx = (NN_VMAP_BATCH,) if fn_str.startswith("Vmapped") else ()
        kx, ky, kw = jax.random.split(key, 3)
        x = jax.random.normal(kx, (*bx, 784))
        y = jax.random.normal(ky, (*bx, 10))
        ws = getattr(examples, _VISION_MODELS[vbase])(kw, **_EQ_VISION_KW[vbase])
        ws = list(ws) if isinstance(ws, (tuple, list)) else [ws]
        return [x, y, *ws]
    else:
        return _BASIC_ARGS[fn_str]

    if fn_str.startswith("Vmapped"):
        shapes[0] = (NN_VMAP_BATCH, *shapes[0])
        if "Encoder" in fn_str or fn_str.endswith(("NeuralNetwork", "Perceptron")):
            shapes[1] = (NN_VMAP_BATCH, *shapes[1])

    keys = jax.random.split(key, len(shapes))
    out = [jax.random.normal(k, s) for k, s in zip(keys, shapes)]
    if fn_str.endswith("NeuralNetwork"):
        # LeCun/Xavier fan-in scaling + zero biases -- the same convention
        # _tlm_dims/TransformerLM already uses above. Unscaled N(0,1) on 784
        # normalized inputs drove the tanh into saturation, which does not
        # change any SHAPE or the vertex/action space (so cost measurements
        # stay comparable) but does move the VALUES the quality channels
        # (cosine / frobenius) and the magnitude-sensitive COMPRESS/QUANT
        # rules are computed from.
        # args 2,4 = W1 (h, in_dim), W2 (out_dim, h); applied as x @ W.T, so
        # fan-in is shape[-1]. args 3,5 = biases.
        for _i in (2, 4):
            out[_i] = out[_i] / jnp.sqrt(jnp.float32(out[_i].shape[-1]))
        for _i in (3, 5):
            out[_i] = jnp.zeros_like(out[_i])
    return out


def get_fn(fn_str: str):
    """Resolve `fn_str` to a Python callable, applying `jax.vmap` for Vmapped variants."""
    base = fn_str[len("Vmapped"):] if fn_str.startswith("Vmapped") else fn_str
    if base.endswith("NeuralNetwork"):
        fn = _neural_network
    elif base == "Perceptron":
        fn = examples.Perceptron
    elif base == "TransformerLM":
        fn = _transformer_lm
    elif base == "TransformerLM3":
        fn = _transformer_lm3
    else:
        # strip the Vmapped prefix so graphax models (ConvNet/MoE/ViT/Encoder/...)
        # resolve by their bare name.
        fn = getattr(examples, base, None)
        if fn is None:
            raise ValueError(f"Target function '{fn_str}' (base '{base}') not found in examples.")

    if fn_str.startswith("Vmapped"):
        num_args = len(inspect.signature(fn).parameters)
        has_y = ("Encoder" in base or base.endswith(("NeuralNetwork", "Perceptron"))
                 or base in _VISION_MODELS)
        mapped_axes = (0, 0) if has_y else (0,)
        static_axes = (None,) * (num_args - len(mapped_axes))
        fn = jax.vmap(fn, in_axes=mapped_axes + static_axes)

    return fn


# ---------------------------------------------------------------------------
# THE SCALAR TRAINING LOSS
#
# The reduction is DECLARED PER FAMILY. It used to be INFERRED from ndim
# (``ndim >= 2 -> mean(sum(out, -1)) else mean(out)``), and ndim cannot tell a
# CLASS axis from a SEQUENCE axis, so the same rule was right for one family
# and wrong for the next:
#
#   VmappedNeuralNetwork  out (B, C)  ndim 2 -> mean_B(sum_C)   CORRECT
#   NeuralNetwork         out (C,)    ndim 1 -> mean_C          WRONG: NLL/C
#   TransformerLM         out (S,)    ndim 1 -> mean_S          CORRECT (the
#       class axis was already summed inside graphax's softmax_cross_entropy,
#       so the surviving axis is the SEQUENCE and its mean is the LM loss)
#
# The unbatched-NeuralNetwork row is the SAME bug this function's previous
# docstring records having fixed for the batched row: averaging the class axis
# makes the loss NLL/C. It survived because the fix was written as a shape
# test rather than as a statement about which axis is which.
# ---------------------------------------------------------------------------

#: ``"class_sum"`` — the raw output is a PER-ELEMENT loss whose LAST axis is
#: the class/feature axis. Reduce it the way every framework reduces a
#: classification loss: SUM the class axis (with one-hot ``y`` that is
#: ``-log p_true``), MEAN every remaining (batch) axis.
#: ``"mean"`` — the last axis is NOT a class axis: either the per-example
#: reduction already happened inside the target (``softmax_cross_entropy``
#: returns a per-POSITION NLL) or the loss is an elementwise MSE whose
#: canonical reduction is the plain mean.
def loss_reduction(fn_str: str | None) -> str:
    """Which scalar reduction turns ``fn_str``'s raw output into THE training
    loss. Declared, never guessed -- see the block comment above.

    ``None`` (caller did not name an example) -> ``"mean"``, the
    ``<ones/N, out>`` adjoint. That is correct for a target that is already a
    per-sample loss and is the only defensible default for the analytic
    examples (Helmholtz / RoeFlux / Lighthouse / BlackScholes), which are not
    losses at all and have no training run to match.
    """
    if not fn_str:
        return "mean"
    base = fn_str[len("Vmapped"):] if fn_str.startswith("Vmapped") else fn_str
    # Per-element loss, last axis = class/feature:
    #   NeuralNetwork / Perceptron / ConvNet / MoE / ViT  -(y*logp) or
    #     0.5*(tanh(z)-y)^2 over the 10 MNIST classes;
    #   EncoderDecoder                                    0.5*(z-y)^2 over the
    #     model dimension (``Encoder`` is NOT here -- it ends in
    #     ``softmax_cross_entropy``, which has already summed the classes).
    if (base.endswith(("NeuralNetwork", "Perceptron"))
            or base in _VISION_MODELS
            or base == "EncoderDecoder"):
        return "class_sum"
    # Everything else. TransformerLM / Encoder: graphax's
    # ``softmax_cross_entropy`` already summed the classes, so what is left is
    # the SEQUENCE axis and the LM loss is its mean (mean NLL per token).
    # LIF_SNN_SHD / ADALIF_SNN_SEQ already return a 0-d loss, so the mean is
    # the identity. The analytic examples are not losses at all.
    return "mean"


def scalar_loss_fn(fn, example: str | None = None):
    """Wrap an example function into THE SCALAR TRAINING LOSS -- the exact
    objective a real training run of this target differentiates.

    This is the ONLY traced target now: ``grad_target_setup`` /
    ``grad_target_fn`` apply it unconditionally, so ``jacve`` of the traced
    graph is a GRADIENT and never a Jacobian, with or without
    ``--measure-grad``. Shared by every measurement site (rollout worker + CPU
    measure-actor + gfn worker + ``downstream_train``) so the jaxpr / order /
    transforms and the downstream optimizer all operate on the SAME graph.

    ``example`` selects the reduction via :func:`loss_reduction`. Omitting it
    falls back to the plain mean; pass it whenever you have it.

    AUX OUTPUTS. ``LIF_SNN`` / ``ADALIF_SNN`` return ``(loss, U1, U2, U3, a1,
    a2, a3)`` -- a loss plus carry state. Only the FIRST leaf is the loss, so
    only the first leaf is reduced. The old ndim test called ``jnp.ndim`` on
    the whole tuple; with all seven leaves the same shape that reported ndim
    2, and the ``jnp.sum(out, axis=-1)`` it then took raised
    ``TypeError: sum requires ndarray or scalar arguments, got tuple``.
    ``--measure-grad`` on those two targets has therefore never run at all
    (the campaign's SNNs are ``LIF_SNN_SHD`` / ``ADALIF_SNN_SEQ``, which
    already return a 0-d loss). Taking the first leaf turns a crash into the
    right answer; no measured number changes, because there were none.
    """
    how = loss_reduction(example)

    def _loss(*a):
        out = fn(*a)
        leaves = jax.tree_util.tree_leaves(out)
        # (loss, *aux): the loss is the first leaf, the rest is carry state.
        out = leaves[0] if leaves else out
        if how == "class_sum" and jnp.ndim(out) >= 1:
            # SUM the class axis...
            out = jnp.sum(out, axis=-1)
            if jnp.ndim(out) == 0:
                # ...and for an UNBATCHED target that sum IS the loss. Return
                # it rather than wrapping it in jnp.mean: the mean of a scalar
                # emits a live reduce_sum + divide-by-1, i.e. a dead vertex in
                # the pointer's action space -- exactly the scaffolding A4
                # removed. The early return is scoped to this branch on
                # purpose, so the graph of every already-0-d target
                # (LIF_SNN_SHD, ADALIF_SNN_SEQ) is untouched.
                return out
        # ...then MEAN whatever batch / sequence axes are left.
        return jnp.mean(out)

    return _loss


def seed_loss_fn(fn, argnums):
    """Scalar loss with the tangent + adjoint seeds as EXACTLY TWO vertices.

    NOT THE TRAINING GRADIENT -- REPLAY ONLY. Both this and
    :func:`scalar_loss_fn` add two equations to the graph (here: a tangent
    ``add`` and an adjoint ``reduce_sum``; there: a ``reduce_sum`` and a
    ``div``), which is why they agree on vertex count -- on TransformerLM,
    96 eqns / 95 valid vertices either way. Equal COUNTS, different
    FUNCTIONS. This one deliberately drops the ``1/N`` of the mean and, worse,
    perturbs the first differentiated argument by ``p + t`` before the forward
    pass, so what it differentiates is a directional derivative along one
    parameter block of an unnormalised sum -- not the objective any training
    run optimizes. ``scalar_loss_fn`` is the correct one and is what every
    non-replay path now builds.

    Target shape: ``N + 2`` eliminable vertices, where N is the model graph and
    the +2 are one tangent seed and one adjoint seed. Measured on nn256/mnist:
    13 -> 15.

    THE PREVIOUS VERSION COST +23, not +2, and the extra 21 were scaffolding:

      * 6 ``broadcast_in_dim`` building ``ones_like``/``zeros_like`` tangent
        DIRECTION constants — pure constants, no derivative content;
      * 12 injection eqns, because ``with_tangent_seed`` emits ``p + t*x`` for
        EVERY primal leaf. Four of those act on x and y (the DATA) whose
        direction is ``zeros_like``, i.e. they computed ``x + t*0`` — an
        IDENTITY on the two largest tensors in the graph ([16,784], [16,10]).
        Those identity vertices carry big intermediate Jacobians that the
        policy then has to eliminate, polluting the cost landscape;
      * 5 for the adjoint contraction (``broadcast(ones)``, ``div``, ``mul``,
        ``reduce_sum``, ``add``) where one reduction suffices.

    The tight form:

      TANGENT SEED (1 eqn) — ``p + t`` on the FIRST DIFFERENTIATED arg. ``t`` is
        a scalar so it broadcasts inside the single ``add`` (scalar-tensor ops
        do not emit a separate broadcast eqn). Direction ``ones`` is implicit:
        ``p + t*1 == p + t``, so no constant tensor is built. Forward mode seeds
        at the differentiated inputs, which is exactly this vertex.

      ADJOINT SEED (1 eqn) — ``reduce_sum(out)``, i.e. ``<ones, out>``. The 1/N
        of a mean is dropped deliberately: it is a constant scale, and every
        quality metric we train on (frob residual, cosine) is scale-invariant,
        so paying two extra eqns to divide would buy nothing.

    SEMANTIC NARROWING, stated plainly: the tangent direction now covers the
    first differentiated argument rather than all of them, so ``d/dt`` is the
    directional derivative along that one parameter block instead of along the
    all-ones direction over every block. The seed's PURPOSE — handing the
    elimination order the freedom to propagate forward, reverse or
    cross-country, and to choose seed timing — is unchanged, and it now costs
    one vertex instead of eighteen. Set ALPHAGRAD_SEED_ALL_ARGS=1 to inject
    into every differentiated arg (one ``add`` each: N + 1 + n_argnums).
    """
    import os as _os
    from graphax.seed_vertices import with_tangent_seed  # noqa: F401 (API ref)
    argset = {int(a) for a in argnums}
    seed_all = _os.environ.get("ALPHAGRAD_SEED_ALL_ARGS", "0") == "1"

    def g(*primals_and_t):
        *primals, t = primals_and_t
        # --- TANGENT SEED ------------------------------------------------
        # `p + t` == `p + t*ones`, one `add` eqn per seeded arg, no constant.
        seeded = list(primals)
        targets = sorted(argset) if seed_all else sorted(argset)[:1]
        for i in targets:
            seeded[i] = seeded[i] + t
        out = fn(*seeded)
        # --- ADJOINT SEED ------------------------------------------------
        # <ones, out> == reduce_sum(out): one eqn. Scale (1/N) omitted — frob
        # and cosine are scale-invariant.
        leaves = jax.tree_util.tree_leaves(out)
        acc = jnp.sum(leaves[0])
        for leaf in leaves[1:]:
            acc = acc + jnp.sum(leaf)
        return acc

    return g


_SEED_VERTICES_WARNED: list = []


def _seed_vertices_requested(_flag) -> bool:
    """``--seed-vertices``: DEPRECATED (workstream A4). Seeds are NOT vertices.

    This is the ONE gate in front of the ONE place the flag is read, and it
    makes two things loud that used to be silent.

    1. THE COUPLING. Every archived run that passed ``--seed-vertices``
       passed ``--measure-grad`` with it -- the shell encoded the rule by hand
       (``run_full_nn256_v2_5seed.sh:305,309``: "--seed-vertices only when
       SEED_VERTICES=1 AND grad mode (grad-only flag)"). The pair is required
       here so a replay rebuilds the archived run's QUALITY CHANNEL as well as
       its graph: ``--measure-grad`` is what puts ``env.quality_metric`` on
       ``loss_drop`` rather than ``cosine``, and the flag's ONLY remaining job
       is that choice. (When A4 wrote this gate the reason was different --
       the seed wrapper was then unreachable without ``--measure-grad``,
       making the flag a silent no-op. The scalar loss is now applied
       unconditionally, so the no-op is gone; the requirement stays because
       replaying half of an archived configuration is still not a replay.)

    2. THE DEPRECATION. ``seed_loss_fn`` appends exactly two eqns -- a
       tangent-seed ``add`` and an adjoint ``reduce_sum`` -- and they are
       ORDINARY ELIMINABLE VERTICES: they enter the pointer's action space
       (``env.py`` ``valid_vertices``, ``masks.build_vertex_valid_static``)
       and enlarge the ``derived_max_faces`` closure bound. Measured on
       nn256: 13 -> 15 vertices. Their stated purpose -- exposing forward /
       reverse / cross-country seeding to the search -- was UNREACHABLE
       anyway under ``ALPHAGRAD_FORCE_REV_ORDER=1``, which keeps only the
       highest remaining vertex (``common/masks.py:145-151``), so the order
       is pinned to reverse and only approximations are learned
       (``docs/CLEAN_DESIGN_AUDIT.md`` row b10). No launcher passes it now.

    The flag is kept ACCEPTED rather than deleted for exactly one reason: the
    forensics / landscape-replay paths (``tools/landscape_map.py``,
    ``ls_verify.sh``) must rebuild the SAME graph as an archived run that was
    launched with it. New work should never set it.
    """
    if not _flag("seed_vertices"):
        return False
    if not _flag("measure_grad"):
        raise ValueError(
            "--seed-vertices requires --measure-grad. The flag exists only to "
            "replay an archived run, every archived run that set it also set "
            "--measure-grad, and --measure-grad is what selects that run's "
            "quality channel (loss_drop, not cosine) -- so half the pair "
            "rebuilds the graph but scores it on the wrong channel. Pass "
            "--measure-grad, or -- preferred -- drop --seed-vertices: "
            "workstream A4 removed it from every launcher because seeds are "
            "NOT vertices."
        )
    if not _SEED_VERTICES_WARNED:
        _SEED_VERTICES_WARNED.append(1)
        print("[A4] WARNING: --seed-vertices is DEPRECATED and is no longer "
              "set by any launcher. It injects 2 eliminable vertices "
              "(tangent-seed add + adjoint reduce_sum) into the action space "
              "and into derived_max_faces, and the forward/reverse/"
              "cross-country freedom they were meant to expose is unreachable "
              "under ALPHAGRAD_FORCE_REV_ORDER=1. Keep it ONLY to reproduce "
              "an archived run's graph.", flush=True)
    return True


def grad_target_setup(args_like, base_fn, xs, example):
    """Shared target builder — returns ``(target_fn, xs, argnums)``.

    THE TARGET IS ALWAYS A SCALAR LOSS. It used to be the RAW example unless
    ``--measure-grad`` was passed, and the raw NeuralNetwork/Encoder/vision
    examples return PER-ELEMENT losses -- so without the flag the measured
    object was a full JACOBIAN (one row per class), not the gradient any
    training run computes. There is now no way to ask for that: ``jacve`` of
    this graph is the gradient, up to a leading size-1 axis.

    ``--measure-grad`` no longer decides that. What it still does is pick the
    quality channel (``env.quality_metric``: ``loss_drop`` vs ``cosine``) and
    gate ``--seed-vertices`` (accepted for archived-run replay, DEPRECATED --
    see ``_seed_vertices_requested``).

    Called identically by the trainer (ppo_ray_worker), the CPU measure-actor
    (cpu_approx_worker) and ``tools/landscape_map`` so all three build the
    IDENTICAL graph (jaxpr / vertex+action space / argnums). ``args_like`` may
    be an argparse Namespace or the actor's args dict."""
    def _flag(name):
        if isinstance(args_like, dict):
            return bool(args_like.get(name, False))
        return bool(getattr(args_like, name, False))

    base_argnums = infer_argnums(example)
    if _seed_vertices_requested(_flag):
        return (
            seed_loss_fn(base_fn, base_argnums),
            tuple(xs) + (jnp.zeros(()),),                 # append tangent seed t=0
            tuple(base_argnums) + (len(xs),),             # differentiate weights + t
        )
    return scalar_loss_fn(base_fn, example), tuple(xs), base_argnums


def grad_target_fn(args_like, base_fn, example):
    """Wrap-only variant of ``grad_target_setup`` for sites that re-swap just the
    target function (the env's args/argnums were fixed at build time).

    Always wraps, for the reason spelled out in ``grad_target_setup``."""
    def _flag(name):
        if isinstance(args_like, dict):
            return bool(args_like.get(name, False))
        return bool(getattr(args_like, name, False))

    if _seed_vertices_requested(_flag):
        return seed_loss_fn(base_fn, infer_argnums(example))
    return scalar_loss_fn(base_fn, example)


def infer_argnums(fn_str: str) -> tuple[int, ...]:
    """Default `argnums` (which input slots are differentiated through) per example name."""
    # Every SNN in this family keeps its weights at 8/9/10 -- LIF and ADALIF
    # alike (ADALIF swaps the synaptic current I1..I3 for the adaptation state
    # a1..a3, which does not move the weight slots). Without the ADALIF names
    # here they fell through to (0,), i.e. differentiating w.r.t. the INPUT
    # SPIKES rather than the weights -- a silently different problem.
    if fn_str in ("LIF_SNN", "LIF_SNN_SHD", "ADALIF_SNN", "ADALIF_SNN_SEQ"):
        return (8, 9, 10)
    if "Encoder" in fn_str or "Decoder" in fn_str:
        # (x, y, *weights) -> every weight arg, matching the vision models
        n = len(inspect.signature(getattr(examples, fn_str)).parameters)
        return tuple(range(2, n))
    if fn_str in _TLM_BLOCKS:
        # (x, y, 7 weights per block, Wout) -> differentiate every weight.
        return tuple(range(2, 2 + 7 * _TLM_BLOCKS[fn_str] + 1))
    if fn_str.endswith("NeuralNetwork"):
        return (2, 3, 4, 5)
    if fn_str.endswith("Perceptron"):
        return (2, 3, 4, 5, 6, 7)
    vbase = _vision_base(fn_str)
    if vbase is not None:
        # differentiate w.r.t. every weight arg (everything after x, y)
        n = len(inspect.signature(getattr(examples, vbase)).parameters)
        return tuple(range(2, n))
    return (0,)


# ---------------------------------------------------------------------------
# TransformerLM (wikitext): N encoder blocks + LM head + xent. x is the
# PRE-EMBEDDED (S, D) window, y the (S, V) one-hot next tokens — the
# embedding gather happens in data_gen, never in the differentiated graph.
# The depth is in the NAME, one entry per registered target: TransformerLM is
# the 2-block target every prior result is measured on and MUST NOT MOVE;
# TransformerLM3 is the deeper variant. Shapes (S, D, V) are shared, so both
# read the same _tlm_dims()/_tlm_embedding data path.
_TLM_BLOCKS = {"TransformerLM": 2, "TransformerLM3": 3}


def _tlm_dims():
    return (int(os.environ.get("ALPHAGRAD_TLM_SEQ", "64")),
            int(os.environ.get("ALPHAGRAD_TLM_DMODEL", "128")),
            int(os.environ.get("ALPHAGRAD_TLM_VOCAB", "2048")))


def _transformer_lm(x, y, WQ1, WK1, WV1, W1, b1, g0, be0,
                    WQ2, WK2, WV2, W2, b2, g1, be1, Wout):
    from graphax.examples.deep_learning import (encoder_block,
                                                softmax_cross_entropy)
    z1 = encoder_block(x, WQ1, WK1, WV1, W1, b1, g0, be0)
    z2 = encoder_block(z1, WQ2, WK2, WV2, W2, b2, g1, be1)
    return softmax_cross_entropy(z2 @ Wout, y)


def _transformer_lm3(x, y, WQ1, WK1, WV1, W1, b1, g0, be0,
                     WQ2, WK2, WV2, W2, b2, g1, be1,
                     WQ3, WK3, WV3, W3, b3, g2, be2, Wout):
    from graphax.examples.deep_learning import (encoder_block,
                                                softmax_cross_entropy)
    z1 = encoder_block(x, WQ1, WK1, WV1, W1, b1, g0, be0)
    z2 = encoder_block(z1, WQ2, WK2, WV2, W2, b2, g1, be1)
    z3 = encoder_block(z2, WQ3, WK3, WV3, W3, b3, g2, be2)
    return softmax_cross_entropy(z3 @ Wout, y)


def _tlm_embedding(D, V):
    # FIXED (seeded) embedding: part of the data pipeline, not learned here.
    return jrand.normal(jrand.PRNGKey(1729), (V, D)) / jnp.sqrt(
        jnp.float32(D))
