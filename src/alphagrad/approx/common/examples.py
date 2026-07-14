"""Lookup helpers for the per-example target function, args, and data generator.

All trainers used to copy these three functions verbatim. They now live here so
adding a new example or a new variant (Vmapped, dataset-backed) is a one-place
change.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import jax.random as jrand
from graphax import examples

from alphagrad.approx.common.datasets import (
    NN_HIDDEN_DIM,
    NN_VMAP_BATCH,
    dataset_dims,
    load_dataset,
)


def _neural_network(x, y, W1, b1, W2, b2):
    a1 = jnp.tanh(x @ W1.T + b1)
    return 0.5 * (jnp.tanh(a1 @ W2.T + b2) - y) ** 2


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
            return x, y

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
    n_in = n_out = h = 16
    shapes = [
        (n_in,), (n_out,),              # S_in, S_target
        (h,), (h,), (n_out,),           # U1, U2, U3
        (h,), (h,), (n_out,),           # I1, I2, I3
        (h, n_in), (h, h), (n_out, h),  # W1, W2, W3
        (), (), (),                     # alpha, beta, thresh
    ]
    keys = jax.random.split(jax.random.PRNGKey(0), len(shapes))
    return tuple(jax.random.normal(kk, s) for kk, s in zip(keys, shapes))


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
            shapes = [(4,), (4,), (8, 4), (8,), (4, 8), (4,)]
    elif fn_str.endswith("Perceptron"):
        shapes = [(4,), (4,), (8, 4), (8,), (4, 8), (4,), (8,), (8,)]
    elif "EncoderDecoder" in fn_str:
        shapes = [(4, 4)] * 13 + [(4,)] * 8
    elif "Encoder" in fn_str:
        shapes = [(4, 4)] * 10 + [(4,)] * 6
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
    return [jax.random.normal(k, s) for k, s in zip(keys, shapes)]


def get_fn(fn_str: str):
    """Resolve `fn_str` to a Python callable, applying `jax.vmap` for Vmapped variants."""
    base = fn_str[len("Vmapped"):] if fn_str.startswith("Vmapped") else fn_str
    if base.endswith("NeuralNetwork"):
        fn = _neural_network
    elif base == "Perceptron":
        fn = examples.Perceptron
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


def scalar_loss_fn(fn):
    """Wrap an example function into a SCALAR training loss by averaging its
    outputs. Required for graphax ``grad`` / ``value_and_grad`` (which need a
    scalar output) when measuring the GRADIENT instead of the full Jacobian.
    The NeuralNetwork examples already return per-element squared errors, so the
    mean is the MSE loss — the gradient that would hit the optimizer. Shared by
    every measurement site (rollout worker + CPU measure-actor + gfn worker) so
    the jaxpr/order/transforms all operate on the SAME scalar-loss graph."""
    def _loss(*a):
        return jnp.mean(fn(*a))

    return _loss


def seed_loss_fn(fn, argnums):
    """Both-seed (tangent + adjoint) scalar loss with the SEEDS AS EXPLICIT
    GRAPH VERTICES (graphax.seed_vertices). Returns ``g(*primals, t)`` where:

      g(*primals, t) = < ones/N , fn( p + t * dir ) >      # == mean(fn) at any t

    ``dir`` is the tangent-seed direction: ones on the DIFFERENTIATED args
    (``argnums`` — the weights) and zero elsewhere (x / y stay fixed), and the
    scalar tangent seed ``t`` is appended LAST so existing arg indices (and the
    data_gen / weight slots) are unchanged. The ``<ones/N, ·>`` is the adjoint
    seed contraction (an explicit elementwise mul + sum) — same value as
    ``scalar_loss_fn`` (mean), but the tangent injection and adjoint contraction
    show up in ``_build_graph`` as ORDINARY eliminable vertices. Differentiated
    w.r.t. ``argnums + (t,)`` the learned order then chooses forward / reverse /
    cross-country seed timing. The trainer + every CPU measure-actor MUST build
    the SAME wrap + appended ``t`` + shifted argnums (see ``grad_target_setup``)."""
    from graphax.seed_vertices import with_tangent_seed
    argset = {int(a) for a in argnums}

    def g(*primals_and_t):
        *primals, t = primals_and_t
        tangent = tuple(
            jnp.ones_like(p) if i in argset else jnp.zeros_like(p)
            for i, p in enumerate(primals)
        )
        out = with_tangent_seed(fn, tangent)(t, *primals)  # tangent seed vertex
        leaves = jax.tree_util.tree_leaves(out)
        N = 0
        for leaf in leaves:
            n = 1
            for d in jnp.shape(leaf):
                n *= int(d)
            N += n
        N = N or 1
        # adjoint seed vertex: <ones/N, out> as an explicit elementwise mul + sum
        return sum(jnp.sum((jnp.ones_like(leaf) / N) * leaf) for leaf in leaves)

    return g


def grad_target_setup(args_like, base_fn, xs, example):
    """Shared grad-mode target builder — returns ``(target_fn, xs, argnums)``.

    Honors ``--measure-grad`` and ``--seed-vertices``. Called identically by the
    trainer (ppo_ray_worker) and the CPU measure-actor (cpu_approx_worker) so
    both build the IDENTICAL graph (jaxpr / vertex+action space / argnums).
    ``args_like`` may be an argparse Namespace or the actor's args dict."""
    def _flag(name):
        if isinstance(args_like, dict):
            return bool(args_like.get(name, False))
        return bool(getattr(args_like, name, False))

    base_argnums = infer_argnums(example)
    if not _flag("measure_grad"):
        return base_fn, tuple(xs), base_argnums
    if _flag("seed_vertices"):
        return (
            seed_loss_fn(base_fn, base_argnums),
            tuple(xs) + (jnp.zeros(()),),                 # append tangent seed t=0
            tuple(base_argnums) + (len(xs),),             # differentiate weights + t
        )
    return scalar_loss_fn(base_fn), tuple(xs), base_argnums


def grad_target_fn(args_like, base_fn, example):
    """Wrap-only variant of ``grad_target_setup`` for sites that re-swap just the
    target function (the env's args/argnums were fixed at build time)."""
    def _flag(name):
        if isinstance(args_like, dict):
            return bool(args_like.get(name, False))
        return bool(getattr(args_like, name, False))

    if not _flag("measure_grad"):
        return base_fn
    if _flag("seed_vertices"):
        return seed_loss_fn(base_fn, infer_argnums(example))
    return scalar_loss_fn(base_fn)


def infer_argnums(fn_str: str) -> tuple[int, ...]:
    """Default `argnums` (which input slots are differentiated through) per example name."""
    if fn_str in ("LIF_SNN", "LIF_SNN_SHD"):
        return (8, 9, 10)
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
