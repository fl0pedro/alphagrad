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

    if fn_str.endswith("NeuralNetwork"):
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


_BASIC_ARGS = {
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
            h = NN_HIDDEN_DIM
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
    if fn_str.endswith("NeuralNetwork"):
        fn = _neural_network
    elif fn_str.endswith("Perceptron"):
        fn = examples.Perceptron
    else:
        fn = getattr(examples, fn_str, None)
        if fn is None:
            raise ValueError(f"Target function '{fn_str}' not found in examples.")

    if fn_str.startswith("Vmapped"):
        num_args = len(inspect.signature(fn).parameters)
        has_y = "Encoder" in fn_str or fn_str.endswith(("NeuralNetwork", "Perceptron"))
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


def maybe_scalar_loss(args, target_fn):
    """Apply the grad-mode wrap consistently across every measurement site.

    Returns ``(target_fn, measure_grad)``. When ``args.measure_grad`` is set,
    ``target_fn`` is wrapped in :func:`scalar_loss_fn` so the jaxpr/order/
    transforms — and thus the policy's vertex/action space — operate on the
    SAME scalar-loss graph in the rollout worker, the CPU measure-actor, and
    the gfn worker. Centralised here so the five call sites can't drift (a
    site that forgot the wrap would build a different graph than its peers).
    """
    if isinstance(args, dict):
        measure_grad = bool(args.get("measure_grad", False))
    else:
        measure_grad = bool(getattr(args, "measure_grad", False))
    if measure_grad:
        target_fn = scalar_loss_fn(target_fn)
    return target_fn, measure_grad


def infer_argnums(fn_str: str) -> tuple[int, ...]:
    """Default `argnums` (which input slots are differentiated through) per example name."""
    if fn_str.endswith("NeuralNetwork"):
        return (2, 3, 4, 5)
    if fn_str.endswith("Perceptron"):
        return (2, 3, 4, 5, 6, 7)
    return (0,)
