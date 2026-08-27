"""Weight initialisation helpers shared across trainers."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp


#: The initialisation schemes ``--init-scheme`` selects between.
#:
#: ``orthogonal`` -- the campaign's: ``orthogonal(sqrt 2)`` weights, zero
#: biases, followed by ``_scale_output_heads``. This is what
#: CLEAN_DESIGN_AUDIT calls a2 (compatible) plus a3 (the asymmetric part).
#:
#: ``glorot`` -- textbook Glorot/Xavier UNIFORM
#: (``U[-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out))]``, i.e.
#: ``jax.nn.initializers.glorot_uniform()``), zero biases, and NO output-head
#: rescaling: every head, the face head included, is treated identically.
#: Chosen over "equinox defaults" deliberately -- equinox's ``Linear`` default
#: is ``U[-1/sqrt(fan_in), +1/sqrt(fan_in)]`` with NON-zero biases, which would
#: put a constant learned offset back into the logits (exactly the prior a2
#: removes).
INIT_SCHEMES: tuple[str, ...] = ("orthogonal", "glorot")


def init_linear_weights(model, key, gain: float | None = None,
                        scheme: str = "orthogonal"):
    """Re-initialise every `eqx.nn.Linear` in `model`, with zero biases.

    ``scheme="orthogonal"`` (default, bit-identical to every prior run):
    orthogonal weights at gain ``sqrt(2)`` unless ``gain`` overrides.
    ``scheme="glorot"``: Glorot/Xavier uniform, ``gain`` scaling the
    distribution (default 1.0, the textbook value).

    Orthogonal init goes through ``jax.numpy.linalg.qr`` which on
    ``jax_cuda12_plugin`` calls ``cusolver_geqrf_ffi`` — that FFI handler
    is missing in jaxlib 0.10.0 + the currently-installed plugin, so we
    pin the init to CPU. The output arrays are transferred back to the
    default device via :func:`eqx.tree_at`'s assignment. ``glorot`` keeps the
    same CPU block: same key consumption, same traversal, same device
    round-trip, so the two schemes differ ONLY in the sampler.
    """
    if scheme not in INIT_SCHEMES:
        raise ValueError(
            f"init_linear_weights: unknown scheme {scheme!r} "
            f"(expected one of {INIT_SCHEMES})")
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [
        x.weight
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]
    get_biases = lambda m: [
        x.bias
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x) and x.bias is not None
    ]

    weights = get_weights(model)
    biases = get_biases(model)
    if scheme == "orthogonal":
        init_gain = jnp.sqrt(2) if gain is None else gain
        init_fn = jnn.initializers.orthogonal(init_gain)
    else:
        init_gain = 1.0 if gain is None else float(gain)
        _glorot = jnn.initializers.glorot_uniform()

        def init_fn(k, shape, _g=init_gain, _f=_glorot):
            return _f(k, shape) * _g

    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        new_weights = [
            init_fn(subkey, weight.shape)
            for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
        ]
    new_biases = [jnp.zeros_like(bias) for bias in biases]

    new_model = eqx.tree_at(get_weights, model, new_weights)
    new_model = eqx.tree_at(get_biases, new_model, new_biases)
    return new_model


def scale_module_weight(model, getter, scale: float):
    """Multiply a single weight tensor (selected by `getter`) by `scale`."""
    return eqx.tree_at(getter, model, getter(model) * scale)
