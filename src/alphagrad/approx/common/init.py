"""Weight initialisation helpers shared across trainers."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp


def init_linear_weights(model, key, gain: float | None = None):
    """Re-initialise every `eqx.nn.Linear` in `model` with orthogonal weights and zero biases."""
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
    init_gain = jnp.sqrt(2) if gain is None else gain
    init_fn = jnn.initializers.orthogonal(init_gain)

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
