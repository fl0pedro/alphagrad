import jax
import jax.numpy as jnp


def symlog(x: jax.Array) -> jax.Array:
    """Symmetric logarithmic transform to compress dynamic range."""
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0)


def symexp(x: jax.Array) -> jax.Array:
    """Inverse of symlog."""
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)


def entropy(probs: jax.Array) -> jax.Array:
    """Calculates the Shannon entropy of a probability distribution."""
    # Using 1e-8 to avoid log(0)
    return -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)


def explained_variance(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    """
    Computes fraction of variance that y_pred explains about y_true.
    Returns 1 - Var[y_true - y_pred] / Var[y_true].
    """
    var_y = jnp.var(y_true)
    # Avoid division by zero if y_true is constant
    return jnp.where(var_y != 0, 1.0 - jnp.var(y_true - y_pred) / var_y, 0.0)
