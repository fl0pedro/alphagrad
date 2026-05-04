"""Dataset loading helpers used by the example data generators."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# Default hidden / vmap sizes for the synthetic NeuralNetwork example. Kept here
# so trainers and data-generators agree on the shapes.
NN_HIDDEN_DIM = 128
NN_VMAP_BATCH = 16

_DATASET_CACHE: dict = {}


def dataset_dims(name: str) -> tuple[int, int]:
    """Return (input_dim, output_dim) for a built-in dataset name."""
    if name == "mnist":
        return 784, 10
    raise ValueError(f"Unknown dataset '{name}'")


def load_dataset(name: str, dataset_size: int | None):
    """Load a dataset (cached) and return a `(x, y)` tuple of jnp arrays.

    `dataset_size` truncates the cached arrays when > 0; `None` or `<= 0` keeps
    the full set.
    """
    cache_key = (name, dataset_size)
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    if name == "mnist":
        import tensorflow_datasets as tfds  # local import keeps tfds optional

        ds = tfds.load("mnist", split="train", as_supervised=True, batch_size=-1)
        x_np, y_np = tfds.as_numpy(ds)
        x_np = x_np.reshape(x_np.shape[0], -1).astype(np.float32) / 255.0
        y_np = np.eye(10, dtype=np.float32)[y_np]
        if dataset_size is not None and dataset_size > 0:
            x_np = x_np[:dataset_size]
            y_np = y_np[:dataset_size]
        result = (jnp.asarray(x_np), jnp.asarray(y_np))
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    _DATASET_CACHE[cache_key] = result
    return result
