"""Dataset loading helpers used by the example data generators.

MNIST is loaded directly from the IDX files (yann.lecun.com layout, served
by the cvdf-datasets GCS mirror) into ``numpy``. We previously routed the
load through ``tensorflow_datasets``, but tfds 4.9.x reaches into protobuf
internals that were removed in protobuf 6.x (``FieldDescriptor.label``),
which crashed every PPO run on environments shipping a recent protobuf.
The native loader is a few dozen lines and removes the tfds dep on this
hot-path entirely.
"""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# Default hidden / vmap sizes for the synthetic NeuralNetwork example. Kept here
# so trainers and data-generators agree on the shapes.
NN_HIDDEN_DIM = 128
NN_VMAP_BATCH = 16

_DATASET_CACHE: dict = {}

# CVDF mirrors yann.lecun.com (which is rate-limited and frequently down).
# Files use the standard MNIST IDX format described at
# http://yann.lecun.com/exdb/mnist/.
_MNIST_MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist"
_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
}


def _mnist_cache_dir() -> Path:
    """Where to put the four IDX files. ``DSNN_MNIST_DIR`` overrides for
    air-gapped / shared-mount setups; default lives under XDG cache."""
    override = os.environ.get("DSNN_MNIST_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "dsnn_mnist"


def _download_mnist(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for fname in _MNIST_FILES.values():
        target = cache / fname
        if target.exists():
            continue
        url = f"{_MNIST_MIRROR}/{fname}"
        print(f"  fetching {url} -> {target}")
        urllib.request.urlretrieve(url, target)


def _read_idx_images(path: Path) -> np.ndarray:
    """Return ``(N, rows, cols)`` uint8 array from a gzipped IDX-3 file."""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"unexpected IDX magic 0x{magic:x} in {path}")
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    """Return ``(N,)`` uint8 array from a gzipped IDX-1 file."""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected IDX magic 0x{magic:x} in {path}")
        return np.frombuffer(f.read(), dtype=np.uint8)


def _load_mnist_native() -> tuple[np.ndarray, np.ndarray]:
    """Train split of MNIST as ``(x_uint8 [N, 28, 28], y_uint8 [N])``."""
    cache = _mnist_cache_dir()
    _download_mnist(cache)
    x = _read_idx_images(cache / _MNIST_FILES["train_images"])
    y = _read_idx_labels(cache / _MNIST_FILES["train_labels"])
    return x, y


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
        x_np, y_np = _load_mnist_native()
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
