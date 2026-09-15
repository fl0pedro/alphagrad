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
NN_HIDDEN_DIM = 256  # matrix+sampler at 256 (was 128)
# ALPHAGRAD_NN_BATCH: vmap batch of the NN target. 16 was DISPATCH-
# BOUND on Blackwell (97-104us wall for 29x flops spreads — the HLO
# audit 2026-08-04); scale it so one execution carries real compute
# and the latency reward can SEE approximation savings.
NN_VMAP_BATCH = int(os.environ.get("ALPHAGRAD_NN_BATCH", "16"))

_DATASET_CACHE: dict = {}

# CVDF mirrors yann.lecun.com (which is rate-limited and frequently down).
# Files use the standard MNIST IDX format described at
# http://yann.lecun.com/exdb/mnist/.
_MNIST_MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist"
_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def loss_mode() -> str:
    """Canonical ALPHAGRAD_LOSS resolution -- the ONLY reader of the var.

    This used to be read independently here and in examples.py with DIFFERENT
    defaults (mse here, xent there). Unset, that combination fed +/-0.9
    targets into the cross-entropy branch, making -(y*logp) negative and
    rewarding wrong classes. One resolver, one default.
    """
    m = os.environ.get("ALPHAGRAD_LOSS", "xent").strip().lower()
    if m not in ("mse", "xent"):
        raise ValueError(f"ALPHAGRAD_LOSS must be mse or xent, got {m!r}")
    return m


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


def _load_mnist_native(subset: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Train or test split of MNIST as ``(x_uint8 [N, 28, 28], y_uint8 [N])``."""
    if subset not in ("train", "test"):
        raise ValueError(f"subset must be 'train' or 'test', got {subset!r}")
    cache = _mnist_cache_dir()
    _download_mnist(cache)
    x = _read_idx_images(cache / _MNIST_FILES[f"{subset}_images"])
    y = _read_idx_labels(cache / _MNIST_FILES[f"{subset}_labels"])
    return x, y


def dataset_dims(name: str) -> tuple[int, int]:
    """Return (input_dim, output_dim) for a built-in dataset name."""
    if name == "mnist":
        return 784, 10
    if name == "shd":
        # ONE TIME BIN's input width and the class count. The time axis is not
        # a "dim" here: the SNN targets consume it one step at a time.
        return SHD_CHANNELS, SHD_CLASSES
    raise ValueError(f"Unknown dataset '{name}'")


def load_dataset(name: str, dataset_size: int | None, subset: str = "train"):
    """Load a dataset (cached) and return a `(x, y)` tuple of jnp arrays.

    `dataset_size` truncates the cached arrays when > 0; `None` or `<= 0` keeps
    the full set. `subset` selects the MNIST split (``"train"`` or ``"test"``).
    Other datasets currently ignore ``subset`` (only ``"train"`` is supported).
    """
    cache_key = (name, dataset_size, subset)
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    if name == "mnist":
        x_np, y_np = _load_mnist_native(subset=subset)
        x_np = x_np.reshape(x_np.shape[0], -1).astype(np.float32) / 255.0
        y_np = np.eye(10, dtype=np.float32)[y_np]
        # Cross-entropy needs a probability vector: -(y * logp) with a
        # NEGATIVE y would reward the wrong classes. So xent pins {0,1}
        # regardless of the +/-s setting rather than producing nonsense.
        _xent = loss_mode() == "xent"
        if not _xent and os.environ.get("ALPHAGRAD_PM1_TARGETS", "1") == "1":
            # {0,1} -> {-s,+s}, INSET from tanh's asymptotes by default.
            # Measured at 40k steps, jax_grad, seed 250197:
            #   {0,1}  acc 0.8602  drawdown 0.1380   <- the accuracy crater
            #   +/-0.9 acc 0.8453  drawdown 0.0053
            #   +/-0.8 acc 0.8443  drawdown 0.0130
            #   +/-0.7 acc 0.8443  drawdown 0.0056
            #   +/-0.6 acc 0.8449  drawdown 0.0057
            #   +/-1.0 acc 0.6162  drawdown 0.0048   <- asymptote, do not use
            # Anything inset removes the crater; the choice between 0.6 and
            # 0.9 is worth 0.001. Exactly 1.0 puts the target ON tanh's limit
            # where tanh'(z) -> 0, and costs 0.23 accuracy.
            # See the module note: with a tanh output and
            # squared error, {0,1} makes "predict 0 everywhere" a large free
            # loss win that destroys argmax accuracy; +/-1 makes it 10x worse
            # and uses tanh's range symmetrically.
            try:
                _s = float(os.environ.get("ALPHAGRAD_TARGET_SCALE", "0.9"))
            except ValueError:
                _s = 1.0
            y_np = ((2.0 * y_np - 1.0) * _s).astype(np.float32)
        if dataset_size is not None and dataset_size > 0:
            x_np = x_np[:dataset_size]
            y_np = y_np[:dataset_size]
        result = (jnp.asarray(x_np), jnp.asarray(y_np))
    elif name == "shd":
        # The binned Spiking Heidelberg Digits. Its x carries a TIME axis
        # (N, 100, 700) where MNIST's carries none, so it has its own loader;
        # this branch is the one door every caller can knock on.
        result = load_shd(dataset_size, subset=subset)
    else:
        if subset != "train":
            raise ValueError(
                f"subset={subset!r} is only supported for MNIST and SHD"
            )
        raise ValueError(f"Unknown dataset '{name}'")

    _DATASET_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# wikitext-2 (word-level) for the TransformerLM example. The corpus ships
# pre-tokenized (space-separated, <unk> included); vocab = the `vocab_size`
# most frequent words, everything else -> <unk>.
_WIKITEXT_URL = "https://wikitext.smerity.com/wikitext-2-v1.zip"


def _wikitext_cache_dir() -> Path:
    override = os.environ.get("DSNN_WIKITEXT_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "dsnn_wikitext"


def load_wikitext2(vocab_size: int, subset: str = "train") -> np.ndarray:
    ck = ("wikitext2", int(vocab_size), subset)
    if ck in _DATASET_CACHE:
        return _DATASET_CACHE[ck]
    cache = _wikitext_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    z = cache / "wikitext-2-v1.zip"
    if not z.exists():
        print(f"  fetching {_WIKITEXT_URL} -> {z}")
        urllib.request.urlretrieve(_WIKITEXT_URL, z)
    import zipfile
    from collections import Counter
    with zipfile.ZipFile(z) as zf:
        with zf.open(f"wikitext-2/wiki.{subset}.tokens") as fh:
            words = fh.read().decode("utf-8").split()
    counts = Counter(words)
    vocab = [w for w, _ in counts.most_common(int(vocab_size))]
    if "<unk>" not in vocab:
        vocab[-1] = "<unk>"
    idx = {w: i for i, w in enumerate(vocab)}
    unk = idx["<unk>"]
    ids = np.asarray([idx.get(w, unk) for w in words], dtype=np.int32)
    _DATASET_CACHE[ck] = ids
    return ids


# ---------------------------------------------------------------------------
# Spiking Heidelberg Digits (SHD) for the temporal SNN targets.
#
# THE RAW FILES, NOT A PACKAGE. ``shd_train.h5`` / ``shd_test.h5`` are read
# with h5py (already in the stack) and binned here. ``tonic`` is deliberately
# NOT a dependency: it pulls torch and a download stack onto the trainer's hot
# path for what is fifty lines of binning.
#
# THE LAYOUT of either file, as the Zenke lab publishes it:
#   spikes/times   (N,) variable-length float64, spike times in SECONDS
#   spikes/units   (N,) variable-length int,     input channel 0..699
#   labels         (N,)  int, the spoken digit 0..19 (0-9 English, 10-19 German)
#
# THE BINNING is the one LIF_SNN_SHD's shape already declares: 700 channels x
# 100 time bins of 10 ms over the FIRST SECOND of each recording. A spike later
# than 1.0 s is dropped (the recordings are about 0.7-1.4 s; the window is the
# stated shape, not a property of the data). A bin holds the COUNT of spikes of
# that channel in those 10 ms, which is 0 or 1 almost everywhere: a sample
# carries about 8k spikes over 70k (bin, channel) cells.
_SHD_MIRROR = "https://zenkelab.org/datasets"
_SHD_FILES = {"train": "shd_train.h5", "test": "shd_test.h5"}

#: The binned shape. n_in = 700 input channels, T = 100 bins, 10 ms each,
#: n_out = 20 classes. LIF_SNN_SHD / ADALIF_SNN_SHD are built to it.
SHD_CHANNELS = 700
SHD_TIME_BINS = 100
SHD_BIN_SECONDS = 0.01
SHD_CLASSES = 20


def _shd_cache_dir() -> Path:
    """Where the raw ``.h5`` files and the binned ``.npz`` cache live.

    ``DSNN_SHD_DIR`` overrides, exactly as ``DSNN_MNIST_DIR`` does, and the
    campaign points all three at one directory on ``/Scratch`` because the GPU
    nodes mount no home (finding 57). The default sits beside the MNIST cache.
    """
    override = os.environ.get("DSNN_SHD_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "dsnn_shd"


def _download_shd(cache: Path, subset: str) -> Path:
    """The raw ``shd_<subset>.h5``, downloading and un-gzipping it once."""
    cache.mkdir(parents=True, exist_ok=True)
    fname = _SHD_FILES[subset]
    target = cache / fname
    if target.exists():
        return target
    url = f"{_SHD_MIRROR}/{fname}.gz"
    gz = cache / f"{fname}.gz"
    if not gz.exists():
        print(f"  fetching {url} -> {gz}", flush=True)
        urllib.request.urlretrieve(url, gz)
    print(f"  unpacking {gz} -> {target}", flush=True)
    tmp = cache / f"{fname}.part"
    with gzip.open(gz, "rb") as src, open(tmp, "wb") as dst:
        while True:
            chunk = src.read(1 << 22)
            if not chunk:
                break
            dst.write(chunk)
    tmp.replace(target)
    return target


def _bin_shd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Bin one raw SHD file to ``(x [N, T, 700] float32, y [N] uint8)``."""
    import h5py

    with h5py.File(path, "r") as fh:
        times = fh["spikes"]["times"]
        units = fh["spikes"]["units"]
        labels = np.asarray(fh["labels"][:], dtype=np.uint8)
        n = int(labels.shape[0])
        x = np.zeros((n, SHD_TIME_BINS, SHD_CHANNELS), dtype=np.float32)
        for i in range(n):
            t = np.asarray(times[i], dtype=np.float64)
            u = np.asarray(units[i], dtype=np.int64)
            keep = (t >= 0.0) & (t < SHD_TIME_BINS * SHD_BIN_SECONDS)
            b = (t[keep] / SHD_BIN_SECONDS).astype(np.int64)
            c = u[keep]
            if c.size and (c.min() < 0 or c.max() >= SHD_CHANNELS):
                raise ValueError(
                    f"{path.name} sample {i}: channel index out of range "
                    f"[0, {SHD_CHANNELS}): min {c.min()}, max {c.max()}")
            np.add.at(x[i], (b, c), 1.0)
    if labels.size and int(labels.max()) >= SHD_CLASSES:
        raise ValueError(
            f"{path.name}: label {int(labels.max())} is outside "
            f"[0, {SHD_CLASSES})")
    return x, labels


def _shd_binned(subset: str) -> tuple[np.ndarray, np.ndarray]:
    """The binned split, from the ``.npz`` cache or built and cached once."""
    if subset not in _SHD_FILES:
        raise ValueError(
            f"SHD subset must be one of {sorted(_SHD_FILES)}, got {subset!r}")
    cache = _shd_cache_dir()
    npz = cache / (f"shd_{subset}_binned_"
                   f"{SHD_TIME_BINS}x{SHD_CHANNELS}.npz")
    if npz.exists():
        with np.load(npz) as z:
            return z["x"], z["y"]
    raw = _download_shd(cache, subset)
    x, y = _bin_shd(raw)
    tmp = npz.with_suffix(".npz.part")
    np.savez(tmp, x=x, y=y)
    tmp.replace(npz)
    return x, y


def load_shd(dataset_size: int | None = -1, subset: str = "train"):
    """SHD as ``(x [N, 100, 700] float32, y [N, 20] float32 one-hot)``.

    ``dataset_size > 0`` keeps the FIRST ``dataset_size`` samples -- the same
    fixed-prefix subset rule :func:`load_dataset` applies to MNIST, so two
    processes that were given the same number see the same samples in the same
    slots and the trainer and its measure actors cannot disagree about which
    recording a gradient was taken on.
    """
    ck = ("shd", int(dataset_size) if dataset_size is not None else None, subset)
    if ck in _DATASET_CACHE:
        return _DATASET_CACHE[ck]
    x_np, y_np = _shd_binned(subset)
    y_np = np.eye(SHD_CLASSES, dtype=np.float32)[y_np]
    if dataset_size is not None and dataset_size > 0:
        x_np = x_np[:dataset_size]
        y_np = y_np[:dataset_size]
    result = (jnp.asarray(x_np), jnp.asarray(y_np))
    _DATASET_CACHE[ck] = result
    return result
