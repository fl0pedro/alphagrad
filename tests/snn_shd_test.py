"""The SHD dataset, and the gradient window as an ARGUMENT.

  * ``--dataset shd`` bins the raw Spiking Heidelberg Digits HDF5 files to
    700 channels x 100 bins of 10 ms over the first second, caches the binned
    tensors beside the MNIST cache, and honours ``--dataset-size`` as a fixed
    prefix subset exactly as MNIST does.
  * ``--target-grad-window N`` sizes the temporal targets: the graph is one
    base block plus N per-step blocks, and N is refused on a target with no
    time steps.
  * ``ALPHAGRAD_SNN_STEPS`` and ``ALPHAGRAD_SNN_TRUNC`` are REFUSED, loudly,
    at import.

The binning tests build a SYNTHETIC raw file with the published layout rather
than downloading 131 MB: what is under test is the binning, the cache and the
subset rule, and a fixture whose spike times are known exactly is the only way
to assert the bin a spike lands in.
"""
import os
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")

import jax                                                        # noqa: E402
import numpy as np                                                # noqa: E402
import pytest                                                     # noqa: E402

from alphagrad.approx.common import datasets as D                 # noqa: E402
from alphagrad.approx.common.snn_shd import (                     # noqa: E402
    SHD_STEPS, has_time_steps, resolve_grad_window)


# ---------------------------------------------------------------------------
# 1. the loader: shape, binning, cache round-trip, subset determinism
# ---------------------------------------------------------------------------

def _write_raw_shd(path, n_samples=6, seed=0):
    """A raw SHD file in the published layout: vlen spike times in SECONDS,
    vlen unit indices, and one integer label per sample."""
    import h5py

    rng = np.random.default_rng(seed)
    times, units, labels = [], [], []
    for i in range(n_samples):
        # Distinct lengths on purpose: a list of EQUAL-length arrays would
        # make numpy build a 2-D object array, which h5py cannot store as vlen.
        m = 20 + 7 * i
        # Deliberately past 1 s for some spikes: the window is the declared
        # shape, and a late spike must be DROPPED, not folded back in.
        t = rng.uniform(0.0, 1.4, size=m)
        u = rng.integers(0, D.SHD_CHANNELS, size=m)
        times.append(t)
        units.append(u.astype(np.int64))
        labels.append(i % D.SHD_CLASSES)
    with h5py.File(path, "w") as fh:
        g = fh.create_group("spikes")
        g.create_dataset("times", data=np.array(times, dtype=object),
                         dtype=h5py.special_dtype(vlen=np.float64))
        g.create_dataset("units", data=np.array(units, dtype=object),
                         dtype=h5py.special_dtype(vlen=np.int64))
        fh.create_dataset("labels", data=np.array(labels, dtype=np.int64))
    return times, units, labels


@pytest.fixture()
def shd_cache(tmp_path, monkeypatch):
    """A cache directory holding a synthetic raw train file, and nothing else,
    so the loader has to bin it and write the ``.npz`` itself."""
    monkeypatch.setenv("DSNN_SHD_DIR", str(tmp_path))
    D._DATASET_CACHE.clear()
    D._SHD_BINNED.clear()
    raw = _write_raw_shd(tmp_path / "shd_train.h5")
    yield tmp_path, raw
    D._DATASET_CACHE.clear()
    D._SHD_BINNED.clear()


def test_shapes_and_dtypes(shd_cache):
    path, _ = shd_cache
    x, y = D.load_shd(-1)
    assert x.shape == (6, D.SHD_TIME_BINS, D.SHD_CHANNELS)
    assert y.shape == (6, D.SHD_CLASSES)
    assert x.dtype == np.float32 and y.dtype == np.float32
    assert np.allclose(np.asarray(y).sum(axis=-1), 1.0)
    assert D.dataset_dims("shd") == (D.SHD_CHANNELS, D.SHD_CLASSES)


def test_a_spike_lands_in_the_bin_its_time_names(shd_cache):
    path, (times, units, _labels) = shd_cache
    x = np.asarray(D.load_shd(-1)[0])
    for i in range(len(times)):
        t, u = np.asarray(times[i]), np.asarray(units[i])
        keep = t < D.SHD_TIME_BINS * D.SHD_BIN_SECONDS
        want = np.zeros((D.SHD_TIME_BINS, D.SHD_CHANNELS), dtype=np.float32)
        np.add.at(want, ((t[keep] / D.SHD_BIN_SECONDS).astype(np.int64),
                         u[keep]), 1.0)
        assert np.array_equal(x[i], want), f"sample {i} binned differently"
        # The spikes past one second are gone, not folded into the last bin.
        assert x[i].sum() == float(keep.sum())


def test_the_binned_split_is_uint8_on_the_host(shd_cache):
    """The trainer never materialises the split as float32: 8156 recordings
    are 2.28 GB that way, once per process. The measured maximum count in a
    10 ms bin is 6, so uint8 is lossless."""
    x, y = D._shd_binned("train")
    assert x.dtype == np.uint8 and isinstance(x, np.ndarray)
    assert y.dtype == np.uint8
    seq, tgt = D.shd_sample(0)
    assert seq.dtype == np.float32 and seq.shape == (D.SHD_TIME_BINS,
                                                     D.SHD_CHANNELS)
    assert tgt.shape == (D.SHD_CLASSES,) and float(tgt.sum()) == 1.0


def test_the_binned_cache_round_trips(shd_cache):
    path, _ = shd_cache
    x1, y1 = D.load_shd(-1)
    npz = path / f"shd_train_binned_{D.SHD_TIME_BINS}x{D.SHD_CHANNELS}.npz"
    assert npz.exists(), "the binned tensors were not cached"
    # Second load must come from the .npz, so remove the raw file first: if
    # anything still reads it the loader would try to download.
    (path / "shd_train.h5").unlink()
    D._DATASET_CACHE.clear()
    D._SHD_BINNED.clear()
    x2, y2 = D.load_shd(-1)
    assert np.array_equal(np.asarray(x1), np.asarray(x2))
    assert np.array_equal(np.asarray(y1), np.asarray(y2))


def test_dataset_size_is_a_fixed_prefix_subset(shd_cache):
    x_all, y_all = D.load_shd(-1)
    D._DATASET_CACHE.clear()
    x3, y3 = D.load_shd(3)
    assert x3.shape[0] == 3 and y3.shape[0] == 3
    assert D.shd_split_size(3) == 3 and D.shd_split_size(-1) == 6
    assert np.array_equal(np.asarray(x3), np.asarray(x_all)[:3])
    assert np.array_equal(np.asarray(y3), np.asarray(y_all)[:3])
    # Deterministic: a second load of the same number gives the same
    # recordings in the same slots, which is what lets the trainer and its
    # measure actors agree about which one a gradient was taken on.
    D._DATASET_CACHE.clear()
    D._SHD_BINNED.clear()
    x3b, _ = D.load_shd(3)
    assert np.array_equal(np.asarray(x3), np.asarray(x3b))
    assert np.array_equal(np.asarray(x3)[1], D.shd_sample(1)[0])


def test_load_dataset_routes_shd(shd_cache):
    x, y = D.load_dataset("shd", 2)
    assert x.shape == (2, D.SHD_TIME_BINS, D.SHD_CHANNELS)


def test_the_target_consumes_the_real_recording(shd_cache):
    """``--dataset shd`` puts SHD bins in the window slot, not Poisson noise."""
    from alphagrad.approx.common.examples import get_args

    xs = get_args("ADALIF_SNN_SHD", jax.random.PRNGKey(0), dataset="shd",
                  grad_window=4, dataset_size=6)
    window = np.asarray(xs[0])
    assert window.shape == (4, D.SHD_CHANNELS)
    assert any(np.array_equal(window, D.shd_sample(i)[0][-4:])
               for i in range(6)), (
        "the window is not the last 4 bins of any recording in the subset")


# ---------------------------------------------------------------------------
# 2. the gradient window
# ---------------------------------------------------------------------------

def _traced_tag_counts(window):
    from alphagrad.approx.common.examples import get_args, get_fn
    from alphagrad.approx.common.temporal_order import BASE_STEP, step_tags
    from graphax import inline_call_primitives

    xs = get_args("ADALIF_SNN_SHD", jax.random.PRNGKey(0), dataset=None,
                  grad_window=window)
    cj = jax.make_jaxpr(get_fn("ADALIF_SNN_SHD"))(*xs)
    jx, _ = inline_call_primitives(cj.jaxpr, cj.literals)
    tags = step_tags(jx)
    return int((tags == BASE_STEP).sum()), len(tags)


def test_the_window_sizes_the_graph():
    """base + N * per-step, measured on the traced equation count.

    MEASURED on ADALIF_SNN_SHD: base 1, per-step 47, so N = 2 is 95 equations,
    N = 3 is 142 and N = 5 is 236. The base is the final ``loss / N``.
    """
    counts = {n: _traced_tag_counts(n) for n in (2, 3, 5)}
    base = counts[2][0]
    per_step = (counts[2][1] - base) // 2
    assert (base, per_step) == (1, 47), counts
    for n, (b, total) in counts.items():
        assert b == base, counts
        assert total == base + n * per_step, counts


def test_window_one_is_one_equation_smaller_than_the_block_predicts():
    """RECORDED, not a defect. At N = 1 the window array is ``(1, 700)`` and
    ``S_in_seq[0]`` is the WHOLE leading axis, so JAX emits the squeeze without
    the slice in front of it: 46 equations in that step instead of 47. Every
    other window traces the uniform block. Pinned so that a change to the
    indexing shows up here rather than as an off-by-one in a step tag."""
    assert _traced_tag_counts(1) == (1, 47)
    assert _traced_tag_counts(2) == (1, 95)


def test_the_window_shortens_the_differentiated_window_only():
    """A shorter window is a smaller GRAPH, not a shorter recording: the
    detached steps still ran, so the carry entering the window is not zero."""
    from alphagrad.approx.common.examples import get_args

    xs = get_args("ADALIF_SNN_SHD", jax.random.PRNGKey(0), dataset=None,
                  grad_window=2)
    assert np.asarray(xs[0]).shape == (2, D.SHD_CHANNELS)
    assert float(np.abs(np.asarray(xs[2])).max()) > 0.0, (
        "the membrane carry is zero: the detached pre-window forward did not "
        "run")


def test_the_window_raises_on_a_target_without_time_steps():
    with pytest.raises(ValueError) as ei:
        resolve_grad_window("Helmholtz", 4)
    assert "has NO time steps" in str(ei.value)
    assert resolve_grad_window("Helmholtz", None) == 0
    assert not has_time_steps("Helmholtz")
    assert has_time_steps("ADALIF_SNN_SHD")


def test_the_window_is_bounded_by_the_sequence():
    assert resolve_grad_window("LIF_SNN_SHD", SHD_STEPS) == SHD_STEPS
    with pytest.raises(ValueError):
        resolve_grad_window("LIF_SNN_SHD", SHD_STEPS + 1)
    with pytest.raises(ValueError):
        resolve_grad_window("LIF_SNN_SHD", 0)


def test_get_args_refuses_the_window_off_a_temporal_target():
    from alphagrad.approx.common.examples import get_args

    with pytest.raises(ValueError):
        get_args("Helmholtz", jax.random.PRNGKey(0), grad_window=3)


# ---------------------------------------------------------------------------
# 3. the two environment variables are gone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name",
                         ["ALPHAGRAD_SNN_STEPS", "ALPHAGRAD_SNN_TRUNC"])
def test_the_replaced_env_vars_fail_loudly_at_import(name):
    env = dict(os.environ, **{name: "1"})
    r = subprocess.run(
        [sys.executable, "-c", "import alphagrad.approx.common.examples"],
        env=env, capture_output=True, text=True)
    assert r.returncode != 0, r.stdout
    assert f"{name} is no longer read" in r.stderr, r.stderr
    assert "--target-grad-window" in r.stderr, r.stderr


def test_lif_snn_shd_still_works():
    """The ticket keeps the existing target running, on the same shapes."""
    from alphagrad.approx.common.examples import get_args, get_fn, infer_argnums

    xs = get_args("LIF_SNN_SHD", jax.random.PRNGKey(0), dataset=None,
                  grad_window=2)
    assert np.asarray(xs[0]).shape == (2, D.SHD_CHANNELS)
    assert np.asarray(xs[8]).shape == (128, D.SHD_CHANNELS)
    assert infer_argnums("LIF_SNN_SHD") == (8, 9, 10)
    assert np.ndim(get_fn("LIF_SNN_SHD")(*xs)) == 0


def test_adalif_shd_has_the_adaptive_signature():
    """ADALIF swaps LIF's synaptic current for the adaptation state and adds
    one decay, so its tuple is one longer -- and the WEIGHTS DO NOT MOVE."""
    from alphagrad.approx.common.examples import get_args, get_fn, infer_argnums

    lif = get_args("LIF_SNN_SHD", jax.random.PRNGKey(0), dataset=None,
                   grad_window=2)
    ada = get_args("ADALIF_SNN_SHD", jax.random.PRNGKey(0), dataset=None,
                   grad_window=2)
    assert len(ada) == len(lif) + 1
    for i in (8, 9, 10):
        assert np.asarray(ada[i]).shape == np.asarray(lif[i]).shape
    assert infer_argnums("ADALIF_SNN_SHD") == (8, 9, 10)
    assert np.ndim(get_fn("ADALIF_SNN_SHD")(*ada)) == 0


def test_a_dataset_the_target_cannot_consume_raises(shd_cache):
    """--dataset mnist on an SHD target used to fall through to the synthetic
    Poisson train, silently. A spike window is (T, 700) and nothing in MNIST
    has that shape, so the name that cannot work says so."""
    from alphagrad.approx.common.examples import get_args

    with pytest.raises(ValueError) as ei:
        get_args("ADALIF_SNN_SHD", jax.random.PRNGKey(0), dataset="mnist",
                 grad_window=2)
    assert "cannot feed an SHD target" in str(ei.value)
