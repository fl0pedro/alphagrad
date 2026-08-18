"""THE VAR PROBE MUST NOT TRAIN THE THING IT MEASURES -- and must actually
measure it.

Same contract as feature_probe_test.py, extended to the new variable-level
module: (1) the gradient of the probe loss w.r.t. the LATENTS is EXACTLY
zero (an algebraic property of stop_gradient -- "the weight is small" is not
accepted); (2) the probe's OWN parameters receive a nonzero gradient (two
zeros would pass "no leak" while learning nothing); (3) invalid variable
slots contribute EXACTLY zero loss with the VALID count as denominator;
(4) the teacher-forced shape GRU is masked past ndim on BOTH the loss and
the input side, so a padded target can never move a scored step; (5) the
episode metrics report majority-class baselines next to every accuracy.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import math  # noqa: E402

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from alphagrad.approx.common import var_probe as VP  # noqa: E402

EMBD = 32
NROWS = 10


@pytest.fixture(scope="module")
def probes():
    return VP.VarProbes(embd_dim=EMBD, key=jax.random.PRNGKey(0))


def _target_rows(rng, n):
    """n random (3, TGT_COLS) target rows + a valid mask with real holes."""
    shapes = [(), (63,), (16, 10), (16, 10, 63), (4, 4, 4, 4), (2, 3, 5, 7, 11)]
    dtypes = ["float32", "bfloat16", "float8_e4m3fn", "int32", "abstract"]
    tgt = np.zeros((n, VP.N_SLOTS, VP.TGT_COLS), np.float32)
    val = np.zeros((n, VP.N_SLOTS), np.float32)
    for i in range(n):
        for s in range(VP.N_SLOTS):
            if rng.random() < 0.25:
                continue  # masked slot (e.g. unary equation's rhs)
            shp = shapes[rng.integers(len(shapes))]
            dt = dtypes[rng.integers(len(dtypes))]
            tgt[i, s] = VP.encode_var(shp, dt)
            val[i, s] = 1.0
    # Guarantee at least one valid and one invalid slot exist.
    val[0, 0] = 1.0
    val[1, 1] = 0.0
    return jnp.asarray(tgt), jnp.asarray(val)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    f_tgt, f_val = _target_rows(rng, NROWS)
    v_tgt, v_val = _target_rows(rng, NROWS)
    return dict(
        f_lat=jnp.asarray(rng.normal(size=(NROWS, EMBD)).astype(np.float32)),
        vctx=jnp.asarray(rng.normal(size=(NROWS, EMBD)).astype(np.float32)),
        f_tgt=f_tgt, f_val=f_val, v_tgt=v_tgt, v_val=v_val,
    )


def _loss(probes, d):
    loss, _ = VP.var_probe_loss(
        probes, d["f_lat"], d["f_tgt"], d["f_val"],
        d["vctx"], d["v_tgt"], d["v_val"])
    return loss


# ------------------------------------------------------------- isolation ---
def test_no_gradient_to_face_latents(probes, data):
    g = jax.grad(lambda lat: _loss(probes, dict(data, f_lat=lat)))(
        data["f_lat"])
    arr = np.asarray(g)
    assert np.all(arr == 0.0), (
        f"var-probe gradient LEAKED into the face latents: "
        f"max|g|={np.abs(arr).max()}")


def test_no_gradient_to_vertex_latents(probes, data):
    g = jax.grad(lambda v: _loss(probes, dict(data, vctx=v)))(data["vctx"])
    arr = np.asarray(g)
    assert np.all(arr == 0.0), (
        f"var-probe gradient LEAKED into the vertex latents: "
        f"max|g|={np.abs(arr).max()}")


def test_probe_parameters_do_get_a_gradient(probes, data):
    g = eqx.filter_grad(lambda p: _loss(p, data))(probes)
    leaves = [x for x in jax.tree_util.tree_leaves(g)
              if eqx.is_inexact_array(x)]
    assert leaves, "no differentiable probe parameters at all"
    tot = sum(float(jnp.sum(jnp.abs(x))) for x in leaves)
    assert tot > 0.0, "the probe's own gradient is identically zero"


def test_every_head_group_gets_gradient(probes, data):
    """All 6 head-groups (face+vertex x lhs/rhs/res) must train -- a dead
    slot would silently report chance accuracy forever."""
    g = eqx.filter_grad(lambda p: _loss(p, data))(probes)
    for level in ("face", "vertex"):
        for s, name in enumerate(VP.VAR_SLOTS):
            head = getattr(g, level)[s]
            tot = sum(float(jnp.sum(jnp.abs(x)))
                      for x in jax.tree_util.tree_leaves(head)
                      if eqx.is_inexact_array(x))
            assert tot > 0.0, f"{level}/{name} head received zero gradient"


# --------------------------------------------------------------- masking ---
def test_invalid_slots_contribute_zero(probes, data):
    """Poisoning an INVALID slot's target must not move the loss at all."""
    tgt = np.asarray(data["f_tgt"]).copy()
    val = np.asarray(data["f_val"])
    inv = np.argwhere(val == 0.0)
    assert inv.size, "fixture must contain invalid slots"
    for i, s in inv[:4]:
        tgt[i, s] = VP.encode_var((4096, 4096, 4096), "float64")
    l0 = float(_loss(probes, data))
    l1 = float(_loss(probes, dict(data, f_tgt=jnp.asarray(tgt))))
    assert l0 == l1, f"invalid slot moved the loss: {l0} != {l1}"


def test_all_invalid_is_zero_not_nan(probes, data):
    z = jnp.zeros_like(data["f_val"])
    zv = jnp.zeros_like(data["v_val"])
    loss = float(_loss(probes, dict(data, f_val=z, v_val=zv)))
    assert math.isfinite(loss) and loss == 0.0, (
        f"empty mask produced {loss}, expected exactly 0")


def test_dims_past_ndim_never_move_the_loss(probes, data):
    """The GRU is teacher-forced; a padded dim target past ndim must not
    change ANY scored step -- neither through the CE mask nor through the
    input side (inputs past ndim are forced to PAD)."""
    tgt = np.asarray(data["f_tgt"]).copy()
    val = np.asarray(data["f_val"])
    changed = 0
    for i in range(tgt.shape[0]):
        for s in range(VP.N_SLOTS):
            nd = int(tgt[i, s, VP.COL_NDIM])
            if val[i, s] > 0 and nd < VP.MAX_NDIM:
                tgt[i, s, VP.COL_DIMS + nd:] = VP.N_DIM_BUCKETS - 1
                changed += 1
    assert changed, "fixture must contain rows with ndim < MAX_NDIM"
    l0 = float(_loss(probes, data))
    l1 = float(_loss(probes, dict(data, f_tgt=jnp.asarray(tgt))))
    assert l0 == l1, (
        f"a padded dim target leaked into the loss: {l0} != {l1}")


# -------------------------------------------------------------- encoding ---
def test_encode_var():
    row = VP.encode_var((16, 10, 63), "float32")
    assert row[VP.COL_NDIM] == 3
    assert row[VP.COL_DTYPE] == VP.dtype_code("float32")
    assert abs(row[VP.COL_LOGSIZE] - math.log10(16 * 10 * 63 + 1)) < 1e-6
    assert list(row[VP.COL_DIMS:VP.COL_DIMS + 3]) == [4.0, 3.0, 5.0]
    assert list(row[VP.COL_DIMS + 3:]) == [0.0, 0.0, 0.0]
    scalar = VP.encode_var((), "float32")
    assert scalar[VP.COL_NDIM] == 0
    assert abs(scalar[VP.COL_LOGSIZE] - math.log10(2)) < 1e-6
    deep = VP.encode_var((2,) * 9, "float32")
    assert deep[VP.COL_NDIM] == VP.MAX_NDIM  # clamped


def test_dim_bucket_edges():
    assert VP.dim_bucket(1) == 0
    assert VP.dim_bucket(2) == 1
    assert VP.dim_bucket(3) == 1
    assert VP.dim_bucket(4096) == 12
    assert VP.dim_bucket(1 << 20) == VP.N_DIM_BUCKETS - 1


def test_dtype_code_families():
    assert VP.dtype_code("float32") == VP.DTYPE_VOCAB.index("float32")
    assert VP.dtype_code("float8_e4m3fn") == VP.DTYPE_VOCAB.index("float8")
    assert VP.dtype_code("float8_e5m2") == VP.DTYPE_VOCAB.index("float8")
    assert VP.dtype_code("float4_e2m1fn") == VP.DTYPE_VOCAB.index("fsub8")
    assert VP.dtype_code("int4") == VP.DTYPE_VOCAB.index("isub8")
    assert VP.dtype_code("int2") == VP.DTYPE_VOCAB.index("isub8")
    assert VP.dtype_code("uint8") == VP.DTYPE_VOCAB.index("uint")
    assert VP.dtype_code("bool") == VP.DTYPE_VOCAB.index("bool")
    assert VP.dtype_code("abstract") == VP.DTYPE_VOCAB.index("abstract")
    assert VP.dtype_code("weird128") == VP.DTYPE_VOCAB.index("other")


def test_vertex_var_table_on_a_real_jaxpr():
    def f(x, w):
        return jnp.tanh(w @ x)

    jaxpr = jax.make_jaxpr(f)(jnp.ones((4,)), jnp.ones((3, 4))).jaxpr
    total_v = len(jaxpr.eqns)
    tgt, val = VP.vertex_var_table(jaxpr, total_v)
    assert tgt.shape == (total_v + 2, 3, VP.TGT_COLS)
    # Row 0 is padding: all-masked.
    assert val[0].sum() == 0.0
    # The dot eqn (vertex 1): binary, res is (3,) float32.
    assert val[1, 0] == 1.0 and val[1, 1] == 1.0 and val[1, 2] == 1.0
    assert tgt[1, 2, VP.COL_NDIM] == 1
    assert tgt[1, 2, VP.COL_DTYPE] == VP.dtype_code("float32")
    # The tanh eqn (vertex 2): unary -> rhs slot MASKED.
    assert val[2, 0] == 1.0 and val[2, 1] == 0.0 and val[2, 2] == 1.0
    assert tgt[2, 0, VP.COL_NDIM] == 1


# ---------------------------------------------------------------- metrics ---
def _perfect_preds(tgt, val):
    tgt = np.asarray(tgt)
    n = tgt.shape[0]
    ndim_t = np.clip(tgt[..., VP.COL_NDIM].astype(np.int64), 0, VP.MAX_NDIM)
    steps = np.arange(VP.MAX_NDIM)[None, None, :]
    return (
        jnp.asarray(ndim_t.astype(np.int32)),
        jnp.asarray(tgt[..., VP.COL_DTYPE].astype(np.int32)),
        jnp.asarray(tgt[..., VP.COL_LOGSIZE].astype(np.float32)),
        jnp.asarray(tgt[..., VP.COL_DIMS:].astype(np.int32)),
        jnp.asarray((steps >= ndim_t[..., None]).astype(np.int32)),
    )


def test_metrics_perfect_decode_scores_one(data):
    preds = _perfect_preds(data["f_tgt"], data["f_val"])
    out = VP.episode_metrics(preds, data["f_tgt"], data["f_val"], "face")
    tested = 0
    for s in VP.VAR_SLOTS:
        pre = f"probe/face/{s}/"
        if out[pre + "n"] == 0.0:
            continue  # slot empty in this fixture draw
        tested += 1
        assert out[pre + "ndim_acc"] == 1.0
        assert out[pre + "dtype_acc"] == 1.0
        assert out[pre + "shape_exact"] == 1.0
        # size_r2 is 1.0 unless the valid targets happen to be constant
        # (then the degenerate-target rule reports 0, never 1).
        assert out[pre + "size_r2"] in (1.0, 0.0)
        # Baselines are frequencies: in (0, 1], and never above a perfect acc.
        for b in ("ndim_base", "dtype_base"):
            assert 0.0 < out[pre + b] <= 1.0
    assert tested, "every slot was empty -- fixture is broken"


def test_metrics_baseline_is_majority_class():
    n = 12
    tgt = np.zeros((n, 3, VP.TGT_COLS), np.float32)
    val = np.ones((n, 3), np.float32)
    # 9 of 12 rows ndim=2, 3 rows ndim=3  -> baseline 0.75 for every slot.
    for i in range(n):
        shp = (4, 4) if i < 9 else (4, 4, 4)
        for s in range(3):
            tgt[i, s] = VP.encode_var(shp, "float32")
    preds = _perfect_preds(tgt, val)
    out = VP.episode_metrics(preds, tgt, val, "vertex")
    for s in VP.VAR_SLOTS:
        assert out[f"probe/vertex/{s}/ndim_base"] == 0.75
        assert out[f"probe/vertex/{s}/dtype_base"] == 1.0  # all float32


def test_metrics_empty_slot_is_all_zero_not_nan():
    tgt = np.zeros((4, 3, VP.TGT_COLS), np.float32)
    val = np.zeros((4, 3), np.float32)
    preds = _perfect_preds(tgt, val)
    out = VP.episode_metrics(preds, tgt, val, "face")
    for k, v in out.items():
        assert np.isfinite(v), f"{k} is not finite on an empty episode"
        assert v == 0.0


def test_metric_keys_are_the_documented_wandb_keys():
    rng = np.random.default_rng(7)
    tgt, val = _target_rows(rng, 5)
    preds = _perfect_preds(tgt, val)
    out = VP.episode_metrics(preds, tgt, val, "face")
    expect = {f"probe/face/{s}/{k}"
              for s in VP.VAR_SLOTS for k in VP.METRIC_KEYS}
    assert set(out) == expect
