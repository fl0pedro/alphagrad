"""Pin the ADDITIVE quality gate (_apply_quality_gate / _exact_cost_reference).

Sign-trap regression: cost channels are PENALTIES, so the gate must FLOOR a
destroyed plan's costs at the exact-reverse reference (destruction pays what
exact pays), never scale them toward zero (which would reward destruction).
"""
import numpy as np
import jax.numpy as jnp
import pytest

from alphagrad.approx import env as env_mod


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    env_mod._COST_REF.clear()
    env_mod._QUALITY_GATE_STATS["clamps"] = 0
    yield
    env_mod._COST_REF.clear()


def _seed_ref(lat, mem):
    env_mod._COST_REF["ref"] = (lat, mem)


def test_gate_off_is_identity(monkeypatch):
    monkeypatch.delenv("ALPHAGRAD_QUALITY_GATE_MIN", raising=False)
    _seed_ref(1e9, 1e9)
    lat, mem = env_mod._apply_quality_gate(
        37e3, 1e6, 0.0, True, True, None, None)
    assert (lat, mem) == (37e3, 1e6)


def test_destroyed_plan_pays_reference(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0.05")
    _seed_ref(132e3, 4.0e9)
    lat, mem = env_mod._apply_quality_gate(
        37e3, 1e6, 0.0, True, True, None, None)
    assert lat == 132e3 and mem == 4.0e9          # floored, not zeroed
    assert env_mod._QUALITY_GATE_STATS["clamps"] == 1


def test_good_plan_keeps_its_real_costs(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0.05")
    _seed_ref(132e3, 4.0e9)
    # pulldown-class plan: genuinely fast AND high quality -> untouched
    lat, mem = env_mod._apply_quality_gate(
        124e3, 3.9e9, 0.99997, True, True, None, None)
    assert (lat, mem) == (124e3, 3.9e9)
    assert env_mod._QUALITY_GATE_STATS["clamps"] == 0


def test_slower_than_ref_stays_slower(monkeypatch):
    """max() only floors: a destroyed plan that is ALSO slow keeps its own
    worse cost -- the gate never improves anyone."""
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0.05")
    _seed_ref(132e3, 4.0e9)
    lat, mem = env_mod._apply_quality_gate(
        400e3, 9.9e9, 0.0, True, True, None, None)
    assert (lat, mem) == (400e3, 9.9e9)


def test_unmeasured_quality_never_gates(monkeypatch):
    """Non-terminal steps / steps without a quality sample must pass through
    even under the env var -- quality 0.0 there means NOT MEASURED."""
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0.05")
    _seed_ref(132e3, 4.0e9)
    lat, mem = env_mod._apply_quality_gate(
        37e3, 1e6, 0.0, False, False, None, None)
    assert (lat, mem) == (37e3, 1e6)
    lat, mem = env_mod._apply_quality_gate(
        37e3, 1e6, 0.0, True, False, None, None)
    assert (lat, mem) == (37e3, 1e6)


def test_reference_failure_fails_open(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0.05")
    env_mod._COST_REF["ref"] = None                # reference build failed
    lat, mem = env_mod._apply_quality_gate(
        37e3, 1e6, 0.0, True, True, None, None)
    assert (lat, mem) == (37e3, 1e6)


def test_reference_measures_on_a_real_target():
    """End-to-end (CPU): the exact-rev reference builds and returns positive
    latency for a tiny scalar-loss target through the real compile path."""
    import jax
    from types import SimpleNamespace as NS
    W = jnp.asarray(np.random.RandomState(0).randn(8, 4).astype(np.float32))
    x = jnp.asarray(np.random.RandomState(1).randn(4).astype(np.float32))

    def loss(x, W):
        y = W @ x
        return jnp.sum(y * y)

    cfg = NS(target_fun=loss, argnums=(1,), has_aux=False, sparse=False)
    ref = env_mod._exact_cost_reference(cfg, [x, W])
    assert ref is not None
    lat, mem = ref
    assert lat > 0.0 and mem >= 0.0


def test_order_floor_preferred_over_rev_reference(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0.05")
    _seed_ref(132e3, 4.0e9)                     # global rev reference
    lat, mem = env_mod._apply_quality_gate(
        37e3, 1e6, 0.0, True, True, None, None,
        order_floor_fn=lambda: (67e6, 9.0e9))   # this order, done exactly
    assert lat == 67e6 and mem == 9.0e9         # order floor wins


def test_order_floor_failure_falls_back_to_rev(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0.05")
    _seed_ref(132e3, 4.0e9)

    def _boom():
        raise RuntimeError("compile died")

    lat, mem = env_mod._apply_quality_gate(
        37e3, 1e6, 0.0, True, True, None, None, order_floor_fn=_boom)
    assert lat == 132e3 and mem == 4.0e9        # fell back, still floored
