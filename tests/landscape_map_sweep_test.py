"""Unit test for landscape_map singleton sweep and F* stack generation."""

import numpy as np
import pytest

from alphagrad.approx.tools.landscape_map import (
    make_argparser,
    build_env,
    markowitz_order,
    rev_order,
    face_inventory,
    build_singleton_sweep_plans,
    build_f_star_stacks,
    measure,
)


@pytest.fixture(scope="module")
def helmholtz_setup():
    parser = make_argparser()
    args = parser.parse_args([
        "--example", "Helmholtz",
        "--out-dir", "/tmp/test_helmholtz_sweep",
        "--singleton-sweep",
    ])
    env, eval_samples, _ = build_env(args)
    order = markowitz_order(env)
    inv = face_inventory(env, order, capture_tensors=True)
    return env, eval_samples, order, inv


def test_markowitz_order(helmholtz_setup):
    env, _, order, _ = helmholtz_setup
    assert len(order) == len(env.valid_vertices)
    assert set(order.tolist()) == set(int(v) for v in env.valid_vertices)
    # On Helmholtz, Markowitz degree order starts from equation 1
    assert order[0] == 1


def test_face_inventory_capture(helmholtz_setup):
    _, _, _, inv = helmholtz_setup
    assert len(inv) > 0
    for entry in inv:
        assert "tensors" in entry
        tensors = entry["tensors"]
        assert isinstance(tensors, dict)
        # Verify at least one slot has captured SparseTensor
        assert any(t is not None for t in tensors.values())


def test_build_singleton_sweep_plans(helmholtz_setup):
    env, _, order, inv = helmholtz_setup
    plans, plan_orders = build_singleton_sweep_plans(env, order, inv)

    assert len(plans) > 0
    assert len(plans) == len(plan_orders)

    ops = {p["op"] for p in plans.values()}
    # Must contain all 4 classes on Helmholtz
    assert "skip" in ops
    assert "quant" in ops
    assert "compress" in ops
    assert "diag" in ops

    # SKIP: exactly 1 per face
    skip_plans = [p for p in plans.values() if p["op"] == "skip"]
    assert len(skip_plans) == len(inv)

    # QUANT: only bf16
    for pid, p in plans.items():
        if p["op"] == "quant":
            assert ":bf16" in pid
            wire = p["wires"][0]
            assert wire["row"][0] == -3  # QUANT_SENTINEL

    # DIAG: explicit factor > 1 (never -1)
    for pid, p in plans.items():
        if p["op"] == "diag":
            wire = p["wires"][0]
            row = wire["row"]
            factor = row[2]
            assert factor > 1


def test_measure_singleton_and_stacks(helmholtz_setup):
    env, eval_samples, order, inv = helmholtz_setup
    plans, plan_orders = build_singleton_sweep_plans(env, order, inv)

    # Pick one of each class to test live measurement
    sampled_pids = []
    for op in ("skip", "quant", "compress", "diag"):
        pid = next(p for p, pl in plans.items() if pl["op"] == op)
        sampled_pids.append(pid)

    measured = {}
    for pid in sampled_pids:
        m = measure(env, eval_samples, plan_orders[pid], plans[pid])
        assert np.isfinite(m["latency_ns"])
        assert np.isfinite(m["quality"])
        assert "static_temp" in m
        assert "peak_memory" in m
        measured[pid] = m

    # Compose pair and stack from measured items
    f_star_items = [(pid, plans[pid]) for pid in sampled_pids]
    stack_plans, stack_orders = build_f_star_stacks(
        env, order, f_star_items, stack_ladder="2", stack_samples=1, pair_samples=1
    )
    assert len(stack_plans) >= 2
    for spid, spl in stack_plans.items():
        sm = measure(env, eval_samples, stack_orders[spid], spl)
        assert np.isfinite(sm["latency_ns"])
        assert np.isfinite(sm["quality"])
