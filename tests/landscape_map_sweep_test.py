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


def test_build_singleton_sweep_plans_avoids_implicit_noop(helmholtz_setup):
    """build_singleton_sweep_plans uses slot_legality.comp so reduce plans
    never target implicit dimensions that collapse to no-ops (dsnn-3qm.73)."""
    from alphagrad.approx.env import slot_rules_for_row
    env, _, order, inv = helmholtz_setup
    plans, _ = build_singleton_sweep_plans(env, order, inv)
    for pid, p in plans.items():
        if p["op"] == "compress":
            w = p["wires"][0]
            k, f, s = int(w["k"]), int(w["f"]), int(w["slot"])
            entry = [e for e in inv if int(e["k"]) == k and int(e["f"]) == f][0]
            st = entry["tensors"][s]
            rules = slot_rules_for_row(st, w["row"])
            assert rules, f"Plan {pid} decoded to empty rules on its live tensor"
            for r in rules:
                assert r.axes, f"Plan {pid} decoded to Compress with empty axes on its live tensor"



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


def test_shard_partitioning(helmholtz_setup):
    env, _, order, inv = helmholtz_setup
    plans, plan_orders = build_singleton_sweep_plans(env, order, inv)
    plans["identity"] = {"op": "identity", "budget": "0", "wires": []}
    
    num_shards = 3
    shards = []
    for shard_idx in range(num_shards):
        ident = plans.get("identity")
        non_ident = [(pid, plans[pid]) for pid in plans if pid != "identity"]
        import math
        chunk_size = math.ceil(len(non_ident) / num_shards)
        shard_items = non_ident[shard_idx * chunk_size : (shard_idx + 1) * chunk_size]
        s_plans = {}
        if ident is not None:
            s_plans["identity"] = ident
        s_plans.update(shard_items)
        shards.append(s_plans)

    # Every shard must have identity
    for s in shards:
        assert "identity" in s

    # Union of all non-identity plans must equal original non-identity plans
    all_sharded_non_ident = set()
    for s in shards:
        shard_non_ident = set(k for k in s if k != "identity")
        # No overlap between shards
        assert not (all_sharded_non_ident & shard_non_ident)
        all_sharded_non_ident.update(shard_non_ident)

    orig_non_ident = set(k for k in plans if k != "identity")
    assert all_sharded_non_ident == orig_non_ident


def test_ref_order_defaults_to_reverse():
    """The default reference order is rev_order (ticket dsnn-3qm.63, owner Q8/Q19)."""
    parser = make_argparser()
    args = parser.parse_args(["--singleton-sweep"])
    assert args.order == "markowitz"
    assert args.ref_order == "reverse"


def test_oracle_b_sparse_vs_dense(helmholtz_setup):
    """Oracle B: verify sparse vs dense representation agreement on sampled plans (dsnn-3qm.63)."""
    from alphagrad.approx.tools.landscape_map import run_oracle_b
    env, eval_samples, order, inv = helmholtz_setup
    plans, _ = build_singleton_sweep_plans(env, order, inv)
    results = run_oracle_b(env, eval_samples, order, plans)
    assert len(results) == 4
    for pid, res in results.items():
        assert res["diff"] < 1e-3, f"Oracle B failed for {pid}: diff={res['diff']}"



def test_singleton_sweep_enumerates_one_quant_per_dtype(helmholtz_setup):
    """``--quant-dtypes`` adds one Quant singleton per legal slot per dtype;
    the bfloat16 ids keep their ``:bf16`` suffix so rows written before
    2026-09-13 still match; ``--singleton-ops quant`` yields quants only."""
    import pytest
    from alphagrad.approx.tools import landscape_map as LM
    env, eval_samples, order, inv = helmholtz_setup
    base, _ = LM.build_singleton_sweep_plans(env, order, inv)
    multi, _ = LM.build_singleton_sweep_plans(
        env, order, inv, quant_dtypes=("bfloat16", "float8_e5m2", "int8"))
    bf16 = [k for k in base if k.startswith("singleton:quant:")]
    assert bf16 and all(k.endswith(":bf16") for k in bf16)
    assert set(bf16) <= set(multi)
    e5 = [k for k in multi if k.endswith(":float8_e5m2")]
    i8 = [k for k in multi if k.endswith(":int8")]
    assert len(e5) == len(bf16) and len(i8) == len(bf16), (len(bf16), len(e5), len(i8))
    non_quant = {k for k in base if not k.startswith("singleton:quant:")}
    assert non_quant == {k for k in multi if not k.startswith("singleton:quant:")}
    only_q, _ = LM.build_singleton_sweep_plans(
        env, order, inv, quant_dtypes=("int8",), ops=("quant",))
    assert only_q and all(k.startswith("singleton:quant:") for k in only_q)
    with pytest.raises(ValueError):
        LM.build_singleton_sweep_plans(env, order, inv, ops=("quant", "bogus"))
    with pytest.raises(ValueError):
        LM.build_singleton_sweep_plans(env, order, inv, quant_dtypes=("float99",))
