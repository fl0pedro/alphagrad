# dsnn-cl0, owner ruling 2026-09-24 Q51: since dsnn-dkz every measured program
# returns the dense form, so Oracle B measured the dense program twice. It now
# builds the plan's gradient with jacve(..., sparse_representation=True),
# executes it, densifies it and compares it with the dense program's gradient,
# outside the measurement and its memory checks: the gradients to float
# rounding, the two qualities to 1e-3.
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import jax                                                      # noqa: E402
import pytest                                                   # noqa: E402

from alphagrad.approx import env as env_mod                     # noqa: E402


@pytest.fixture
def lm(monkeypatch):
    # The sweep rewrites the face width and exports its knobs at import; both
    # are put back so no later module of this worker inherits them.
    saved_env = dict(os.environ)
    monkeypatch.setattr(env_mod, "MAX_FACES", env_mod.MAX_FACES)
    import alphagrad.approx.tools.landscape_map as lm
    try:
        yield lm
    finally:
        env_mod.set_measure_timeout_s(None)
        env_mod.consume_refused_counts()
        env_mod.consume_plan_records()
        for key in set(os.environ) - set(saved_env):
            del os.environ[key]
        os.environ.update(saved_env)


def _sweep(lm, tmp_path):
    args = lm.make_argparser().parse_args([
        "--example", "Helmholtz", "--dataset", "none",
        "--out-dir", str(tmp_path), "--order", "reverse",
        "--num-data-points", "2", "--num-eval-samples", "2"])
    env, eval_samples, _cj = lm.build_env(args)
    order = lm.rev_order(env)
    plans = {}
    for op, budget in (("quant", "all"), ("diag", "all"),
                       ("compress", "all"), ("skip", 1)):
        pl = lm.build_ladder_plan(env, order, op, budget, args)
        pl["op"], pl["budget"] = op, str(budget)
        plans[f"{op}@{budget}"] = pl
    return env, eval_samples, order, plans


@pytest.mark.parametrize("metric", ["grad_cosine", "jac_cosine"])
def test_oracle_b_checks_the_densified_sparse_gradient_outside_the_measurement(
        lm, monkeypatch, tmp_path, metric):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", metric)
    env, eval_samples, order, plans = _sweep(lm, tmp_path)
    built = []
    real_jacve = lm.jacve

    def spy(*a, **kw):
        built.append(bool(kw.get("sparse_representation")))
        return real_jacve(*a, **kw)

    measured = []
    real_callback = env_mod._callback

    def no_measurement(*a, **kw):
        measured.append(1)
        return real_callback(*a, **kw)

    monkeypatch.setattr(lm, "jacve", spy)
    monkeypatch.setattr(env_mod, "_callback", no_measurement)
    res = lm.run_oracle_b(env, eval_samples, order, plans)
    assert measured == [], "Oracle B runs outside the measurement"
    assert sorted(r["op"] for r in res.values()) == [
        "compress", "diag", "quant", "skip"]
    assert built.count(True) == 4, "one sparse program per class"
    for pid, r in res.items():
        print(f"[oracle-b-test] {metric} {pid}: grad_rel_l2 "
              f"{r['grad_rel_l2']:.3e} bar {r['grad_bar']:.3e} step "
              f"{r['rounding_step']:.3e} quality {r['dense_quality']} vs "
              f"{r['sparse_quality']} sparse_leaves {r['sparse_leaves']}")
        assert r["sparse_leaves"] > 0
        assert r["points"] == 2
        assert r["grad_rel_l2"] <= r["grad_bar"]
        assert r["diff"] is not None and r["diff"] < 1e-3
    quant = next(r for r in res.values() if r["op"] == "quant")
    assert quant["rounding_step"] == float(
        jax.numpy.finfo(jax.numpy.bfloat16).eps)


def test_oracle_b_fails_when_the_sparse_gradient_is_not_the_dense_one(
        lm, monkeypatch, tmp_path):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "grad_cosine")
    env, eval_samples, order, plans = _sweep(lm, tmp_path)
    real = lm._densify

    def off(out, like):
        dense, n = real(out, like)
        return jax.tree_util.tree_map(lambda x: x * 1.05, dense), n

    monkeypatch.setattr(lm, "_densify", off)
    with pytest.raises(AssertionError, match="relative L2"):
        lm.run_oracle_b(env, eval_samples, order, plans)
