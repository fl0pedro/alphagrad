"""THE PAIRED REFERENCE IS ONE PROGRAM PER GRAPH, WHATEVER THE CONTAINER.

dsnn-biw. A plan's classes on the carried-Jacobian face move the
approximation out of the graph and into the argument, and the measurement
compiles the program that reads that container. The paired reference used to
be the rev-exact elimination of THAT program, so a diag plan was divided by
the compact traces (0.9 MB of arguments) and a quant plan by the upcast ones:
e-prop read 88 times the memory of exact RTRL and a narrow carry read half.

The reference has to be the rev-exact elimination of the graph the policy
acted on -- the dense carry, the base arguments -- for every container.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import jax                                                     # noqa: E402
import jax.random as jrand                                     # noqa: E402
import numpy as np                                             # noqa: E402
import pytest                                                  # noqa: E402

import alphagrad.approx.env as envmod                          # noqa: E402
from alphagrad.approx.common import carry_plan as CP           # noqa: E402
from alphagrad.approx.common import compile_cache as _cc       # noqa: E402

T_PIN = 7
_ENV: dict = {}


@pytest.fixture(autouse=True)
def _the_paired_log_form(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_MEM_CHANNEL", "temp")
    monkeypatch.setenv("ALPHAGRAD_PAIRED_COST_FLOOR", "byte")
    monkeypatch.delenv("ALPHAGRAD_PLAN_LOG", raising=False)
    envmod.consume_plan_records()
    yield
    envmod.consume_plan_records()


def _rtrl_env():
    import alphagrad.approx.tools.landscape_map as lm
    hit = _ENV.get("rtrl")
    if hit is not None:
        CP.register(hit["args_ns"], hit["key"], "RSNN_SHD", "rtrl",
                    hit["env"].config, hit["env"].args, hit["env"].consts,
                    dataset=None, dataset_size=-1, step_position=T_PIN)
        return lm, hit["env"], hit["eval"]
    argv = ["--example", "RSNN_SHD", "--dataset", "none",
            "--temporal-rule", "rtrl", "--step-position", str(T_PIN),
            "--num-eval-samples", "1", "--num-data-points", "1",
            "--reps-per-point", "1", "--latency-inner-reps", "1",
            "--out-dir", "/tmp/paired_reference_container_test"]
    args = lm.make_argparser().parse_args(argv)
    env, eval_samples, _cj = lm.build_env(args)
    _key, args_key = jrand.split(jrand.PRNGKey(args.seed))
    _ENV["rtrl"] = {"env": env, "args_ns": args, "key": args_key,
                    "eval": eval_samples}
    return lm, env, eval_samples


def _carry_first_order(env):
    jx = env.config.jaxpr
    valid = sorted(int(v) for v in env.valid_vertices)
    mask = CP.carry_scope_mask(jx)
    return ([v for v in valid if mask[v - 1]]
            + sorted((v for v in valid if not mask[v - 1]), reverse=True))


def _carry_wires(lm, env, order, row=None, skip=False, slots=(0,)):
    jx = env.config.jaxpr
    mask = CP.carry_scope_mask(jx)
    inv = lm.face_inventory(env, np.asarray(order, dtype=np.int32))
    picked = [e for e in inv
              if mask[int(e["vertex"]) - 1]
              and jx.eqns[int(e["vertex"]) - 1].primitive.name == "dot_general"]
    assert picked, "the dense carry block contracts with a dot_general"
    if skip:
        return [{"k": int(e["k"]), "f": int(e["f"]), "kind": "SKIP"}
                for e in picked]
    return [{"k": int(e["k"]), "f": int(e["f"]), "slot": int(s),
             "row": list(row), "kind": "X"} for e in picked for s in slots]


def _plans(lm, env, order):
    from graphax.sparse.micro_actions import QUANT_DTYPES
    return {
        "exact": [],
        "diag": _carry_wires(lm, env, order, [0, 0, -1]),
        "quant": _carry_wires(lm, env, order,
                              [envmod.QUANT_SENTINEL,
                               QUANT_DTYPES.index("bfloat16"), 0],
                              slots=(0, 1)),
        "skip": _carry_wires(lm, env, order, skip=True),
    }


def test_every_container_is_divided_by_the_base_programs_reference(
        monkeypatch):
    lm, env, eval_samples = _rtrl_env()
    order = _carry_first_order(env)
    ref_ex: dict = {}
    real_cc = _cc.cached_compile

    def _spy(key, fn):
        out = real_cc(key, fn)
        if bytes(key).startswith(b"paired-ref:"):
            ref_ex[bytes(key)] = out
        return out
    monkeypatch.setattr(_cc, "cached_compile", _spy)

    got = {}
    for name, wires in _plans(lm, env, order).items():
        ref_ex.clear()
        plan = {"specs": None, "face_specs": None, "face_skips": None,
                "wires": wires}
        specs, faces, skips = lm.get_plan_arrays(plan, len(order))
        container = CP.container_for_plan(env.config, order, faces, skips,
                                          specs)
        assert container == name, (name, container)
        lm.measure(env, eval_samples, order, plan)
        recs = envmod.consume_plan_records()["paired_ref"]["records"]
        assert len(recs) == 1, (name, len(recs))
        assert len(ref_ex) == 1, (name, list(ref_ex))
        ex = next(iter(ref_ex.values()))
        ma = ex.memory_analysis()
        got[name] = {
            "key": next(iter(ref_ex)),
            "args": int(ma.argument_size_in_bytes),
            "temp": int(ma.temp_size_in_bytes),
            "order": list(recs[0]["order"]),
        }

    base = got["exact"]
    assert base["order"] == sorted(order, reverse=True)
    for name, g in got.items():
        # ONE executable, one vertex set, one argument tuple: the dense
        # carry of the graph the policy acted on, whatever this plan's
        # container made of it.
        assert g["key"] == base["key"], (name, got)
        assert g["args"] == base["args"], (name, got)
        assert g["temp"] == base["temp"], (name, got)
        assert g["order"] == base["order"], (name, got)


def test_the_candidate_is_still_the_containers_own_program(monkeypatch):
    """The fix moves the REFERENCE only. The candidate half of the pair is
    the variant's program, read on the variant's own eval samples."""
    lm, env, eval_samples = _rtrl_env()
    order = _carry_first_order(env)
    cand_ex: dict = {}
    real_cc = _cc.cached_compile

    def _spy(key, fn):
        out = real_cc(key, fn)
        if bytes(key).startswith(b"approx:"):
            cand_ex[bytes(key)] = out
        return out
    monkeypatch.setattr(_cc, "cached_compile", _spy)

    plans = _plans(lm, env, order)
    sizes = {}
    for name in ("exact", "diag"):
        cand_ex.clear()
        plan = {"specs": None, "face_specs": None, "face_skips": None,
                "wires": plans[name]}
        lm.measure(env, eval_samples, order, plan)
        assert len(cand_ex) == 1, (name, list(cand_ex))
        ma = next(iter(cand_ex.values())).memory_analysis()
        sizes[name] = int(ma.argument_size_in_bytes)
    var = CP.measurement_env("diag", env.config)
    dense = sum(int(np.asarray(a).nbytes) for a in env.args)
    compact = sum(int(np.asarray(a).nbytes) for a in var["args"])
    assert compact < dense / 50, (compact, dense)
    assert sizes["diag"] < sizes["exact"] / 50, sizes
