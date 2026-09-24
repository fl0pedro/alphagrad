from __future__ import annotations

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")

import equinox as eqx                                           # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import jax.random as jrand                                      # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

import alphagrad.approx.env as envmod                           # noqa: E402
import graphax.core as gxc                                      # noqa: E402
from graphax.sparse.micro_actions import Quant                  # noqa: E402
from alphagrad.approx.common import carry_plan as CP            # noqa: E402
from alphagrad.approx.common import masks as M                  # noqa: E402
from alphagrad.approx.common.face_driver import (               # noqa: E402
    build_live_face_stream)
from alphagrad.approx.heads import (                            # noqa: E402
    AXIS_TAG_BITS, MAX_PRIMES, AxisTokenFeatures, MicroAction, OP_END,
    OP_QUANT, precompute_factor_tables, quant_hardware_masks)
from alphagrad.approx.unified_face_head import O_QUANT, QUANT_SLOTS  # noqa: E402
from alphagrad.approx.unified_face_policy import (              # noqa: E402
    UnifiedFacePolicy, _NARROW_SLOT)

T_PIN = 7
QROW = (int(envmod.QUANT_SENTINEL), int(_NARROW_SLOT), 1)
_CACHE: dict = {}


@pytest.fixture(autouse=True)
def _campaign_flags(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", "lossless")
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.delenv("ALPHAGRAD_PLAN_LOG", raising=False)
    for name in ("ALPHAGRAD_PER_FACE_MASKS", "ALPHAGRAD_PER_FACE_REPAIR_AXIS"):
        monkeypatch.setenv(name, os.environ.get(name, "0"))
    flags = (M._PER_FACE_MASKS[0], M._PER_FACE_REPAIR_AXIS[0])
    M.set_per_face_masks(True)
    yield
    M._PER_FACE_MASKS[0], M._PER_FACE_REPAIR_AXIS[0] = flags


def _rtrl():
    import alphagrad.approx.tools.landscape_map as lm
    hit = _CACHE.get("rtrl")
    if hit is None:
        ns = lm.make_argparser().parse_args(
            ["--example", "RSNN_SHD", "--dataset", "none",
             "--temporal-rule", "rtrl", "--step-position", str(T_PIN),
             "--num-eval-samples", "1", "--num-data-points", "1",
             "--reps-per-point", "1", "--latency-inner-reps", "1",
             "--out-dir", "/tmp/face_quant_identity_cast_test"])
        env, eval_samples, _cj = lm.build_env(ns)
        _k, key = jrand.split(jrand.PRNGKey(ns.seed))
        cfg = env.config
        order = sorted((int(v) for v in env.valid_vertices), reverse=True)
        stream = build_live_face_stream(
            cfg.jaxpr, tuple(cfg.argnums), list(env.consts), list(env.args),
            max_faces=envmod.MAX_FACES, max_axes=envmod.MAX_AXES_PER_VERTEX,
            window=envmod.MAX_DELTA_TOKENS, cache=64)
        inv = lm.face_inventory(env, np.asarray(order, np.int32),
                                capture_tensors=True)
        hit = _CACHE["rtrl"] = dict(lm=lm, env=env, ns=ns, key=key,
                                    eval=eval_samples, stream=stream,
                                    order=order, inv=inv)
    env = hit["env"]
    CP.register(hit["ns"], hit["key"], "RSNN_SHD", "rtrl", env.config,
                env.args, env.consts, dataset=None, dataset_size=-1,
                step_position=T_PIN)
    return hit


def _kind(st):
    if st is None:
        return "absent"
    val = getattr(st, "val", None)
    return "noval" if val is None else str(val.dtype)


def _carried(env, v):
    return bool(CP.carry_scope_mask(env.config.jaxpr)[int(v) - 1])


def _exact_specs(T):
    specs = np.full((T, envmod.MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[..., 2] = 0
    return specs


def _exact_face(c, kinds, carried=None):
    for e in c["inv"]:
        t = e["tensors"]
        got = {_kind(t.get(0)), _kind(t.get(1))}
        if got == set(kinds) and carried in (None, _carried(c["env"],
                                                            e["vertex"])):
            return e["k"], e["f"], e["vertex"], tuple(e["key"])
    raise AssertionError(f"no {kinds} face with carried={carried} on the "
                         f"reverse order of the rtrl graph")


def _prefix_wires(c, n):
    return [{"k": e["k"], "f": e["f"], "slot": s, "row": list(QROW),
             "kind": "X"}
            for e in c["inv"] if e["k"] < n for s in QUANT_SLOTS]


def _quantized_prefix_faces(c):
    if "qprefix" not in c:
        c["qprefix"] = _scan_quantized_prefix(c)
    return c["qprefix"]


def _scan_quantized_prefix(c):
    lm, stream, order = c["lm"], c["stream"], c["order"]
    T = len(order)
    oa = np.asarray(order, np.int32)
    specs = _exact_specs(T)
    _s, frh, fsh = lm.get_plan_arrays(
        {"specs": None, "face_specs": None, "face_skips": None,
         "wires": _prefix_wires(c, T)}, T)
    found = {}
    for n in range(1, T):
        v = int(order[n])
        tk = stream._tokenizer_at(oa, specs, n, frh, fsh)
        keys = list(tk.ij.faces(v))
        src = stream._probe_faces(tk, v, keys, slots=True, stat="slot")
        for f, key in enumerate(keys):
            by = src.get(key) or {}
            ks = (_kind(by.get("lhs")), _kind(by.get("rhs")))
            if "absent" in ks or "bfloat16" not in ks:
                continue
            cls = "one_narrow" if "float32" in ks else "no_cast"
            found.setdefault(cls, (n, f, v, tuple(key), ks, by))
        if len(found) == 2:
            break
    return found, frh, fsh


def _slot_arrays(c, n, v, frh=None, fsh=None):
    order = c["order"]
    oa = np.asarray(order, np.int32)
    specs = _exact_specs(len(order))
    sizes, quant, pair, comp, _nout, nf = c["stream"].face_slot_legality(
        oa, specs, n, v, frh, fsh)
    dec = c["stream"].vertex_face_decisions(
        oa, specs, n, v, lambda f, s, L: None, face_rows_hist=frh,
        face_skips_hist=fsh)
    for s in QUANT_SLOTS:
        np.testing.assert_array_equal(dec.quant[:, s], quant[:, s])
    return (sizes, quant, pair, comp), int(nf)


def _policy(n_faces):
    pol = UnifiedFacePolicy(32, num_heads=2, max_faces=n_faces,
                            key=jrand.PRNGKey(0), approx_add="lossless")
    bias = pol.head.proj.layers[-1].bias
    return eqx.tree_at(lambda p: p.head.proj.layers[-1].bias, pol,
                       bias.at[0].add(-30.0).at[O_QUANT].add(30.0))


def _features():
    n = envmod.MAX_AXES_PER_VERTEX
    return AxisTokenFeatures(
        size=jnp.ones((n,), jnp.int32), log_size=jnp.zeros((n,), jnp.float32),
        tag_bits=jnp.zeros((n, AXIS_TAG_BITS), jnp.float32),
        group_id=-jnp.ones((n,), jnp.int32),
        valid_mask=jnp.ones((n,), jnp.float32))


def _bit_legal(arrays, f, n_faces):
    pol = _policy(n_faces)
    sizes, quant, pair, comp = (jnp.asarray(a[f]) for a in arrays)
    ff, pv, cv, qlm = pol._slot_inputs(_features(), pair, comp,
                                       quant_hardware_masks()[0], sizes,
                                       quant)
    return float(pol._face_masks(ff, pv, cv, qlm, None,
                                 precompute_factor_tables(64))[5])


def _drawn_rows(arrays, n_faces):
    from alphagrad.approx.ppo import Agent
    pol = _policy(n_faces)
    sizes, quant, pair, comp = (jnp.asarray(a[:n_faces]) for a in arrays)
    fa, *_ = pol.sample(None, _features(), precompute_factor_tables(64),
                        jrand.PRNGKey(1), pair, comp,
                        jnp.ones((n_faces,), jnp.float32),
                        face_sizes=sizes, face_quant=quant)
    zero = jnp.zeros((1,), jnp.int32)
    act = MicroAction(
        op_type=jnp.full((1,), OP_END, jnp.int32), i=zero, j=zero,
        exponents=jnp.zeros((1, MAX_PRIMES), jnp.int32), factor=zero,
        compress_kind=zero, quant_dtype=zero,
        quant_scale_sign=jnp.ones((1,), jnp.int32),
        quant_scale_frac=jnp.zeros((1,), jnp.float32))
    ax = jnp.zeros((2, envmod.MAX_AXES_PER_VERTEX, envmod.AXIS_FEATURE_DIM),
                   jnp.int32)
    sa = Agent.to_env_action_dynamic(None, 0, act, ax, face_action=fa)
    return fa, np.asarray(sa.face_rows)


def _measure(c, wires, monkeypatch):
    seen = []
    real = gxc._check_face_quant

    def _spy(lq, rq, v, key):
        seen.append((int(v), tuple(int(x) for x in key), lq, rq))
        return real(lq, rq, v, key)

    monkeypatch.setattr(gxc, "_check_face_quant", _spy)
    plan = {"specs": None, "face_specs": None, "face_skips": None,
            "wires": wires}
    out = c["lm"].measure(c["env"], c["eval"], list(c["order"]), plan)
    monkeypatch.setattr(gxc, "_check_face_quant", real)
    return out, seen, c["lm"].get_plan_arrays(plan, len(c["order"]))


def _offered_drawn_and_measured(c, n, f, v, key, arrays, n_faces, prefix,
                                monkeypatch):
    sizes, quant = arrays[0], arrays[1]
    casts = [bool(quant[f, s, M.FACE_QUANT_NARROW] > 0.5)
             for s in QUANT_SLOTS]
    assert sorted(casts) == [False, True], (n, f, v, quant[f])
    assert _bit_legal(arrays, f, n_faces) == 1.0, (n, f, v, quant[f])
    fa, rows = _drawn_rows(arrays, n_faces)
    assert int(fa.quant[f]) == 1 and int(fa.skip[f]) == 0, (n, f, fa.quant)
    for s in QUANT_SLOTS:
        assert int(fa.op_type[f, s]) == OP_QUANT
        assert tuple(int(x) for x in rows[f, s]) == QROW, rows[f]
    envmod.check_face_quant_rows(rows[f], where=f"face {f} of vertex {v}")
    wires = prefix + [{"k": n, "f": f, "slot": s,
                       "row": [int(x) for x in rows[f, s]], "kind": "X"}
                      for s in QUANT_SLOTS]
    out, seen, (specs, fs, sk) = _measure(c, wires, monkeypatch)
    assert out["refused"] == "", (out["refused"], out["refusal"])
    assert json.loads(out["applied_detail"]).get("applied_quant", 0) >= 1, (
        out["applied_detail"])
    at = [(lq, rq) for vv, kk, lq, rq in seen if (vv, kk) == (v, key)]
    assert at, f"vertex {v} never contracted face {key}"
    for lq, rq in at:
        assert isinstance(lq, Quant) and isinstance(rq, Quant), (lq, rq)
        assert str(lq.dtype) == str(rq.dtype) == "bfloat16", (lq, rq)
    return CP.container_for_plan(c["env"].config, list(c["order"]), fs, sk,
                                 specs)


def test_a_body_face_with_a_value_free_operand_is_offered_and_measured(
        monkeypatch):
    c = _rtrl()
    n, f, v, key = _exact_face(c, ("float32", "noval"), carried=False)
    arrays, nf = _slot_arrays(c, n, v)
    container = _offered_drawn_and_measured(c, n, f, v, key, arrays, nf, [],
                                            monkeypatch)
    assert container == "exact", container


def test_a_face_with_an_operand_already_narrow_is_offered_and_measured(
        monkeypatch):
    c = _rtrl()
    found, frh, fsh = _quantized_prefix_faces(c)
    assert "one_narrow" in found, sorted(found)
    n, f, v, key, _ks, _by = found["one_narrow"]
    arrays, nf = _slot_arrays(c, n, v, frh, fsh)
    _offered_drawn_and_measured(c, n, f, v, key, arrays, nf,
                                _prefix_wires(c, n), monkeypatch)


def test_a_carried_face_with_a_value_free_operand_takes_the_quant_container(
        monkeypatch):
    c = _rtrl()
    n, f, v, key = _exact_face(c, ("float32", "noval"), carried=True)
    arrays, nf = _slot_arrays(c, n, v)
    container = _offered_drawn_and_measured(c, n, f, v, key, arrays, nf, [],
                                            monkeypatch)
    assert container == "quant", container


def test_a_face_where_no_operand_casts_is_not_offered_the_bit():
    c = _rtrl()
    n, f, v, _key = _exact_face(c, ("noval",))
    arrays, nf = _slot_arrays(c, n, v)
    q = arrays[1][f, list(QUANT_SLOTS), M.FACE_QUANT_NARROW]
    assert not q.any(), arrays[1][f]
    assert _bit_legal(arrays, f, nf) == 0.0
    found, frh, fsh = _quantized_prefix_faces(c)
    assert "no_cast" in found, sorted(found)
    n, f, v, _key, ks, _by = found["no_cast"]
    arrays, nf = _slot_arrays(c, n, v, frh, fsh)
    q = arrays[1][f, list(QUANT_SLOTS), M.FACE_QUANT_NARROW]
    assert not q.any(), (ks, arrays[1][f])
    assert _bit_legal(arrays, f, nf) == 0.0, ks


def test_the_slot_hook_holds_the_narrow_quant_on_every_operand_kind():
    c = _rtrl()
    tensors = {}
    for e in c["inv"]:
        for s in QUANT_SLOTS:
            st = e["tensors"].get(s)
            tensors.setdefault(_kind(st), st)
    found, _frh, _fsh = _quantized_prefix_faces(c)
    for _n, _f, _v, _key, _ks, by in found.values():
        for site in ("lhs", "rhs"):
            tensors.setdefault(_kind(by.get(site)), by.get(site))
    assert {"float32", "bfloat16", "noval"} <= set(tensors), sorted(tensors)
    for kind in ("float32", "bfloat16", "noval"):
        got = envmod.make_slot_frame_hook(QROW)(tensors[kind])
        assert isinstance(got, Quant) and str(got.dtype) == "bfloat16", (
            kind, got)
