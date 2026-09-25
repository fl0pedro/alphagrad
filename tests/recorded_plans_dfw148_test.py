import hashlib
import json
import pathlib

import jax
import numpy as np
import pytest

import alphagrad.approx.env as E

DATA = pathlib.Path(__file__).resolve().parent / "recorded_plans"

TLM_SHA = "a03aadb62563788914a0fabeb1306bb1e8faee427d0dc3c86211cbb4027a9da9"
TLM_B16_SHA = "0248e9440115a62a3203a7a87a5c1bb55ea840a5853d91f5b1e88d53fe8c5d4d"
NN256_SHA = "94beb003fc52accd71f8bc0b7e1865cb913e007776a853c8e6124583aa831a26"

ORDER89 = [68, 70, 72, 87, 89, 80, 67, 94, 78, 66, 85, 11, 34, 73, 9, 22, 69,
           47, 31, 83, 84, 92, 76, 65, 21, 93, 90, 62, 77, 54, 6, 49, 29, 3,
           10, 81, 59, 12, 46, 44, 8, 1, 56, 86, 57, 61, 82, 17, 5, 7, 58, 50,
           19, 79, 88, 26, 91, 16, 20, 60, 45, 2, 15, 53, 39, 48, 36, 13, 37,
           4, 38, 74, 71, 28, 55, 63, 30, 64, 14, 75, 25, 18, 40, 51, 95, 27,
           23, 33, 52]
ORDER25 = [68, 70, 72, 87, 80, 94, 73, 92, 65, 93, 90, 54, 81, 44, 56, 86, 50,
           91, 45, 53, 55, 75, 51, 95, 52]
ORDER89_B16 = [71, 74, 77, 93, 95, 86, 70, 100, 84, 69, 91, 11, 37, 78, 9, 22,
               72, 50, 33, 89, 90, 98, 82, 68, 21, 99, 96, 65, 83, 57, 6, 52,
               29, 3, 10, 87, 62, 12, 49, 47, 8, 1, 59, 92, 60, 64, 88, 17, 5,
               7, 61, 53, 19, 85, 94, 26, 97, 16, 20, 63, 48, 2, 15, 56, 42,
               51, 39, 13, 40, 4, 41, 80, 76, 28, 58, 66, 31, 67, 14, 81, 25,
               18, 43, 54, 101, 27, 23, 35, 55]
ORDER25_B16 = [71, 74, 77, 93, 86, 100, 78, 98, 68, 99, 96, 57, 87, 47, 59, 92,
               53, 97, 48, 56, 58, 81, 54, 101, 55]

COMMON = [
    "--seed", "250197", "--measure-latency", "--latency-inner-reps", "50",
    "--num-data-points", "5", "--reps-per-point", "4",
    "--ref-num-data-points", "5", "--ref-reps-per-point", "32",
    "--measure-budget-secs", "1.0", "--measure-window-secs", "0.05",
    "--incremental-encode", "--cmp-type", "latency",
    "--mem-type", "peak_memory", "--terminal-rewards-only",
    "--rewards", "cmp", "mem", "acc", "--reward-mode", "lagrangian",
    "--quality-metric", "grad_cosine", "--approx-add", "lossless",
    "--fixed-order", "free", "--cost-form", "paired-log",
    "--reduce-axis-space", "physical", "--face-read", "last-row",
    "--set-pointer", "--face-actions", "--per-face-masks",
    "--unified-face-head", "--live-faces", "--dynamic-substeps",
    "--hidden-dim", "256", "--vocab-size", "256", "--num-layers", "3",
    "--tokenize-where", "local", "--face-wire-faces", "64",
    "--approx-profile", "all", "--wandb", "disabled",
]
ARGV_NN256 = ["--example", "NeuralNetwork", "--dataset", "mnist",
              "--mem-channel", "watermark"] + COMMON
ARGV_TLM = ["--example", "TransformerLM", "--dataset", "wikitext2",
            "--mem-channel", "temp"] + COMMON

QUANT = -3


@pytest.fixture(autouse=True)
def _restore_width(monkeypatch):
    monkeypatch.delenv("ALPHAGRAD_MAX_FACES", raising=False)
    keep = E.MAX_FACES
    yield
    E.MAX_FACES = keep
    E._LIVE_CHAINS.clear()


@pytest.fixture
def tlm_dims(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_TLM_SEQ", "32")
    monkeypatch.setenv("ALPHAGRAD_TLM_DMODEL", "128")
    monkeypatch.setenv("ALPHAGRAD_TLM_VOCAB", "1024")


@pytest.fixture
def nn256(monkeypatch):
    from alphagrad.approx.common import examples
    monkeypatch.setattr(examples, "_EQ_NN_HIDDEN", 256)


@pytest.fixture
def census(monkeypatch):
    import graphax.core as gc
    rec = {"stores": 0, "queued": 0, "bad": []}
    orig = gc._set_inner

    def counted(outer, k1, k2, v, is_transpose=False):
        if (not is_transpose and hasattr(v, "shape")
                and hasattr(k1, "aval") and hasattr(k2, "aval")):
            rec["stores"] += 1
            queue = (tuple(getattr(v, "pre_transforms", ()) or ())
                     + tuple(getattr(v, "post_transforms", ()) or ()))
            nominal = tuple(k2.aval.shape) + tuple(k1.aval.shape)
            if queue:
                rec["queued"] += 1
                got = tuple(int(s) for s in
                            gc._drain_transforms(v.copy()).shape)
            else:
                got = tuple(int(s) for s in v.shape)
            if got != nominal:
                rec["bad"].append((str(k2), str(k1), nominal, got))
        return orig(outer, k1, k2, v, is_transpose=is_transpose)

    monkeypatch.setattr(gc, "_set_inner", counted)
    return rec


@pytest.fixture
def faces_seen(monkeypatch):
    import graphax
    seen = {"max": 0}
    orig = graphax.faces_of

    def counted(*a, **kw):
        keys = orig(*a, **kw)
        seen["max"] = max(seen["max"], len(keys))
        return keys

    monkeypatch.setattr(graphax, "faces_of", counted)
    return seen


def _sha(jaxpr):
    src = {}
    for i, v in enumerate(jaxpr.invars):
        src[id(v)] = f"in{i}"
    for i, v in enumerate(jaxpr.constvars):
        src[id(v)] = f"c{i}"
    keep = ("permutation", "dimension_numbers", "axes", "shape",
            "broadcast_dimensions", "new_dtype", "dimensions", "new_sizes")
    rows = []
    for k, e in enumerate(jaxpr.eqns, start=1):
        ins = ["lit" if type(a).__name__ == "Literal" else src.get(id(a), "?")
               for a in e.invars]
        outs = []
        for o in e.outvars:
            src[id(o)] = f"v{k}"
            outs.append([list(getattr(o.aval, "shape", ())),
                         str(getattr(o.aval, "dtype", ""))])
        params = {p: str(e.params[p]) for p in keep if p in e.params}
        rows.append([k, str(e.primitive), ins, outs, params])
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode()).hexdigest()


def _graph(example):
    from graphax import inline_call_primitives
    from alphagrad.approx.common.examples import (get_args, get_fn,
                                                  infer_argnums)
    xs = get_args(example, jax.random.PRNGKey(42), dataset="wikitext2")
    closed = jax.make_jaxpr(get_fn(example))(*xs)
    jaxpr, cst = inline_call_primitives(closed.jaxpr, list(closed.literals))
    return jaxpr, list(cst), list(xs), tuple(infer_argnums(example))


def _tokenize(jaxpr, cst, xs, argnums, order):
    from graphax import IncrementalPathTokenizer
    from alphagrad.approx.common.token_vocab import incr_token_vocab
    tk = IncrementalPathTokenizer(jaxpr, argnums, list(cst), list(xs),
                                  vocab_size=incr_token_vocab())
    list(tk.base_tokens())
    for v in order:
        list(tk.eliminate(int(v), (), None))


def _actor_env(argv):
    from alphagrad.approx import ppo
    from alphagrad.approx.cpu_approx_worker import _build_env_from_args
    ns, unknown = ppo.make_argparser().parse_known_args(argv)
    assert unknown == [], unknown
    d = dict(vars(ns))
    d["exec_on_gpu"] = False
    return _build_env_from_args(d, None, seed=int(d["seed"]))


def _records(name):
    with open(DATA / name) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _wires_without_quant(rec):
    from alphagrad.approx.common import plan_log
    order, rule_specs, fs, fk = plan_log.decode_wires(rec)
    fs = np.array(fs, copy=True)
    fs[fs[..., 0] == QUANT] = (-1, -1, 0)
    return [int(v) for v in order], rule_specs, fs, fk


def _leaves(tree):
    from graphax.sparse.tensor import SparseTensor
    return jax.tree_util.tree_leaves(
        tree, is_leaf=lambda x: x is None or isinstance(x, SparseTensor))


def _dense(x):
    from graphax.sparse.tensor import SparseTensor
    return np.asarray(x.dense() if isinstance(x, SparseTensor) else x,
                      dtype=np.float64)


def _measure(env, order, rule_specs, fs, fk):
    cfg = env.config
    specs = [np.asarray(rule_specs[k]) for k in range(len(order))]
    have = bool(np.any(fk == 1) or np.any(fs[..., 0] != -1))
    ft = (E._face_transforms_for_order(cfg, list(env.consts), list(env.args),
                                       order, specs, fs, fk)
          if have else None)
    transforms, _ = E._decode_vertex_transforms(cfg, order, specs)
    fn = E.measured_program(cfg, order, list(env.consts),
                            transforms=transforms, face_transforms=ft)
    out = jax.jit(fn, keep_unused=True).lower(*env.args).compile()(*env.args)
    ref = jax.jacrev(cfg.target_fun, argnums=tuple(cfg.argnums))(*env.args)
    return _leaves(out), _leaves(ref)


@pytest.mark.parametrize("order", [ORDER89, ORDER25],
                         ids=["order89", "order25"])
def test_dfw10_the_recorded_tlm_order_stores_every_edge_nominal(
        tlm_dims, census, order):
    jaxpr, cst, xs, argnums = _graph("TransformerLM")
    assert _sha(jaxpr) == TLM_SHA, (
        "the unbatched TransformerLM graph changed since job 66745 recorded "
        "this order, so its vertex ids name other equations now")
    _tokenize(jaxpr, cst, xs, argnums, order)
    assert census["queued"] > 0
    assert census["bad"] == []


@pytest.mark.parametrize("order", [ORDER89_B16, ORDER25_B16],
                         ids=["order89", "order25"])
def test_dfw10_the_recorded_order_on_the_batched_tlm_stores_every_edge_nominal(
        tlm_dims, census, order):
    from alphagrad.approx.common import datasets
    assert datasets.NN_VMAP_BATCH == 16, "the batched fingerprint is B=16"
    jaxpr, cst, xs, argnums = _graph("VmappedTransformerLM")
    assert _sha(jaxpr) == TLM_B16_SHA, (
        "VmappedTransformerLM changed; the translated order no longer names "
        "the equations of the recorded one")
    _tokenize(jaxpr, cst, xs, argnums, order)
    assert census["queued"] > 0
    assert census["bad"] == []


@pytest.mark.parametrize("line", range(5))
def test_dfw70_the_recorded_nn256_plan_without_its_quant_rows_is_nominal(
        nn256, census, line):
    rec = _records("dfw70_66769_refused.jsonl")[line]
    assert rec["refused"].startswith("raised:")
    env = _actor_env(ARGV_NN256)
    assert _sha(env.config.jaxpr) == NN256_SHA, (
        "the NN256 graph changed since job 66769 recorded this plan")
    order, rule_specs, fs, fk = _wires_without_quant(rec)
    got, want = _measure(env, order, rule_specs, fs, fk)
    assert [tuple(np.shape(g)) for g in got] == [
        tuple(np.shape(w)) for w in want]
    assert census["bad"] == []

    exact_fs = np.full_like(fs, -1)
    exact_fs[..., 2] = 0
    got, want = _measure(env, order, rule_specs, exact_fs, np.zeros_like(fk))
    num = sum(float(np.sum((_dense(g) - _dense(w)) ** 2))
              for g, w in zip(got, want))
    den = sum(float(np.sum(_dense(w) ** 2)) for w in want)
    assert (num / den) ** 0.5 < 1e-5


def test_dfw36_the_actor_face_width_admits_the_recorded_tlm_plans(
        tlm_dims, faces_seen):
    E.MAX_FACES = 16
    env = _actor_env(ARGV_TLM)
    assert _sha(env.config.jaxpr) == TLM_SHA, (
        "the TransformerLM graph changed since jobs 66314 and 66466 recorded "
        "these plans")
    bound = int(E.derived_max_faces(env.config.jaxpr, env.config.argnums,
                                    list(env.consts), list(env.args)))
    assert bound > 16
    assert E.MAX_FACES == bound
    recs = (_records("dfw36_66466_refused.jsonl")
            + _records("dfw36_66314_refused.jsonl"))
    assert len(recs) == 8
    for rec in recs:
        order, rule_specs, fs, fk = _wires_without_quant(rec)
        specs = [np.asarray(rule_specs[k]) for k in range(len(order))]
        faces_seen["max"] = 0
        E._LIVE_CHAINS.clear()
        E._face_transforms_for_order(env.config, list(env.consts),
                                     list(env.args), order, specs, fs, fk)
        assert faces_seen["max"] > 16


@pytest.mark.parametrize("argv", [
    ["--example", "VmappedNeuralNetwork", "--dataset", "mnist"],
    ["--example", "VmappedTransformerLM", "--dataset", "wikitext2"],
    ["--example", "VmappedRSNN_SHD", "--dataset", "none",
     "--temporal-rule", "bptt"],
], ids=["nn256", "tlm", "rsnn-bptt"])
def test_dfw36_the_actor_derives_the_face_width_on_the_batched_targets(
        tlm_dims, nn256, monkeypatch, argv):
    from alphagrad.approx.common import rsnn_shd as R
    monkeypatch.setattr(R, "RSNN_HIDDEN", 6)
    E.MAX_FACES = 16
    env = _actor_env(argv + ["--face-actions", "--wandb", "disabled"])
    bound = int(E.derived_max_faces(env.config.jaxpr, env.config.argnums,
                                    list(env.consts), list(env.args)))
    assert bound != 16, "this graph cannot tell its bound from the default"
    assert E.MAX_FACES == bound


def _landscape_env(example, extra=()):
    import alphagrad.approx.tools.landscape_map as lm
    argv = ["--example", example, "--dataset", "none", "--num-eval-samples",
            "1", "--num-data-points", "1", "--reps-per-point", "1",
            "--out-dir", "/tmp/dfw148_test"] + list(extra)
    env, _samples, _cj = lm.build_env(lm.make_argparser().parse_args(argv))
    return env


def _jaxpr_the_measurement_walks(env, order, monkeypatch):
    import alphagrad.approx.tools.landscape_map as lm
    import graphax.core as gxcore
    seen = {}

    def stub(jaxpr, *a, **kw):
        seen.setdefault("jaxpr", jaxpr)
        raise ValueError("stubbed elimination (test): jaxpr recorded")

    plan = {"specs": None, "face_specs": None, "face_skips": None,
            "n_faces_approx": 0, "n_slot_rows": 0, "total_live_faces": 0,
            "per_vertex_faces": [], "wires": [], "op": "identity",
            "budget": "identity"}
    with monkeypatch.context() as m:
        m.setattr(gxcore, "vertex_elimination_jaxpr", stub)
        try:
            lm.measure(env, env.eval_args_samples, list(order), plan)
        except ValueError:
            pass
    return seen.get("jaxpr")


@pytest.mark.parametrize("example,extra", [
    ("VmappedNeuralNetwork", ()),
    ("VmappedTransformerLM", ("--hidden-dim", "16", "--vocab-size", "16",
                              "--num-layers", "1")),
    ("VmappedRSNN_SHD", ("--temporal-rule", "tbptt", "--step-position", "7")),
    ("VmappedRSNN_SHD", ("--temporal-rule", "bptt", "--step-position", "7")),
    ("VmappedRSNN_SHD", ("--temporal-rule", "rtrl", "--step-position", "7")),
], ids=["nn", "tlm", "rsnn-tbptt", "rsnn-bptt", "rsnn-rtrl"])
def test_dfw24_the_measurement_walks_the_env_jaxpr_on_the_batched_targets(
        tlm_dims, monkeypatch, example, extra):
    from alphagrad.approx.common import rsnn_shd as R
    monkeypatch.setattr(R, "RSNN_HIDDEN", 6)
    env = _landscape_env(example, extra)
    order = [int(v) for v in sorted(env.valid_vertices, reverse=True)]
    walked = _jaxpr_the_measurement_walks(env, order, monkeypatch)
    assert walked is not None, f"{example}: the measurement never reached jacve"
    assert walked is env.config.jaxpr, (
        f"{example}: the measurement eliminates {len(walked.eqns)} equations "
        f"and the environment numbered {len(env.config.jaxpr.eqns)}")
