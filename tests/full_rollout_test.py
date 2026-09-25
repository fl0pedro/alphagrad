"""The full rollout on the recurrent SHD target (owner ruling 2026-09-25 Q24 a).

The measured program runs the plan's step at every step of each recording, as
one scan, and the plan produces its own given value from zero at the first
step. Exactness claims run in a float64 subprocess, the pattern
``temporal_rule_test.py`` uses.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest


def _run(src, x64=True):
    env = dict(os.environ)
    if x64:
        env["JAX_ENABLE_X64"] = "1"
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(src)],
                         capture_output=True, text=True, env=env)
    lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, (out.stdout[-4000:], out.stderr[-6000:])
    return json.loads(lines[-1][len("RESULT "):])


_SETUP = r'''
import os, json
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
import numpy as np, jax, jax.numpy as jnp
from alphagrad.approx.common import carry_plan as CP
from alphagrad.approx.common import datasets as ds
from alphagrad.approx.common import examples as ex
from alphagrad.approx.common import order as O
from alphagrad.approx.common import rsnn_shd as R
import alphagrad.approx.env as E
from graphax.examples.neuromorphic import rsnn_cell

H, T, B = 6, 9, 3
R.RSNN_HIDDEN = H
R.SHD_TIME_BINS = T
ds.NN_VMAP_BATCH = B
KEY = jax.random.PRNGKey(5)


def env_for(example, rule, container=None):
    xs = ex.get_args(example, KEY, dataset=None, temporal_rule=rule,
                     carry_container=container)
    gen = ex.data_gen(example, key=KEY, temporal_rule=rule,
                      carry_container=container)
    fn, xs, argnums = ex.grad_target_setup({}, ex.get_fn(example), xs,
                                           example)
    cj = CP._traced_inlined(fn, xs)
    return E.VertexEliminationEnv.from_jaxpr(
        cj, args=xs, argnums=argnums, num_envs=0, data_gen=gen,
        target_fun=fn, scalar_target=True, per_face=True)


def sample(env, s):
    a = list(R.measure_args(env.config, env.args))
    gen = env.config.data_gen
    for slot, d in zip(gen.data_slots,
                       gen(jax.random.split(jax.random.PRNGKey(s), 5))):
        a[slot] = jnp.asarray(d)
    return tuple(a)


def bptt_oracle(a):
    seqs, ys, W = a[0], a[1], tuple(a[7:10])
    if seqs.ndim == 3:
        f = lambda *w: jnp.mean(jax.vmap(
            lambda s, y: R.sequence_loss(s, y, w))(seqs, ys))
    else:
        f = lambda *w: R.sequence_loss(seqs, ys, w)
    return jax.grad(f, argnums=(0, 1, 2))(*W)


def tbptt_one(seq, y, W, V, Wo):
    c = R._consts()

    def body(st, x):
        st = tuple(jax.lax.stop_gradient(v) for v in st)
        nxt = rsnn_cell(x, *st, W, V, Wo, *c)
        return nxt, R._step_loss(nxt[4], y)
    _, ls = jax.lax.scan(body, R.zero_state(), seq)
    return jnp.sum(ls)


def tbptt_oracle(a):
    seqs, ys, W = a[0], a[1], tuple(a[7:10])
    if seqs.ndim == 3:
        f = lambda *w: jnp.mean(jax.vmap(
            lambda s, y: tbptt_one(s, y, *w))(seqs, ys))
    else:
        f = lambda *w: tbptt_one(seqs, ys, *w)
    return jax.grad(f, argnums=(0, 1, 2))(*W)


def eprop_one(seq, y, W, V, Wo):
    c = R._consts()

    def fwd(st, x):
        nxt = rsnn_cell(x, *st, W, V, Wo, *c)
        return nxt, nxt[4]
    _, Uos = jax.lax.scan(fwd, R.zero_state(), seq)
    d = jax.nn.softmax(Uos, axis=-1) * jnp.sum(y) - y
    tr = jax.vmap(lambda t: R.carry_traces(seq, t, (W, V, Wo)))(
        jnp.arange(1, seq.shape[0] + 1))
    s = jnp.einsum("tm,mj->tj", d, Wo)
    return (jnp.einsum("tj,tji->ji", s, tr[2]),
            jnp.einsum("tj,tji->ji", s, tr[3]),
            jnp.einsum("tk,tkj->kj", d, tr[4]))


def eprop_oracle(a):
    seqs, ys, W = a[0], a[1], tuple(a[7:10])
    if seqs.ndim == 3:
        g = [eprop_one(seqs[b], ys[b], *W) for b in range(seqs.shape[0])]
        return tuple(jnp.mean(jnp.stack([x[k] for x in g]), axis=0)
                     for k in range(3))
    return eprop_one(seqs, ys, *W)


def rel(got, want):
    got = jax.tree_util.tree_leaves(got)
    want = jax.tree_util.tree_leaves(want)
    assert len(got) == len(want) == 3, (len(got), len(want))
    num = sum(float(jnp.sum((jnp.asarray(g) - w) ** 2)) for g, w in zip(got, want))
    den = sum(float(jnp.sum(w ** 2)) for w in want)
    return (num / den) ** 0.5 if den else num ** 0.5


def empty_plan(env, order_name):
    order = [int(v) for v in O.fixed_order_for_env(order_name, env)]
    return E.measured_program(env.config, order, env.consts, transforms=[])
'''


_EXACT = _SETUP + r'''
assert jax.config.jax_enable_x64
out = {}
for example in ("RSNN_SHD", "VmappedRSNN_SHD"):
    for rule, oracle in (("rtrl", bptt_oracle), ("bptt", bptt_oracle),
                         ("tbptt", tbptt_oracle)):
        env = env_for(example, rule)
        ref = jax.jit(E.reference_program(env.config))
        for order_name in ("reverse", "markowitz"):
            prog = jax.jit(empty_plan(env, order_name))
            worst = worst_ref = 0.0
            for s in range(2):
                a = sample(env, s)
                worst = max(worst, rel(prog(*a), oracle(a)))
                worst_ref = max(worst_ref, rel(ref(*a), bptt_oracle(a)))
            out[f"{example}/{rule}/{order_name}"] = worst
            out[f"{example}/{rule}/reference"] = worst_ref
        out[f"{example}/{rule}/argnums"] = list(env.config.argnums)
        out[f"{example}/{rule}/reference_kind"] = E.reference_kind(env.config)
        # the asynchronous oracle checks the STEP on its own tuple
        seed, a_np, _batch = E.grad_oracle_submission(env.config, env.args, 0)
        status, rel_l2 = E.grad_oracle_cpu_check(
            env.config, a_np, [int(v) for v in O.reverse_order(
                env.valid_vertices)], seed)
        out[f"{example}/{rule}/oracle"] = [status, rel_l2]
print("RESULT " + json.dumps(out))
'''


@pytest.fixture(scope="module")
def exact():
    return _run(_EXACT)


@pytest.mark.parametrize("example", ["RSNN_SHD", "VmappedRSNN_SHD"])
@pytest.mark.parametrize("order", ["reverse", "markowitz"])
def test_the_empty_rtrl_plans_rollout_is_the_gradient_of_the_sequence_loss(
        exact, example, order):
    assert exact[f"{example}/rtrl/{order}"] < 1e-10, exact


@pytest.mark.parametrize("example", ["RSNN_SHD", "VmappedRSNN_SHD"])
@pytest.mark.parametrize("order", ["reverse", "markowitz"])
def test_the_empty_bptt_plans_rollout_is_the_gradient_of_the_sequence_loss(
        exact, example, order):
    assert exact[f"{example}/bptt/{order}"] < 1e-10, exact


@pytest.mark.parametrize("example", ["RSNN_SHD", "VmappedRSNN_SHD"])
@pytest.mark.parametrize("order", ["reverse", "markowitz"])
def test_the_empty_tbptt_plans_rollout_sums_the_spatial_gradients(
        exact, example, order):
    assert exact[f"{example}/tbptt/{order}"] < 1e-10, exact


@pytest.mark.parametrize("example", ["RSNN_SHD", "VmappedRSNN_SHD"])
@pytest.mark.parametrize("rule", ["tbptt", "bptt", "rtrl"])
def test_the_reference_is_jax_grad_of_the_sequence_loss(exact, example, rule):
    assert exact[f"{example}/{rule}/reference"] < 1e-12, exact
    assert exact[f"{example}/{rule}/reference_kind"] == "jax.grad"


@pytest.mark.parametrize("example", ["RSNN_SHD", "VmappedRSNN_SHD"])
@pytest.mark.parametrize("rule", ["tbptt", "bptt", "rtrl"])
def test_the_gradient_oracle_checks_the_step_and_passes(exact, example, rule):
    status, rel_l2 = exact[f"{example}/{rule}/oracle"]
    assert status == "pass" and rel_l2 < 1e-10, exact


@pytest.mark.parametrize("example", ["RSNN_SHD", "VmappedRSNN_SHD"])
def test_bptt_differentiates_the_carried_state(exact, example):
    assert exact[f"{example}/bptt/argnums"] == [2, 3, 4, 5, 6, 7, 8, 9]
    assert exact[f"{example}/rtrl/argnums"] == [7, 8, 9]
    assert exact[f"{example}/tbptt/argnums"] == [7, 8, 9]


_EPROP = _SETUP + r'''
assert jax.config.jax_enable_x64
from alphagrad.approx.env import FACE_SLOTS, MAX_RULES_PER_VERTEX
from graphax import faces_of
from graphax.incremental import IncrementalJaxpr

out = {}
for example in ("RSNN_SHD", "VmappedRSNN_SHD"):
    CP.reset()
    env = env_for(example, "rtrl")
    cfg = env.config
    E.configure_max_faces(E.derived_max_faces(cfg.jaxpr, cfg.argnums,
                                              env.consts, env.args))
    CP.register({}, KEY, example, "rtrl", cfg, env.args, env.consts)
    jx = cfg.jaxpr
    order = sorted((int(v) for v in env.valid_vertices), reverse=True)
    mask = CP.carry_scope_mask(jx)
    ij = IncrementalJaxpr(jx, tuple(cfg.argnums), list(env.consts),
                          list(env.args))
    carry_dot = [v for v in order if mask[v - 1]
                 and jx.eqns[v - 1].primitive.name == "dot_general"]
    rec_v = [v for v in order if not mask[v - 1]
             and jx.eqns[v - 1].primitive.name == "dot_general"
             and any(iv is jx.invars[8] for iv in jx.eqns[v - 1].invars)]
    assert len(rec_v) == 1, rec_v
    specs = np.full((len(order), MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[:, :, 2] = 0
    faces = np.full((len(order), E.MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    skips = np.zeros((len(order), E.MAX_FACES), np.int32)
    for k, v in enumerate(order):
        keys = faces_of(ij.graph, ij.tgraph, v, jx)
        seen = {}

        def hook(fk):
            def h(t):
                seen[fk] = t
                return t
            return h
        ij.eliminate(v, (), face_transforms={
            fk: ((hook(fk), None, None), (None, None, None)) for fk in keys})
        for f, key in enumerate(keys[:E.MAX_FACES]):
            if v in carry_dot:
                faces[k, f, 0] = [0, 0, -1]
            elif v == rec_v[0] and int(key[0]) not in (7, 8):
                t = seen[key]
                outs = [int(d.logical_size) for d in t.out_dims]
                prims = [int(d.logical_size) for d in t.primal_dims]
                faces[k, f, 0] = [outs.index(H), prims.index(H), -1]
    container = CP.container_for_plan(cfg, order, faces, skips, specs)
    var = CP.measurement_env(container, cfg)
    o2, m_specs, m_faces, m_skips, m_joins = CP.transport_wires(
        order, var, specs, faces, skips, None)
    sl = m_specs.tolist()
    transforms, _ = E._decode_vertex_transforms(var["config"], o2, sl)
    ft = E._face_transforms_for_order(
        var["config"], var["consts"], var["args"], o2, sl, m_faces, m_skips,
        wire_sig=E._face_wire_keys(m_faces, m_skips, len(o2), m_joins),
        face_joins_list=m_joins)
    prog = jax.jit(E.measured_program(var["config"], o2, var["consts"],
                                      transforms=transforms,
                                      face_transforms=ft))
    worst = 0.0
    for s in range(2):
        a = sample(env, s)
        worst = max(worst, rel(prog(*a), eprop_oracle(a)))
    out[f"{example}/container"] = container
    out[f"{example}/eprop_vs_oracle"] = worst
    out[f"{example}/eprop_vs_bptt"] = rel(prog(*sample(env, 0)),
                                          bptt_oracle(sample(env, 0)))
print("RESULT " + json.dumps(out))
'''


@pytest.fixture(scope="module")
def eprop():
    return _run(_EPROP)


@pytest.mark.parametrize("example", ["RSNN_SHD", "VmappedRSNN_SHD"])
def test_the_two_diag_plans_rollout_is_the_eprop_gradient(eprop, example):
    # container Diag + Diag on the recurrent face's state edge, measured
    # through the variant, the transport and the face transforms
    assert eprop[f"{example}/container"] == "diag", eprop
    assert eprop[f"{example}/eprop_vs_oracle"] < 1e-10, eprop
    assert eprop[f"{example}/eprop_vs_bptt"] > 1e-6, eprop


_MEASURE = r'''
import os, json
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
import numpy as np, jax
from alphagrad.approx.common import carry_plan as CP
from alphagrad.approx.common import datasets as ds
from alphagrad.approx.common import rsnn_shd as R
R.RSNN_HIDDEN = 6
R.SHD_TIME_BINS = 9
ds.NN_VMAP_BATCH = 3
import alphagrad.approx.tools.landscape_map as lm
import alphagrad.approx.env as E
os.environ["ALPHAGRAD_QUALITY_METRIC"] = "grad_cosine"
os.environ["ALPHAGRAD_COST_FORM"] = "paired-log"
os.environ["ALPHAGRAD_PAIRED_COST_FLOOR"] = "byte"
os.environ["ALPHAGRAD_MEM_CHANNEL"] = "watermark"
out = {}
for rule in ("rtrl", "bptt", "tbptt"):
    CP.reset()
    argv = ["--example", "VmappedRSNN_SHD", "--dataset", "none",
            "--temporal-rule", rule, "--num-eval-samples", "2",
            "--num-data-points", "2", "--reps-per-point", "1",
            "--latency-inner-reps", "1", "--measure-budget-secs", "0.05",
            "--out-dir", "/tmp/full_rollout_test"]
    args = lm.make_argparser().parse_args(argv)
    env, eval_samples, _cj = lm.build_env(args)
    order = [int(v) for v in lm.rev_order(env)]
    m = lm.measure(env, eval_samples, order,
                   {"specs": None, "face_specs": None, "face_skips": None,
                    "wires": []})
    out[rule] = {"quality": m["quality"], "refused": m["refused"],
                 "latency_ns": m["latency_ns"],
                 "samples": [list(np.shape(x)) for x in eval_samples[:2]]}
print("RESULT " + json.dumps(out))
'''


@pytest.fixture(scope="module")
def measured():
    return _run(_MEASURE, x64=False)


@pytest.mark.parametrize("rule", ["rtrl", "bptt", "tbptt"])
def test_the_measurement_runs_the_rollout_on_batches_of_recordings(
        measured, rule):
    m = measured[rule]
    assert m["refused"] == "", m
    assert m["latency_ns"] > 0, m
    # five eval samples are five batches of whole recordings
    assert m["samples"][0] == [2, 3, 9, 700], m
    assert m["samples"][1] == [2, 3, 20], m
    if rule == "tbptt":
        assert 0.0 < m["quality"] < 0.9999, m
    else:
        assert m["quality"] > 0.9999, m
