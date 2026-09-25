"""The two-Diag plan's measured step forms no readout row (dsnn-dfw.217).

The diag container's readout trace against a hidden weight is the filter of
the given trace and the plan's S row, the attachment's zero-valued forward is
never computed, each block of a stack is read from its own slot, and the
transported carry block runs from the attached states back to the weights.
Built through the pipeline in a float64 subprocess, as full_rollout_test does.
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest


def _run(src):
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(src)],
                         capture_output=True, text=True, env=env)
    lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, (out.stdout[-4000:], out.stderr[-6000:])
    return json.loads(lines[-1][len("RESULT "):])


_STEP = r'''
import os, json
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
import numpy as np, jax, jax.numpy as jnp
from jax._src.interpreters import partial_eval as pe
from graphax import faces_of, jacve
from graphax.incremental import IncrementalJaxpr
from graphax.examples.neuromorphic import rsnn_cell
from alphagrad.approx.common import carry_plan as CP
from alphagrad.approx.common import datasets as ds
from alphagrad.approx.common import examples as ex
from alphagrad.approx.common import rsnn_shd as R
import alphagrad.approx.env as E
from alphagrad.approx.env import FACE_SLOTS, MAX_RULES_PER_VERTEX

H, T, B = 6, 9, 3
R.RSNN_HIDDEN = H
R.SHD_TIME_BINS = T
ds.NN_VMAP_BATCH = B
KEY = jax.random.PRNGKey(5)
N_IN, N_OUT = 700, 20


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
    num = sum(float(jnp.sum((jnp.asarray(g) - w) ** 2)) for g, w in zip(got, want))
    den = sum(float(jnp.sum(w ** 2)) for w in want)
    return (num / den) ** 0.5 if den else num ** 0.5


def eprop_plan(example):
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
    assert container == "diag", container
    var = CP.measurement_env(container, cfg)
    o2, m_specs, m_faces, m_skips, m_joins = CP.transport_wires(
        order, var, specs, faces, skips, None)
    sl = m_specs.tolist()
    transforms, _ = E._decode_vertex_transforms(var["config"], o2, sl)
    ft = E._face_transforms_for_order(
        var["config"], var["consts"], var["args"], o2, sl, m_faces, m_skips,
        wire_sig=E._face_wire_keys(m_faces, m_skips, len(o2), m_joins),
        face_joins_list=m_joins)
    vc = var["config"]
    fnj = jacve(vc.target_fun, list(o2), argnums=vc.argnums, has_aux=True,
                sparse_representation=True, jaxpr=vc.jaxpr,
                consts=list(var["consts"]), transforms=transforms,
                face_transforms=ft)
    step = R.rsnn_one_call_step(fnj)
    roll = E.measured_program(vc, o2, var["consts"], transforms=transforms,
                              face_transforms=ft)
    return env, var, step, roll


def dced(fn, args):
    closed = jax.make_jaxpr(fn)(*args)
    jx, _ = pe.dce_jaxpr(closed.jaxpr, [True] * len(closed.jaxpr.outvars))
    return jx


out = {}
for example in ("RSNN_SHD", "VmappedRSNN_SHD"):
    env, var, step, roll = eprop_plan(example)
    args = tuple(var["args"])
    jx = dced(step, args)
    weights = [jx.invars[i] for i in (7, 8, 9)]
    stacks = [tuple(a.shape) for a in args[16:18]]
    row_tails = {(N_OUT, H, N_IN), (N_OUT, H, H)}

    def tail(v, n):
        return tuple(v.aval.shape[-n:])
    readout_rows = sum(
        1 for e in jx.eqns for v in e.outvars if tail(v, 3) in row_tails)
    weight_subs = sum(
        1 for e in jx.eqns if e.primitive.name == "sub"
        and any(iv is w for iv in e.invars for w in weights))
    stacked_muls = sum(
        1 for e in jx.eqns if e.primitive.name == "mul"
        and any(hasattr(iv, "aval") and tuple(iv.aval.shape) in stacks
                for iv in e.invars))
    prog = jax.jit(roll)
    worst = max(rel(prog(*sample(env, s)), eprop_oracle(sample(env, s)))
                for s in range(2))
    out[example] = {"readout_rows": readout_rows, "weight_subs": weight_subs,
                    "stacked_muls": stacked_muls, "eqns": len(jx.eqns),
                    "eprop_vs_oracle": worst}
print("RESULT " + json.dumps(out))
'''


@pytest.fixture(scope="module")
def step():
    return _run(_STEP)


@pytest.mark.parametrize("example", ["RSNN_SHD", "VmappedRSNN_SHD"])
def test_the_two_diag_step_forms_no_readout_row(step, example):
    # the (n_out, h, n_in) and (n_out, h, h) rows of the readout membrane
    # are dead once the container reads the filter, and dead code is gone
    assert step[example]["readout_rows"] == 0, step


@pytest.mark.parametrize("example", ["RSNN_SHD", "VmappedRSNN_SHD"])
def test_the_attachments_zero_forward_is_not_computed(step, example):
    # W - stop_gradient(W) is the attachment's only sub on a weight
    assert step[example]["weight_subs"] == 0, step


@pytest.mark.parametrize("example", ["RSNN_SHD", "VmappedRSNN_SHD"])
def test_the_stacked_carry_is_read_one_block_at_a_time(step, example):
    assert step[example]["stacked_muls"] == 0, step


@pytest.mark.parametrize("example", ["RSNN_SHD", "VmappedRSNN_SHD"])
def test_the_two_diag_plans_rollout_is_still_the_eprop_gradient(step, example):
    assert step[example]["eprop_vs_oracle"] < 1e-10, step


def _attachment_jaxpr():
    # a readout chain through a contraction and a hidden chain, sharing the
    # weight's difference vertex, like attach_rsnn_past's diag container
    import jax
    import jax.numpy as jnp

    def f(w, J, Jh, Wo, s, sh):
        d = w - jax.lax.stop_gradient(w)               # 1 sg, 2 sub
        rows = jnp.sum(J * d, axis=-1)                  # 3 mul, 4 reduce_sum
        s_att = s + Wo @ rows                           # 5 dot_general, 6 add
        sh_att = sh + jnp.sum(Jh * d, axis=-1)          # 7 mul, 8 reduce_sum, 9 add
        return s_att, sh_att
    args = (jnp.ones((4, 6)), jnp.ones((4, 6)), jnp.ones((4, 6)),
            jnp.ones((3, 4)), jnp.ones((3,)), jnp.ones((4,)))
    return jax.make_jaxpr(f)(*args).jaxpr


def test_a_contractions_chain_stops_at_the_shared_difference():
    from alphagrad.approx.common import carry_plan as CP
    jx = _attachment_jaxpr()
    names = [e.primitive.name for e in jx.eqns]
    assert names == ["stop_gradient", "sub", "mul", "reduce_sum",
                     "dot_general", "add", "mul", "reduce_sum", "add"], names
    chain = CP.contraction_chains(jx, range(1, 10))
    assert chain == {3, 4, 5, 6}, chain


def _variant(container):
    import types
    jx = _attachment_jaxpr()
    return {"container": container, "vertex_map": {11: 11, 12: 12},
            "alt_carry": tuple(range(1, 10)),
            "valid": set(range(1, 10)) | {11, 12},
            "config": types.SimpleNamespace(jaxpr=jx)}


def test_the_transported_block_runs_a_contractions_chain_backward_first():
    from alphagrad.approx.common import carry_plan as CP
    # the policy's carry vertices 20 and 21 sit between its body vertices
    for container in ("diag", "diag+reduce", "diag+quant"):
        moved = CP.transport_order([12, 20, 21, 11], _variant(container))
        assert moved == [12, 6, 5, 4, 3, 1, 2, 7, 8, 9, 11], (container, moved)
        assert sorted(moved) == sorted(_variant(container)["valid"])


def test_the_chain_rule_serves_the_diag_container_only():
    # the other containers attach the dense rows through a tensordot, a
    # contraction too, and their block stays ascending (the quant plan ran
    # 24 percent slower under the chain, job 68169)
    from alphagrad.approx.common import carry_plan as CP
    for container in ("exact", "reduce", "quant", "reduce+quant"):
        moved = CP.transport_order([12, 20, 21, 11], _variant(container))
        assert moved == [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11], (container, moved)
        assert not CP.chain_applies(container)
    assert CP.chain_applies("diag") and CP.chain_applies("diag+quant")
