"""A face decision crosses to the measured graph by the face's IDENTITY.

OWNER RULING 2026-09-25 (dsnn-rbjh, option a). A face is the triple of its
eliminated vertex, its predecessor and its successor (CONTEXT.md, Face), and
its ids are those of the graph at hand, so the carry transport carries each
decision to the triple of the counterparts (``carry_plan._counterparts``) and
raises when there is none. Until then it carried a decision by (step, face
position). On RSNN_SHD rtrl under the static Markowitz order and the forward
order the two graphs list different faces at one position, and e-prop's Diag
on the recurrent face's state edge was applied to its V in-edge face: 3.1e-5
from the hand e-prop gradient where rounding is 2e-16, with no error raised
(job 68087, H 6, T 9, --dataset none, seed 5, CPU, float64).

1. That check, in a float64 subprocess (the pattern ``full_rollout_test``
   uses), with its hand e-prop gradient (``eprop_one``): under the reverse,
   the static Markowitz and the forward order the transported e-prop plan
   decides the same faces as the policy's, named by what their vertices are,
   and its gradient is the hand gradient to rounding, step by step and over
   the recording. A decision whose face the measured program does not have
   raises, with the face named.
2. The rules, on the two-program toy of ``_carry_transport_toy``.
"""

import json
import os
import subprocess
import sys
import textwrap

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np                                             # noqa: E402
import pytest                                                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _carry_transport_toy as toy                             # noqa: E402
from full_rollout_test import _SETUP                           # noqa: E402

TOL = 1e-12


def _run(src):
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(src)],
                         capture_output=True, text=True, env=env)
    lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, (out.stdout[-4000:], out.stderr[-6000:])
    return json.loads(lines[-1][len("RESULT "):])


# ---------------------------------------------------------------------------
# 1. JOB 68087, ON THE CPU IN FLOAT64
# ---------------------------------------------------------------------------
_ORDERS = _SETUP + r'''
assert jax.config.jax_enable_x64
import types
import alphagrad.approx.tools.landscape_map as lm
from graphax import jacve
from graphax.core import _stable_var_index

# THE PROBE'S TARGET: RSNN_SHD rtrl, H 6, T 9 (set by the setup above), the
# synthetic recording (--dataset none), seed 5, through the sweep tool's own
# build, which registers the graph with the carry plan as the trainer does.
argv = ["--example", "RSNN_SHD", "--dataset", "none", "--temporal-rule",
        "rtrl", "--step-position", "3", "--seed", "5", "--num-eval-samples",
        "1", "--num-data-points", "1", "--reps-per-point", "1", "--out-dir",
        "/tmp/face_identity_transport_test"]
env, _samples, _cj = lm.build_env(lm.make_argparser().parse_args(argv))
cfg = env.config
jx = cfg.jaxpr
spec = CP._entry(cfg)["spec"]
k = jax.random.split(spec["key"], 3)
seq, y, _ = R._draw_recording(k[0], None, -1)
W = R.rsnn_weights(k[1])
for a, b in zip(env.args[7:10], W):
    assert np.array_equal(np.asarray(a), np.asarray(b))
c = R._consts()
valid = sorted(int(v) for v in env.valid_vertices)
mask = CP.carry_scope_mask(jx)
vidx = _stable_var_index(jx)
carry_dot = [v for v in valid if mask[v - 1]
             and jx.eqns[v - 1].primitive.name == "dot_general"]
rec_v = [v for v in valid if not mask[v - 1]
         and jx.eqns[v - 1].primitive.name == "dot_general"
         and any(iv is jx.invars[8] for iv in jx.eqns[v - 1].invars)]
assert len(rec_v) == 1, rec_v
rec_v = rec_v[0]
state = [iv for iv in jx.eqns[rec_v - 1].invars if iv is not jx.invars[8]]
assert len(state) == 1, state
state = state[0]
producer = {ov: i for i, e in enumerate(jx.eqns, 1) for ov in e.outvars}
s_att = producer[state]
assert mask[s_att - 1], "V @ S reads the attached S"


def described(jaxpr):
    """variable -> what it IS, named alike on either graph: an input by its
    slot, a step value by its rank in the step body, an attached state by the
    step equation and operand slot that first read it. A variable inside the
    carry block is absent: it has no name the other graph shares."""
    m = CP.carry_scope_mask(jaxpr)
    names = {v: ("input", n) for n, v in enumerate(jaxpr.invars)}
    body = [i for i in range(1, len(jaxpr.eqns) + 1) if not m[i - 1]]
    for r, i in enumerate(body):
        e = jaxpr.eqns[i - 1]
        for p, v in enumerate(e.invars):
            if not hasattr(v, "val") and v not in names:
                names[v] = ("attached", r, p)
        for o, v in enumerate(e.outvars):
            names[v] = ("step", r, o)
    return names


def decisions(prog, order, faces, skips):
    """The face decisions of the step body, each as the (vertex,
    predecessor, successor) triple it sits on and its wire row, read on the
    program's own face inventory along ``order``."""
    jaxpr = prog.config.jaxpr
    names = described(jaxpr)
    var_of = {i: v for v, i in _stable_var_index(jaxpr).items()}
    m = CP.carry_scope_mask(jaxpr)
    out = set()
    for e in lm.face_inventory(prog, np.asarray(order, dtype=np.int32)):
        k, v, f = int(e["k"]), int(e["vertex"]), int(e["f"])
        if m[v - 1] or not (skips[k, f] == 1
                            or np.any(faces[k, f][..., 0] != -1)):
            continue
        a, b = (names.get(var_of[int(x)]) for x in e["key"])
        out.add((names[jaxpr.eqns[v - 1].outvars[0]], a, b,
                 tuple(faces[k, f].ravel().tolist()), int(skips[k, f])))
    return out


def eprop_plan(order, state_edge=True, skip_carry=False):
    """The named e-prop plan (CONTEXT.md, Named plans): a Diag on the carried
    faces, the container, plus a Diag on the state edge of the recurrent
    face, on whatever faces ``order`` gives the policy's graph."""
    inv = lm.face_inventory(env, np.asarray(order, dtype=np.int32))
    carried = [e for e in inv if int(e["vertex"]) in carry_dot]
    edge = [e for e in inv if int(e["vertex"]) == rec_v
            and int(e["key"][0]) == vidx[state]]
    assert edge, "no state-edge face on the recurrent vertex"
    wires = [{"k": int(e["k"]), "f": int(e["f"]), "slot": 0,
              "row": [0, 0, -1], "kind": "DIAG"}
             for e in carried + (edge if state_edge else [])]
    if skip_carry:
        wires.append({"k": int(carried[0]["k"]), "f": int(carried[0]["f"]),
                      "kind": "SKIP"})
    plan = {"specs": None, "face_specs": None, "face_skips": None,
            "wires": wires}
    specs, faces, skips = (np.asarray(a) for a in
                           lm.get_plan_arrays(plan, len(order)))
    return specs, faces, skips, edge


def measured(order, specs, faces, skips):
    """The callback's chain: container, variant, transport, face
    transforms."""
    container = CP.container_for_plan(cfg, order, faces, skips, specs)
    var = CP.measurement_env(container, cfg)
    o2, m_specs, m_faces, m_skips, m_joins = CP.transport_wires(
        order, var, specs, faces, skips, None)
    vc = var["config"]
    sl = m_specs.tolist()
    transforms, _ = E._decode_vertex_transforms(vc, o2, sl)
    ft = E._face_transforms_for_order(
        vc, var["consts"], var["args"], o2, sl, m_faces, m_skips,
        wire_sig=E._face_wire_keys(m_faces, m_skips, len(o2), m_joins),
        face_joins_list=m_joins)
    return container, var, o2, m_faces, m_skips, transforms, ft


reverse = [int(v) for v in O.reverse_order(env.valid_vertices)]
markowitz = [int(v) for v in O.fixed_order_for_env("markowitz", env)]
# THE FORWARD ORDER OF THE PROBE: ascending, with the attached S right after
# V @ S, so the recurrent face still has its state edge.
forward = [v for v in sorted(valid) if v != s_att]
forward.insert(forward.index(rec_v) + 1, s_att)
orders = {"reverse": reverse, "markowitz": markowitz, "forward": forward}
out = {}
for name, order in orders.items():
    row = {}
    pos = {v: i for i, v in enumerate(order)}
    # does the order interleave the carry block with the step body?
    row["carry_before_rec"] = sum(1 for v in order[:pos[rec_v]]
                                  if mask[v - 1])
    row["attached_after_rec"] = pos[s_att] > pos[rec_v]
    specs, faces, skips, edge = eprop_plan(order)
    container, var, o2, m_faces, m_skips, transforms, ft = measured(
        order, specs, faces, skips)
    row["container"] = container
    vc = var["config"]
    shim = types.SimpleNamespace(config=vc, consts=var["consts"],
                                 args=var["args"])
    policy = decisions(env, order, faces, skips)
    moved = decisions(shim, o2, m_faces, m_skips)
    row["same_faces"] = policy == moved
    row["n_decisions"] = len(policy)
    rank = [i for i in range(1, len(jx.eqns) + 1)
            if not mask[i - 1]].index(rec_v)
    row["state_edge_decided"] = sorted(
        {d[1] == ("attached", rank, 1) and d[0] == ("step", rank, 0)
         for d in moved})
    row["n_state_edge"] = len(edge)
    # (i) THE PROBE'S CHECK: the plan's gradient at step t, from the carry
    # its own program produced over the prefix, against the hand e-prop
    # gradient of that step (the step's term of eprop_one).
    program = jacve(vc.target_fun, list(o2), argnums=vc.argnums,
                    has_aux=vc.has_aux, sparse_representation=False,
                    jaxpr=vc.jaxpr, consts=list(var["consts"]),
                    transforms=transforms, face_transforms=ft)
    for t in (3, 5, 7):
        given = R.carry_from_program(seq, y, t, W, program, container)
        st = R.prefix_state(seq, t, W)(*W)
        got = program(seq[t], y, *st, *W, *c, *given)[0]
        hand = tuple(a - b for a, b in zip(eprop_one(seq[:t + 1], y, *W),
                                           eprop_one(seq[:t], y, *W)))
        row[f"step_t{t}"] = rel(got, hand)
    # (ii) THE MEASURED QUANTITY: the plan's full rollout of the recording
    # against the hand e-prop gradient of the recording.
    roll = jax.jit(E.measured_program(vc, o2, var["consts"],
                                      transforms=transforms,
                                      face_transforms=ft))
    got = roll(*R.rollout_tuple(seq, y, tuple(W), tuple(var["args"][10:16])))
    row["rollout"] = rel(got, eprop_one(seq, y, *W))
    out[name] = row


def raised(order, specs, faces, skips):
    try:
        measured(order, specs, faces, skips)
    except ValueError as exc:
        return str(exc)
    return None


# A DECISION WITH NO FACE TO LAND ON. (a) Under the static Markowitz order a
# step vertex meets the dense rows' stacked tensordot, which the diag
# container rewrites: a Diag on that face raises.
inv = lm.face_inventory(env, np.asarray(markowitz, dtype=np.int32))
names = described(jx)
var_of = {i: v for v, i in vidx.items()}
inside = [e for e in inv if not mask[int(e["vertex"]) - 1]
          and names.get(var_of[int(e["key"][0])]) is None]
assert inside, "no step face reaches into the carry block under markowitz"
e = inside[0]
specs, faces, skips, _edge = eprop_plan(markowitz)
faces[int(e["k"]), int(e["f"]), 0] = [0, 0, -1]
pred = producer[var_of[int(e["key"][0])]]
out["rewritten"] = {"message": raised(markowitz, specs, faces, skips),
                    "vertex": int(e["vertex"]), "pred": pred,
                    "pred_prim": jx.eqns[pred - 1].primitive.name}
# (b) Under the skip container the program is the truncated one: no
# attached state, so e-prop's state-edge Diag has no face there.
specs, faces, skips, _edge = eprop_plan(reverse, skip_carry=True)
out["skip"] = {"container": CP.container_for_plan(cfg, reverse, faces, skips,
                                                  specs),
               "message": raised(reverse, specs, faces, skips),
               "vertex": rec_v, "pred": s_att}
print("RESULT " + json.dumps(out))
'''


@pytest.fixture(scope="module")
def orders():
    return _run(_ORDERS)


@pytest.mark.parametrize("order", ["reverse", "markowitz", "forward"])
def test_the_eprop_plan_moves_to_the_same_faces(orders, order):
    row = orders[order]
    assert row["container"] == "diag", row
    assert row["same_faces"], row
    # the decisions of the step body are e-prop's state-edge Diags, and each
    # sits on V @ S with the attached S as its predecessor
    assert row["n_decisions"] == row["n_state_edge"] > 0, row
    assert row["state_edge_decided"] == [True], row


def test_the_markowitz_and_forward_orders_interleave_the_carry_block(orders):
    # the case job 68087 found: most of the block goes before V @ S, the
    # attached S after it; the reverse order puts the whole block last
    assert orders["reverse"]["carry_before_rec"] == 0, orders["reverse"]
    for name in ("markowitz", "forward"):
        row = orders[name]
        assert row["carry_before_rec"] > 0 and row["attached_after_rec"], row


@pytest.mark.parametrize("order", ["reverse", "markowitz", "forward"])
def test_the_transported_eprop_plan_is_the_hand_gradient(orders, order):
    row = orders[order]
    for t in (3, 5, 7):
        assert row[f"step_t{t}"] <= TOL, row
    assert row["rollout"] <= TOL, row


def test_a_decision_on_a_face_the_container_rewrites_raises(orders):
    got = orders["rewritten"]
    msg = got["message"]
    assert msg is not None, got
    assert "no counterpart" in msg, msg
    assert (f"{got['pred']} ({got['pred_prim']}) -> {got['vertex']} ->"
            in msg), msg


def test_a_decision_the_truncated_program_does_not_have_raises(orders):
    got = orders["skip"]
    assert got["container"] == "skip", got
    msg = got["message"]
    assert msg is not None and "no counterpart" in msg, got
    assert f"{got['pred']} (add) -> {got['vertex']} ->" in msg, msg


# ---------------------------------------------------------------------------
# 2. THE RULES, ON THE TOY
# ---------------------------------------------------------------------------
def _order(var, order):
    from alphagrad.approx.common import carry_plan as CP
    return CP.transport_order(order, var)


def _rows(n, width=4):
    """An all-exact plan of ``n`` steps: per-vertex rows, face rows, skips,
    join bits."""
    from alphagrad.approx.env import FACE_SLOTS, MAX_RULES_PER_VERTEX
    rs = np.full((n, MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    rs[:, :, 2] = 0
    fs = np.full((n, width, FACE_SLOTS, 3), -1, np.int32)
    sk = np.zeros((n, width), np.int32)
    jn = np.zeros((n, width), np.int32)
    return rs, fs, sk, jn


def _quant_row():
    from alphagrad.approx.env import QUANT_SENTINEL
    from alphagrad.approx.unified_face_policy import _NARROW_SLOT
    return (QUANT_SENTINEL, int(_NARROW_SLOT), 0)


def _faces(prog, order):
    """vertex -> its faces along ``order``, as (predecessor, successor)
    vertex ids (an input as ``("input", slot)``)."""
    from graphax import face_specs_of
    from graphax.incremental import IncrementalJaxpr
    cfg = prog["config"]
    jx = cfg.jaxpr
    prod = {ov: i for i, e in enumerate(jx.eqns, 1) for ov in e.outvars}

    def name(x):
        return prod[x] if x in prod else ("input", jx.invars.index(x))
    ij = IncrementalJaxpr(jx, tuple(cfg.argnums), list(prog["consts"]),
                          list(prog["args"]), track_faces=False)
    out = {}
    for v in order:
        out[v] = [(name(s.in_edge), name(s.out_edge))
                  for s in face_specs_of(ij.graph, ij.tgraph, v, jx)]
        ij.eliminate(v, (), None)
    return out


def _measured(var):
    return {"config": var["config"], "consts": var["consts"],
            "args": var["args"]}


def test_the_counterparts_are_the_body_the_attached_states_and_the_difference():
    var = toy.variant()
    assert var["vertex_map"] == {**toy.BODY, **toy.OUTPUT}
    # stop_gradient(w) and d are the same operation on the same operands, an
    # attached state is what the step body reads, and the contractions the
    # container rewrote have no counterpart
    assert var["carry_map"] == {**toy.DIFFERENCE, **toy.ATTACHED}
    assert not set(toy.REWRITTEN) & set(var["carry_map"])


def test_an_order_that_does_not_interleave_is_carried_as_before():
    var = toy.variant()
    # the block in one run after the step body (reverse) or before it: the
    # whole measured block goes there, in the block's own order
    assert _order(var, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]) == [
        14, 13, 12, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert _order(var, list(range(1, 11))) == list(range(1, 15))


def test_the_attached_states_and_the_rewritten_blocks_keep_their_segments():
    var = toy.variant()
    # the policy eliminates K @ d (5) and u' (6) before the step body, J @ d
    # (3) and s' (4) after its first vertex (7), d and stop_gradient(w) last:
    # the measured u' (6) and its row sum (3, 4, 5) go first, s' (10) and
    # its row sum (7, 8, 9) after vertex 11, and 1, 2 last
    order = [5, 6, 7, 3, 4, 8, 9, 10, 1, 2]
    o2 = _order(var, order)
    assert o2 == [3, 4, 5, 6, 11, 7, 8, 9, 10, 12, 13, 14, 1, 2]
    # ... so every step vertex meets the same faces on both graphs
    pf, mf = _faces(var["base"], order), _faces(_measured(var), o2)
    back = {j: i for i, j in {**toy.BODY, **toy.OUTPUT, **toy.ATTACHED,
                              **toy.DIFFERENCE}.items()}
    for i, j in toy.BODY.items():
        assert [tuple(back.get(x, x) for x in f) for f in mf[j]] == pf[i], (
            i, pf[i], mf[j])


def test_a_decision_moves_to_the_face_that_is_its_face(monkeypatch):
    from alphagrad.approx.common import carry_plan as CP
    # the join bit rides the wire only when the join is decided per face
    monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", "choose")
    var = toy.variant()
    # UNDER THE REVERSE ORDER the faces of vertex 7 through the attached
    # states are (s', out) and (u', out) on the policy's graph and (u', out)
    # and (s', out) on the measured one, which attaches u' first: carried by
    # position, a decision on one would land on the other
    order = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    o2 = _order(var, order)
    out, out2 = next(iter(toy.OUTPUT.items()))
    assert _faces(var["base"], order)[7] == [(4, out), (6, out)]
    assert _faces(_measured(var), o2)[11] == [(6, out2), (10, out2)]
    rs, fs, sk, jn = _rows(len(order))
    k = order.index(7)
    for s in (0, 1):
        fs[k, 0, s] = _quant_row()          # a Quant on (s', 7, out) ...
    jn[k, 0] = 1                            # ... joined lossless
    sk[k, 1] = 1                            # a Skip on (u', 7, out)
    o2, _rs2, fs2, sk2, jn2 = CP.transport_wires(order, var, rs, fs, sk, jn)
    k2 = o2.index(11)
    np.testing.assert_array_equal(fs2[k2, 1], fs[k, 0])
    assert jn2[k2, 1] == 1
    assert sk2[k2, 0] == 1
    decided = np.any(fs2[..., 0] != -1, axis=-1) | (sk2 == 1)
    assert set(zip(*np.nonzero(decided))) == {(k2, 0), (k2, 1)}, decided


def test_a_decision_on_a_face_through_a_rewritten_block_raises():
    from alphagrad.approx.common import carry_plan as CP
    var = toy.variant()
    # s' goes before vertex 7 and J @ d after it: vertex 7 meets J @ d, which
    # the measured graph does not have
    order = [4, 7, 3, 5, 6, 8, 9, 10, 1, 2]
    assert _faces(var["base"], order)[7] == [(3, 8), (6, 8)]
    rs, fs, sk, _jn = _rows(len(order))
    for s in (0, 1):
        fs[order.index(7), 0, s] = _quant_row()
    with pytest.raises(ValueError, match=r"3 \(dot_general\) -> 7 -> 8 "
                                         r"\(mul\).*no counterpart"):
        CP.transport_wires(order, var, rs, fs, sk)
    # the face through u' has one, and moves
    fs[:] = -1
    for s in (0, 1):
        fs[order.index(7), 1, s] = _quant_row()
    o2, _rs, fs2, _sk, _jn2 = CP.transport_wires(order, var, rs, fs, sk)
    f2 = _faces(_measured(var), o2)[11].index((6, 12))
    assert (fs2[o2.index(11), f2, 0] == np.asarray(_quant_row())).all()
