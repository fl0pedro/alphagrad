"""M2 feature-extraction spot checks (CPU-fast, tiny target).

(a1) a dot_general vertex gets the right dimension-number one-hots;
(a2) a diagonal SparseTensor edge classifies diagonal, a dense one dense;
(a3) fill edges are flagged after a non-absorb face elimination;
plus role flags / topo / vocab sanity.
"""

import numpy as np
import pytest

from alphagrad.elimrl.baselines import tiny_target
from alphagrad.elimrl.env import ElimEnv
from alphagrad.elimrl import features as F


@pytest.fixture(scope="module")
def env_static():
    fn, args, argnums = tiny_target()
    env = ElimEnv(fn, args, argnums=argnums)
    static = F.build_static(env)
    return env, static


def test_dot_general_dimnum_onehots(env_static):
    env, static = env_static
    vids = [j for j, eqn in enumerate(env.jaxpr.eqns, start=1)
            if eqn.primitive.name == "dot_general"]
    assert vids, "tiny target must contain dot_general"
    j = vids[0]
    eqn = env.jaxpr.eqns[j - 1]
    (lc, rc), (lb, rb) = eqn.params["dimension_numbers"]
    row = static.feat[static.row_of(j)]
    for sl, axes in ((F.S_DOT_LHS_C, lc), (F.S_DOT_RHS_C, rc),
                     (F.S_DOT_LHS_B, lb), (F.S_DOT_RHS_B, rb)):
        expect = np.zeros(F.MAX_RANK, np.float32)
        for a in axes:
            expect[min(int(a), F.MAX_RANK - 1)] = 1.0
        np.testing.assert_array_equal(row[sl], expect)
    # a non-dot vertex has all-zero dimnum slots
    k = next(j2 for j2, e in enumerate(env.jaxpr.eqns, start=1)
             if e.primitive.name != "dot_general")
    assert not static.feat[static.row_of(k), 0:4 * F.MAX_RANK].any()


def test_sparsity_classification(env_static):
    env, static = env_static
    env.reset()
    feat = F.extract(env.state(), static)
    diag, dense = [], []
    for n in env.state().nodes:
        if n.kind != "Z":
            continue
        rows = n.out_dims + n.primal_dims
        if not rows:
            continue
        p = feat.nid_index[n.nid]
        if any(r[4] is not None for r in rows):
            diag.append(p)
        elif all(r[0] == "Index" for r in rows):
            dense.append(p)
    assert diag and dense, "tiny target should have both edge kinds"
    cls = feat.edge_feat[:, F.E_CLS]
    for p in diag:
        assert cls[p, F.CLS_DIAGONAL] == 1.0 and cls[p].sum() == 1.0
    for p in dense:
        assert cls[p, F.CLS_DENSE] == 1.0 and cls[p].sum() == 1.0
    # stored <= logical for every stored buffer; gap on a diagonal edge < 0
    stored_gap = feat.edge_feat[diag, F.E_GAP]
    assert (stored_gap <= 0).all()


def test_fill_edge_flagged(env_static):
    env, static = env_static
    env.reset()
    st = env.state()
    feat0 = F.extract(st, static)
    assert not feat0.edge_feat[:, F.E_IS_FILL].any()
    fm = next(f for f in st.faces if not f.absorb)
    env.apply(("F", fm.u, fm.w))
    st1 = env.state()
    feat1 = F.extract(st1, static)
    new_nids = set(feat1.nid_index) - set(feat0.nid_index)
    assert new_nids, "non-absorb face must create a fill node"
    for nid in new_nids:
        assert feat1.edge_feat[feat1.nid_index[nid], F.E_IS_FILL] == 1.0
    for nid in set(feat1.nid_index) & set(feat0.nid_index):
        assert feat1.edge_feat[feat1.nid_index[nid], F.E_IS_FILL] == 0.0
    # the fill also shows up in the endpoint vertices' fill counters
    d0, ep = F.changed_rows(feat0, feat1)
    assert len(ep) > 0
    env.reset()


def test_roles_topo_vocab(env_static):
    env, static = env_static
    roles = static.feat[:, F.S_ROLE]
    for i in range(static.n_invars):
        is_param = i in env.argnums
        assert roles[i, F.ROLE_PARAM] == (1.0 if is_param else 0.0)
        assert roles[i, F.ROLE_DATA] == (0.0 if is_param else 1.0)
    out_rows = [static.row_of(j) for j, eqn in enumerate(env.jaxpr.eqns, start=1)
                if any(id(o) in {id(v) for v in env.jaxpr.outvars}
                       for o in eqn.outvars)]
    assert out_rows and all(roles[r, F.ROLE_OUTPUT] == 1.0 for r in out_rows)
    sg_rows = [static.row_of(j) for j, eqn in enumerate(env.jaxpr.eqns, start=1)
               if eqn.primitive.name == "stop_gradient"]
    assert all(roles[r, F.ROLE_STOPGRAD] == 1.0 for r in sg_rows)
    topo = static.feat[:, F.S_TOPO]
    assert topo.min() >= 0.0 and topo.max() <= 1.0
    assert static.prim_vocab[F.PRIM_UNK] == 0
    assert "dot_general" in static.prim_vocab


def test_vertex_dynamics(env_static):
    env, static = env_static
    env.reset()
    st = env.state()
    feat = F.extract(st, static)
    # degrees must match a hand count over Z nodes
    z = [n for n in st.nodes if n.kind == "Z"]
    for vid in list(st.legal_vertices)[:5]:
        row = static.row_of(vid)
        ind = sum(1 for n in z if static.row_of(n.snk_vertex) == row)
        outd = sum(1 for n in z if static.row_of(n.src_vertex) == row)
        assert feat.vert_dyn[row, F.DV_IN_DEG] == pytest.approx(np.log2(1 + ind))
        assert feat.vert_dyn[row, F.DV_OUT_DEG] == pytest.approx(np.log2(1 + outd))
        assert feat.vert_dyn[row, F.DV_MARKOWITZ] == pytest.approx(
            np.log2(1 + ind * outd))
        assert feat.vert_dyn[row, F.DV_IS_LEGAL] == 1.0
    # eliminating a vertex sets its flag
    j = st.legal_vertices[0]
    env.apply(("V", j))
    feat1 = F.extract(env.state(), static)
    assert feat1.vert_dyn[static.row_of(j), F.DV_ELIMINATED] == 1.0
    env.reset()
