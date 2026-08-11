"""M2 encoder tests (CPU-fast, tiny target).

(b) incremental == full recompute after each of 20 random eliminations;
(c) bucketed == unbucketed outputs under masking;
(d) compile-signature count across a full episode <= (#buckets)^2;
(e) permutation-equivariance sanity of the GNN core.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from alphagrad.elimrl.baselines import tiny_target
from alphagrad.elimrl.env import ElimEnv
from alphagrad.elimrl import features as F
from alphagrad.elimrl.encoder import (
    CAND_BUCKETS, EDGE_BUCKETS, NODE_BUCKETS, TGT_BUCKETS,
    EncoderRuntime, full_forward, make_model, pick_bucket,
)


@pytest.fixture(scope="module")
def setup():
    fn, args, argnums = tiny_target()
    env = ElimEnv(fn, args, argnums=argnums)
    static = F.build_static(env)
    model = make_model(static, seed=0, hidden=64, width=64)
    return env, static, model


def _cmp(out_a, out_b, atol=2e-4):
    n = out_a["n_rows"]
    np.testing.assert_allclose(np.asarray(out_a["h"][:n]),
                               np.asarray(out_b["h"][:n]), atol=atol)
    np.testing.assert_allclose(np.asarray(out_a["graph"]),
                               np.asarray(out_b["graph"]), atol=atol)
    nf, nv = out_a["n_faces"], out_a["n_vertices"]
    if nf:
        np.testing.assert_allclose(np.asarray(out_a["face_emb"][:nf]),
                                   np.asarray(out_b["face_emb"][:nf]), atol=atol)
    if nv:
        np.testing.assert_allclose(np.asarray(out_a["vertex_emb"][:nv]),
                                   np.asarray(out_b["vertex_emb"][:nv]), atol=atol)


def test_incremental_matches_full(setup):
    env, static, model = setup
    env.reset()
    rt_inc = EncoderRuntime(model, static)
    rt_ref = EncoderRuntime(model, static)
    rng = np.random.default_rng(0)
    rt_inc.encode_incremental(env.state())          # prime the cache
    steps = 0
    while steps < 20:
        faces = env.faces()
        if not faces:
            break
        env.apply(("F",) + faces[rng.integers(len(faces))])
        st = env.state()
        out_i = rt_inc.encode_incremental(st)
        out_f = rt_ref.encode_full(st)
        _cmp(out_i, out_f)
        steps += 1
    assert steps >= 10, "tiny episode ended suspiciously early"
    env.reset()


def test_incremental_matches_full_vertex_macros(setup):
    env, static, model = setup
    env.reset()
    rt_inc = EncoderRuntime(model, static)
    rt_ref = EncoderRuntime(model, static)
    rng = np.random.default_rng(1)
    rt_inc.encode_incremental(env.state())
    for _ in range(6):
        lv = env.legal_vertices()
        if not lv:
            break
        env.apply(("V", int(lv[rng.integers(len(lv))])))
        st = env.state()
        _cmp(rt_inc.encode_incremental(st), rt_ref.encode_full(st))
    env.reset()


def test_bucketed_equals_unbucketed(setup):
    env, static, model = setup
    env.reset()
    rt_b = EncoderRuntime(model, static)
    rt_e = EncoderRuntime(model, static, node_buckets=None, edge_buckets=None,
                          cand_buckets=None, tgt_buckets=None)
    rng = np.random.default_rng(2)
    for _ in range(4):
        st = env.state()
        _cmp(rt_b.encode_full(st), rt_e.encode_full(st), atol=1e-5)
        faces = env.faces()
        if not faces:
            break
        env.apply(("F",) + faces[rng.integers(len(faces))])
    env.reset()


def test_compile_count_bound(setup):
    env, static, model = setup
    env.reset()
    rt = EncoderRuntime(model, static)
    rng = np.random.default_rng(3)
    rt.encode_incremental(env.state())
    while True:
        faces = env.faces()
        if not faces:
            break
        env.apply(("F",) + faces[rng.integers(len(faces))])
        rt.encode_incremental(env.state())
    n_buckets = (len(NODE_BUCKETS) + len(EDGE_BUCKETS) + len(CAND_BUCKETS)
                 + len(TGT_BUCKETS))
    assert rt.compile_signatures <= n_buckets ** 2, (
        rt.compile_signatures, n_buckets ** 2)
    # and in practice far fewer -- a loose sanity ceiling on absolute count
    assert rt.compile_signatures < 60
    env.reset()


def test_permutation_equivariance(setup):
    env, static, model = setup
    env.reset()
    feat = F.extract(env.state(), static)
    n = static.n_rows
    nmask = np.ones(n, np.float32)
    emask = np.ones(len(feat.edge_src), np.float32)
    hs, pooled, _ = full_forward(
        model, static.prim_idx, static.feat, feat.vert_dyn,
        feat.edge_feat, feat.edge_src, feat.edge_dst, emask, nmask)

    rng = np.random.default_rng(4)
    perm = rng.permutation(n).astype(np.int32)      # old row i -> new row perm[i]
    order = np.argsort(perm)                        # new row p holds old row order[p]
    src_p = perm[feat.edge_src].astype(np.int32)
    dst_p = perm[feat.edge_dst].astype(np.int32)
    hs_p, pooled_p, _ = full_forward(
        model, static.prim_idx[order], static.feat[order],
        feat.vert_dyn[order], feat.edge_feat, src_p, dst_p, emask, nmask)
    for h, hp in zip(hs, hs_p):
        np.testing.assert_allclose(np.asarray(hp)[perm], np.asarray(h),
                                   atol=2e-4)
    np.testing.assert_allclose(np.asarray(pooled_p)[perm], np.asarray(pooled),
                               atol=2e-4)
