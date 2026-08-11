"""alphagrad.elimrl.encoder -- M2: edge-conditioned message-passing GNN over
the PRIMAL graph, bucket-compiled and incrementally updatable.

Model (:class:`ElimGNN`, equinox): learned primitive embedding + input
projection over [prim_embed ; static ; dynamic] vertex features, then 3
edge-conditioned layers (per edge s->d: msg_fwd([h_s, e]) summed into d and
msg_bwd([h_d, e]) summed into s; residual update MLP + LayerNorm), plus

    face candidate  e_a = face_head([h_i; h_j; h_k; edge_ij; edge_jk;
                                     log2cost; absorb])
    vertex candidate e_a = vertex_head([h_j; pooled incident edge features])
    graph embedding      = graph_head([masked mean ; masked max] of h^3)

Bucketed compilation (:class:`EncoderRuntime`): node / edge / candidate /
incremental-target counts are padded up to fixed bucket sizes and dispatched
to the smallest fitting bucket, so the number of XLA retraces is bounded by
the bucket grid, not by the episode. Padding is exact: padded edges carry
zero features and a zero mask (their messages are multiplied to 0 before the
segment sum -- adding 0.0 is exact in fp), padded candidate rows are masked
out, and padded incremental-target indices point one past the last row, which
JAX scatter DROPS (out-of-bounds updates are dropped; out-of-bounds gathers
clamp, and the resulting garbage rows are precisely the dropped ones).

Incremental update: after an elimination only the pred(j) x succ(j)
neighbourhood changes. The runtime diffs the extracted features
(features.changed_rows), seeds D0 = {rows with changed vertex features},
EP = {rows incident to changed edges}, and expands

    T1 = D0 u N(D0) u EP,   T_{l+1} = T_l u N(T_l)      (k = 3 hops)

re-embedding exactly the affected rows per layer against cached embeddings;
everything else is carried over. Full recompute stays available as the
reference path (encode_full).
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from alphagrad.elimrl.features import (
    DYN_VERTEX_DIM, EDGE_DIM, FACE_SCALAR_DIM, STATIC_DIM,
    StaticFeatures, StepFeatures, changed_rows, extract,
)

# default bucket grids (spec candidate buckets + power-of-2 overflow at run
# time for graphs that exceed the largest listed size)
CAND_BUCKETS = (2, 4, 8, 32, 128)
EDGE_BUCKETS = (32, 64, 128, 256, 512)
NODE_BUCKETS = (32, 64, 128, 256, 512)
TGT_BUCKETS = (2, 4, 8, 32, 128)


def pick_bucket(n: int, buckets: Optional[Sequence[int]]) -> int:
    """Smallest listed bucket >= n; doubles past the end; None = exact fit."""
    n = max(int(n), 1)
    if buckets is None:
        return n
    for b in buckets:
        if n <= b:
            return b
    b = buckets[-1]
    while b < n:
        b *= 2
    return b


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class EdgeCondLayer(eqx.Module):
    msg_fwd: eqx.nn.MLP
    msg_bwd: eqx.nn.MLP
    update: eqx.nn.MLP
    norm: eqx.nn.LayerNorm

    def __init__(self, hidden: int, edge_dim: int, width: int, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.msg_fwd = eqx.nn.MLP(hidden + edge_dim, hidden, width, 2, key=k1)
        self.msg_bwd = eqx.nn.MLP(hidden + edge_dim, hidden, width, 2, key=k2)
        self.update = eqx.nn.MLP(3 * hidden, hidden, width, 2, key=k3)
        self.norm = eqx.nn.LayerNorm(hidden)


class ElimGNN(eqx.Module):
    """3-layer edge-conditioned GNN + candidate/graph heads (see module doc)."""
    prim_embed: eqx.nn.Embedding
    in_proj: eqx.nn.Linear
    layers: Tuple[EdgeCondLayer, ...]
    face_head: eqx.nn.MLP
    vertex_head: eqx.nn.MLP
    graph_head: eqx.nn.MLP
    hidden: int = eqx.field(static=True)
    cand_dim: int = eqx.field(static=True)

    def __init__(self, n_prims: int, hidden: int = 128, cand_dim: int = 64,
                 graph_dim: int = 64, prim_dim: int = 16, width: int = 128,
                 n_layers: int = 3, *, key):
        ks = jax.random.split(key, n_layers + 5)
        self.hidden = hidden
        self.cand_dim = cand_dim
        self.prim_embed = eqx.nn.Embedding(n_prims, prim_dim, key=ks[0])
        self.in_proj = eqx.nn.Linear(prim_dim + STATIC_DIM + DYN_VERTEX_DIM,
                                     hidden, key=ks[1])
        self.layers = tuple(EdgeCondLayer(hidden, EDGE_DIM, width, ks[2 + i])
                            for i in range(n_layers))
        self.face_head = eqx.nn.MLP(3 * hidden + 2 * EDGE_DIM + FACE_SCALAR_DIM,
                                    cand_dim, width, 2, key=ks[n_layers + 2])
        self.vertex_head = eqx.nn.MLP(hidden + 2 * EDGE_DIM, cand_dim, width, 2,
                                      key=ks[n_layers + 3])
        self.graph_head = eqx.nn.MLP(2 * hidden, graph_dim, width, 2,
                                     key=ks[n_layers + 4])


# ---------------------------------------------------------------------------
# pure forward pieces (jitted by the runtime; model passed as a pytree so all
# same-shaped calls share one trace)
# ---------------------------------------------------------------------------
def embed_rows(model: ElimGNN, prim_idx, stat, dyn):
    pe = jax.vmap(model.prim_embed)(prim_idx)
    x = jnp.concatenate([pe, stat, dyn], axis=-1)
    return jax.vmap(model.in_proj)(x)


def layer_rows(layer: EdgeCondLayer, h_prev, e, src, dst, emask, tgt):
    """New layer-l rows for the target rows, aggregating over the given edges
    (which must include every edge incident to a target)."""
    n = h_prev.shape[0]
    mf = jax.vmap(layer.msg_fwd)(jnp.concatenate([h_prev[src], e], -1))
    mb = jax.vmap(layer.msg_bwd)(jnp.concatenate([h_prev[dst], e], -1))
    agg_in = jax.ops.segment_sum(mf * emask[:, None], dst, num_segments=n)
    agg_out = jax.ops.segment_sum(mb * emask[:, None], src, num_segments=n)
    h_t = h_prev[tgt]
    upd = jax.vmap(layer.update)(
        jnp.concatenate([h_t, agg_in[tgt], agg_out[tgt]], -1))
    return jax.vmap(layer.norm)(h_t + upd)


def pool_edges(e, src, dst, emask, n):
    """Per-vertex pooled incident edge features [in-sum ; out-sum]."""
    em = e * emask[:, None]
    return jnp.concatenate([jax.ops.segment_sum(em, dst, num_segments=n),
                            jax.ops.segment_sum(em, src, num_segments=n)], -1)


def full_forward(model: ElimGNN, prim_idx, stat, dyn, e, src, dst, emask, nmask):
    """Reference path: all layers over all rows. Returns (H0..H3, pooled_inc,
    graph_emb)."""
    n = prim_idx.shape[0]
    all_rows = jnp.arange(n)
    hs = [embed_rows(model, prim_idx, stat, dyn)]
    for layer in model.layers:
        hs.append(layer_rows(layer, hs[-1], e, src, dst, emask, all_rows))
    pooled = pool_edges(e, src, dst, emask, n)
    return tuple(hs), pooled, graph_pool(model, hs[-1], nmask)


def graph_pool(model: ElimGNN, h, nmask):
    m = nmask[:, None]
    mean = jnp.sum(h * m, 0) / jnp.maximum(jnp.sum(nmask), 1.0)
    mx = jnp.max(jnp.where(m > 0, h, -1e30), 0)
    return model.graph_head(jnp.concatenate([mean, mx], -1))


def inc_embed(model: ElimGNN, prim_idx_t, stat_t, dyn_t, tgt, h0_old):
    """Re-embed target rows into the cached H0 (padded tgt = n -> dropped)."""
    rows = embed_rows(model, prim_idx_t, stat_t, dyn_t)
    return h0_old.at[tgt].set(rows)


def inc_layer(layer: EdgeCondLayer, h_prev, h_out_old, e, src, dst, emask, tgt):
    rows = layer_rows(layer, h_prev, e, src, dst, emask, tgt)
    return h_out_old.at[tgt].set(rows)


def face_embed(model: ElimGNN, h, e, ui, wi, rows, scal, cmask):
    x = jnp.concatenate([h[rows[:, 0]], h[rows[:, 1]], h[rows[:, 2]],
                         e[ui], e[wi], scal], -1)
    return jax.vmap(model.face_head)(x) * cmask[:, None]


def vertex_embed(model: ElimGNN, h, pooled, rows, cmask):
    x = jnp.concatenate([h[rows], pooled[rows]], -1)
    return jax.vmap(model.vertex_head)(x) * cmask[:, None]


# ---------------------------------------------------------------------------
# runtime: bucketing, jit caches, incremental state
# ---------------------------------------------------------------------------
def _pad1(a: np.ndarray, n: int, fill=0):
    if len(a) == n:
        return a
    out = np.full((n,) + a.shape[1:], fill, a.dtype)
    out[:len(a)] = a
    return out


def _pad2(a: np.ndarray, n: int):
    if a.shape[0] == n:
        return a
    out = np.zeros((n,) + a.shape[1:], a.dtype)
    out[:a.shape[0]] = a
    return out


class EncoderRuntime:
    """Owns the jitted kernels, the bucket grids and the incremental cache.

    encode_full(state)         -- reference path, refreshes the cache.
    encode_incremental(state)  -- k-hop re-embedding against the cache
                                  (falls back to full on the first call).

    Every output dict carries `timings` (feat/forward/cand/total, ms) and
    `new_compiles` (bucket signatures first seen on this call).
    """

    def __init__(self, model: ElimGNN, static: StaticFeatures,
                 node_buckets=NODE_BUCKETS, edge_buckets=EDGE_BUCKETS,
                 cand_buckets=CAND_BUCKETS, tgt_buckets=TGT_BUCKETS):
        self.model = model
        self.static = static
        self.edge_buckets = edge_buckets
        self.cand_buckets = cand_buckets
        self.tgt_buckets = tgt_buckets
        self.n_pad = pick_bucket(static.n_rows, node_buckets)

        self._full = eqx.filter_jit(full_forward)
        self._inc_embed = eqx.filter_jit(inc_embed)
        self._inc_layer = eqx.filter_jit(inc_layer)
        self._pool_edges = eqx.filter_jit(pool_edges)
        self._graph_pool = eqx.filter_jit(graph_pool)
        self._face = eqx.filter_jit(face_embed)
        self._vertex = eqx.filter_jit(vertex_embed)

        n = self.n_pad
        self._prim_pad = _pad1(static.prim_idx, n)
        self._stat_pad = _pad2(static.feat, n)
        self._nmask = np.zeros(n, np.float32)
        self._nmask[:static.n_rows] = 1.0

        self._sigs: set = set()
        self._cache: Optional[dict] = None

    # -- bucket-signature accounting -------------------------------------
    def _sig(self, name: str, *dims: int) -> int:
        key = (name,) + dims
        if key in self._sigs:
            return 0
        self._sigs.add(key)
        return 1

    @property
    def compile_signatures(self) -> int:
        """Distinct (kernel, bucket-shape) signatures dispatched so far --
        an upper bound on and proxy for the number of XLA retraces."""
        return len(self._sigs)

    # -- shared padding helpers -------------------------------------------
    def _pad_edges(self, feat: StepFeatures):
        eb = pick_bucket(len(feat.edge_src), self.edge_buckets)
        e = _pad2(feat.edge_feat, eb)
        src = _pad1(feat.edge_src, eb)
        dst = _pad1(feat.edge_dst, eb)
        emask = np.zeros(eb, np.float32)
        emask[:len(feat.edge_src)] = 1.0
        return eb, e, src, dst, emask

    def _heads(self, feat: StepFeatures, h, pooled):
        new = 0
        nF, nV = len(feat.face_edge_u), len(feat.vert_rows)
        eb, e, _, _, _ = self._pad_edges(feat)
        fb = pick_bucket(nF, self.cand_buckets)
        new += self._sig("face", fb, eb)
        cmask = np.zeros(fb, np.float32)
        cmask[:nF] = 1.0
        face = self._face(self.model, h, jnp.asarray(e),
                          _pad1(feat.face_edge_u, fb), _pad1(feat.face_edge_w, fb),
                          _pad2(feat.face_rows, fb), _pad2(feat.face_scal, fb),
                          cmask)
        vb = pick_bucket(nV, self.cand_buckets)
        new += self._sig("vertex", vb)
        vmask = np.zeros(vb, np.float32)
        vmask[:nV] = 1.0
        vert = self._vertex(self.model, h, pooled, _pad1(feat.vert_rows, vb), vmask)
        return face, vert, new

    def _out(self, feat, hs, pooled, graph, face, vert, new, t0, t1, t2, t3):
        self._cache = {"feat": feat, "hs": hs, "pooled": pooled}
        return {
            "h": hs[-1], "graph": graph, "face_emb": face, "vertex_emb": vert,
            "n_rows": self.static.n_rows, "n_edges": len(feat.edge_src),
            "n_faces": len(feat.face_edge_u), "n_vertices": len(feat.vert_rows),
            "new_compiles": new,
            "timings": {"feat_ms": (t1 - t0) * 1e3, "forward_ms": (t2 - t1) * 1e3,
                        "cand_ms": (t3 - t2) * 1e3, "total_ms": (t3 - t0) * 1e3},
        }

    # -- full path ----------------------------------------------------------
    def encode_full(self, state) -> dict:
        t0 = time.perf_counter()
        feat = extract(state, self.static)
        t1 = time.perf_counter()
        eb, e, src, dst, emask = self._pad_edges(feat)
        new = self._sig("full", eb)
        hs, pooled, graph = self._full(
            self.model, self._prim_pad, self._stat_pad,
            _pad2(feat.vert_dyn, self.n_pad), jnp.asarray(e), src, dst,
            emask, self._nmask)
        jax.block_until_ready(hs[-1])
        t2 = time.perf_counter()
        face, vert, n2 = self._heads(feat, hs[-1], pooled)
        jax.block_until_ready((face, vert, graph))
        t3 = time.perf_counter()
        return self._out(feat, hs, pooled, graph, face, vert, new + n2,
                         t0, t1, t2, t3)

    # -- incremental path ----------------------------------------------------
    def _neighbors(self, rows: np.ndarray, src, dst) -> np.ndarray:
        m = np.isin(src, rows)
        m2 = np.isin(dst, rows)
        return np.union1d(dst[m], src[m2]).astype(np.int32)

    def encode_incremental(self, state) -> dict:
        if self._cache is None:
            return self.encode_full(state)
        t0 = time.perf_counter()
        feat = extract(state, self.static)
        prev = self._cache["feat"]
        d0, ep = changed_rows(prev, feat)
        src, dst = feat.edge_src, feat.edge_dst
        # affected sets per layer: T1 = D0 u N(D0) u EP, T_{l+1} = T_l u N(T_l)
        tsets = [d0]
        t = np.union1d(np.union1d(d0, self._neighbors(d0, src, dst)), ep)
        tsets.append(t.astype(np.int32))
        for _ in range(len(self.model.layers) - 1):
            t = np.union1d(t, self._neighbors(t, src, dst)).astype(np.int32)
            tsets.append(t)
        t1 = time.perf_counter()

        new = 0
        hs = list(self._cache["hs"])
        # layer 0: re-embed changed-input rows
        if len(d0):
            tb = pick_bucket(len(d0), self.tgt_buckets)
            new += self._sig("inc_embed", tb)
            tgt = _pad1(d0, tb, fill=self.n_pad)          # OOB pad -> dropped
            hs[0] = self._inc_embed(
                self.model, self._prim_pad[_pad1(d0, tb)],
                self._stat_pad[_pad1(d0, tb)],
                _pad2(feat.vert_dyn[d0], tb), tgt, hs[0])
        # layers 1..k: re-aggregate over edges incident to the affected rows
        for li, layer in enumerate(self.model.layers):
            tl = tsets[li + 1]
            if not len(tl):
                continue
            inc = np.isin(src, tl) | np.isin(dst, tl)
            idx = np.nonzero(inc)[0]
            seb = pick_bucket(len(idx), self.edge_buckets)
            tb = pick_bucket(len(tl), self.tgt_buckets)
            new += self._sig("inc_layer", seb, tb)
            emask = np.zeros(seb, np.float32)
            emask[:len(idx)] = 1.0
            hs[li + 1] = self._inc_layer(
                layer, hs[li], hs[li + 1],
                _pad2(feat.edge_feat[idx], seb), _pad1(src[idx], seb),
                _pad1(dst[idx], seb), emask, _pad1(tl, tb, fill=self.n_pad))
        # pooled incident edges + graph embedding (cheap O(E) refresh)
        eb, e, esrc, edst, emask = self._pad_edges(feat)
        new += self._sig("pool_edges", eb)
        pooled = self._pool_edges(jnp.asarray(e), esrc, edst, emask, self.n_pad)
        new += self._sig("graph_pool", self.n_pad)
        graph = self._graph_pool(self.model, hs[-1], self._nmask)
        jax.block_until_ready(hs[-1])
        t2 = time.perf_counter()
        face, vert, n2 = self._heads(feat, hs[-1], pooled)
        jax.block_until_ready((face, vert, graph))
        t3 = time.perf_counter()
        out = self._out(feat, tuple(hs), pooled, graph, face, vert, new + n2,
                        t0, t1, t2, t3)
        out["n_affected"] = [int(len(s)) for s in tsets]
        return out

    def reset(self):
        self._cache = None


def make_model(static: StaticFeatures, seed: int = 0, **kw) -> ElimGNN:
    return ElimGNN(len(static.prim_vocab), key=jax.random.PRNGKey(seed), **kw)
