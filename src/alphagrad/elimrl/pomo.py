"""alphagrad.elimrl.pomo -- M3: POMO over VERTEX macro-actions (task #117).

Policy
------
The M2 :class:`~alphagrad.elimrl.encoder.ElimGNN` is reused through its PURE
functions (``full_forward`` + ``vertex_embed``) -- not through
:class:`~alphagrad.elimrl.encoder.EncoderRuntime` -- so sampling and the
update share one bucketed, vmapped forward and the whole policy stays
differentiable end-to-end. On top sits a preference-conditioned pointer head:

    logit_a = score_mlp([vertex_emb_a ; graph_emb ; lam_mlp(lam)])

lam = (lam_latency, lam_memory) on the 2-simplex, drawn Dirichlet(1, 1) once
per update. CONDITIONING POINT = the HEAD: the lam embedding is concatenated
onto every candidate row next to the (pooled) graph embedding, so the scores
-- not the graph representation -- bend with the preference; the GNN trunk
stays preference-free and its cached structure keeps one meaning. Illegal and
padded candidates are masked to ``-inf`` (NOT a finite sentinel -- the -1e9
mask/sentinel collision that produced cos==0 across v22-v28).

Start-vertex scheme
-------------------
M1 measured the coloured automorphism group of this target to be TRIVIAL, so
POMO's symmetry-induced orbit starts do not exist here. Documented fallback:
the N trajectories of an update are forced to N DISTINCT first vertices drawn
WITHOUT REPLACEMENT from the policy's own first-step distribution (one shared
Gumbel perturbation + top-N == exact Gumbel-top-k sampling without
replacement). Because these starts are policy choices rather than
symmetry-imposed relabelings, the first step's log-prob IS kept inside the
trajectory log-prob -- a deliberate deviation from POMO's convention of
dropping the forced start; the shared-mean baseline still centres the
advantage, and keeping the term lets the policy learn which starts are good.

POMO update
-----------
    R_n  = -(lam1 * ln latency_n + lam2 * ln memory_bytes_n)   REAL measurements
    b    = mean_n R_n                 (shared mean; NO critic, no PopArt)
    loss = -mean_n (R_n - b) * log p(tau_n)  -  ent_coef * mean-step-entropy

Both terms decompose additively over trajectories, so
:meth:`PomoRunner.update` can accumulate the gradient in trajectory chunks
(``grad_chunk``) at identical arithmetic -- the [N*L]-row backward otherwise
dominates trainer memory.

Infeasible scoring (scale-free, documented choice): an infeasible / errored /
missing measurement gets

    R_n = min(feasible R in this update) - margin,
    margin = max(std(feasible R), 0.05)

R lives in log space, so the 0.05 floor is a ~5% multiplicative penalty and
the std term adapts to the batch spread -- no invented huge constant, no
dependence on absolute latency/memory units, and (unlike -inf or a fixed
-1e9) it cannot NaN or swamp the update. An all-infeasible update produces no
usable advantage and is SKIPPED by the trainer.

Episodes: nominal length L = |jacve_vertices| with a per-step mask. An episode
whose line dag terminates early (argnums-pruned no-op vertices left over) pads
the remaining steps with a dummy row carrying ONE fake legal candidate, so the
padded softmax stays finite; the step mask zeroes its loss contribution.

Measurement economy: :class:`MeasureCache` keys results on the M1 Foata
``trace_key`` of the realized order (restricted to the pruned elimination
graph's eliminable set -- vertices outside it are structural no-ops for the
measured artifact), so trace-equivalent plans cost ONE unique measurement.
Passing ``graph=None`` falls back to exact-order keys.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from alphagrad.elimrl.encoder import (
    CAND_BUCKETS, EDGE_BUCKETS, NODE_BUCKETS, ElimGNN, _pad1, _pad2,
    full_forward, pick_bucket, vertex_embed,
)
from alphagrad.elimrl.features import (
    DYN_VERTEX_DIM, EDGE_DIM, StaticFeatures, StepFeatures, extract,
)
from alphagrad.elimrl.symmetry import ElimGraph, eliminate_vertex, trace_key

INFEASIBLE_MARGIN_FLOOR = 0.05        # log-space ~5% multiplicative penalty


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------
class PomoPolicy(eqx.Module):
    """ElimGNN trunk + preference-conditioned pointer head over legal vertices."""
    gnn: ElimGNN
    lam_mlp: eqx.nn.MLP
    score_mlp: eqx.nn.MLP

    def __init__(self, gnn: ElimGNN, lam_dim: int = 16, width: int = 64, *, key):
        k1, k2 = jax.random.split(key)
        self.gnn = gnn
        self.lam_mlp = eqx.nn.MLP(2, lam_dim, width, 1, key=k1)
        self.score_mlp = eqx.nn.MLP(
            gnn.cand_dim + gnn.graph_head.out_size + lam_dim, 1, width, 2,
            key=k2)


def step_scores(policy: PomoPolicy, prim, stat, dyn, e, src, dst, emask,
                nmask, vrows, vmask, lam):
    """Masked logits over the (padded) legal-vertex candidates of ONE state."""
    hs, pooled, graph = full_forward(
        policy.gnn, prim, stat, dyn, e, src, dst, emask, nmask)
    vert = vertex_embed(policy.gnn, hs[-1], pooled, vrows, vmask)
    ctx = jnp.concatenate([graph, policy.lam_mlp(lam)])
    x = jnp.concatenate(
        [vert, jnp.broadcast_to(ctx, (vert.shape[0], ctx.shape[0]))], axis=-1)
    raw = jax.vmap(policy.score_mlp)(x)[:, 0]
    return jnp.where(vmask > 0, raw, -jnp.inf)


def batch_scores(policy: PomoPolicy, prim, stat, nmask, lam,
                 dyn, e, src, dst, emask, vrows, vmask):
    """vmap of :func:`step_scores` over a leading step-batch axis."""
    def one(dyn_, e_, s_, d_, em_, vr_, vm_):
        return step_scores(policy, prim, stat, dyn_, e_, s_, d_, em_,
                           nmask, vr_, vm_, lam)
    return jax.vmap(one)(dyn, e, src, dst, emask, vrows, vmask)


def _logp_ent(logits, aidx, vmask):
    """(log p(a), entropy) of one masked categorical. ``-inf``-safe: the
    entropy uses where-guarded log-probs so ``0 * -inf`` never appears."""
    lp = jax.nn.log_softmax(logits)
    safe = jnp.where(vmask > 0, lp, 0.0)
    p = jnp.where(vmask > 0, jnp.exp(safe), 0.0)
    ent = -jnp.sum(p * safe)
    return jnp.take(lp, aidx), ent


def shared_baseline_advantage(rewards) -> jnp.ndarray:
    """POMO's critic-free advantage: R_n - mean_n R_n (exactly zero-mean)."""
    r = jnp.asarray(rewards)
    return r - jnp.mean(r)


def pomo_surrogate(traj_logp, rewards):
    """POMO policy-gradient surrogate ``-mean_n (R_n - mean R) logp(tau_n)``.
    ``rewards`` are constants (stop_gradient); the baseline is the shared
    mean, so no critic and no value loss exist anywhere in M3."""
    adv = shared_baseline_advantage(jax.lax.stop_gradient(jnp.asarray(rewards)))
    return -jnp.mean(adv * traj_logp)


def _loss_core(policy, prim, stat, nmask, lam,
               dyn, e, src, dst, emask, vrows, vmask, aidx,
               adv, step_mask, ent_coef, n_total, ent_denom):
    """Chunk-additive POMO loss over an [n*L]-row step batch (n-major).

    ``adv`` are the (already centred, constant) advantages of the chunk's
    trajectories; ``n_total`` / ``ent_denom`` are the FULL-update normalizers,
    so summing this over a partition of the trajectories reproduces the
    single-shot loss (and hence its gradient) exactly.
    """
    logits = batch_scores(policy, prim, stat, nmask, lam,
                          dyn, e, src, dst, emask, vrows, vmask)
    lp, ent = jax.vmap(_logp_ent)(logits, aidx, vmask)
    n, l = step_mask.shape
    traj_lp = (lp.reshape(n, l) * step_mask).sum(axis=1)
    pg = -jnp.sum(jax.lax.stop_gradient(adv) * traj_lp) / n_total
    ent_sum = (ent.reshape(n, l) * step_mask).sum()
    return pg - ent_coef * ent_sum / ent_denom


def pomo_loss(policy: PomoPolicy, prim, stat, nmask, lam,
              dyn, e, src, dst, emask, vrows, vmask, aidx,
              rewards, step_mask, ent_coef):
    """Full-batch POMO loss (shared-mean baseline computed internally)."""
    adv = shared_baseline_advantage(
        jax.lax.stop_gradient(jnp.asarray(rewards)))
    n = step_mask.shape[0]
    return _loss_core(policy, prim, stat, nmask, lam, dyn, e, src, dst, emask,
                      vrows, vmask, aidx, adv, step_mask, ent_coef,
                      jnp.asarray(n, jnp.float32),
                      jnp.maximum(step_mask.sum(), 1.0))


# ---------------------------------------------------------------------------
# rewards
# ---------------------------------------------------------------------------
def score_rewards(results: Sequence[Optional[dict]], lam,
                  margin_floor: float = INFEASIBLE_MARGIN_FLOOR):
    """``R_n = -(lam1 ln latency_ns + lam2 ln mem_total_bytes)`` per result.

    Infeasible / errored / missing results score ``min(feasible R) - margin``
    with ``margin = max(std(feasible R), margin_floor)``. Returns
    ``(R, feasible_mask)``; ``(None, mask)`` when NO result is feasible (the
    trainer then skips the update -- there is no usable advantage).
    """
    lam1, lam2 = float(lam[0]), float(lam[1])
    n = len(results)
    R = np.full(n, np.nan, np.float64)
    feas = np.zeros(n, bool)
    for i, r in enumerate(results):
        if (r and r.get("status") == "ok" and r.get("executed")
                and r.get("latency_ns") and r.get("mem_total_bytes")):
            R[i] = -(lam1 * math.log(float(r["latency_ns"]))
                     + lam2 * math.log(max(float(r["mem_total_bytes"]), 1.0)))
            feas[i] = True
    if not feas.any():
        return None, feas
    margin = max(float(np.std(R[feas])), float(margin_floor))
    R[~feas] = float(R[feas].min()) - margin
    return R, feas


# ---------------------------------------------------------------------------
# measurement cache (trace-key'd)
# ---------------------------------------------------------------------------
class MeasureCache:
    """Dedup REAL measurements on the Foata trace key of the realized order.

    ``measure_fn(order)`` is invoked ONLY on cache misses and must return the
    measurement-worker result dict. Orders are restricted to
    ``graph.eliminable`` before keying (vertices pruned out of the elimination
    graph are structural no-ops for the measured artifact). ``graph=None``
    keys on the exact order tuple instead.
    """

    def __init__(self, graph: Optional[ElimGraph],
                 measure_fn: Callable[[list], dict]):
        self.graph = graph
        self._elim = frozenset(graph.eliminable) if graph is not None else None
        self.measure_fn = measure_fn
        self.cache: dict = {}
        self.unique = 0
        self.hits = 0

    def key(self, order: Sequence[int]):
        if self.graph is None:
            return tuple(int(v) for v in order)
        return trace_key([v for v in order if v in self._elim], self.graph)

    def measure_order(self, order: Sequence[int]) -> Tuple[dict, bool]:
        """Returns (result, was_cache_hit)."""
        k = self.key(order)
        if k in self.cache:
            self.hits += 1
            return self.cache[k], True
        res = self.measure_fn([int(v) for v in order])
        res = {kk: v for kk, v in res.items() if kk != "latency_samples_ns"}
        self.cache[k] = res
        self.unique += 1
        return res, False

    @property
    def hit_rate(self) -> float:
        tot = self.unique + self.hits
        return self.hits / tot if tot else 0.0


# ---------------------------------------------------------------------------
# order helpers (baselines + plan realization)
# ---------------------------------------------------------------------------
def markowitz_order(graph: ElimGraph) -> List[int]:
    """Greedy min-Markowitz order over the elimination-graph connectivity
    (the cautionary baseline; ties break to the smallest vertex id)."""
    succ, pred = graph.mutable()
    remaining = set(graph.eliminable)
    order: List[int] = []
    while remaining:
        v = min(remaining, key=lambda x: (len(pred[x]) * len(succ[x]), x))
        eliminate_vertex(succ, pred, v)
        remaining.remove(v)
        order.append(v)
    return order


def order_to_plan(env, order: Sequence[int]) -> Tuple[list, List[int]]:
    """Replay ``order`` (any priority list) on a fresh env, skipping already
    eliminated vertices, appending the unmentioned jacve vertices in id order,
    and stopping at termination. Returns ``(plan, realized_order)`` with the
    plan in the ``elim_plan`` wire format."""
    env.reset()
    seen = set(int(v) for v in order)
    full = [int(v) for v in order] + [j for j in sorted(env.jacve_vertices)
                                      if j not in seen]
    realized: List[int] = []
    for j in full:
        if env.done:
            break
        if j not in env.jacve_vertices or j in env.eliminated_vertices():
            continue
        env.apply(("V", int(j)))
        realized.append(int(j))
    if not env.done:
        raise RuntimeError("order_to_plan did not terminate the elimination")
    return [["V", j] for j in realized], realized


# ---------------------------------------------------------------------------
# rollout / update runner
# ---------------------------------------------------------------------------
@dataclass
class StepBatch:
    """[S = N*L]-row step batch, n-major (row n*L + t = trajectory n, step t)."""
    dyn: np.ndarray        # [S, n_pad, DYN_VERTEX_DIM]
    e: np.ndarray          # [S, EB, EDGE_DIM]
    src: np.ndarray        # [S, EB]
    dst: np.ndarray        # [S, EB]
    emask: np.ndarray      # [S, EB]
    vrows: np.ndarray      # [S, VB]
    vmask: np.ndarray      # [S, VB]
    aidx: np.ndarray       # [S]

    def rows_of(self, idx: Sequence[int], L: int) -> "StepBatch":
        """Sub-batch holding only the trajectories in ``idx``."""
        take = np.concatenate([np.arange(i * L, (i + 1) * L) for i in idx])
        return StepBatch(*(getattr(self, f)[take] for f in
                           ("dyn", "e", "src", "dst", "emask", "vrows",
                            "vmask", "aidx")))


class PomoRunner:
    """Owns the padded static tables, the bucket state and the jitted sampling
    / update kernels. One runner per (target, n_traj)."""

    def __init__(self, static: StaticFeatures, n_traj: int, n_steps: int,
                 node_buckets=NODE_BUCKETS, edge_buckets=EDGE_BUCKETS,
                 cand_buckets=CAND_BUCKETS, edge_hint: int = 1,
                 cand_hint: int = 0):
        self.static = static
        self.N = int(n_traj)
        self.L = int(n_steps)
        self.edge_buckets = edge_buckets
        self.cand_buckets = cand_buckets
        self.EB = pick_bucket(edge_hint, edge_buckets)
        # candidate count is |legal vertices|, which is INDEPENDENT of the
        # nominal episode length (they coincide only for a full episode).
        self.VB = pick_bucket(max(int(cand_hint) or self.L, 1), cand_buckets)
        self.n_pad = pick_bucket(static.n_rows, node_buckets)
        self.prim = jnp.asarray(_pad1(static.prim_idx, self.n_pad))
        self.stat = jnp.asarray(_pad2(static.feat, self.n_pad))
        nmask = np.zeros(self.n_pad, np.float32)
        nmask[:static.n_rows] = 1.0
        self.nmask = jnp.asarray(nmask)
        self._scores = eqx.filter_jit(batch_scores)
        self._vg = eqx.filter_jit(eqx.filter_value_and_grad(_loss_core))

    # -- padding ------------------------------------------------------------
    def _pad_feat(self, feat: StepFeatures, aidx: int):
        dyn = _pad2(feat.vert_dyn, self.n_pad)
        e = _pad2(feat.edge_feat, self.EB)
        src = _pad1(feat.edge_src, self.EB)
        dst = _pad1(feat.edge_dst, self.EB)
        emask = np.zeros(self.EB, np.float32)
        emask[:len(feat.edge_src)] = 1.0
        vrows = _pad1(feat.vert_rows, self.VB)
        vmask = np.zeros(self.VB, np.float32)
        vmask[:len(feat.vert_rows)] = 1.0
        return dyn, e, src, dst, emask, vrows, vmask, np.int32(aidx)

    def _dummy_row(self):
        """Padded no-op row: ONE fake legal candidate keeps the softmax finite;
        the step mask zeroes its loss contribution."""
        vmask = np.zeros(self.VB, np.float32)
        vmask[0] = 1.0
        return (np.zeros((self.n_pad, DYN_VERTEX_DIM), np.float32),
                np.zeros((self.EB, EDGE_DIM), np.float32),
                np.zeros(self.EB, np.int32), np.zeros(self.EB, np.int32),
                np.zeros(self.EB, np.float32),
                np.zeros(self.VB, np.int32), vmask, np.int32(0))

    # -- rollout ------------------------------------------------------------
    def rollout(self, envs, policy: PomoPolicy, lam, rng,
                greedy: bool = False, forced_starts: bool = True):
        """Lockstep episodes over ``envs`` (len <= N; missing slots padded).

        Returns ``(StepBatch, step_mask [len(envs), L], orders)``; episode n's
        realized vertex order is ``orders[n]``.
        """
        n_env = len(envs)
        assert n_env <= self.N
        lam_j = jnp.asarray(np.asarray(lam, np.float32))
        for env in envs:
            env.reset()
        step_mask = np.zeros((n_env, self.L), np.float32)
        orders: List[List[int]] = [[] for _ in range(n_env)]
        recorded: List[List[Optional[tuple]]] = [[] for _ in range(n_env)]
        done = [False] * n_env

        for t in range(self.L):
            feats: List[Optional[tuple]] = []
            for i, env in enumerate(envs):
                if done[i] or env.done:
                    done[i] = True
                    feats.append(None)
                    continue
                st = env.state()
                if st.done or not st.legal_vertices:
                    done[i] = True
                    feats.append(None)
                    continue
                feats.append((extract(st, self.static), st.legal_vertices))
            if all(f is None for f in feats):
                for i in range(n_env):
                    recorded[i].append(None)
                continue
            max_e = max(len(f[0].edge_src) for f in feats if f is not None)
            if max_e > self.EB:
                self.EB = pick_bucket(max_e, self.edge_buckets)
            max_v = max(len(f[1]) for f in feats if f is not None)
            if max_v > self.VB:
                self.VB = pick_bucket(max_v, self.cand_buckets)

            rows = [self._dummy_row() if feats[i] is None
                    else self._pad_feat(feats[i][0], 0) for i in range(n_env)]
            while len(rows) < self.N:
                rows.append(self._dummy_row())
            dyn, e, src, dst, emask, vrows, vmask, _ = (
                np.stack(x) for x in zip(*rows))
            logits = np.asarray(self._scores(
                policy, self.prim, self.stat, self.nmask, lam_j,
                jnp.asarray(dyn), jnp.asarray(e), jnp.asarray(src),
                jnp.asarray(dst), jnp.asarray(emask), jnp.asarray(vrows),
                jnp.asarray(vmask)))

            # forced DISTINCT first vertices: one shared Gumbel perturbation,
            # top-N == sampling without replacement from p(first vertex).
            ranked0 = None
            if t == 0 and forced_starts and not greedy and feats[0] is not None:
                lv0 = feats[0][1]
                g = rng.gumbel(size=len(lv0))
                ranked0 = np.argsort(logits[0, :len(lv0)] + g)[::-1]
            for i in range(n_env):
                if feats[i] is None:
                    recorded[i].append(None)
                    continue
                legal = feats[i][1]
                nv = len(legal)
                if ranked0 is not None:
                    a = int(ranked0[i % len(ranked0)])
                elif greedy:
                    a = int(np.argmax(logits[i, :nv]))
                else:
                    a = int(np.argmax(logits[i, :nv] + rng.gumbel(size=nv)))
                j = int(legal[a])
                recorded[i].append((feats[i][0], a))
                envs[i].apply(("V", j))
                orders[i].append(j)
                step_mask[i, t] = 1.0

        # finalize against the FINAL edge bucket (monotone within a rollout)
        all_rows = []
        for i in range(n_env):
            for t in range(self.L):
                rec = recorded[i][t] if t < len(recorded[i]) else None
                all_rows.append(self._dummy_row() if rec is None
                                else self._pad_feat(rec[0], rec[1]))
        cols = [np.stack(x) for x in zip(*all_rows)]
        batch = StepBatch(dyn=cols[0], e=cols[1], src=cols[2], dst=cols[3],
                          emask=cols[4], vrows=cols[5], vmask=cols[6],
                          aidx=cols[7])
        return batch, step_mask, orders

    # -- update -------------------------------------------------------------
    def _chunk_grad(self, policy, batch: StepBatch, lam_j, adv, step_mask,
                    ent_coef, n_total, ent_denom):
        return self._vg(
            policy, self.prim, self.stat, self.nmask, lam_j,
            jnp.asarray(batch.dyn), jnp.asarray(batch.e),
            jnp.asarray(batch.src), jnp.asarray(batch.dst),
            jnp.asarray(batch.emask), jnp.asarray(batch.vrows),
            jnp.asarray(batch.vmask), jnp.asarray(batch.aidx),
            jnp.asarray(np.asarray(adv, np.float32)),
            jnp.asarray(np.asarray(step_mask, np.float32)),
            jnp.asarray(ent_coef, jnp.float32),
            jnp.asarray(n_total, jnp.float32),
            jnp.asarray(ent_denom, jnp.float32))

    def update(self, policy: PomoPolicy, optim, opt_state,
               batch: StepBatch, lam, rewards, step_mask,
               ent_coef: float, grad_chunk: int = 0):
        """One POMO gradient step (shared-mean baseline, no critic).

        ``grad_chunk`` > 0 accumulates the gradient over chunks of that many
        trajectories -- arithmetically identical, bounded memory.
        Returns ``(policy, opt_state, loss, grad_norm)``.
        """
        rewards = np.asarray(rewards, np.float64)
        adv_all = rewards - rewards.mean()
        n_total = float(len(rewards))
        ent_denom = max(float(np.asarray(step_mask).sum()), 1.0)
        lam_j = jnp.asarray(np.asarray(lam, np.float32))

        chunk = int(grad_chunk) if grad_chunk and grad_chunk > 0 else len(rewards)
        loss_tot = 0.0
        grads = None
        for s in range(0, len(rewards), chunk):
            idx = list(range(s, min(s + chunk, len(rewards))))
            sub = batch.rows_of(idx, self.L)
            loss_c, g_c = self._chunk_grad(
                policy, sub, lam_j, adv_all[idx], np.asarray(step_mask)[idx],
                ent_coef, n_total, ent_denom)
            loss_tot += float(loss_c)
            grads = g_c if grads is None else jax.tree_util.tree_map(
                lambda a, b: a + b, grads, g_c)

        gnorm = float(jnp.sqrt(sum(
            jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads))))
        params = eqx.filter(policy, eqx.is_inexact_array)
        updates, opt_state = optim.update(grads, opt_state, params)
        policy = eqx.apply_updates(policy, updates)
        return policy, opt_state, loss_tot, gnorm


# ---------------------------------------------------------------------------
# Pareto front + hypervolume (2D, minimization)
# ---------------------------------------------------------------------------
def pareto_front(points: Sequence[Tuple[float, float]]) -> List[int]:
    """Indices of the nondominated points (both axes minimized)."""
    idx = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    front: List[int] = []
    best_mem = math.inf
    for i in idx:
        if points[i][1] < best_mem:
            front.append(i)
            best_mem = points[i][1]
    return front


def hypervolume_2d(front_points: Sequence[Tuple[float, float]],
                   ref: Tuple[float, float]) -> float:
    """Dominated hypervolume vs the reference (nadir) point, normalized by
    ``ref`` on both axes so the result lies in [0, 1]; points that do not
    dominate the reference contribute nothing."""
    pts = sorted((p[0] / ref[0], p[1] / ref[1]) for p in front_points
                 if p[0] < ref[0] and p[1] < ref[1])
    hv = 0.0
    for k, (x, y) in enumerate(pts):
        x_next = pts[k + 1][0] if k + 1 < len(pts) else 1.0
        hv += max(0.0, min(x_next, 1.0) - x) * max(0.0, 1.0 - y)
    return hv
