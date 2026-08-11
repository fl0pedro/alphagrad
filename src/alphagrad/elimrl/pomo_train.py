"""alphagrad.elimrl.pomo_train -- M3 CLI: POMO over VERTEX macro-actions.

Process layout (ONE sbatch job, free measurement isolation)
-----------------------------------------------------------
    trainer  (this process)   JAX_PLATFORMS=cpu   tiny GNN policy, no CUDA ctx
      `-- measure worker      JAX_PLATFORMS=cuda  owns the GPU exclusively

The trainer pins itself to the CPU backend BEFORE importing jax and overrides
``JAX_PLATFORMS`` for the worker child; the handshake backend is asserted to
be non-CPU under ``--require-gpu`` (default), so a silently-CPU worker can
never produce a "GPU" campaign.

Run (smoke)::

    uv run --no-sync python -m alphagrad.elimrl.pomo_train \
        --out-dir ~/dsnn/elimrl_m3_smoke --updates 20 --n-traj 8 --baselines

Outputs in ``--out-dir``: ``measurements.jsonl`` (one row per measured
trajectory, incl. cache hits), ``updates.jsonl`` (per-update telemetry) and
``summary.json`` (baselines, best-so-far, Pareto front, hypervolume, gates).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


# ---------------------------------------------------------------------------
def _p(*a):
    print(*a, flush=True)


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", default="tlm", choices=("tlm", "tiny"))
    p.add_argument("--seq", type=int, default=32)
    p.add_argument("--dmodel", type=int, default=128)
    p.add_argument("--vocab", type=int, default=1024)
    # POMO
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--n-traj", type=int, default=8)
    p.add_argument("--max-unique", type=int, default=4000)
    p.add_argument("--max-hours", type=float, default=23.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-chunk", type=int, default=2,
                   help="trajectories per gradient-accumulation chunk (0=all)")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--greedy-every", type=int, default=10,
                   help="every k updates also measure ONE greedy latency-only "
                        "rollout (lam=(1,0)); 0 disables")
    p.add_argument("--no-forced-starts", action="store_true")
    p.add_argument("--exact-cache-key", action="store_true",
                   help="key the cache on the exact order, not the trace key")
    # measurement
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--budget-gb", type=float, default=24.0)
    p.add_argument("--timeout", type=float, default=1500.0)
    p.add_argument("--worker-platforms", default="cuda")
    p.add_argument("--require-gpu", dest="require_gpu", action="store_true",
                   default=True)
    p.add_argument("--no-require-gpu", dest="require_gpu",
                   action="store_false")
    # baselines
    p.add_argument("--baselines", action="store_true",
                   help="measure jacve rev/fwd, jacfe rev, jax.grad, "
                        "jax.jacrev, min-Markowitz and --random-baselines "
                        "random orders through the SAME worker first")
    p.add_argument("--random-baselines", type=int, default=32)
    p.add_argument("--ref-latency-ns", type=float, default=128400.0)
    p.add_argument("--ref-mem-bytes", type=float, default=34.8 * 2 ** 20)
    return p


# ---------------------------------------------------------------------------
def _target_spec(a):
    if a.target == "tiny":
        return {"builder": "alphagrad.elimrl.baselines:tiny_target"}, {}
    kw = {"seq": a.seq, "dmodel": a.dmodel, "vocab": a.vocab}
    return {"builder": "alphagrad.elimrl.baselines:tlm_target",
            "kwargs": kw}, kw


def _lat(r):
    return None if not r else r.get("latency_ns")


def _mem(r):
    return None if not r else r.get("mem_total_bytes")


def _fmt(r):
    l, m = _lat(r), _mem(r)
    return (f"{(l or 0) / 1e3:9.1f}us {(m or 0) / 2 ** 20:8.1f}MB "
            f"{r.get('status', '?') if r else 'none':10s}")


# ---------------------------------------------------------------------------
def main(argv=None):
    a = build_argparser().parse_args(argv)
    # trainer stays on CPU (the worker child owns the GPU)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.makedirs(a.out_dir, exist_ok=True)

    import numpy as np
    import jax
    import optax
    import equinox as eqx

    from alphagrad.elimrl.baselines import tiny_target, tlm_target
    from alphagrad.elimrl.env import ElimEnv
    from alphagrad.elimrl.encoder import ElimGNN
    from alphagrad.elimrl.features import build_static
    from alphagrad.elimrl.measure_worker import MeasureClient
    from alphagrad.elimrl.symmetry import build_elim_graph
    from alphagrad.elimrl import pomo as P

    t_start = time.time()
    rng = np.random.default_rng(a.seed)
    target, kw = _target_spec(a)

    # -- target + envs (trainer side, CPU, SYMBOLIC: graph dynamics only) ----
    build = tiny_target if a.target == "tiny" else tlm_target
    fn, args_, argnums = build(**kw)
    t0 = time.perf_counter()
    envs = [ElimEnv(fn, args_, argnums, vertex_only=True, symbolic=True)
            for _ in range(a.n_traj)]
    env0 = envs[0]
    L = len(env0.jacve_vertices)
    static = build_static(env0)
    graph = None if a.exact_cache_key else build_elim_graph(fn, args_, argnums)
    if graph is not None and not set(graph.eliminable) <= set(env0.jacve_vertices):
        _p("[warn] elimination-graph eliminable set is NOT a subset of the "
           "env's jacve vertices -- falling back to EXACT cache keys")
        graph = None
    _p(f"[m3] target={a.target} L={L} rows={static.n_rows} "
       f"envs={len(envs)} build={time.perf_counter() - t0:.1f}s "
       f"cache_key={'trace' if graph is not None else 'exact'}")

    # -- policy ---------------------------------------------------------------
    key = jax.random.PRNGKey(a.seed)
    k_gnn, k_pol = jax.random.split(key)
    gnn = ElimGNN(len(static.prim_vocab), hidden=a.hidden, width=a.width,
                  key=k_gnn)
    policy = P.PomoPolicy(gnn, width=a.width, key=k_pol)
    optim = optax.chain(optax.clip_by_global_norm(a.grad_clip),
                        optax.adam(a.lr))
    opt_state = optim.init(eqx.filter(policy, eqx.is_inexact_array))
    runner = P.PomoRunner(static, a.n_traj, L, edge_hint=len(env0.state().edges))
    _p(f"[m3] policy hidden={a.hidden} n_pad={runner.n_pad} EB={runner.EB} "
       f"VB={runner.VB} params="
       f"{sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(policy, eqx.is_inexact_array)))}")

    # -- measurement worker (GPU) --------------------------------------------
    client = MeasureClient(env={"JAX_PLATFORMS": a.worker_platforms},
                           startup_timeout=900.0,
                           default_timeout=a.timeout)
    _p(f"[m3] worker backend={client.backend!r}")
    if a.require_gpu and str(client.backend).lower() in ("cpu", "none", ""):
        client.close()
        raise SystemExit(f"worker backend is {client.backend!r}, expected a "
                         "cuda/gpu device (pass --no-require-gpu to override)")

    budget = a.budget_gb * 2 ** 30
    common = dict(target=target, budget_bytes=budget, reps=a.reps,
                  inner=a.inner, timeout=a.timeout)

    meas_path = os.path.join(a.out_dir, "measurements.jsonl")
    upd_path = os.path.join(a.out_dir, "updates.jsonl")
    meas_f = open(meas_path, "a", buffering=1)
    upd_f = open(upd_path, "a", buffering=1)

    meas_wall = []          # per-unique-measurement wall seconds

    def measure_fn(order):
        t = time.perf_counter()
        r = client.measure(method="elim_plan",
                           plan=[["V", int(j)] for j in order],
                           vertex_only=True, **common)
        r["wall_s"] = time.perf_counter() - t
        meas_wall.append(r["wall_s"])
        return r

    cache = P.MeasureCache(graph, measure_fn)
    archive = []          # (latency_ns, mem_bytes) of POMO-measured feasible plans
    archive_meta = []

    def record(tag, upd, order, res, hit, lam):
        row = {"tag": tag, "update": upd, "lam": [float(x) for x in lam],
               "cached": bool(hit), "unique": cache.unique,
               "status": res.get("status"), "reason": res.get("reason"),
               "latency_ns": _lat(res), "mem_total_bytes": _mem(res),
               "compile_s": res.get("compile_s"), "wall_s": res.get("wall_s"),
               "order": [int(j) for j in order]}
        meas_f.write(json.dumps(row) + "\n")
        # the archive = everything the POLICY produced (sampled + greedy);
        # baseline rows ("random", "markowitz") are deliberately excluded.
        if tag in ("pomo", "greedy") and res.get("status") == "ok" and _lat(res):
            archive.append((float(_lat(res)), float(_mem(res) or 0.0)))
            archive_meta.append({"update": upd, "lam": [float(x) for x in lam]})

    # -- baselines through the SAME worker -----------------------------------
    baselines = {}
    if a.baselines:
        _p("\n[m3] baselines (same worker, same reps/inner):")
        specs = {
            "jacve_rev": dict(method="jacve", order="rev"),
            "jacve_fwd": dict(method="jacve", order="fwd"),
            "jacfe_rev": dict(method="jacfe", order="rev",
                              check_against={"method": "jacve", "order": "rev"}),
            "jax.grad": dict(method="jax.grad"),
            "jax.jacrev": dict(method="jax.jacrev"),
        }
        for name, spec in specs.items():
            r = client.measure(**{**common, **spec})
            r.pop("latency_samples_ns", None)
            baselines[name] = r
            _p(f"  {name:22s} {_fmt(r)} compile={r.get('compile_s', 0):6.1f}s"
               + (f" cos={r['check_cos']:.6f}" if "check_cos" in r else ""))
        # min-Markowitz (cautionary) + random orders -- measured as elim_plans
        if graph is not None:
            mk = P.markowitz_order(graph)
            _, mk_real = P.order_to_plan(env0, mk)
            r = measure_fn(mk_real)
            r.pop("latency_samples_ns", None)
            baselines["min_markowitz"] = r
            _p(f"  {'min_markowitz':22s} {_fmt(r)}")
        rand_rows = []
        for i in range(a.random_baselines):
            perm = list(rng.permutation(sorted(env0.jacve_vertices)))
            _, real = P.order_to_plan(env0, perm)
            r = measure_fn(real)
            r.pop("latency_samples_ns", None)
            rand_rows.append(r)
            record("random", -1, real, r, False, (1.0, 0.0))
        ok = [r for r in rand_rows if r.get("status") == "ok" and _lat(r)]
        if ok:
            lat = np.array([_lat(r) for r in ok])
            mem = np.array([_mem(r) or 0.0 for r in ok])
            baselines["random"] = {
                "n": len(rand_rows), "n_ok": len(ok),
                "latency_ns": {"min": float(lat.min()), "median": float(np.median(lat)),
                               "mean": float(lat.mean()), "max": float(lat.max()),
                               "std": float(lat.std())},
                "mem_total_bytes": {"min": float(mem.min()),
                                    "median": float(np.median(mem)),
                                    "max": float(mem.max())}}
            _p(f"  {'random x' + str(len(ok)):22s} "
               f"min={lat.min() / 1e3:.1f}us median={np.median(lat) / 1e3:.1f}us "
               f"max={lat.max() / 1e3:.1f}us")
        _p("")

    ref_lat = (_lat(baselines.get("jacve_rev")) or a.ref_latency_ns)
    ref_mem = (_mem(baselines.get("jacve_rev")) or a.ref_mem_bytes)
    _p(f"[m3] reverse reference: {ref_lat / 1e3:.1f}us {ref_mem / 2 ** 20:.1f}MB "
       f"(gate <= {1.05 * ref_lat / 1e3:.1f}us)")

    def summarize(n_upd, stopped=None):
        best_lat = min((p[0] for p in archive), default=None)
        best_mem = min((p[1] for p in archive), default=None)
        front_idx = P.pareto_front(archive) if archive else []
        front = [archive[i] for i in front_idx]
        hv = (P.hypervolume_2d(front, (2 * ref_lat, 2 * ref_mem))
              if front else 0.0)
        dom = [p for p in front if p[0] <= ref_lat and p[1] <= ref_mem]
        rnd = baselines.get("random", {}).get("latency_ns", {})
        out = {
            "seed": a.seed, "target": a.target, "kwargs": kw,
            "updates_done": n_upd, "stopped": stopped,
            "unique_measurements": cache.unique,
            "cache_hits": cache.hits, "cache_hit_rate": cache.hit_rate,
            "measure_wall_s": {
                "n": len(meas_wall), "total": float(sum(meas_wall)),
                "mean": float(sum(meas_wall) / max(len(meas_wall), 1)),
                "max": float(max(meas_wall)) if meas_wall else 0.0},
            "worker_respawns": client.respawns,
            "wall_hours": (time.time() - t_start) / 3600.0,
            "baselines": baselines,
            "ref_latency_ns": ref_lat, "ref_mem_bytes": ref_mem,
            "best_latency_ns": best_lat, "best_mem_bytes": best_mem,
            "front": [{"latency_ns": p[0], "mem_total_bytes": p[1],
                       **archive_meta[i]} for i, p in zip(front_idx, front)],
            "front_size": len(front),
            "hypervolume_norm": hv,
            "front_frac_dominating_reverse": (len(dom) / len(front)) if front else 0.0,
            "gate_within_5pct_of_reverse": (
                bool(best_lat is not None and best_lat <= 1.05 * ref_lat)),
            "gate_beats_random": (
                bool(best_lat is not None and rnd
                     and best_lat < rnd.get("min", float("inf")))),
            "gate_unique_budget_ok": cache.unique <= a.max_unique,
        }
        with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
            json.dump(out, f, indent=2, default=str)
        return out

    # -- POMO loop -------------------------------------------------------------
    stopped = "updates"
    u = 0
    try:
        for u in range(1, a.updates + 1):
            if cache.unique >= a.max_unique:
                stopped = "max_unique"
                break
            if (time.time() - t_start) / 3600.0 > a.max_hours:
                stopped = "max_hours"
                break
            lam = rng.dirichlet((1.0, 1.0))
            t_roll = time.perf_counter()
            batch, step_mask, orders = runner.rollout(
                envs, policy, lam, rng,
                forced_starts=not a.no_forced_starts)
            t_meas = time.perf_counter()
            results, hits = [], 0
            for n, order in enumerate(orders):
                res, hit = cache.measure_order(order)
                hits += int(hit)
                results.append(res)
                record("pomo", u, order, res, hit, lam)
            t_upd = time.perf_counter()

            R, feas = P.score_rewards(results, lam)
            if R is None:
                _p(f"[u{u:04d}] ALL {len(results)} measurements infeasible -- "
                   "update skipped (no usable advantage)")
                upd_f.write(json.dumps(
                    {"update": u, "skipped": "all_infeasible",
                     "lam": list(map(float, lam)),
                     "unique": cache.unique}) + "\n")
                continue
            policy, opt_state, loss, gnorm = runner.update(
                policy, optim, opt_state, batch, lam, R, step_mask,
                a.ent_coef, grad_chunk=a.grad_chunk)
            t_end = time.perf_counter()

            lats = [_lat(r) for r in results if r.get("status") == "ok" and _lat(r)]
            best_lat = min((p[0] for p in archive), default=float("nan"))
            best_mem = min((p[1] for p in archive), default=float("nan"))
            row = {"update": u, "lam": list(map(float, lam)),
                   "mean_R": float(np.mean(R)), "max_R": float(np.max(R)),
                   "n_feasible": int(feas.sum()), "loss": loss,
                   "grad_norm": gnorm,
                   "batch_min_latency_ns": (min(lats) if lats else None),
                   "best_latency_ns": best_lat, "best_mem_bytes": best_mem,
                   "unique": cache.unique, "hits": cache.hits,
                   "hit_rate": cache.hit_rate,
                   "respawns": client.respawns,
                   "s_rollout": t_meas - t_roll, "s_measure": t_upd - t_meas,
                   "s_update": t_end - t_upd}
            upd_f.write(json.dumps(row) + "\n")
            _p(f"[u{u:04d}] lam=({lam[0]:.2f},{lam[1]:.2f}) meanR={row['mean_R']:+.3f} "
               f"loss={loss:+.4f} |g|={gnorm:.2e} feas={int(feas.sum())}/{len(results)} "
               f"batch_min={(min(lats) / 1e3 if lats else float('nan')):.1f}us "
               f"best={best_lat / 1e3:.1f}us/{best_mem / 2 ** 20:.1f}MB "
               f"uniq={cache.unique} hit={cache.hit_rate:.2f} "
               f"t={row['s_rollout']:.1f}/{row['s_measure']:.1f}/{row['s_update']:.1f}s")

            if a.greedy_every and u % a.greedy_every == 0:
                gl = (1.0, 0.0)
                _b, _sm, gorders = runner.rollout(
                    envs[:1], policy, gl, rng, greedy=True, forced_starts=False)
                gres, ghit = cache.measure_order(gorders[0])
                record("greedy", u, gorders[0], gres, ghit, gl)
                _p(f"          greedy(lam=1,0): {_fmt(gres)} "
                   f"{'(cached)' if ghit else ''}")
            if u % 10 == 0:
                summarize(u, None)
    except KeyboardInterrupt:
        stopped = "interrupt"
    finally:
        out = summarize(u, stopped)
        meas_f.close()
        upd_f.close()
        try:
            client.close()
        except Exception:
            pass

    _p("\n[m3] ===== SUMMARY =====")
    _p(f"  updates={out['updates_done']} stopped={out['stopped']} "
       f"unique={out['unique_measurements']} hit_rate={out['cache_hit_rate']:.3f} "
       f"respawns={out['worker_respawns']} wall={out['wall_hours']:.2f}h")
    if out["best_latency_ns"]:
        _p(f"  best latency {out['best_latency_ns'] / 1e3:.1f}us "
           f"({out['best_latency_ns'] / ref_lat:.3f}x reverse), "
           f"best mem {out['best_mem_bytes'] / 2 ** 20:.1f}MB")
    _p(f"  front={out['front_size']} hv={out['hypervolume_norm']:.4f} "
       f"frac_dominating_reverse={out['front_frac_dominating_reverse']:.3f}")
    _p(f"  GATE beats_random={out['gate_beats_random']} "
       f"within_5pct_reverse={out['gate_within_5pct_of_reverse']} "
       f"unique_budget_ok={out['gate_unique_budget_ok']}")
    _p("M3_SUMMARY " + json.dumps({k: v for k, v in out.items()
                                   if k not in ("front", "baselines")}))
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
