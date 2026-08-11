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

Measurement protocol (M3 fixes #119 / #120 / #121)
--------------------------------------------------
#121  PAIRED: candidates are never compared against a job-start reference.
      A fresh ``jacve`` reverse reference is re-measured in the SAME worker
      (``--pair-mode``) and the optimised/archived quantity is the RATIO
      candidate/reference. Absolute microseconds and every reference sample
      stay in ``measurements.jsonl`` (``tag="ref"``) so drift is plottable.
#120  VERIFIED: a plan that sets a new best is re-measured interleaved with
      references AND numerically checked against ``jacve`` rev; a plan that
      fails the check is invalidated, excluded and loudly logged. Optional
      1-in-k auditing of ordinary plans via ``--check-every``.
#119  DROPPED, not floored: measurements that produced no verdict (Triton
      compile failure, worker death, broken pairing, failed numeric check)
      are removed from the POMO batch and the shared baseline is renormalised
      over the survivors; OOM / predicted-memory / timeout keep the floor
      score. Fewer than ``--min-survivors`` survivors skips the update.

Outputs in ``--out-dir``: ``measurements.jsonl`` (one row per measured
trajectory incl. cache hits, plus ``tag="ref"`` reference and ``tag="confirm"``
rows), ``updates.jsonl`` (per-update telemetry incl. drop rate) and
``summary.json`` (baselines, best PAIRED / best CONFIRMED, Pareto front over
ratios, hypervolume, drop rate, reference drift, gates).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

# mirror of pomo.NUMERIC_CHECK_MIN_COS -- the argparser must not import pomo
# (that would pull jax in before JAX_PLATFORMS is pinned to cpu).
P_MIN_COS = 1.0 - 1e-6


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
    # measurement protocol (M3 fixes #119/#120/#121)
    p.add_argument("--pair-mode", default="bracket",
                   choices=("every", "bracket", "update", "off"),
                   help="paired-measurement scheme: a fresh jacve-rev "
                        "reference per candidate ('every', 2x cost), one "
                        "before+after each update's candidate block with "
                        "log-linear interpolation ('bracket', default), one "
                        "per update ('update'), or none ('off' = the old, "
                        "drift-contaminated protocol)")
    p.add_argument("--min-survivors", type=int, default=2,
                   help="skip the update when fewer trajectories survive the "
                        "unmeasurable-drop (no meaningful shared baseline)")
    p.add_argument("--check-every", type=int, default=0,
                   help="numerically verify every k-th unique plan against "
                        "jacve rev (0 = audit only new bests, which are "
                        "always verified)")
    p.add_argument("--numeric-min-cos", type=float, default=P_MIN_COS,
                   help="cosine floor for a plan's Jacobian vs jacve rev")
    p.add_argument("--no-confirm-best", dest="confirm_best",
                   action="store_false", default=True,
                   help="do NOT re-measure a new best interleaved with fresh "
                        "references before archiving it")
    p.add_argument("--confirm-margin", type=float, default=0.01,
                   help="a provisional best must beat the CONFIRMED best by "
                        "this fraction before a confirmation is spent on it")
    p.add_argument("--confirm-budget-frac", type=float, default=0.25,
                   help="hard cap on confirmation wall time as a fraction of "
                        "candidate-measurement wall time")
    p.add_argument("--confirm-reps", type=int, default=5,
                   help="candidate/reference alternations in a confirmation; "
                        "the confirmed ratio is their MEDIAN (one ratio "
                        "carries ~1%% measurement noise)")
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
    ref_wall = []           # per-reference-measurement wall seconds

    def _now():
        return time.time() - t_start

    audit = {"n": 0}

    def measure_fn(order, check=False):
        t = time.perf_counter()
        kw = dict(common)
        # #120: plans were NEVER verified -- a wrong Jacobian would be logged
        # as fast and would train the policy. New bests are always verified
        # (see confirm_plan); --check-every additionally audits a 1-in-k
        # sample of unique plans (an extra reference compile per checked plan).
        if a.check_every and audit["n"] % int(a.check_every) == 0:
            check = True
        audit["n"] += 1
        if check:
            kw["check_against"] = {"method": "jacve", "order": "rev"}
        r = client.measure(method="elim_plan",
                           plan=[["V", int(j)] for j in order],
                           vertex_only=True, **kw)
        r["wall_s"] = time.perf_counter() - t
        r["t_s"] = _now() - 0.5 * r["wall_s"]      # midpoint of the window
        r["checked"] = bool(check)
        meas_wall.append(r["wall_s"])
        return r

    def measure_ref():
        """One jacve-REVERSE reference through the SAME worker. Logged with an
        absolute latency every time, so reference drift is an explicit,
        plottable channel rather than a hidden confounder (#121)."""
        t = time.perf_counter()
        r = client.measure(**{**common, "method": "jacve", "order": "rev"})
        r.pop("latency_samples_ns", None)
        r["wall_s"] = time.perf_counter() - t
        r["t_s"] = _now() - 0.5 * r["wall_s"]
        ref_wall.append(r["wall_s"])
        meas_f.write(json.dumps(
            {"tag": "ref", "update": ref_state["update"], "t_s": r["t_s"],
             "status": r.get("status"), "reason": r.get("reason"),
             "latency_ns": _lat(r), "mem_total_bytes": _mem(r),
             "compile_s": r.get("compile_s"), "wall_s": r["wall_s"]}) + "\n")
        return r

    ref_state = {"update": 0}
    tracker = P.ReferenceTracker(measure_ref, mode=a.pair_mode, clock=_now)

    cache = P.MeasureCache(graph, measure_fn)
    archive = []          # (lat_ratio, mem_ratio) of POMO-measured feasible plans
    archive_meta = []
    drop_totals: dict = {}
    n_measured_traj = 0
    n_dropped_traj = 0
    n_rejected_numeric = 0
    best = {"ratio": float("inf"), "confirmed": None}
    confirmed_keys = set()      # each unique plan is confirmed at most once
    confirm_wall = [0.0]

    def record(tag, upd, order, res, hit, lam, extra=None):
        row = {"tag": tag, "update": upd, "lam": [float(x) for x in lam],
               "cached": bool(hit), "unique": cache.unique,
               "status": res.get("status"), "reason": res.get("reason"),
               "latency_ns": _lat(res), "mem_total_bytes": _mem(res),
               "lat_ratio": res.get("lat_ratio"),
               "mem_ratio": res.get("mem_ratio"),
               "ref_latency_ns": res.get("ref_latency_ns"),
               "ref_mem_bytes": res.get("ref_mem_bytes"),
               "ref_mode": res.get("ref_mode"),
               "check_cos": res.get("check_cos"),
               "check_maxdiff": res.get("check_maxdiff"),
               "check_error": res.get("check_error"),
               "t_s": res.get("t_s"),
               "compile_s": res.get("compile_s"), "wall_s": res.get("wall_s"),
               "order": [int(j) for j in order]}
        if extra:
            row.update(extra)
        meas_f.write(json.dumps(row) + "\n")

    def archive_add(upd, res, lam, confirmed=False, order=None):
        """The archive holds RATIOS (candidate / paired reverse reference), so
        entries measured hours apart remain comparable. Returns the index."""
        archive.append((float(res["lat_ratio"]), float(res["mem_ratio"])))
        archive_meta.append({"update": upd, "lam": [float(x) for x in lam],
                             "latency_ns": _lat(res),
                             "mem_total_bytes": _mem(res),
                             "ref_latency_ns": res.get("ref_latency_ns"),
                             "confirmed": bool(confirmed),
                             "check_cos": res.get("check_cos"),
                             "check_maxdiff": res.get("check_maxdiff"),
                             "order": ([int(j) for j in order]
                                       if order is not None else None)})
        return len(archive) - 1

    def confirm_new_best(upd, order, lam, prov_lat, prov_mem, arc_i):
        """A provisional new best must survive a CONFIRMATION before it is
        allowed to stand: re-measured interleaved with fresh references (#121)
        and numerically verified against jacve rev (#120). Returns the seconds
        spent.

        Triggering: the provisional ratio has to beat the CONFIRMED best by
        ``--confirm-margin`` (a single ratio carries ~1% noise, so a bare
        improvement is mostly noise), and each unique plan is confirmed at
        most once. Both bounds keep the confirmation cost near zero.
        """
        nonlocal n_rejected_numeric
        t0 = time.perf_counter()
        k = cache.key(order)
        thr = best["ratio"] * (1.0 - a.confirm_margin)
        over_budget = (confirm_wall[0] > a.confirm_budget_frac
                       * max(sum(meas_wall), 1e-9))
        if (not a.confirm_best or not prov_lat or k in confirmed_keys
                or prov_lat >= thr or over_budget):
            if over_budget and prov_lat and prov_lat < thr:
                _p(f"[u{upd:04d}] confirmation budget spent "
                   f"({confirm_wall[0] / 60:.1f}min > "
                   f"{100 * a.confirm_budget_frac:.0f}% of measurement); "
                   "provisional best left UNCONFIRMED")
            return time.perf_counter() - t0
        confirmed_keys.add(k)
        conf, _first = confirm_plan(upd, order, lam,
                                    threshold=(None if best["confirmed"] is None
                                               else thr))
        if not conf["ok"]:
            n_rejected_numeric += 1
            drop_totals["numeric_check_failed"] = drop_totals.get(
                "numeric_check_failed", 0) + 1
            P.invalidate(cache.cache[cache.key(order)], conf["check"])
            if arc_i is not None:
                archive.pop(arc_i)
                archive_meta.pop(arc_i)
            _p(f"[u{upd:04d}] !! NEW BEST REJECTED: numeric check "
               f"{conf['check']} cos={conf['check_cos']} "
               f"maxdiff={conf['check_maxdiff']} -- plan INVALIDATED "
               "(never archived, never a good plan)")
        else:
            won = conf["lat_ratio"] < best["ratio"]
            if won:                                    # best stays MONOTONE
                best.update(ratio=conf["lat_ratio"], confirmed={
                    "update": upd, "lat_ratio": conf["lat_ratio"],
                    "mem_ratio": conf["mem_ratio"],
                    "lat_ratios": conf["lat_ratios"],
                    "latency_ns": conf["latency_ns"],
                    "check_cos": conf["check_cos"],
                    "check_maxdiff": conf["check_maxdiff"], "ok": True,
                    "order": [int(j) for j in order]})
            if arc_i is not None:
                archive[arc_i] = (conf["lat_ratio"],
                                  conf["mem_ratio"] or float(prov_mem))
                archive_meta[arc_i].update(
                    confirmed=True, check_cos=conf["check_cos"],
                    check_maxdiff=conf["check_maxdiff"])
            _p(f"[u{upd:04d}] "
               + ("NEW BEST confirmed" if won else
                  "provisional best NOT confirmed (its interleaved re-measure "
                  "does not beat the incumbent -- the seed-2 artifact class)")
               + f" ratio={conf['lat_ratio']:.4f} (interleaved A/B "
               f"{['%.4f' % x for x in conf['lat_ratios']]}"
               + (", early stop" if conf.get("early_stop") else "") + ") "
               f"cos={conf['check_cos'] or 0.0:.8f} "
               f"maxdiff={conf['check_maxdiff'] or 0.0:.3e}")
        dt = time.perf_counter() - t0
        confirm_wall[0] += dt
        return dt

    def archive_and_confirm(upd, results, rb, orders, lam):
        """Archive every feasible RATIO, then confirm the batch's best."""
        arc_idx = {n: archive_add(upd, results[n], lam, order=orders[n])
                   for n in range(len(results)) if rb.feasible[n]}
        if not arc_idx:
            return 0.0
        b = min(arc_idx, key=lambda n: rb.lat_ratio[n])
        return confirm_new_best(upd, orders[b], lam, float(rb.lat_ratio[b]),
                                float(rb.mem_ratio[b]), arc_idx[b])

    def audit_numeric(upd, res):
        """A plan whose Jacobian was checked and is WRONG is invalidated in
        place -- it is excluded from the batch (unmeasurable class) and stays
        excluded through the measurement cache, which holds this same dict."""
        nonlocal n_rejected_numeric
        if res is None or ("check_cos" not in res
                           and "check_error" not in res):
            return True
        ok, label = P.numeric_check(res, a.numeric_min_cos)
        if ok:
            return True
        n_rejected_numeric += 1
        _p(f"[u{upd:04d}] !! NUMERIC CHECK FAILED ({label}) "
           f"cos={res.get('check_cos')} maxdiff={res.get('check_maxdiff')} "
           f"-- plan INVALIDATED and excluded")
        P.invalidate(res, label)
        return False

    # -- new-best confirmation: interleaved A/B + numeric verification -------
    def confirm_plan(upd, order, lam, threshold=None):
        """Re-measure ``order`` INTERLEAVED with fresh references in one
        worker (ref, cand, ref, cand, ...) and verify its Jacobian against
        jacve rev (#120). This is the protocol that showed the seed-2 "record"
        (128.2us vs a job-start reference of 155.4us, an apparent 17.5% win)
        to be an artifact: ratio 1.001-1.010, i.e. NOT faster. New bests are
        rare, so paying ~2x here is nearly free."""
        import numpy as _np
        ratios, mratios, first = [], [], None
        r_prev = measure_ref()
        for k in range(max(int(a.confirm_reps), 1)):
            c = measure_fn(order, check=(k == 0))
            r_next = measure_ref()
            if k == 0:
                first = c
            lp, ln_, cl = _lat(r_prev), _lat(r_next), _lat(c)
            if cl and lp and ln_:
                ref_lat_g = math.sqrt(float(lp) * float(ln_))
                ratios.append(float(cl) / ref_lat_g)
                mp, mn, cm = _mem(r_prev), _mem(r_next), _mem(c)
                if cm and mp and mn:
                    mratios.append(float(cm) / math.sqrt(float(mp) * float(mn)))
                P.attach_ratio(c, ref_lat_g, math.sqrt(float(mp) * float(mn))
                               if (mp and mn) else None,
                               {"ref_mode": "confirm"})
            record("confirm", upd, order, c, False, lam, {"confirm_rep": k})
            r_prev = r_next
            # sequential early abort: two interleaved reps that already put
            # the plan at/above the incumbent are enough to say "not a new
            # best". Only genuine improvements pay the full confirm_reps.
            if (threshold is not None and k >= 1 and ratios
                    and float(_np.median(ratios)) >= threshold):
                break
        ok, label = P.numeric_check(first, a.numeric_min_cos)
        out = {"ok": bool(ok and ratios), "check": label,
               "early_stop": bool(threshold is not None
                                  and len(ratios) < max(int(a.confirm_reps), 1)),
               "check_cos": (first or {}).get("check_cos"),
               "check_maxdiff": (first or {}).get("check_maxdiff"),
               "lat_ratio": float(_np.median(ratios)) if ratios else None,
               "mem_ratio": float(_np.median(mratios)) if mratios else None,
               "lat_ratios": [float(x) for x in ratios],
               "latency_ns": _lat(first), "n_ref": len(ratios) + 1}
        return out, first

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
            r["t_s"] = _now()
            baselines[name] = r
            if name == "jacve_rev":
                # anchor the tracker so the pre-loop rows are paired too
                tracker.add_sample(r["t_s"], _lat(r), _mem(r))
            _p(f"  {name:22s} {_fmt(r)} compile={r.get('compile_s', 0):6.1f}s"
               + (f" cos={r['check_cos']:.6f}" if "check_cos" in r else ""))
        # min-Markowitz (cautionary) + random orders -- measured as elim_plans
        if graph is not None:
            mk = P.markowitz_order(graph)
            _, mk_real = P.order_to_plan(env0, mk)
            r = measure_fn(mk_real)
            r.pop("latency_samples_ns", None)
            tracker.pair(r, r.get("t_s", _now()))
            baselines["min_markowitz"] = r
            _p(f"  {'min_markowitz':22s} {_fmt(r)}")
        rand_rows = []
        for i in range(a.random_baselines):
            perm = list(rng.permutation(sorted(env0.jacve_vertices)))
            _, real = P.order_to_plan(env0, perm)
            r = measure_fn(real)
            r.pop("latency_samples_ns", None)
            tracker.pair(r, r.get("t_s", _now()))
            rand_rows.append(r)
            record("random", -1, real, r, False, (1.0, 0.0))
        ok = [r for r in rand_rows if r.get("status") == "ok" and _lat(r)]
        if ok:
            lat = np.array([_lat(r) for r in ok])
            mem = np.array([_mem(r) or 0.0 for r in ok])
            rat = np.array([r["lat_ratio"] for r in ok if r.get("lat_ratio")])
            baselines["random"] = {
                "n": len(rand_rows), "n_ok": len(ok),
                "latency_ns": {"min": float(lat.min()), "median": float(np.median(lat)),
                               "mean": float(lat.mean()), "max": float(lat.max()),
                               "std": float(lat.std())},
                "lat_ratio": ({"min": float(rat.min()),
                               "median": float(np.median(rat)),
                               "max": float(rat.max())} if rat.size else {}),
                "mem_total_bytes": {"min": float(mem.min()),
                                    "median": float(np.median(mem)),
                                    "max": float(mem.max())}}
            _p(f"  {'random x' + str(len(ok)):22s} "
               f"min={lat.min() / 1e3:.1f}us median={np.median(lat) / 1e3:.1f}us "
               f"max={lat.max() / 1e3:.1f}us")
        _p("")

    ref_lat = (_lat(baselines.get("jacve_rev")) or a.ref_latency_ns)
    ref_mem = (_mem(baselines.get("jacve_rev")) or a.ref_mem_bytes)
    _p(f"[m3] reverse reference at t=0: {ref_lat / 1e3:.1f}us "
       f"{ref_mem / 2 ** 20:.1f}MB -- DIAGNOSTIC ONLY. Rewards, the archive "
       f"and every gate use the PAIRED ratio against a reference re-measured "
       f"in mode={a.pair_mode!r}; a job-start reference is exactly what made "
       "the previous campaign's headline a drift artifact (#121).")

    if a.pair_mode == "off" and not tracker.samples:
        client.close()
        raise SystemExit(
            "--pair-mode off needs a reference to pair against, but no "
            "baseline was measured: every candidate would be unpaired and "
            "every update skipped. Pass --baselines (reproducing the old, "
            "job-start-reference protocol) or use a real pair mode.")

    def summarize(n_upd, stopped=None):
        # EVERYTHING below is in RATIO space (candidate / paired jacve-rev
        # reference): 1.0 IS reverse, by construction and at the same clock.
        best_lat = min((p[0] for p in archive), default=None)
        best_mem = min((p[1] for p in archive), default=None)
        front_idx = P.pareto_front(archive) if archive else []
        front = [archive[i] for i in front_idx]
        hv = P.hypervolume_2d(front, (2.0, 2.0)) if front else 0.0
        dom = [p for p in front if p[0] <= 1.0 and p[1] <= 1.0]
        rnd = baselines.get("random", {}).get("lat_ratio", {})
        cb = best["confirmed"]
        out = {
            "seed": a.seed, "target": a.target, "kwargs": kw,
            "updates_done": n_upd, "stopped": stopped,
            "unique_measurements": cache.unique,
            "cache_hits": cache.hits, "cache_hit_rate": cache.hit_rate,
            "measure_wall_s": {
                "n": len(meas_wall), "total": float(sum(meas_wall)),
                "mean": float(sum(meas_wall) / max(len(meas_wall), 1)),
                "max": float(max(meas_wall)) if meas_wall else 0.0},
            "reference_wall_s": {
                "n": len(ref_wall), "total": float(sum(ref_wall)),
                "mean": float(sum(ref_wall) / max(len(ref_wall), 1))},
            "pair_mode": a.pair_mode,
            "reference_drift": tracker.drift_stats(),
            "reference_failures": tracker.n_failed,
            "unmeasurable_dropped": n_dropped_traj,
            "numeric_check_rejected": n_rejected_numeric,
            "trajectories_scored": n_measured_traj,
            "drop_rate": (n_dropped_traj / n_measured_traj
                          if n_measured_traj else 0.0),
            "drop_counts": dict(drop_totals),
            "worker_respawns": client.respawns,
            "wall_hours": (time.time() - t_start) / 3600.0,
            "baselines": baselines,
            "ref_latency_ns": ref_lat, "ref_mem_bytes": ref_mem,
            "best_lat_ratio": best_lat, "best_mem_ratio": best_mem,
            "best_confirmed": cb,
            "confirmations": len(confirmed_keys),
            "confirm_wall_s": confirm_wall[0],
            "front": [{"lat_ratio": p[0], "mem_ratio": p[1],
                       **archive_meta[i]} for i, p in zip(front_idx, front)],
            "front_size": len(front),
            "hypervolume_norm": hv,
            "front_frac_dominating_reverse": (len(dom) / len(front)) if front else 0.0,
            "gate_within_5pct_of_reverse": (
                bool(best_lat is not None and best_lat <= 1.05)),
            # the headline claim: only a CONFIRMED (interleaved A/B +
            # numerically verified) ratio below 1 counts as beating reverse.
            "gate_beats_reverse_confirmed": (
                bool(cb and cb.get("ok") and cb.get("lat_ratio") is not None
                     and cb["lat_ratio"] < 1.0)),
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
            ref_state["update"] = u
            t_roll = time.perf_counter()
            batch, step_mask, orders = runner.rollout(
                envs, policy, lam, rng,
                forced_starts=not a.no_forced_starts)
            t_meas = time.perf_counter()

            # PAIRED measurement (#121): open the update with a fresh
            # reference, measure the candidates, close with a second one; each
            # fresh candidate is then paired with the reference interpolated
            # at its own timestamp. Cached results keep the pairing they were
            # measured with -- ratios are drift-free, absolutes are not.
            tracker.sample()
            results, hit_flags, fresh = [], [], []
            for n, order in enumerate(orders):
                res, hit = cache.measure_order(order)
                hit_flags.append(hit)
                results.append(res)
                if not hit:
                    fresh.append(res)
                    if tracker.mode == "every":
                        tracker.sample()
            if tracker.mode == "bracket":
                tracker.sample()
            for res in fresh:
                tracker.pair(res, res.get("t_s", _now()))
                audit_numeric(u, res)
            for n, order in enumerate(orders):
                record("pomo", u, order, results[n], hit_flags[n], lam)
            t_upd = time.perf_counter()

            rb = P.score_rewards_paired(results, lam)
            n_measured_traj += len(results)
            n_dropped_traj += rb.n_dropped
            for lab, c in rb.drop_counts().items():
                drop_totals[lab] = drop_totals.get(lab, 0) + c
            if rb.n_dropped:
                _p(f"[u{u:04d}] DROPPED {rb.n_dropped}/{len(results)} "
                   f"unmeasurable trajectories {rb.drop_counts()} -- excluded "
                   "from the batch (NOT scored worst-in-batch)")
            # archive + confirm BEFORE the skip test: a real paired measurement
            # is worth keeping even when its update carries no gradient.
            t_conf = archive_and_confirm(u, results, rb, orders, lam)
            if not rb.usable(a.min_survivors):
                why = ("no_feasible" if not rb.feasible.any()
                       else "too_few_survivors")
                _p(f"[u{u:04d}] update SKIPPED ({why}): "
                   f"{int(rb.keep.sum())} survivors, "
                   f"{int(rb.feasible.sum())} feasible of {len(results)}")
                upd_f.write(json.dumps(
                    {"update": u, "skipped": why,
                     "lam": list(map(float, lam)),
                     "n_dropped": rb.n_dropped,
                     "drop_counts": rb.drop_counts(),
                     "n_survivors": int(rb.keep.sum()),
                     "unique": cache.unique}) + "\n")
                continue

            keep_idx, R = rb.survivors()
            sub = (batch if len(keep_idx) == len(results)
                   else batch.rows_of(keep_idx, runner.L))
            policy, opt_state, loss, gnorm = runner.update(
                policy, optim, opt_state, sub, lam, R,
                np.asarray(step_mask)[keep_idx], a.ent_coef,
                grad_chunk=a.grad_chunk)
            t_end = time.perf_counter()

            lats = [_lat(r) for i, r in enumerate(results) if rb.feasible[i]]
            best_lat = min((p[0] for p in archive), default=float("nan"))
            best_mem = min((p[1] for p in archive), default=float("nan"))
            drift = tracker.drift_stats()
            row = {"update": u, "lam": list(map(float, lam)),
                   "mean_R": float(np.mean(R)), "max_R": float(np.max(R)),
                   "n_feasible": int(rb.feasible.sum()),
                   "n_dropped": rb.n_dropped,
                   "drop_counts": rb.drop_counts(),
                   "drop_rate_cum": (n_dropped_traj / n_measured_traj
                                     if n_measured_traj else 0.0),
                   "n_survivors": len(keep_idx),
                   "loss": loss, "grad_norm": gnorm,
                   "batch_min_lat_ratio": float(np.nanmin(rb.lat_ratio))
                   if rb.feasible.any() else None,
                   "batch_min_latency_ns": (min(lats) if lats else None),
                   "ref_latency_ns": (tracker.last[1] if tracker.last else None),
                   "ref_drift_span_frac": drift.get("span_frac"),
                   "best_lat_ratio": best_lat, "best_mem_ratio": best_mem,
                   "best_confirmed_ratio": best["confirmed"] and
                   best["confirmed"]["lat_ratio"],
                   "unique": cache.unique, "hits": cache.hits,
                   "hit_rate": cache.hit_rate,
                   "respawns": client.respawns,
                   "s_rollout": t_meas - t_roll, "s_measure": t_upd - t_meas,
                   "s_update": t_end - t_upd, "s_confirm": t_conf}
            upd_f.write(json.dumps(row) + "\n")
            _p(f"[u{u:04d}] lam=({lam[0]:.2f},{lam[1]:.2f}) meanR={row['mean_R']:+.3f} "
               f"loss={loss:+.4f} |g|={gnorm:.2e} "
               f"feas={int(rb.feasible.sum())}/{len(results)} "
               f"drop={rb.n_dropped} "
               f"batch_min={(row['batch_min_lat_ratio'] or float('nan')):.4f}x "
               f"best={best_lat:.4f}x/{best_mem:.4f}x "
               f"ref={(tracker.last[1] / 1e3 if tracker.last else float('nan')):.1f}us "
               f"uniq={cache.unique} hit={cache.hit_rate:.2f} "
               f"t={row['s_rollout']:.1f}/{row['s_measure']:.1f}/{row['s_update']:.1f}s")

            if a.greedy_every and u % a.greedy_every == 0:
                gl = (1.0, 0.0)
                _b, _sm, gorders = runner.rollout(
                    envs[:1], policy, gl, rng, greedy=True, forced_starts=False)
                gres, ghit = cache.measure_order(gorders[0])
                if not ghit:
                    tracker.pair(gres, gres.get("t_s", _now()))
                    audit_numeric(u, gres)
                record("greedy", u, gorders[0], gres, ghit, gl)
                if not ghit and gres.get("lat_ratio"):
                    gi = archive_add(u, gres, gl, order=gorders[0])
                    confirm_new_best(u, gorders[0], gl, gres["lat_ratio"],
                                     gres.get("mem_ratio") or 1.0, gi)
                _p(f"          greedy(lam=1,0): {_fmt(gres)} "
                   f"ratio={gres.get('lat_ratio') or float('nan'):.4f} "
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
    if out["best_lat_ratio"]:
        _p(f"  best PAIRED ratio {out['best_lat_ratio']:.4f}x reverse "
           f"(mem {out['best_mem_ratio']:.4f}x)")
    cb = out["best_confirmed"]
    _p(f"  best CONFIRMED (interleaved A/B + numeric check): "
       + (f"{cb['lat_ratio']:.4f}x reverse, reps={cb['lat_ratios']}, "
          f"cos={cb['check_cos']}" if cb else "none"))
    d = out["reference_drift"]
    _p(f"  reference drift: n={d.get('n')} "
       f"{(d.get('min_ns') or 0) / 1e3:.1f}-{(d.get('max_ns') or 0) / 1e3:.1f}us "
       f"span={100 * (d.get('span_frac') or 0):.1f}% "
       f"max_rate={100 * (d.get('max_rate_frac_per_min') or 0):.2f}%/min "
       f"(pair_mode={out['pair_mode']}, ref_wall="
       f"{out['reference_wall_s']['total'] / 60.0:.1f}min)")
    _p(f"  UNMEASURABLE dropped {out['unmeasurable_dropped']}/"
       f"{out['trajectories_scored']} = {100 * out['drop_rate']:.2f}% "
       f"{out['drop_counts']}   <-- report this in any results table")
    _p(f"  front={out['front_size']} hv={out['hypervolume_norm']:.4f} "
       f"frac_dominating_reverse={out['front_frac_dominating_reverse']:.3f}")
    _p(f"  GATE beats_random={out['gate_beats_random']} "
       f"beats_reverse_confirmed={out['gate_beats_reverse_confirmed']} "
       f"within_5pct_reverse={out['gate_within_5pct_of_reverse']} "
       f"unique_budget_ok={out['gate_unique_budget_ok']}")
    _p("M3_SUMMARY " + json.dumps({k: v for k, v in out.items()
                                   if k not in ("front", "baselines")}))
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
