#!/usr/bin/env python3
"""v3 optimizer branch: FULL per-vertex micro-actions (none/quant/diag/compress)
with EXPLICIT param ranges + redraw-on-invalid, and a two-level search that fixes
an order and redraws random approximations for it (focus). ALPHAGRAD_MF_APPROX=1.
Replaces the quant-only v2 branch."""
import pathlib, sys
p = pathlib.Path("src/alphagrad/approx/autoscheduler_loop.py")
s = p.read_text()
START = "# ============ MODEL-FREE OPTIMIZERS (--optimizer random|sa|ga) ============"
END = "# HOISTED to module scope so @eqx.filter_jit compiles ONCE."
assert START in s and END in s, "markers"
i0 = s.index(START); i1 = s.index(END)

NEW = r'''# ============ MODEL-FREE OPTIMIZERS (--optimizer random|sa|ga|bo) ============
# Genome = (order, per-vertex micro-action). ALPHAGRAD_MF_APPROX=1 searches the
# FULL micro-action space {none, quant(dtype), diag(i,j,factor), compress(axis,
# kind)}; params drawn from EXPLICIT ranges (ALPHAGRAD_MF_MAX_AX / _FACTORS) and
# REDRAWN on an invalid measure (bounded retries). Two-level focus: an order can
# be revisited with fresh random approximations. =0 keeps pure order.
if getattr(A, "optimizer", "none") != "none":
    import os as _os_mf
    from graphax.sparse.micro_actions import COMPRESS_KINDS as _CKS
    _os_mf.makedirs(A.out, exist_ok=True)
    _opt = A.optimizer
    _budget = A.total_measurements if A.total_measurements > 0 else (A.rounds * A.topk)
    _APPROX = _os_mf.environ.get("ALPHAGRAD_MF_APPROX", "0") == "1"
    _QDr = [d.strip() for d in _os_mf.environ.get(
        "ALPHAGRAD_QUANT_ALLOWED", "int8,int16,float8_e4m3fn,float8_e5m2,bfloat16,float16"
        ).split(",") if d.strip()]
    _MAXAX = int(_os_mf.environ.get("ALPHAGRAD_MF_MAX_AX", "3"))
    _FACS = [int(x) for x in _os_mf.environ.get("ALPHAGRAD_MF_FACTORS", "2,3,4").split(",") if x]
    _OPS = [o.strip() for o in _os_mf.environ.get(
        "ALPHAGRAD_MF_MICRO_OPS", "none,quant,diag,compress").split(",") if o.strip()]
    _REDRAW = int(_os_mf.environ.get("ALPHAGRAD_MF_REDRAW", "3"))       # retries on invalid measure
    _orng = np.random.default_rng(A.seed)
    _mf_bufY = []
    _mf_best = {"scalar": -1e18, "raw": None, "order": None, "approx": None, "at": 0}
    _mf_hist = []
    _n = [0]
    print(f"[MF] optimizer={_opt} approx={_APPROX} budget={_budget} NV={NV} "
          f"ops={_OPS} maxax={_MAXAX} facs={_FACS}", flush=True)

    def _mf_reverse():
        return list(range(NV))[::-1]
    def _mf_random_order(rng):
        graph, tg = copy_g(GRAPH0), copy_g(TG0); ca = []
        while True:
            legal = legal_set(graph)
            if not legal:
                break
            v = legal[int(rng.integers(len(legal)))]
            ca.append(VALID.index(v))
            _eliminate_vertex(v, jaxpr, graph, tg, VO, count_ops=False, transforms=())
        return ca
    def _rand_micro(rng):
        op = _OPS[int(rng.integers(len(_OPS)))]
        if op == "none":
            return None
        if op == "quant":
            return ("q", _QDr[int(rng.integers(len(_QDr)))])
        if op == "diag":
            return ("d", int(rng.integers(_MAXAX)), int(rng.integers(_MAXAX)),
                    _FACS[int(rng.integers(len(_FACS)))])
        if op == "compress":
            return ("c", int(rng.integers(_MAXAX)), int(rng.integers(len(_CKS))))
        return None
    def _rand_approx(rng, n):
        return [_rand_micro(rng) for _ in range(n)] if _APPROX else None
    def _mf_rand_cand(rng):
        o = _mf_random_order(rng)
        return (o, _rand_approx(rng, len(o)))
    def _micro_str(m):
        if m is None: return []
        if m[0] == "q": return ["quant('%s')" % m[1]]
        if m[0] == "d": return ["diag(%d,%d,%d)" % (m[1], m[2], m[3])]
        if m[0] == "c": return ["compress('%s',%d)" % (_CKS[m[2]], m[1])]
        return []
    def _mf_seq(cand):
        o, ap = cand
        if ap is None:
            return [(int(a), []) for a in o]
        return [(int(a), _micro_str(ap[k])) for k, a in enumerate(o)]
    def _mf_feat(cand):
        o, ap = cand
        pos = np.zeros(NV, dtype=np.float64)
        for rank, a in enumerate(o):
            pos[int(a)] = rank / max(NV - 1, 1)
        opv = np.zeros(NV, dtype=np.float64)  # per-vertex micro-op class (0 none..3 compress)
        if ap is not None:
            _m = {None: 0}
            for k, a in enumerate(o):
                m = ap[k]
                opv[int(a)] = {None: 0, "q": 1, "d": 2, "c": 3}.get(m[0] if m else None, 0)
        return np.concatenate([pos, opv / 3.0])
    def _measure1(cand):
        try:
            order_arr, specs, _ = build_order_specs(_mf_seq(cand), env)
            rs = {}
            _callback(env.config, env.args, env.consts, jnp.asarray(order_arr),
                      jnp.asarray(specs), len(order_arr), *ev, raw_sink=rs)
        except Exception:
            return None
        lat_s = [x for x in rs.get("latency_ns_samples", []) if x > 0 and np.isfinite(x)]
        lat = float(np.mean(lat_s)) if lat_s else float("nan")
        peak = float(rs.get("xla_peak_memory", float("nan")))
        flops = float(rs.get("flops", float("nan")))
        cos_pp = rs.get("cosine_sim_per_point", [])
        cos = float(np.mean(cos_pp)) if cos_pp else float("nan")
        r = np.array([lat, peak, flops, cos], dtype=np.float64)
        return r if np.all(np.isfinite(r)) else None
    def _mf_measure(cands):
        # measure each candidate; REDRAW its approximations on invalid (bounded).
        out = []
        for cand in cands:
            r = _measure1(cand); tries = 0
            while r is None and _APPROX and cand[1] is not None and tries < _REDRAW:
                cand = (cand[0], _rand_approx(_orng, len(cand[0]))); r = _measure1(cand); tries += 1
                _n[0] += 1  # a redraw is a fresh measurement
            _n[0] += 1
            if r is None:
                out.append((cand, None)); continue
            _mf_bufY.append(r); out.append((cand, r))
        _refresh_az_norm(_mf_bufY)
        for cand, r in out:
            if r is None:
                continue
            sc = scalarize(r)
            if sc > _mf_best["scalar"]:
                _mf_best.update(scalar=float(sc), raw=r.tolist(),
                                order=list(map(int, cand[0])),
                                approx=([list(m) if m else None for m in cand[1]] if cand[1] else None),
                                at=_n[0])
        if _mf_best["raw"] is not None:
            b = _mf_best["raw"]
            _mf_hist.append((_n[0], b[0] / 1e3, b[1] / 1e6, b[3], _mf_best["scalar"]))
        return out
    def _mf_log(tag):
        b = _mf_best["raw"]
        if b is not None:
            na = 0 if not _mf_best["approx"] else sum(1 for x in _mf_best["approx"] if x)
            print(f"[MF-{_opt}] {tag} n={_n[0]}/{_budget} best_lat={b[0]/1e3:.1f}us "
                  f"best_peak={b[1]/1e6:.2f}MB best_cos={b[3]:+.4f} napprox={na} "
                  f"(at n={_mf_best['at']})", flush=True)

    _P_REUSE = float(_os_mf.environ.get("ALPHAGRAD_MF_REUSE", "0.5"))  # focus: revisit an order
    _seen_orders = []
    def _pick_order(rng):
        if _APPROX and _seen_orders and rng.random() < _P_REUSE:
            return list(_seen_orders[int(rng.integers(len(_seen_orders)))])
        o = _mf_random_order(rng); _seen_orders.append(o); return o

    if _opt == "random":
        while _n[0] < _budget:
            k = min(max(A.topk, 1), max(_budget - _n[0], 1))
            cands = [(_pick_order(_orng), _rand_approx(_orng, NV)) for _ in range(k)]
            _mf_measure(cands); _mf_log("iter")
    elif _opt == "sa":
        cur = (_mf_reverse(), _rand_approx(_orng, NV))
        cr = _mf_measure([cur]); craw = cr[0][1] if cr and cr[0][1] is not None else None
        if craw is None:
            cur = _mf_rand_cand(_orng); craw = _mf_measure([cur])[0][1]
        T = float(_os_mf.environ.get("ALPHAGRAD_SA_T0", "1.0"))
        cool = float(_os_mf.environ.get("ALPHAGRAD_SA_COOL", "0.98"))
        while _n[0] < _budget:
            if _APPROX and _orng.random() < 0.5:
                nxt = (list(cur[0]), _rand_approx(_orng, NV))     # FOCUS: same order, redraw approx
            else:
                o2 = list(cur[0]); a, b = int(_orng.integers(NV)), int(_orng.integers(NV))
                o2[a], o2[b] = o2[b], o2[a]; nxt = (o2, _rand_approx(_orng, NV))
            mr = _mf_measure([nxt]); nraw = mr[0][1] if mr else None
            if nraw is None:
                T *= cool; continue
            csc = scalarize(craw); nsc = scalarize(nraw)
            if nsc > csc or _orng.random() < np.exp((nsc - csc) / max(T, 1e-6)):
                cur, craw = mr[0][0], nraw
            T *= cool; _mf_log(f"T={T:.3f}")
    elif _opt == "ga":
        P = int(_os_mf.environ.get("ALPHAGRAD_GA_POP", "10"))
        pop0 = [_mf_rand_cand(_orng) for _ in range(P)]
        mr = _mf_measure(pop0)
        pop = [c for c, r in mr if r is not None]; praws = [r for c, r in mr if r is not None]
        def _ox(p1, p2, rng):
            n = len(p1); i, j = sorted(int(rng.integers(0, n)) for _ in range(2))
            child = [-1] * n; child[i:j + 1] = p1[i:j + 1]
            fill = [x for x in p2 if x not in child]; k = 0
            for t in range(n):
                if child[t] == -1:
                    child[t] = fill[k]; k += 1
            return child
        def _tourn(rng):
            a, b = int(rng.integers(0, len(pop))), int(rng.integers(0, len(pop)))
            return pop[a] if scalarize(praws[a]) >= scalarize(praws[b]) else pop[b]
        while _n[0] < _budget:
            kids = []
            for _ in range(min(P, max(_budget - _n[0], 1))):
                pa, pb = _tourn(_orng), _tourn(_orng)
                co = _ox(pa[0], pb[0], _orng)
                ap = _rand_approx(_orng, NV) if _APPROX else None    # mutate = redraw approx
                if _orng.random() < 0.3:
                    a, b = int(_orng.integers(NV)), int(_orng.integers(NV)); co[a], co[b] = co[b], co[a]
                kids.append((co, ap))
            mr = _mf_measure(kids)
            allc = list(zip(pop, praws)) + [(c, r) for c, r in mr if r is not None]
            allc.sort(key=lambda t: -scalarize(t[1])); allc = allc[:P]
            pop = [c for c, _ in allc]; praws = [r for _, r in allc]; _mf_log("gen")
    if _opt == "bo":
        _bo = []   # (feat, cand, raw)
        init = [_mf_rand_cand(_orng) for _ in range(max(A.topk, 8))]
        for c, r in _mf_measure(init):
            if r is not None: _bo.append((_mf_feat(c), c, r))
        _beta = float(_os_mf.environ.get("ALPHAGRAD_BO_BETA", "0.3"))
        while _n[0] < _budget and _bo:
            Xa = np.asarray([x[0] for x in _bo]); ya = np.asarray([scalarize(x[2]) for x in _bo])
            cands = [(_pick_order(_orng), _rand_approx(_orng, NV)) for _ in range(max(A.pool, A.topk))]
            scr = []
            for c in cands:
                d = np.linalg.norm(Xa - _mf_feat(c), axis=1); idx = np.argsort(d)[:min(5, len(d))]
                w = 1.0 / (d[idx] + 1e-6); scr.append(float(np.sum(w * ya[idx]) / np.sum(w)) + _beta * float(d[idx].min()))
            top = [cands[i] for i in np.argsort(scr)[::-1][:A.topk]]
            for c, r in _mf_measure(top):
                if r is not None: _bo.append((_mf_feat(c), c, r))
            _mf_log("bo")

    import json as _json_mf
    _json_mf.dump({"optimizer": _opt, "approx": _APPROX, "seed": int(A.seed),
                   "budget": int(_budget), "best": _mf_best, "history": _mf_hist},
                  open(_os_mf.path.join(A.out, f"mf_{_opt}.json"), "w"), indent=2, default=float)
    _mf_log("DONE")
    _wbr = globals().get("_wb_run", None)
    if _wbr is not None:
        try: _wbr.finish()
        except Exception: pass
    sys.exit(0)

'''
s = s[:i0] + NEW + s[i1:]
p.write_text(s)
print("v3 optimizer branch (full micro-actions + focus + redraw) installed")
