#!/usr/bin/env python3
"""Replace the model-free optimizer branch with a version whose genome is
(order, per-vertex QUANT). ALPHAGRAD_MF_APPROX=1 searches order+quant; =0 (default)
keeps pure-order (quant=None) = same objective as before. QUANT is index-free
(quant('dtype')), so blind optimizers can generate it; DIAG/COMPRESS need
state-dependent axis indices (left to the policies). Measures explicit seqs via
build_order_specs + _callback (serial; the campaign uses --measure-workers 0)."""
import pathlib, sys
p = pathlib.Path("src/alphagrad/approx/autoscheduler_loop.py")
s = p.read_text()
START = "# ============ MODEL-FREE OPTIMIZERS (--optimizer random|sa|ga) ============"
END = "# HOISTED to module scope so @eqx.filter_jit compiles ONCE."
assert START in s and END in s, "markers not found"
i0 = s.index(START); i1 = s.index(END)

NEW = r'''# ============ MODEL-FREE OPTIMIZERS (--optimizer random|sa|ga) ============
# Genome = (elimination order, per-vertex QUANT). ALPHAGRAD_MF_APPROX=1 searches
# order+quant (single approx/vertex); =0 = pure order (quant=None). Reuses
# scalarize (equal-weight PopArt via AZ_ALIGN). Measures EXPLICIT seqs so the
# chosen quant is applied (not the auto-random _seq_from_order). Model-free.
if getattr(A, "optimizer", "none") != "none":
    import os as _os_mf
    _os_mf.makedirs(A.out, exist_ok=True)
    _opt = A.optimizer
    _budget = A.total_measurements if A.total_measurements > 0 else (A.rounds * A.topk)
    _APPROX = _os_mf.environ.get("ALPHAGRAD_MF_APPROX", "0") == "1"
    # None = no quant; else a promotion-safe dtype name (quant('dtype'))
    _QD = [None] + [d.strip() for d in _os_mf.environ.get(
        "ALPHAGRAD_QUANT_ALLOWED", "int8,int16,float8_e4m3fn,float8_e5m2,bfloat16,float16"
        ).split(",") if d.strip()]
    _orng = np.random.default_rng(A.seed)
    _mf_bufY = []
    _mf_best = {"scalar": -1e18, "raw": None, "order": None, "quant": None, "at": 0}
    _mf_hist = []
    _n = [0]
    print(f"[MF] optimizer={_opt} approx={_APPROX} budget={_budget} NV={NV} "
          f"quant_opts={len(_QD)}", flush=True)

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
    def _mf_rand_quant(rng, n):
        return [_QD[int(rng.integers(len(_QD)))] for _ in range(n)] if _APPROX else None
    def _mf_rand_cand(rng):
        o = _mf_random_order(rng)
        return (o, _mf_rand_quant(rng, len(o)))
    def _mf_seq(cand):
        o, q = cand
        if q is None:
            return [(int(a), []) for a in o]
        return [(int(a), ([] if qq is None else [f"quant('{qq}')"]))
                for a, qq in zip(o, q)]
    def _mf_measure_seq(seq):
        try:
            order_arr, specs, _ = build_order_specs(seq, env)
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
        out = []
        for cand in cands:
            r = _mf_measure_seq(_mf_seq(cand))
            if r is None:
                out.append(None); continue
            _mf_bufY.append(r); _n[0] += 1; out.append(r)
        _refresh_az_norm(_mf_bufY)
        for cand, r in zip(cands, out):
            if r is None:
                continue
            sc = scalarize(r)
            if sc > _mf_best["scalar"]:
                _mf_best.update(scalar=float(sc), raw=r.tolist(),
                                order=list(map(int, cand[0])),
                                quant=(list(cand[1]) if cand[1] is not None else None),
                                at=_n[0])
        if _mf_best["raw"] is not None:
            b = _mf_best["raw"]
            _mf_hist.append((_n[0], b[0] / 1e3, b[1] / 1e6, b[3], _mf_best["scalar"]))
        return out
    def _mf_log(tag):
        b = _mf_best["raw"]
        if b is not None:
            nq = 0 if not _mf_best["quant"] else sum(1 for x in _mf_best["quant"] if x)
            print(f"[MF-{_opt}] {tag} n={_n[0]}/{_budget} best_lat={b[0]/1e3:.1f}us "
                  f"best_peak={b[1]/1e6:.2f}MB best_cos={b[3]:+.4f} nquant={nq} "
                  f"(at n={_mf_best['at']})", flush=True)

    if _opt == "random":
        while _n[0] < _budget:
            k = min(max(A.topk, 1), _budget - _n[0])
            _mf_measure([_mf_rand_cand(_orng) for _ in range(k)])
            _mf_log("iter")
    elif _opt == "sa":
        cur = (_mf_reverse(), _mf_rand_quant(_orng, NV))
        cr = _mf_measure([cur])
        craw = cr[0] if cr and cr[0] is not None else _mf_measure([_mf_rand_cand(_orng)])[0]
        T = float(_os_mf.environ.get("ALPHAGRAD_SA_T0", "1.0"))
        cool = float(_os_mf.environ.get("ALPHAGRAD_SA_COOL", "0.98"))
        while _n[0] < _budget:
            o2 = list(cur[0]); q2 = (list(cur[1]) if cur[1] is not None else None)
            if _APPROX and _orng.random() < 0.5:
                idx = int(_orng.integers(NV)); q2[idx] = _QD[int(_orng.integers(len(_QD)))]
            else:
                a, b = int(_orng.integers(NV)), int(_orng.integers(NV))
                o2[a], o2[b] = o2[b], o2[a]
            nxt = (o2, q2)
            nr = _mf_measure([nxt]); nraw = nr[0] if nr else None
            if nraw is None:
                T *= cool; continue
            csc = scalarize(craw); nsc = scalarize(nraw)
            if nsc > csc or _orng.random() < np.exp((nsc - csc) / max(T, 1e-6)):
                cur, craw = nxt, nraw
            T *= cool
            _mf_log(f"T={T:.3f}")
    elif _opt == "ga":
        P = int(_os_mf.environ.get("ALPHAGRAD_GA_POP", "10"))
        pop = [_mf_rand_cand(_orng) for _ in range(P)]
        praws = _mf_measure(pop)
        keep = [(c, r) for c, r in zip(pop, praws) if r is not None]
        pop = [c for c, _ in keep]; praws = [r for _, r in keep]
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
            for _ in range(min(P, _budget - _n[0])):
                pa, pb = _tourn(_orng), _tourn(_orng)
                co = _ox(pa[0], pb[0], _orng)
                if _APPROX:
                    cq = [(pa[1][t] if _orng.random() < 0.5 else pb[1][t]) for t in range(NV)]
                else:
                    cq = None
                if _orng.random() < 0.3:  # mutate
                    if _APPROX and _orng.random() < 0.5:
                        idx = int(_orng.integers(NV)); cq[idx] = _QD[int(_orng.integers(len(_QD)))]
                    else:
                        a, b = int(_orng.integers(NV)), int(_orng.integers(NV))
                        co[a], co[b] = co[b], co[a]
                kids.append((co, cq))
            kraws = _mf_measure(kids)
            allc = pop + kids; allr = praws + kraws
            pairs = [(c, r) for c, r in zip(allc, allr) if r is not None]
            pairs.sort(key=lambda t: -scalarize(t[1]))
            pairs = pairs[:P]
            pop = [c for c, _ in pairs]; praws = [r for _, r in pairs]
            _mf_log("gen")

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
print("optimizer branch v2 (order+quant genome) installed")
