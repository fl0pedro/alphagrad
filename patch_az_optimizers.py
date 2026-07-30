#!/usr/bin/env python3
"""Add model-free blackbox optimizers (--optimizer random|sa|ga) to the
autoscheduler. They reuse the SAME measurement env (measure_order /
measure_topk_parallel), the SAME scalarize objective (equal-weight PopArt via
AZ_ALIGN), and the SAME budget accounting -> directly comparable to PPO/AZ/BO.
Genome = elimination ORDER (permutation of NV action-idx); micro-actions are the
deterministic function of the order (micro_budget). No learned policy/surrogate
= model-free, no amortization. Branch runs then exits (like --fixed-orders)."""
import pathlib, sys
p = pathlib.Path("src/alphagrad/approx/autoscheduler_loop.py")
s = p.read_text()
if "MODEL-FREE OPTIMIZERS" in s:
    print("ALREADY PATCHED"); sys.exit(0)

# 1) add --optimizer arg next to --policy-prior
o_arg = 'ap.add_argument("--policy-prior", default="markowitz", choices=["markowitz", "trained"])'
n_arg = (o_arg + '\n'
         'ap.add_argument("--optimizer", default="none", choices=["none", "random", "sa", "ga"],\n'
         '                help="model-free blackbox order search (no policy/surrogate); none=AZ/BO loop")')
assert o_arg in s, "policy-prior arg anchor"; s = s.replace(o_arg, n_arg, 1)

# 2) insert the optimizer branch right before the hoisted _tp_ce block
anchor = "# HOISTED to module scope so @eqx.filter_jit compiles ONCE."
branch = r'''# ============ MODEL-FREE OPTIMIZERS (--optimizer random|sa|ga) ============
# Reuse measure_order / measure_topk_parallel + scalarize (equal-weight PopArt).
# Genome = elimination order (permutation of NV action indices). No learned
# policy or surrogate -> model-free, no amortization. Runs then exits.
if getattr(A, "optimizer", "none") != "none":
    import os as _os_mf
    _os_mf.makedirs(A.out, exist_ok=True)
    _opt = A.optimizer
    _budget = A.total_measurements if A.total_measurements > 0 else (A.rounds * A.topk)
    _mb = A.micro_budget
    _orng = np.random.default_rng(A.seed)
    _mf_bufY = []
    _mf_best = {"scalar": -1e18, "raw": None, "order": None, "at": 0}
    _mf_hist = []
    _n = [0]
    print(f"[MF] optimizer={_opt} budget={_budget} micro_budget={_mb} "
          f"measure_workers={A.measure_workers} NV={NV}", flush=True)

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

    def _mf_measure(orders):
        if A.measure_workers > 0:
            raws = measure_topk_parallel(orders, _mb, A.seed + _n[0],
                                         A.measure_workers, A.measure_gpu_base)
        else:
            raws = [measure_order(o, _mb, _orng) for o in orders]
        out = []
        for o, raw in zip(orders, raws):
            if raw is None or not np.all(np.isfinite(raw)):
                out.append(None); continue
            r = np.asarray(raw, dtype=np.float64)
            _mf_bufY.append(r); _n[0] += 1; out.append(r)
        _refresh_az_norm(_mf_bufY)
        for o, r in zip(orders, out):
            if r is None:
                continue
            sc = scalarize(r)
            if sc > _mf_best["scalar"]:
                _mf_best.update(scalar=float(sc), raw=r.tolist(),
                                order=list(map(int, o)), at=_n[0])
        if _mf_best["raw"] is not None:
            b = _mf_best["raw"]
            _mf_hist.append((_n[0], b[0] / 1e3, b[1] / 1e6, b[3], _mf_best["scalar"]))
        return out

    def _mf_log(tag):
        b = _mf_best["raw"]
        if b is not None:
            print(f"[MF-{_opt}] {tag} n={_n[0]}/{_budget} best_lat={b[0]/1e3:.1f}us "
                  f"best_peak={b[1]/1e6:.2f}MB best_cos={b[3]:+.4f} (at n={_mf_best['at']})", flush=True)

    if _opt == "random":
        while _n[0] < _budget:
            k = min(max(A.topk, 1), _budget - _n[0])
            _mf_measure([_mf_random_order(_orng) for _ in range(k)])
            _mf_log("iter")
    elif _opt == "sa":
        cur = _mf_reverse()
        _r = _mf_measure([cur])
        craw = _r[0] if _r and _r[0] is not None else _mf_measure([_mf_random_order(_orng)])[0]
        T = float(_os_mf.environ.get("ALPHAGRAD_SA_T0", "1.0"))
        cool = float(_os_mf.environ.get("ALPHAGRAD_SA_COOL", "0.98"))
        while _n[0] < _budget:
            nxt = list(cur)
            a, b = int(_orng.integers(0, NV)), int(_orng.integers(0, NV))
            nxt[a], nxt[b] = nxt[b], nxt[a]
            nr = _mf_measure([nxt])
            nraw = nr[0] if nr else None
            if nraw is None:
                T *= cool; continue
            csc = scalarize(craw); nsc = scalarize(nraw)
            if nsc > csc or _orng.random() < np.exp((nsc - csc) / max(T, 1e-6)):
                cur, craw = nxt, nraw
            T *= cool
            _mf_log(f"T={T:.3f}")
    elif _opt == "ga":
        P = int(_os_mf.environ.get("ALPHAGRAD_GA_POP", "10"))
        pop = [_mf_random_order(_orng) for _ in range(P)]
        praws = _mf_measure(pop)
        keep = [(o, r) for o, r in zip(pop, praws) if r is not None]
        pop = [o for o, _ in keep]; praws = [r for _, r in keep]
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
            fa, fb = scalarize(praws[a]), scalarize(praws[b])
            return pop[a] if fa >= fb else pop[b]
        while _n[0] < _budget:
            kids = []
            for _ in range(min(P, _budget - _n[0])):
                c = _ox(_tourn(_orng), _tourn(_orng), _orng)
                if _orng.random() < 0.3:
                    a, b = int(_orng.integers(0, NV)), int(_orng.integers(0, NV))
                    c[a], c[b] = c[b], c[a]
                kids.append(c)
            kraws = _mf_measure(kids)
            allo = pop + kids; allr = praws + kraws
            pairs = [(o, r) for o, r in zip(allo, allr) if r is not None]
            pairs.sort(key=lambda t: -scalarize(t[1]))
            pairs = pairs[:P]
            pop = [o for o, _ in pairs]; praws = [r for _, r in pairs]
            _mf_log("gen")

    import json as _json_mf
    _json_mf.dump({"optimizer": _opt, "seed": int(A.seed), "budget": int(_budget),
                   "micro_budget": int(_mb), "best": _mf_best, "history": _mf_hist},
                  open(_os_mf.path.join(A.out, f"mf_{_opt}.json"), "w"), indent=2, default=float)
    _mf_log("DONE")
    _wbr = globals().get("_wb_run", None)
    if _wbr is not None:
        try: _wbr.finish()
        except Exception: pass
    sys.exit(0)

'''
assert anchor in s, "hoisted anchor"
s = s.replace(anchor, branch + anchor, 1)
p.write_text(s)
print("optimizer branch + --optimizer arg added")
