"""M1 orbit + trace-dedup report on the TransformerLM elimination graph.

Run offline on a CPU node:

    ALPHAGRAD_TLM_SEQ=32 ALPHAGRAD_TLM_DMODEL=128 ALPHAGRAD_TLM_VOCAB=1024 \
    JAX_PLATFORMS=cpu uv run --no-sync python -m alphagrad.elimrl.orbit_report

Prints:
  (a) coloured vs uncoloured automorphism orbit counts / sizes / histogram
      (method: pynauty if importable, else WL refinement + explicit
      per-pair verification -- unverified WL classes are flagged, never
      reported as orbits);
  (b) trace-dedup rate on 1000 uniform random elimination orders and on
      1000 orders sampled near reverse (k in [1,10] random adjacent swaps),
      unique exact orders vs unique Foata trace keys;
  (c) augment sanity: a non-trivial automorphism (if any) applied to random
      trajectories preserves the Markowitz cost profile.
"""
import os
import random
import time
from collections import Counter

# spec'd target dims (env wins if already set)
os.environ.setdefault("ALPHAGRAD_TLM_SEQ", "32")
os.environ.setdefault("ALPHAGRAD_TLM_DMODEL", "128")
os.environ.setdefault("ALPHAGRAD_TLM_VOCAB", "1024")

from alphagrad.elimrl.symmetry import (_automorphism_encoding, augment,
                                       build_example_graph, is_automorphism,
                                       markowitz_profile, orbits, trace_key,
                                       wl_refine)

N_SAMPLES = 1000
MAX_SWAPS = 10


def _orbit_block(graph, colored, budget=500_000):
    t0 = time.perf_counter()
    res = orbits(graph, colored=colored, budget=budget)
    dt = time.perf_counter() - t0
    label = "coloured" if colored else "uncoloured"
    hist = res.histogram()
    nontriv = [o for o in res.orbits if len(o) > 1]
    print(f"\n--- {label} orbits ({res.method}, exact={res.exact}, {dt:.1f}s) ---")
    if res.group_size is not None:
        print(f"  |Aut| = {res.group_size:.6g}")
    print(f"  orbit count: {len(res.orbits)}  "
          f"(non-trivial: {len(nontriv)}, largest: {max(res.sizes()) if res.orbits else 0})")
    print(f"  size histogram {{size: count}}: {dict(sorted(hist.items()))}")
    if res.unverified:
        print(f"  UNVERIFIED WL classes (unions of orbits, NOT orbits): "
              f"{[sorted(c) for c in res.unverified]}")
    prim = graph.meta.get("prim_names", {})
    for o in sorted(nontriv, key=lambda s: (-len(s), min(s)))[:12]:
        names = sorted({prim.get(v, graph.colors.get(v, ("?",))[0]) for v in o})
        print(f"    orbit size {len(o)}: {sorted(o)}  [{', '.join(map(str, names))}]")
    return res


def _near_orbit_block(graph):
    """NEAR-ORBIT diagnostic. These are NOT orbits: vertices sharing a
    structural colour (same primitive/params/shapes) or a stable WL class are
    locally exchangeable-looking but need not admit any global automorphism
    (e.g. the two encoder blocks sit at different depths of one chain). They
    are candidate seeds for a near-orbit POMO variant, nothing stronger."""
    nodes, succ, pred, cint = _automorphism_encoding(graph, True)
    real = set(graph.nodes)
    prim = graph.meta.get("prim_names", {})

    def _classes(coloring, tag):
        by = {}
        for n in nodes:
            if n in real:
                by.setdefault(coloring[n], []).append(n)
        multi = sorted((vs for vs in by.values() if len(vs) > 1),
                       key=lambda vs: (-len(vs), min(vs)))
        hist = Counter(len(vs) for vs in by.values())
        print(f"  {tag}: {len(by)} classes, histogram {dict(sorted(hist.items()))}")
        for vs in multi[:10]:
            names = sorted({str(prim.get(v, graph.colors.get(v, ('?',))[0])) for v in vs})
            print(f"    class size {len(vs)}: {sorted(vs)}  [{', '.join(names)}]")
        return multi

    print("\n--- near-orbit diagnostic (structural classes, NOT orbits) ---")
    multi0 = _classes(cint, "identical structural colour (WL depth 0)")
    stable = wl_refine(nodes, succ, pred, cint)
    _classes(stable, "stable 1-WL classes")
    return multi0


def _dedup_block(graph, tag, sampler, rng):
    t0 = time.perf_counter()
    orders = [tuple(sampler(rng)) for _ in range(N_SAMPLES)]
    uniq_exact = len(set(orders))
    keys = [trace_key(o, graph) for o in orders]
    uniq_trace = len(set(keys))
    dt = time.perf_counter() - t0
    n = len(orders)
    print(f"\n--- trace dedup, {tag} ({n} samples, {dt:.1f}s) ---")
    print(f"  unique exact orders : {uniq_exact:5d}  "
          f"(exact-order caching removes {100.0 * (n - uniq_exact) / n:.1f}%)")
    print(f"  unique trace keys   : {uniq_trace:5d}  "
          f"(trace caching removes       {100.0 * (n - uniq_trace) / n:.1f}%)")
    extra = uniq_exact - uniq_trace
    print(f"  extra dedup from traces over exact caching: {extra} orders "
          f"({100.0 * extra / max(uniq_exact, 1):.1f}% of the exact-unique set)")
    cls = Counter(keys)
    print(f"  trace-class sizes among samples: {dict(sorted(Counter(cls.values()).items()))}")
    return uniq_exact, uniq_trace


def main():
    t0 = time.perf_counter()
    print("building TransformerLM elimination graph "
          f"(SEQ={os.environ['ALPHAGRAD_TLM_SEQ']}, "
          f"DMODEL={os.environ['ALPHAGRAD_TLM_DMODEL']}, "
          f"VOCAB={os.environ['ALPHAGRAD_TLM_VOCAB']}) ...")
    graph = build_example_graph("TransformerLM")
    n_edges = sum(len(s) for s in graph.succ.values())
    print(f"graph built in {time.perf_counter() - t0:.1f}s: "
          f"{len(graph.nodes)} nodes ({graph.meta.get('n_eqns')} eqns), "
          f"{n_edges} edges, {len(graph.eliminable)} eliminable vertices")

    # ---------------- (a) orbits, coloured vs uncoloured ----------------
    res_c = _orbit_block(graph, colored=True)
    res_u = _orbit_block(graph, colored=False)

    nontriv_c = [o for o in res_c.orbits if len(o) > 1]
    print("\n=== POMO gate ===")
    if nontriv_c:
        print(f"  coloured orbits ARE non-trivial: {len(nontriv_c)} orbits of "
              f"sizes {sorted((len(o) for o in nontriv_c), reverse=True)} -> "
              "orbit-seeded POMO multi-start is viable.")
    else:
        print("  coloured automorphism group is TRIVIAL on this graph: every "
              "orbit is a singleton -> the POMO orbit-start variant is dead; "
              "fall back to arbitrary distinct starts.")
    _near_orbit_block(graph)

    # ---------------- (b) trace dedup rates ----------------
    elim = list(graph.eliminable)
    rev = sorted(elim, reverse=True)

    def sample_random(rng):
        o = list(elim)
        rng.shuffle(o)
        return o

    def sample_near_reverse(rng):
        o = list(rev)
        for _ in range(rng.randint(1, MAX_SWAPS)):
            i = rng.randrange(len(o) - 1)
            o[i], o[i + 1] = o[i + 1], o[i]
        return o

    _dedup_block(graph, "1000 uniform random orders", sample_random,
                 random.Random(0))
    _dedup_block(graph, f"1000 near-reverse orders (1..{MAX_SWAPS} adjacent swaps)",
                 sample_near_reverse, random.Random(1))

    # ---------------- (c) augment sanity on TLM ----------------
    perm = next((a for a in res_c.automorphisms
                 if any(a.get(k, k) != k for k in a)), None)
    print("\n--- augment sanity ---")
    if perm is None:
        print("  no non-trivial coloured automorphism -> augment reduces to identity here.")
    else:
        ok_auto = is_automorphism(graph, perm, colored=True)
        rng = random.Random(2)
        traj = list(elim)
        rng.shuffle(traj)
        aug = augment(traj, perm)
        legal = sorted(aug) == sorted(elim)
        same_cost = markowitz_profile(graph, aug) == markowitz_profile(graph, traj)
        moved = sum(1 for k in perm if perm[k] != k)
        print(f"  automorphism check: {ok_auto}; moves {moved} vertices")
        print(f"  augmented trajectory legal: {legal}; "
              f"Markowitz profile identical: {same_cost}")

    print(f"\ntotal {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
