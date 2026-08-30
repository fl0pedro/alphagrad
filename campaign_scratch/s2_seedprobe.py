"""Which seeds keep the policy gate's trace NON-TRIVIAL after the face-path
collapse? Reports per-step (vertex, live faces) so a seed can be picked on
evidence rather than on the first one that stops raising."""
import sys
sys.argv = [sys.argv[0]]
import tests.policy_regression_gate as G

for seed in (20260810, 20260814, 1, 2, 3, 7, 11, 13):
    G.SEED = seed
    try:
        tr = G.run_trace()
    except Exception as e:
        print(f"seed {seed}: RAISED {type(e).__name__}: {str(e)[:120]}")
        continue
    rows = [(s["vertex"], (s["face"] or {}).get("n_live", 0)) for s in tr["steps"]]
    try:
        G.assert_nontrivial(tr)
        ok = "NONTRIVIAL"
    except AssertionError as e:
        ok = "TRIVIAL: " + str(e).split("- ")[-1].strip()[:80]
    print(f"seed {seed}: steps={rows} -> {ok}", flush=True)
