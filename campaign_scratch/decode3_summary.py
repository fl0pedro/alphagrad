"""decode3_summary: steps-to-threshold + final, train beside test.

Reads every decode3 run json in a directory and prints, per target, the first
gradient step at which the held-out WITHIN-STEP R2 crosses 0.6 / 0.8 / 0.9,
the final test value and the final train value.  Steps, not final R2, is the
comparison that matters: gradient steps are the binding budget in RL.
"""
import json, sys, os, glob

THR = [0.6, 0.8, 0.9]


def steps_to(curve, name, key, q=None):
    out = []
    for th in THR:
        hit = None
        for c in curve:
            r = c[key][name]
            v = r["r2w"] if q is None else r[q]["r2w"]
            if v == v and v >= th:
                hit = c["step"]; break
        out.append(hit)
    return out


def final(curve, name, key, q=None):
    r = curve[-1][key][name]
    return r["r2w"] if q is None else r[q]["r2w"]


for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    D = json.load(open(path))
    curve = D["curve"]
    if not curve:
        continue
    a = D["args"]
    kind = "face" if "chunk_cap" in a else "vertex"
    q = None if kind == "face" else "tALL"
    names = list(curve[-1]["test"].keys())
    degen = set(curve[-1].get("degenerate", []))
    print(f"\n=== {os.path.basename(path)}  ({kind}, arm "
          f"{a.get('arm', a.get('variant'))}, {curve[-1]['step']} steps, "
          f"{curve[-1]['wall']:.0f}s) ===")
    print(f"  {'target':13s} {'->0.6':>7s} {'->0.8':>7s} {'->0.9':>7s} "
          f"{'final te':>9s} {'final tr':>9s}")
    for nm in names:
        if nm in degen:
            print(f"  {nm:13s}    DEGENERATE (target identically constant)")
            continue
        s = steps_to(curve, nm, "test", q)
        ft = final(curve, nm, "test", q)
        fr = final(curve, nm, "train", q)
        f = lambda x: ("never" if x is None else str(x))
        print(f"  {nm:13s} {f(s[0]):>7s} {f(s[1]):>7s} {f(s[2]):>7s} "
              f"{ft:9.3f} {fr:9.3f}")
