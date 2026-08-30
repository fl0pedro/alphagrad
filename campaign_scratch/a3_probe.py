"""A3 validation: per-episode spread, SNR, and the frozen-gradient test.

Runs the SAME measurement path the campaign uses (`landscape_map.build_env` ->
`env._callback`), so the numbers are comparable with the singleton-skip sweep
that produced "identity 0.9260 vs skip-k24/f0 0.9258".

Arms (all on one process, one env, one compile cache):

    base   rotate=0 heldout=0   the pre-A3 channel; one deterministic reading
    rot    rotate=1 heldout=0   (b) alone: per-episode probe rotation
    held0  rotate=0 heldout=1   (a) alone: held-out scoring, fixed batches
    held   rotate=1 heldout=1   (a)+(b): the configuration A3 proposes

Emits one CSV row per (arm, episode, plan) as it goes, so a truncated run is
still usable.
"""
import csv
import io
import os
import sys
import time

# landscape_map parses argv AT IMPORT (its CLI must run before the heavy
# imports, see its module comment), so build the argv it should see first.
_EPISODES = int(os.environ.get("A3_EPISODES", "10"))
_ARMS = os.environ.get("A3_ARMS", "base,held0,rot,held").split(",")
_PLANS = os.environ.get("A3_PLANS", "identity,24:0,21:1").split(",")
_OUT = os.environ.get("A3_OUT", "a3_probe_rows.csv")

sys.argv = [
    "landscape_map",
    "--example", os.environ.get("A3_EXAMPLE", "TransformerLM"),
    "--quality-metric", "loss_drop",
    "--walk-steps", os.environ.get("A3_WALK_STEPS", "200"),
    # Cost channels are not the question here; one point / one rep / one inner
    # rep keeps every measurement's non-walk time small. The WALK is untouched
    # (200 Adam steps at lr 1e-3 -- the measured configuration).
    "--num-data-points", "1",
    "--reps-per-point", "1",
    "--latency-inner-reps", "1",
    "--num-eval-samples", "1",
    "--warmup-trials", "0",
    "--reps", "1",
    "--no-figure",
    "--dry-run",                 # never let main() run; we drive it ourselves
]

import numpy as np                                              # noqa: E402

from alphagrad.approx.tools import landscape_map as LM          # noqa: E402
import alphagrad.approx.env as envmod                           # noqa: E402


# ---------------------------------------------------------------- timing
# The wall cost of (a) and (b) is a DELIVERABLE, and it is not visible in
# `_callback`'s total (which is dominated by compile + the cost channels), so
# instrument the two functions directly.
_T = {"walk_s": 0.0, "walk_n": 0, "probe_s": 0.0, "probe_n": 0}
_orig_walk = envmod._loss_drop_quality
_orig_probe = envmod._probe_batch


def _timed_walk(*a, **k):
    t0 = time.perf_counter()
    try:
        return _orig_walk(*a, **k)
    finally:
        _T["walk_s"] += time.perf_counter() - t0
        _T["walk_n"] += 1


def _timed_probe(*a, **k):
    t0 = time.perf_counter()
    n0 = len(envmod._PROBE_BATCH)
    try:
        return _orig_probe(*a, **k)
    finally:
        dt = time.perf_counter() - t0
        if len(envmod._PROBE_BATCH) != n0 or n0 == 0:
            _T["probe_s"] += dt          # only count actual DRAWS
            _T["probe_n"] += 1


envmod._loss_drop_quality = _timed_walk
envmod._probe_batch = _timed_probe


def _pop_timing():
    out = dict(_T)
    _T.update({"walk_s": 0.0, "walk_n": 0, "probe_s": 0.0, "probe_n": 0})
    return out


# ---------------------------------------------------------------- setup
def main():
    args = LM.ARGS
    print(f"[a3] example={args.example} walk_steps={args.walk_steps} "
          f"probe_seed={args.walk_probe_seed} episodes={_EPISODES} "
          f"arms={_ARMS} plans={_PLANS}", flush=True)
    t0 = time.perf_counter()
    env, eval_samples, _cj = LM.build_env(args)
    order = LM.rev_order(env)
    print(f"[a3] env built in {time.perf_counter()-t0:.1f}s; "
          f"{len(order)} valid vertices", flush=True)

    # A SKIP on a (k, f) that is not a LIVE face is a silent no-op, which would
    # read as "held-out scoring cannot tell them apart". Confirm liveness
    # before spending an hour measuring.
    inv_l = LM.face_inventory(env, order)
    inv = {(int(e["k"]), int(e["f"])): e for e in inv_l}
    prims = {}
    for e in inv_l:
        prims.setdefault(e["prim"], []).append((int(e["k"]), int(e["f"])))
    print(f"[a3] {len(inv)} live faces; by primitive: "
          + ", ".join(f"{p}={len(v)}" for p, v in sorted(prims.items())),
          flush=True)

    def _resolve(spec):
        """``K:F`` or ``prim:<name>[:n]`` -> (k, f) or None."""
        parts = spec.split(":")
        if parts[0] == "prim":
            cand = prims.get(parts[1], [])
            idx = int(parts[2]) if len(parts) > 2 else 0
            return cand[idx] if idx < len(cand) else None
        return (int(parts[0]), int(parts[1]))

    plans = {}
    for spec in _PLANS:
        if spec == "identity":
            plans["identity"] = LM.build_skip_only_plan(env, order, [])
            continue
        kf = _resolve(spec)
        if kf is None or kf not in inv:
            print(f"[a3] target {spec}: *** NOT A LIVE FACE -- a skip there "
                  f"is a no-op, SKIPPING this plan ***", flush=True)
            continue
        k, f = kf
        e = inv[kf]
        print(f"[a3] target {spec} -> k{k}/f{f}: LIVE vertex={e['vertex']} "
              f"prim={e['prim']}", flush=True)
        plans[f"skip-k{k}/f{f}[{e['prim']}]"] = LM.build_skip_only_plan(
            env, order, [(k, f)])
    print(f"[a3] plans: {list(plans)}", flush=True)

    fields = ["arm", "rotate", "heldout", "episode", "plan", "quality",
              "latency_ns", "peak_memory", "cb_wall_s", "walk_s", "walk_n",
              "probe_draw_s", "probe_draws"]
    # RESUMABLE. Rows are flushed as they are produced and a re-run skips
    # whatever is already on disk -- the cluster cancels jobs out from under
    # this often enough that a non-resumable 60-measurement sweep never
    # finishes.
    fresh = not os.path.exists(_OUT)
    have = set()
    if not fresh:
        for r in csv.DictReader(io.open(_OUT)):
            have.add((r["arm"], int(r["episode"]), r["plan"]))
        print(f"[a3] resuming: {len(have)} rows already on disk", flush=True)
    fh = io.open(_OUT, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=fields)
    if fresh:
        wr.writeheader()

    arm_cfg = {
        "base":  (0, 0),
        "rot":   (1, 0),
        "held0": (0, 1),
        "held":  (1, 1),
    }
    for arm in _ARMS:
        rot, held = arm_cfg[arm]
        os.environ["ALPHAGRAD_WALK_ROTATE"] = str(rot)
        os.environ["ALPHAGRAD_WALK_HELDOUT"] = str(held)
        eps = range(_EPISODES) if rot else [0]
        for ep in eps:
            envmod.set_walk_episode(ep)
            for pid, pl in plans.items():
                if (arm, ep, pid) in have:
                    continue
                _pop_timing()
                m = LM.measure(env, eval_samples, order, pl)
                tm = _pop_timing()
                row = {
                    "arm": arm, "rotate": rot, "heldout": held,
                    "episode": ep, "plan": pid,
                    "quality": m["quality"],
                    "latency_ns": m["latency_ns"],
                    "peak_memory": m["peak_memory"],
                    "cb_wall_s": round(m["wall_s"], 4),
                    "walk_s": round(tm["walk_s"], 4),
                    "walk_n": tm["walk_n"],
                    "probe_draw_s": round(tm["probe_s"], 4),
                    "probe_draws": tm["probe_n"],
                }
                wr.writerow(row)
                fh.flush()
                print(f"[a3] {arm} ep={ep} {pid:16s} q={m['quality']:.8f} "
                      f"walk={tm['walk_s']:.2f}s draws={tm['probe_n']} "
                      f"({tm['probe_s']:.3f}s) cb={m['wall_s']:.1f}s",
                      flush=True)
    fh.close()
    print(f"[a3] wrote {_OUT}", flush=True)
    report(_OUT)


# ---------------------------------------------------------------- report
def report(path):
    rows = list(csv.DictReader(io.open(path)))
    if not rows:
        return
    plans = sorted({r["plan"] for r in rows})
    print("\n================ A3 RESULTS ================")
    for arm in ("base", "held0", "rot", "held"):
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        print(f"\n--- arm {arm} (rotate={sub[0]['rotate']} "
              f"heldout={sub[0]['heldout']}, n_ep="
              f"{len({r['episode'] for r in sub})}) ---")
        stats = {}
        for p in plans:
            q = np.array([float(r["quality"]) for r in sub if r["plan"] == p])
            if not len(q):
                continue
            stats[p] = q
            print(f"  {p:16s} mean={q.mean():.8f} sd={q.std(ddof=1) if len(q)>1 else 0.0:.3e} "
                  f"min={q.min():.8f} max={q.max():.8f} n={len(q)}")
        base = stats.get("identity")
        if base is None:
            continue
        for p, q in stats.items():
            if p == "identity":
                continue
            gap = base.mean() - q.mean()
            sd = max(base.std(ddof=1) if len(base) > 1 else 0.0,
                     q.std(ddof=1) if len(q) > 1 else 0.0)
            n = min(len(base), len(q))
            if n > 1:
                # PAIRED by episode: both plans see the SAME batch in a given
                # episode, so the paired sd is the honest noise floor for the
                # comparison, not the marginal per-plan sd.
                pb = {int(r["episode"]): float(r["quality"])
                      for r in sub if r["plan"] == "identity"}
                pq = {int(r["episode"]): float(r["quality"])
                      for r in sub if r["plan"] == p}
                d = np.array([pb[e] - pq[e] for e in sorted(pb) if e in pq])
                psd = d.std(ddof=1) if len(d) > 1 else 0.0
                snr_u = abs(gap) / sd if sd > 0 else float("inf")
                snr_p = abs(d.mean()) / psd if psd > 0 else float("inf")
                print(f"  GAP identity - {p}: {gap:+.6f}   "
                      f"unpaired SNR={snr_u:.2f} (sd {sd:.3e})   "
                      f"paired mean_d={d.mean():+.6f} sd_d={psd:.3e} "
                      f"SNR={snr_p:.2f}")
                if snr_u > 0 and snr_u < 2.0:
                    need = (2.0 / max(snr_u, 1e-9)) ** 2
                    print(f"       -> unpaired SNR < 2: would need ~{need:.1f} "
                          f"batches averaged to reach SNR 2")
            else:
                print(f"  GAP identity - {p}: {gap:+.6f}  (single reading)")
    # wall cost
    print("\n--- wall cost ---")
    for arm in ("base", "held0", "rot", "held"):
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        w = np.array([float(r["walk_s"]) for r in sub])
        d = np.array([float(r["probe_draw_s"]) for r in sub])
        nd = np.array([int(r["probe_draws"]) for r in sub])
        print(f"  {arm:6s} walk {w.mean():.3f}s/plan (n={len(w)})  "
              f"probe draws {nd.sum()} total, {d.sum():.3f}s total, "
              f"{d.sum()/max(len(w),1):.4f}s/plan amortised")


if __name__ == "__main__":
    if os.environ.get("A3_REPORT_ONLY", "0") == "1":
        report(_OUT)
    else:
        main()
