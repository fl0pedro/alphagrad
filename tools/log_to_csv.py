"""Turn a run's stdout log into a CSV for matplotlib.

The trainers log per-episode lines but nothing writes a machine-readable file,
so plotting meant re-parsing ad hoc. This extracts every ``key=value`` pair from
lines carrying an ``ep=`` marker and writes one row per episode.

  python tools/log_to_csv.py RUN.out OUT.csv [--warmup 50] [--drop-zeros]

--warmup N        skip the first N episodes
--warmup-frac F   skip the first F of the episodes ACTUALLY PRESENT, e.g.
                  0.05 -> 50 of 1000, 13 of 250. Scales with the run rather
                  than assuming its length: an absolute --warmup 50 against a
                  run that only reached 5 episodes silently produces an EMPTY
                  csv, which is what happened when exact_mlp crashed early.
                  Overrides --warmup when given.
--drop-zeros drop rows whose numeric fields are all zero -- unmeasured steps log
             zeros, and averaging those in silently drags every curve down
"""
import argparse
import csv
import re
import sys

KV = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)=([-+0-9.eE]+|nan|inf|-inf)")


def parse(path):
    rows = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if "ep=" not in line:
                continue
            pairs = dict(KV.findall(line))
            if "ep" not in pairs:
                continue
            try:
                pairs["ep"] = int(float(pairs["ep"]))
            except ValueError:
                continue
            rows.append(pairs)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("out")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--warmup-frac", type=float, default=None)
    ap.add_argument("--drop-zeros", action="store_true")
    a = ap.parse_args()

    rows = parse(a.log)
    if not rows:
        print(f"[log_to_csv] no ep= rows in {a.log}", file=sys.stderr)
        return 1

    merged = {}
    for r in rows:                       # several lines can share one episode
        merged.setdefault(r["ep"], {}).update(r)
    eps = sorted(merged)

    warmup = a.warmup
    if a.warmup_frac is not None:
        if not (0.0 <= a.warmup_frac < 1.0):
            print(f"[log_to_csv] --warmup-frac must be in [0,1), got "
                  f"{a.warmup_frac}", file=sys.stderr)
            return 2
        # fraction of the episodes PRESENT, not of those requested -- a run
        # that stopped early still yields a sane cut rather than nothing
        warmup = int(round(a.warmup_frac * len(eps)))

    kept = []
    for e in eps:
        if e < warmup:
            continue
        r = merged[e]
        if a.drop_zeros:
            vals = [float(v) for k, v in r.items()
                    if k != "ep" and _isnum(v)]
            if vals and all(v == 0.0 for v in vals):
                continue
        kept.append(r)

    cols, seen = ["ep"], {"ep"}
    for r in kept:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in kept:
            w.writerow(r)
    print(f"[log_to_csv] {a.log} -> {a.out}: {len(kept)} rows "
          f"(of {len(eps)} episodes, warmup={warmup}, "
          f"drop_zeros={a.drop_zeros}), {len(cols)} columns")
    return 0


def _isnum(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
