import collections
import json
import statistics as st
import sys

corr, costf = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
d = json.load(open(corr))
cost = {}
if costf:
    try:
        c = json.load(open(costf))
        cost = {(r["pos"], r["config"]): r for r in c["rows"]}
    except Exception as e:
        print("no cost file:", e)

print(f"TARGET {d['target']}  plan_len={d['plan_len']}  "
      f"control cos={d['control']['cos']:.7f}")
inter = sorted({r["pos"] for r in d["rows"] if not r.get("terminal_position")})
term = sorted({r["pos"] for r in d["rows"] if r.get("terminal_position")})
print(f"INTERMEDIATE positions audited: {inter}   TERMINAL control: {term}")
print()

agg = collections.defaultdict(collections.Counter)
best = collections.defaultdict(list)
lat = collections.defaultdict(list)
mem = collections.defaultdict(list)
tagg = collections.defaultdict(collections.Counter)
for r in d["rows"]:
    tgt = tagg if r.get("terminal_position") else agg
    if "verdict" not in r:
        tgt[r["config"]]["ERROR"] += 1
        continue
    tgt[r["config"]][r["verdict"]] += 1
    if r.get("terminal_position"):
        continue
    if r.get("cos_env") is not None:
        best[r["config"]].append(r["cos_env"])
    k = (r["pos"], r["config"])
    if k in cost:
        if cost[k].get("lat_ratio"):
            lat[r["config"]].append(cost[k]["lat_ratio"])
        if cost[k].get("mem_ratio"):
            mem[r["config"]].append(cost[k]["mem_ratio"])

hdr = (f"{'config':34s} {'(a)dec':>6s} {'verdicts @ INTERMEDIATE':44s} "
       f"{'(c)best cos':>11s} {'(d)lat':>7s} {'(e)mem':>7s}")
print(hdr)
print("-" * len(hdr))
for cfg in agg:
    v = agg[cfg]
    tot = sum(v.values())
    acts = v.get("ACTS", 0)
    s = " ".join(f"{k}:{n}" for k, n in v.most_common())
    bc = f"{min(best[cfg]):.6f}" if best[cfg] else "   -   "
    L = f"{st.median(lat[cfg]):.3f}" if lat[cfg] else "   -   "
    M = f"{st.median(mem[cfg]):.3f}" if mem[cfg] else "   -   "
    ndec = sum(1 for r in d["rows"]
               if r["config"] == cfg and not r.get("terminal_position")
               and r.get("decode_survives") is False)
    print(f"{cfg:34s} {('OK' if ndec == 0 else 'FAIL'):>6s} {s:44s} "
          f"{bc:>11s} {L:>7s} {M:>7s}   [{acts}/{tot} act]")

print()
print("TERMINAL-position control verdicts:")
for cfg in tagg:
    print(f"  {cfg:34s} " + " ".join(f"{k}:{n}"
                                     for k, n in tagg[cfg].most_common()))
print()
nfalse = sum(1 for r in d["rows"] if r.get("decode_survives") is False)
print(f"(a) decode_survives FALSE: {nfalse} of {len(d['rows'])} rows")
print(f"SILENTLY_DISCARDED rows  : "
      f"{sum(1 for r in d['rows'] if r.get('verdict') == 'SILENTLY_DISCARDED')}")
print(f"errors                   : "
      f"{sum(1 for r in d['rows'] if 'error' in r)}")
