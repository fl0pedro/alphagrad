import json, glob, os, statistics

def latest(pat):
    fs = [f for f in glob.glob(os.path.expanduser(pat)) if os.path.getsize(f) > 0]
    return max(fs, key=os.path.getmtime) if fs else None

def real(p,k):
    v=p["obj"][k]; return v if k=="cosine_sim" else -v

def nacts(p):  # total micro-actions across all vertices
    return sum(len(calls) for _,calls in p["seq"])

BINS=[(0.99,1.01,"exact >=0.99"),(0.9,0.99,"0.90-0.99"),(0.5,0.9,"0.50-0.90"),
      (0.05,0.5,"0.05-0.50"),(-1,0.05,"degenerate <0.05")]

def report(path,name):
    d=json.load(open(path)); front=d["front"]
    print(f"\n===== {name}  ({len(front)} pts) =====")
    print(f"  overall latency: min={min(real(p,'latency_ns') for p in front)/1e3:.1f}µs  max={max(real(p,'latency_ns') for p in front)/1e3:.0f}µs")
    print(f"  avg micro-actions/plan: {statistics.mean(nacts(p) for p in front):.1f}")
    print(f"  {'cos band':<18}{'n':>4}{'min_lat':>11}{'med_lat':>11}{'med_acts':>9}")
    for lo,hi,lab in BINS:
        b=[p for p in front if lo<=real(p,"cosine_sim")<hi]
        if not b:
            print(f"  {lab:<18}{0:>4}"); continue
        lats=[real(p,"latency_ns")/1e3 for p in b]
        print(f"  {lab:<18}{len(b):>4}{min(lats):>10.1f}µ{statistics.median(lats):>10.1f}µ{statistics.median(nacts(p) for p in b):>9.0f}")

report(latest("~/dsnn/wandb/offline-run-*/files/cmorl_pareto_front.json"),"C-MORL")
report(latest("~/dsnn/wandb/offline-run-*/files/mogfn_pareto_front.*.json"),"MOGFN")
