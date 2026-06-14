import json, glob, os
from collections import Counter

def latest(pat):
    fs = [f for f in glob.glob(os.path.expanduser(pat)) if os.path.getsize(f) > 0]
    return max(fs, key=os.path.getmtime) if fs else None

VLAB = {1:"Tw1",2:"mm1",4:"+b1",5:"th1",6:"Tw2",7:"mm2",9:"+b2",10:"th2",11:"-y",12:"po2",13:"x.5"}

def acts(call_list):
    c = Counter()
    for s in call_list:
        if s.startswith("quant"): c["Q:"+s[s.find("'")+1:s.rfind("'")]] += 1
        elif s.startswith("compress"): c["C"] += 1
        elif s.startswith("diag"): c["D"] += 1
    return c

def real(p,k):
    v=p["obj"][k]; return v if k=="cosine_sim" else -v

def analyze(path, name):
    d=json.load(open(path)); front=d["front"]
    exact=[p for p in front if real(p,"cosine_sim")>=0.99]
    ne=[p for p in front if real(p,"cosine_sim")<0.99]
    ex_minlat=min((real(p,"latency_ns") for p in exact), default=float("inf"))
    ex_minpeak=min((real(p,"xla_peak_memory") for p in exact), default=float("inf"))
    ne.sort(key=lambda p: real(p,"cosine_sim"), reverse=True)
    print(f"\n================= {name} : best 5 non-exact (cos<0.99, by cosine) =================")
    print(f"   [exact corner refs: min-latency={ex_minlat/1e3:.0f}µs  min-peak={ex_minpeak/1e6:.2f}MB]")
    for p in ne[:5]:
        lat=real(p,"latency_ns"); xp=real(p,"xla_peak_memory"); cos=real(p,"cosine_sim")
        bargain = (lat < ex_minlat) or (xp < ex_minpeak)
        # collect non-empty per-vertex actions
        tags=[]
        for v,calls in p["seq"]:
            a=acts(calls)
            if a: tags.append("%s[%s]"%(VLAB.get(v,"v%d"%v), ",".join(f"{k}x{n}" if n>1 else k for k,n in a.items())))
        print(f"\n  cos={cos:.4f}  lat={lat/1e3:.1f}µs  peak={xp/1e6:.2f}MB  {'<< cheaper than exact' if bargain else '(not cheaper than exact)'}")
        print("    actions: " + ("; ".join(tags) if tags else "(none — order-only approx?)"))

for nm,pat in [("C-MORL","~/dsnn/wandb/offline-run-*/files/cmorl_pareto_front.json"),
               ("MOGFN","~/dsnn/wandb/offline-run-*/files/mogfn_pareto_front.*.json")]:
    p=latest(pat)
    if p: analyze(p,nm)
