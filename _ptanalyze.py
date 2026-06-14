import json, glob, os

def latest(pat):
    fs = [f for f in glob.glob(os.path.expanduser(pat)) if os.path.getsize(f) > 0]
    return max(fs, key=os.path.getmtime) if fs else None

VLAB = {1:"Tᵂ¹", 2:"mm1", 4:"+b1", 5:"tanh1", 6:"Tᵂ²", 7:"mm2",
        9:"+b2", 10:"tanh2", 11:"-y", 12:"pow2", 13:"·½"}

def summ(call_list):
    # compress a per-vertex action list to a short tag
    from collections import Counter
    c = Counter()
    for s in call_list:
        if s.startswith("quant"): c["Q:"+s[s.find("'")+1:s.rfind("'")]] += 1
        elif s.startswith("compress"): c["C"] += 1
        elif s.startswith("diag"): c["D"] += 1
        else: c[s[:6]] += 1
    return ",".join(f"{k}x{v}" if v>1 else k for k,v in c.items())

def show(path, picks):
    d = json.load(open(path))
    front = d["front"]
    def real(p,k):
        v=p["obj"][k]; return v if k=="cosine_sim" else -v
    for label, keyfn in picks:
        cand = [p for p in front if keyfn[0](p)]
        if not cand:
            print(f"\n### {label}: none"); continue
        p = min(cand, key=keyfn[1])
        lat=real(p,"latency_ns"); xp=real(p,"xla_peak_memory"); cos=real(p,"cosine_sim")
        print(f"\n### {label}")
        print(f"   latency={lat/1e3:.1f}µs  xla_peak={xp/1e6:.2f}MB  cos={cos:.4f}")
        order=[v[0] for v in p["seq"]]
        print("   elim order:", " ".join(VLAB.get(v,str(v)) for v in order))
        for v,calls in p["seq"]:
            print(f"     v{v:<2}{('='+VLAB[v]) if v in VLAB else '':<7} {summ(calls)}")

C = latest("~/dsnn/wandb/offline-run-*/files/cmorl_pareto_front.json")
print("===================== C-MORL front =====================")
show(C, [
  ("BEST EXACT (cheapest latency, cos>=0.99)", (lambda p: p["obj"]["cosine_sim"]>=0.99, lambda p: -p["obj"]["latency_ns"])),
  ("LOW-MEM EXACT (cos>=0.99)",               (lambda p: p["obj"]["cosine_sim"]>=0.99, lambda p: -p["obj"]["xla_peak_memory"])),
  ("BEST LOSSY BARGAIN (0.85<=cos<0.99, cheapest lat)", (lambda p: 0.85<=p["obj"]["cosine_sim"]<0.99, lambda p: -p["obj"]["latency_ns"])),
  ("MID-COS (0.4<=cos<0.7)",                  (lambda p: 0.4<=p["obj"]["cosine_sim"]<0.7, lambda p: -p["obj"]["latency_ns"])),
  ("DEGENERATE CHEAP (cos<0.05, cheapest lat)",(lambda p: p["obj"]["cosine_sim"]<0.05, lambda p: -p["obj"]["latency_ns"])),
])
