import json, glob, os

def latest(pat):
    fs = [f for f in glob.glob(os.path.expanduser(pat)) if os.path.getsize(f) > 0]
    return max(fs, key=os.path.getmtime) if fs else None

VLAB = {1:"transpose(W1ᵀ)",2:"matmul x·W1ᵀ",3:"b1-broadcast/aux",4:"add b1",5:"tanh₁",
        6:"transpose(W2ᵀ)",7:"matmul a1·W2ᵀ",8:"b2-broadcast/aux",9:"add b2",10:"tanh₂",
        11:"sub (−y)",12:"pow2",13:"mul ½",14:"aux/reshape"}

def real(p,k):
    v=p["obj"][k]; return v if k=="cosine_sim" else -v

d=json.load(open(latest("~/dsnn/wandb/offline-run-*/files/mogfn_pareto_front.*.json")))
ne=[p for p in d["front"] if real(p,"cosine_sim")<0.99]
ne.sort(key=lambda p: real(p,"cosine_sim"), reverse=True)

for rank,p in enumerate(ne[:2],1):
    lat=real(p,"latency_ns"); xp=real(p,"xla_peak_memory"); cos=real(p,"cosine_sim")
    print("="*70)
    print(f"MOGFN non-exact #{rank}:  cos={cos:.4f}  latency={lat/1e3:.1f}µs  xla_peak={xp/1e6:.2f}MB")
    print(f"elimination order: {' → '.join('v%d'%v[0] for v in p['seq'])}")
    print("-"*70)
    for step,(v,calls) in enumerate(p["seq"],1):
        lab=VLAB.get(v,"v%d"%v)
        print(f"  [{step:>2}] eliminate v{v} = {lab}")
        if not calls:
            print("        (no micro-actions — exact local partial)")
        for j,c in enumerate(calls,1):
            print(f"        {j}. {c}")
