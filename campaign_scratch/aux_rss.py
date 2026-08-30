import json,os,resource,time
import numpy as np
from alphagrad.elimrl.baselines import tlm_target
from alphagrad.elimrl.env import ElimEnv
from alphagrad.elimrl.features import build_static, extract
fn,a,ag = tlm_target(seq=32,dmodel=128,vocab=1024)
env = ElimEnv(fn,a,ag,vertex_only=True,symbolic=True)
static = build_static(env)
rows=[]
for s in (0,1,2):
    for li,l in enumerate(open(os.path.expanduser(f"~/dsnn/elimrl_m3_s{s}/measurements.jsonl"))):
        d=json.loads(l)
        if d.get("status")=="ok" and d.get("latency_ns") and d["tag"]=="pomo": rows.append((d["latency_ns"],d["order"]))
sel=sorted(rows)[:400]
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
print("rss after setup GB", rss(), flush=True)
Xs=[]
t0=time.time()
for pi,(lat,order) in enumerate(sel):
    env.reset()
    nZ=0
    for t,vid in enumerate(order):
        st=env.state(); lg=st.legal_vertices
        if len(lg)<2: break
        sf=extract(st,static); nZ=max(nZ,sf.edge_feat.shape[0])
        lr=np.asarray([static.row_of(j) for j in lg],np.int32)
        Xs.append(np.concatenate([sf.vert_dyn[lr],static.feat[lr]],1).astype(np.float32))
        env.step(("V",int(vid)))
    if pi%25==0: print(f"plan {pi} rssGB {rss():.2f} maxZ {nZ} nblocks {len(Xs)} t {time.time()-t0:.0f}s",flush=True)
