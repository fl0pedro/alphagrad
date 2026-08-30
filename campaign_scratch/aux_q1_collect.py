"""aux_q1: replay POMO plans, collect per-step candidate features + chosen label."""
import json, os, sys, time, glob
import numpy as np

from alphagrad.elimrl.baselines import tlm_target
from alphagrad.elimrl.env import ElimEnv
from alphagrad.elimrl.features import build_static, extract, DYN_VERTEX_DIM

OUT = sys.argv[1]
NPLAN = int(sys.argv[2])
MODE = sys.argv[3] if len(sys.argv) > 3 else "best"   # best | late | random

fn, args_, argnums = tlm_target(seq=32, dmodel=128, vocab=1024)
env = ElimEnv(fn, args_, argnums, vertex_only=True, symbolic=True)
static = build_static(env)
print("n_rows", static.n_rows, "n_eqns", static.n_eqns, "n_invars", static.n_invars, flush=True)
env.reset()
st = env.state()
print("legal0", len(st.legal_vertices), "nodes0", len(st.nodes), flush=True)

# --- load plans ---
rows = []
for s in (0, 1, 2):
    p = os.path.expanduser(f"~/dsnn/elimrl_m3_s{s}/measurements.jsonl")
    for li, line in enumerate(open(p)):
        d = json.loads(line)
        if d.get("status") != "ok" or not d.get("latency_ns"):
            continue
        rows.append((d["tag"], s, li, d["latency_ns"], d["order"]))
print("total ok", len(rows), flush=True)

pomo = [r for r in rows if r[0] == "pomo"]
if MODE == "best":
    sel = sorted(pomo, key=lambda r: r[3])[:NPLAN]
elif MODE == "late":
    sel = sorted(pomo, key=lambda r: -r[2])[:NPLAN]
else:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pomo), size=min(NPLAN, len(pomo)), replace=False)
    sel = [pomo[i] for i in idx]
print("selected", len(sel), "lat range", sel[0][3], sel[-1][3], flush=True)

SD = static.feat.shape[1]
D = DYN_VERTEX_DIM + SD
N = static.n_eqns  # steps per plan (== len(order))

allX, allmask, allchosen, allplan, allstep, alllat = [], [], [], [], [], []
t0 = time.time()
for pi, (tag, s, li, lat, order) in enumerate(sel):
    env.reset()
    for t, vid in enumerate(order):
        stt = env.state()
        legal = stt.legal_vertices
        if not legal:
            break
        sf = extract(stt, static)
        lr = np.asarray([static.row_of(j) for j in legal], np.int32)
        X = np.concatenate([sf.vert_dyn[lr], static.feat[lr]], axis=1).astype(np.float32)
        try:
            ci = legal.index(vid)
        except ValueError:
            print("  !! chosen not legal", pi, t, vid, flush=True)
            break
        C = len(legal)
        Xp = np.zeros((N, D), np.float32); Xp[:C] = X
        m = np.zeros(N, bool); m[:C] = True
        allX.append(Xp); allmask.append(m); allchosen.append(ci)
        allplan.append(pi); allstep.append(t); alllat.append(lat)
        env.step(("V", int(vid)))
    if pi % 10 == 0:
        print(f"plan {pi}/{len(sel)} elapsed {time.time()-t0:.1f}s", flush=True)

np.savez_compressed(OUT,
    X=np.stack(allX), mask=np.stack(allmask),
    chosen=np.asarray(allchosen, np.int32),
    plan=np.asarray(allplan, np.int32),
    step=np.asarray(allstep, np.int32),
    lat=np.asarray(alllat, np.float64),
    plan_lat=np.asarray([r[3] for r in sel], np.float64))
print("saved", OUT, np.stack(allX).shape, "wall", time.time()-t0, flush=True)
