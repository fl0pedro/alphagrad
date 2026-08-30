"""decode_probe: sizes/timings for the palimpsa-vmem decodability study."""
import os, sys, time
import numpy as np
import jax

from alphagrad.elimrl.baselines import tlm_target
from alphagrad.elimrl.env import ElimEnv
from alphagrad.elimrl.features import build_static, extract
from graphax import IncrementalPathTokenizer

T0 = time.time()
fn, args, argnums = tlm_target(seq=32, dmodel=128, vocab=1024)
env = ElimEnv(fn, args, argnums, vertex_only=True, symbolic=True)
static = build_static(env)
closed = jax.make_jaxpr(fn)(*args)
jaxpr, consts = closed.jaxpr, closed.literals
print(f"[{time.time()-T0:.0f}s] n_eqns={len(jaxpr.eqns)} n_rows={static.n_rows} "
      f"legal0={len(env.state().legal_vertices)} jacve={len(env.jacve_vertices)}", flush=True)

tk = IncrementalPathTokenizer(jaxpr, tuple(argnums), list(consts), list(args),
                              vocab_size=512)
t = time.time()
base = [int(x) for x in tk.base_tokens()]
own = [int(x) for x in tk.last_owner_ids()]
ids = [int(x) for x in tk.last_eqn_ids()]
print(f"base_tokens={len(base)} owners={len(own)} max_owner={max(own)} "
      f"n_owned={sum(1 for o in own if o>0)} eqn_ids_uniq={len(set(ids))} "
      f"max_tok={max(base)} wall={time.time()-t:.1f}s", flush=True)

MODE = sys.argv[1] if len(sys.argv) > 1 else "rev"
rng = np.random.default_rng(0)
env.reset()
lens, tv = [], []
t = time.time()
tex = 0.0
for i in range(len(env.jacve_vertices)):
    st = env.state()
    legal = st.legal_vertices
    if not legal:
        break
    te = time.time(); sf = extract(st, static); tex += time.time() - te
    if MODE == "rev":
        v = max(legal)
    else:
        v = int(rng.choice(legal))
    d = [int(x) for x in tk.eliminate(int(v))]
    lens.append(len(d)); tv.append(v)
    env.step(("V", int(v)))
    if i in (0, 10, 20, 50, 70):
        print(f"  step {i}: nlegal={len(legal)} delta={len(d)} "
              f"cumtok={len(base)+sum(lens)} fill_nonzero="
              f"{int((sf.vert_dyn[:,3]>0).sum())} wall={time.time()-t:.1f}s", flush=True)
print(f"MODE={MODE} steps={len(lens)} sum_delta={sum(lens)} max_delta={max(lens)} "
      f"total_tokens={len(base)+sum(lens)} elim_wall={time.time()-t:.1f}s "
      f"extract_wall={tex:.1f}s", flush=True)
