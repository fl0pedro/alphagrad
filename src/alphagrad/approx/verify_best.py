"""Measure TRUE single-point cosine + cost of the trajectory-run best sequence."""
import os, json
import numpy as np
import jax, jax.numpy as jnp, equinox as eqx
from alphagrad.approx.env import VertexEliminationEnv, _callback, REWARD_INDEX
from alphagrad.approx.common.examples import get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.verify_pareto_solution import build_order_specs

MODEL="VmappedNeuralNetwork"
LOSS=scalar_loss_fn(get_fn(MODEL)); ARGN=infer_argnums(MODEL)
def build_env():
    k=jax.random.PRNGKey(0); ak,ek=jax.random.split(k)
    xs=get_args(MODEL,ak,dataset="mnist"); gen=data_gen(MODEL,dataset="mnist",dataset_size=128)
    closed=jax.make_jaxpr(LOSS)(*xs)
    env=VertexEliminationEnv.from_jaxpr(closed,args=xs,argnums=ARGN,num_envs=0,data_gen=gen,target_fun=LOSS,
        cmp_type="latency",mem_type="peak_memory",measure_latency=True,latency_samples=1,
        num_data_points=5,reps_per_point=2,percentile_keep=0.60,slow_exec_cutoff_seconds=0.0,
        flop_gate_threshold=0.0,measure_grad=True,latency_timer="perf_counter",
        latency_inner_reps=5,latency_warmup=2,latency_winsor=0.2)
    ev=generate_eval_samples(env,ek,5)
    return eqx.tree_at(lambda e:e.eval_args_samples,env,ev),ev
env,ev=build_env()

# best seq from best.json: convert {vertex,ops} -> seq format [vid, [calls]]
bo=json.load(open(os.path.expanduser("~/dsnn/campaign_traj/VmappedNeuralNetwork/slot0/best.json")))["best_overall"]
def op_to_call(o):
    op=o["op"]
    if op=="Compress": return f"compress('{o['kind']}', 1)"
    if op=="Quant": return f"quant('{o['dtype']}')"
    if op=="Diag": return f"diag({o.get('i',0)}, {o.get('j',0)}, {o.get('factor',0)})"
    raise ValueError(op)
seq=[[s["vertex"],[op_to_call(o) for o in s.get("ops",[])]] for s in bo["seq"]]

# TRUE single-point cosine (proxy OFF)
os.environ.pop("ALPHAGRAD_QUALITY_PROXY",None)
order,specs,_=build_order_specs(seq,env)
_,_,rew=_callback(env.config,env.args,env.consts,jnp.asarray(order),jnp.asarray(specs),len(order),*ev,raw_sink={})
rew=np.asarray(rew)
CI=REWARD_INDEX["cosine_sim"]; FI=REWARD_INDEX["frob_residual"]
LI=REWARD_INDEX["latency_ns"]; MI=REWARD_INDEX["peak_memory"]; XI=REWARD_INDEX["xla_peak_memory"]
print("=== TRUE quality of trajectory-run best (single-point, proxy OFF) ===")
print("real single-point cosine_sim:", round(float(rew[CI]),4))
print("frob_residual:               ", round(float(-rew[FI]),4))
print("latency_ns:                  ", round(float(-rew[LI]),1))
print("peak_memory:                 ", round(float(-rew[MI]),0))
print("xla_peak_memory:             ", round(float(-rew[XI]),0))
print("gated value reported in run (traj cosine):", round(bo["rewards_raw"]["cosine_sim"],4))
