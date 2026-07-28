#!/usr/bin/env python3
"""Collapse/degeneracy story straight from WANDB history (not tqdm)."""
import sys
import wandb

RUN = sys.argv[1] if len(sys.argv) > 1 else "dgk45yyz"
LO = int(sys.argv[2]) if len(sys.argv) > 2 else 15
HI = int(sys.argv[3]) if len(sys.argv) > 3 else 64

COLS = [
    ("collapse/count_this_ep", "collapse"),
    ("collapse/degenerate_plans_this_ep", "degen"),
    ("mean_muls_adds_fmas", "muls"),
    ("mean_cosine_sim", "cos"),
    ("mean_frob_residual", "frob"),
    ("mean_return", "return"),
    ("entropy/macro_vertex", "H_vertex"),
    ("entropy/micro_approx", "H_micro"),
    ("value loss", "vloss"),
    ("KL divergence", "KL"),
    ("op_marginal/quant", "p_quant"),
    ("op_marginal/compress", "p_compr"),
    ("op_marginal/diag", "p_diag"),
    ("op_marginal/end", "p_end"),
    ("sub_episode_length", "sublen"),
    ("per_face/skipped", "fskip"),
    ("per_face/applied", "fappl"),
]

api = wandb.Api()
run = api.run(f"dll-streetview/dsnn-vertex/{RUN}")
print(f"run={run.name} state={run.state}\n")

rows = list(run.scan_history(keys=["_step"] + [k for k, _ in COLS],
                             page_size=10000))
print(f"scanned {len(rows)} rows\n")

hdr = f"{'ep':>4}" + "".join(f"{lbl:>10}" for _, lbl in COLS)
print(hdr)
print("-" * len(hdr))
for r in rows:
    ep = r.get("_step")
    if ep is None or not (LO <= ep <= HI):
        continue
    line = f"{ep:>4}"
    for k, _ in COLS:
        v = r.get(k)
        if isinstance(v, (int, float)):
            line += f"{v:>10.3g}" if abs(v) >= 0.01 or v == 0 else f"{v:>10.2e}"
        else:
            line += f"{'-':>10}"
    print(line)
