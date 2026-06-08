import glob, os, json
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
# newest matrix run (47290-era)
d = sorted(glob.glob(os.path.expanduser("~/dsnn/wandb/run-*")), key=os.path.getmtime, reverse=True)[0]
wf = glob.glob(os.path.join(d, "*.wandb"))[0]
print("run:", os.path.basename(d))
ds = datastore.DataStore(); ds.open_for_scan(wf)
last = {}
while True:
    try: x = ds.scan_data()
    except Exception: break
    if x is None: break
    r = pb.Record()
    try: r.ParseFromString(x)
    except Exception: continue
    if r.WhichOneof("record_type") == "history":
        for it in r.history.item:
            k = it.key or "/".join(it.nested_key)
            if k.startswith("reward_norm_symlog/") or k.startswith("reward_mean/"):
                try: last[k] = float(json.loads(it.value_json))
                except Exception: pass
CH = ["muls_adds_fmas","flops","latency_ns","max_io_sum","bytes_accessed","peak_memory","cosine_sim","frob_residual"]
print(f"\n{'channel':16} {'symlog_mean':>12} {'symlog_std(σ)':>13} {'raw_mean':>12} {'1/σ (static λ)':>14}")
for c in CH:
    sm = last.get(f"reward_norm_symlog/{c}/mean", float('nan'))
    ss = last.get(f"reward_norm_symlog/{c}/std", float('nan'))
    rm = last.get(f"reward_mean/{c}_raw", last.get(f"reward_mean/{c}", float('nan')))
    inv = (1.0/ss) if (ss==ss and ss>1e-9) else float('nan')
    print(f"{c:16} {sm:>12.4g} {ss:>13.4g} {rm:>12.4g} {inv:>14.4g}")
