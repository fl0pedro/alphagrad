import json, os, sys
RUN = os.path.expanduser("~/dsnn/wandb/run-20260608_091602-e924phix/files/best_sequences.json")
b = json.load(open(RUN))
# write a synthetic best_sequences.json with the muls/max Pareto pick as best_overall
pick = b["quantile_sequences"]["muls_adds_fmas"]["max"]
synth_dir = os.path.expanduser("~/dsnn/found_train/diag_factor/mulsmax/files")
os.makedirs(synth_dir, exist_ok=True)
json.dump({"best_overall": {"seq": pick["seq"], "return": 0, "ep": -1,
                            "rewards_raw": {}, "rewards_weighted": {}}},
          open(os.path.join(synth_dir, "best_sequences.json"), "w"))
print("wrote synthetic best_sequences.json for muls/max pick (Diag x{})".format(
    sum(len(v.get("ops",[])) for v in pick["seq"] if isinstance(v,dict))))
