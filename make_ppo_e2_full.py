#!/usr/bin/env python3
"""Derive run_ppo_e2_full.sh from run_ppo_e2_smoke.sh — the FAIR E2 PPO baseline.
Objective ALIGNED to AZ (user-directed): cosine_sim quality + latency+peak cost,
FLOPS DROPPED, PopArt on both. Exact weights via ALPHAGRAD_REWARD_CHANNELS.
Upgrades: cost-head OFF, clearN=64, APPEND_ONLY_JAXPR=1 (both smokes healthy),
MAX_SUBSTEPS=1 (match AZ --micro-budget 1). ~200 eps -> >=3400 measures.
"""
import pathlib
s = pathlib.Path("run_ppo_e2_smoke.sh").read_text()

# --- SBATCH header ---
s = s.replace("#SBATCH --job-name=ppo_e2_smoke", "#SBATCH --job-name=ppo_e2_full")
s = s.replace("#SBATCH --output=/Users/assmuth/dsnn/ppo_e2_smoke_%j.out",
              "#SBATCH --output=/Users/assmuth/dsnn/ppo_e2_full_%j.out")
s = s.replace("#SBATCH --time=03:00:00", "#SBATCH --time=12:00:00")

# --- override block: add APPEND_ONLY_JAXPR + REWARD_CHANNELS (aligned obj) +
#     switch smoke knobs to full-run values. Both smokes validated healthy. ---
anchor = "export ALPHAGRAD_COST_HEAD_AUX=0"
add = (anchor
       + "\nexport ALPHAGRAD_APPEND_ONLY_JAXPR=1   # recompile-free (Smoke-2: healthy, ~1.6x faster)"
       + "\n# ALIGNED E2 OBJECTIVE (user-directed): match AZ = cosine quality + lat+peak cost,"
       + "\n#   FLOPS DROPPED, PopArt normalises each channel. Exact weights (precedence over"
       + "\n#   cmp/mem/acc slots). Costs are env-negated so POSITIVE weights penalise them."
       + '\nexport ALPHAGRAD_REWARD_CHANNELS="cosine_sim:1.0,latency_ns:0.06,peak_memory:0.06"')
s = s.replace(anchor, add)

# smoke->full: EPISODES 25->200, own CAMP
s = s.replace('export EPISODES="${EPISODES:-25}"', 'export EPISODES="${EPISODES:-200}"')
s = s.replace('export CAMP="${CAMP:-campaign_ppo_e2_smoke}"', 'export CAMP="${CAMP:-campaign_ppo_e2_full}"')
s = s.replace('NAME="ppo_e2_smoke', 'NAME="ppo_e2_full')

# --- CRITICAL: the base launcher hardcodes bkstep AFTER the override block;
#     flip them so quality = cosine (not bkstep). ---
s = s.replace("export ALPHAGRAD_ACC_PROXY=bkstep", "export ALPHAGRAD_ACC_PROXY=cosine  # E2: quality=cosine (aligned to AZ), NOT bkstep")
s = s.replace("export ALPHAGRAD_BKSTEP=1", "export ALPHAGRAD_BKSTEP=0  # E2: bkstep probe OFF (cosine is the quality channel; saves ~80 train calls/terminal)")

pathlib.Path("run_ppo_e2_full.sh").write_text(s)
print("wrote run_ppo_e2_full.sh")
for ln in s.splitlines():
    if any(k in ln for k in ("job-name=ppo_e2_full","--time=12","REWARD_CHANNELS","APPEND_ONLY_JAXPR=1",
            "COST_HEAD_AUX=0","MEASURE_CACHE_CLEAR_EVERY=64","EPISODES:-200","MAX_SUBSTEPS:-1",
            "ACC_PROXY=cosine","BKSTEP=0","CAMP:-campaign_ppo_e2_full")):
        print("  |", ln.strip()[:110])
