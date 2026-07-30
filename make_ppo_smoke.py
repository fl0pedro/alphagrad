#!/usr/bin/env python3
"""Derive run_ppo_e2_smoke.sh from run_full_nn256_v2_gpu.sh:
   - SBATCH: job-name/output/nodelist(gpu17)/time(3h)
   - inject E2 clean-baseline + AZ-parity overrides right after `set -uo pipefail`
   - Smoke-1: clean PPO (cost-head OFF) + clearN=64 (pure perf) + MAX_SUBSTEPS=1
     (match AZ --micro-budget 1). APPEND_ONLY_JAXPR deliberately LEFT OFF here
     (validated in Smoke-2 so it can't confound the health signal).
"""
import re, sys, pathlib

src = pathlib.Path("run_full_nn256_v2_gpu.sh").read_text()

# --- SBATCH header edits ---
src = src.replace("#SBATCH --job-name=full_nn256_v2", "#SBATCH --job-name=ppo_e2_smoke")
src = src.replace("#SBATCH --output=/Users/assmuth/dsnn/full_nn256_v2_%j.out",
                  "#SBATCH --output=/Users/assmuth/dsnn/ppo_e2_smoke_%j.out")
src = src.replace("#SBATCH --nodelist=pgi15-gpu15", "#SBATCH --nodelist=pgi15-gpu17")
src = src.replace("#SBATCH --time=24:00:00", "#SBATCH --time=03:00:00")

OVR = r'''
# ================= E2 FAIR-BASELINE + AZ-PARITY OVERRIDES (bridge-cse 2026-07-11) =================
# Author 2026-07-11: PPO must be a PLAIN, CORRECT baseline for E2 (AZ vs PPO
# sample-efficiency) and brought up to the efficiency upgrades AZ got.
#   * SURROGATE HEAD OFF: cost_head aux (ALPHAGRAD_COST_HEAD_AUX) is aux-loss-only
#     + default-off (never feeds reward), but we set it 0 EXPLICITLY so the E2
#     baseline is unambiguous / provably plain PPO.
#   * clearN (AZ had it, PPO=0): ALPHAGRAD_MEASURE_CACHE_CLEAR_EVERY=64 bounds the
#     measure-GPU XLA-executable leak. PURE PERFORMANCE — zero learning impact.
#   * FAIRNESS: MAX_SUBSTEPS=1 matches AZ's --micro-budget 1 (same action space).
#   * SMOKE-1 leaves ALPHAGRAD_APPEND_ONLY_JAXPR OFF (the recompile-free state
#     upgrade CHANGES the policy input encoding -> validated separately in
#     Smoke-2 so it cannot confound "is the clean baseline healthy?").
export ALPHAGRAD_COST_HEAD_AUX=0
export ALPHAGRAD_MEASURE_CACHE_CLEAR_EVERY=64
export EPISODES="${EPISODES:-25}"
export MAX_SUBSTEPS="${MAX_SUBSTEPS:-1}"
export CAMP="${CAMP:-campaign_ppo_e2_smoke}"
export WANDB="${WANDB:-online}"
export SEED="${SEED:-7}"
# =================================================================================================
'''

marker = "set -uo pipefail"
idx = src.index(marker) + len(marker)
src = src[:idx] + "\n" + OVR + src[idx:]

# make the wandb run name reflect the smoke (NAME line uses full_nn256_v2 prefix)
src = src.replace('NAME="full_nn256_v2', 'NAME="ppo_e2_smoke')

pathlib.Path("run_ppo_e2_smoke.sh").write_text(src)
print("wrote run_ppo_e2_smoke.sh")
# sanity: echo the injected block + key knobs
for ln in src.splitlines():
    if any(k in ln for k in ("job-name","nodelist","--time=","COST_HEAD_AUX=0",
            "MEASURE_CACHE_CLEAR_EVERY=64","EPISODES:-25","MAX_SUBSTEPS:-1","APPEND_ONLY_JAXPR")):
        print("  |", ln.strip())
