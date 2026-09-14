> **HISTORICAL SNAPSHOT — 2026-08-15. NOT CURRENT.** Kept only as provenance
> for the results committed then; SUPERSEDED by `docs/EXPERIMENT_PLAN.md` (the
> current plan) and `docs/WAVE1_RESULTS.md` (the current results) — where they
> disagree, those win. This file now lives at `docs/CAMPAIGN_STATE.md`, and the
> scratch scripts it names by bare filename (`lean_probe.py`, `decode4_audit.py`,
> `bwd_kernel_probe.py`, …) now live in `campaign_scratch/`. Body unchanged.

# Autonomous campaign state (owner checked out 2026-08-15)

Canonical copy lives on pgi15 at `~/dsnn/alphagrad/CAMPAIGN_STATE.md`.
If context was compacted, READ THAT FILE FIRST, then `squeue -u assmuth`.

## Owner's instruction (verbatim intent)
1. When the four-pool results land, pick the best pool, PREFERRING one that is
   DAG-agnostic (no finetuning per DAG).
2. For the best TWO pools, investigate adding MESSAGE PASSING AFTER pooling.
3. Pick the best combination.
4. Rerun PPO: 3-layer transformer, wikitext, **seq len 64**, **1000 episodes**.
5. GAZ: measure the speedup of our changes (if measurable), update GAZ to the
   new simplified PPO model, rerun **1000 episodes** on the SAME problem/DAG.
6. Full report as the VERY LAST step. Keep this md updated as we go.
7. Prefer agents (sequential/parallel per dependencies), then double-check their
   work myself.

## UNRESOLVED AMBIGUITY — decided, flag in final report
Owner said seq len **64** for PPO but "same problem and DAG (**seq len 1024**)"
for GAZ. Contradiction. DECISION: run BOTH at **seq 64**, because "same problem
and DAG" is the comparability requirement and arm divergence has invalidated
results in this repo repeatedly (#84/#85/#86). Trivially rerunnable at another
length. STATE THIS IN THE REPORT.

## Stages
- [ ] **A** four-pool A/B (mean / lastrow / sumtok / attn), 5 seeds. Jobs 61233
      (RUNNING gpu19) + 61234 (dependent summary). Pick best, prefer DAG-agnostic.
- [ ] **B** best-2 pools + message passing after pooling. Pick best combination.
- [ ] **C** PPO 1000 episodes, 3-layer TLM wikitext seq 64.
- [ ] **D** GAZ: speedup measurement + port to simplified model + 1000 episodes,
      same config as C.
- [ ] **E** final report.

## Landed already (verified by me, not just claimed)
- 6-commit architecture refactor `e171a30`..`78e4100`; gate **PASS bit-identical**
  (job 61221), imports OK. residual_state / _data_embedding / face AxisSetEncoder
  gone; participation + [identity||dynamic] concat + --grad-window K (default 1).
- `aa0774c` token budgets: MAX_TOKENS deleted (was keeping OLDEST tokens ->
  froze the observation); MAX_DELTA_TOKENS 1024 -> 32768; overflow now RAISES.
- **BUDGET TRAP**: raised bound is free at `ALPHAGRAD_EXTEND_CHUNK=256`
  (4.1 ms both windows) but a **29x FLAT penalty at chunk=0** (529 ms whether 78
  or 25,737 tokens; bwd 2966 ms). CHECK EVERY LAUNCHER SETS CHUNK != 0.
- palimpsa single-recurrence `e171a30`; interpret-vs-ref PASS (fwd 1.0e-7,
  grad 1.8e-7).
- **EPISODE TOKEN STREAM** (owner rulings 2026-09-13, Q1, Q3 and Q4). The
  rollout no longer stores a padded token window per step.
  `Trajectory.delta_tokens` and `Trajectory.face_delta_tokens` were
  `(MAX_DELTA_TOKENS,)` uint8 each. They are now the int32 spans
  `delta_offset` and `face_offset`. Both index one uint8 row per environment
  per episode. There are two rows, one for the step deltas and one for the
  face chunks.
  At the transformer width (MAX_DELTA_TOKENS 32768, T 95) the token storage
  was 65 536 B per step per environment. It is now about 11.7 kB. The
  `--grad-window` K gather materialised every window K times. It is now K
  offsets into one contiguous span.
  `MAX_DELTA_TOKENS` is unchanged. It is still the per-step transport width
  and the write window, and its overflow contract is the same.
  The row is `2^n + MAX_DELTA_TOKENS` slots. The FIRST `n` comes from
  `--episode-tokens-log2`. If that is 0, it comes from
  `ALPHAGRAD_EPISODE_TOKENS_LOG2`. If that is unset, it is MAX_DELTA_TOKENS
  times the episode length, rounded up to the next power of two, divided by
  8. That is 19 at the transformer width.
  THE DRIVER THEN CHOOSES A BIN PER EPISODE (owner clarification
  2026-09-14). The bins are a small set of compiled programs, one per power
  of two. Before each rollout the driver takes the smallest bin that holds
  the longest episode stream of the last `ALPHAGRAD_EPISODE_TOKENS_HISTORY`
  episodes (default 8) times `ALPHAGRAD_EPISODE_TOKENS_MARGIN` (default 2).
  So a run drifts back down to a smaller bin when the deltas shrink. The
  compile per bin happens once, because jit keys on the static shape and the
  persistent JAX compilation cache carries it across runs.
  A step that would pass `2^n` raises on the host before the write. The
  driver logs one `[episode-stream]` line, re-runs that episode one bin up,
  and records the length that overflowed in the history. The hard cap is
  `ALPHAGRAD_EPISODE_TOKENS_LOG2_MAX`, which defaults to 24.
  The first bin holds the measured episode with a factor of 1.84. A per-step
  length that doubled therefore costs exactly one step up.
  Design: `.scratch/trustworthy-approx-search/episode-stream-design.md`.
  Code: `src/alphagrad/approx/common/episode_stream.py`. Tests:
  `tests/episode_stream_test.py`.

## Key measured facts to carry
- Vertex side: tokens-only reaches DV_FILL within-step R2 **0.933 in 7 steps**,
  beating the hand-feature anchor (10/30/310, 0.932). Hand features buy nothing.
- `new_slots` 0.933 vs live-path `new` 0.879 => **SetPointer+ctx_proj is a
  bottleneck**, separate open item.
- Face side tokens-only FAILS every live-contraction target (never crosses 0.6);
  train 0.57-0.73 vs ~0 test = OVERFITTING not under-training.
- Arm B (shapes explicit in the face's own chunk) still only 0.158 on ln_stored
  => NOT an information gap. Root cause = `_face_pool` is an unweighted MEAN
  (ppo.py:2227) over ~760 rows into E=32.
- Bar to beat (within-step R2): ln_factor .600 / n_diag .477 / n_paired .498 /
  n_comp .557 / ln_stored .671 (the `+extents` / `*_sizes` range).
- DAG-agnosticism audit already flagged: `op_embedding.weight (71,8) -> NSTEP`
  and `mlp.layers[0].weight (256,96) -> V(vertices)`. MUST determine whether
  that is a POOL arm or an artifact of the probe harness.

## Standing constraints
NEVER local python/jax; NEVER pgi15 access node; NEVER pgi14. Both CPU nodes
allocated to other users into late Aug -> use idle GPU node, partition `pgi15`
(NOT `pgi15-gpu`, invalid). ALWAYS `sbatch` + `-o` log, never blocking srun
(has cost 2 results). Every sbatch needs `export PATH=$HOME/.local/bin:$PATH`.
Blackwell: `--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true`;
unknown XLA_FLAGS are fatal at import. One GPU job per node. graphax READ ONLY,
never commit/stash. Commits: one line, scoped, NO co-author trailer.

## Log
- 2026-08-15: created. Stage A running (61233 45min in, producing per-seed
  face_{mean,sumtok}_s*.json). Monitor armed on 61233/61234.

## 2026-08-15 update: DAG-agnosticism flag RESOLVED as a false positive
decode4_audit.py docstring says it outright: on THIS graph two numbers collide.
  - op_embedding.weight (71,8): 71 = OP VOCABULARY size, and NSTEP is also 71.
    A bounded primitive vocabulary is DAG-agnostic by construction.
  - mlp.layers[0].weight (256,96): 96 = 3*embd_dim (the [ctx_i||ctx_j||face_latent]
    concat width), and the TLM happens to have 96 vertices.
NEITHER is real V-dependence. CONFIRM by re-running the audit on a graph with
V != 96 and n_ops != 71 -- if the flags vanish, settled. Fold into the transfer check.

## 2026-08-15: STAGE C/D BLOCKERS FOUND EARLY (examples.py)
1. **seq 64 is ALREADY THE DEFAULT.** `_tlm_dims()` reads
   ALPHAGRAD_TLM_SEQ (default "64"), ALPHAGRAD_TLM_DMODEL ("128"),
   ALPHAGRAD_TLM_VOCAB ("2048"). So the owner asking for "seq len 64 instead
   of 1024" needs NO change -- the 1024 in their message is most likely the
   VOCAB of the older seq32/dmodel128/vocab1024 config. Set the env vars
   EXPLICITLY in the launchers anyway so the run is self-documenting.
   => This also DISSOLVES the seq-64-vs-1024 ambiguity: run both arms at the
      default seq 64. Still note it in the report.
2. **A 3-LAYER TRANSFORMER DOES NOT EXIST.** `_transformer_lm` is hardcoded to
   TWO encoder blocks (z1, z2) with a fixed 16-arg signature; docstring says
   "2 encoder blocks + LM head + xent". Registered at examples.py lines
   ~127 / ~361 / ~432 / ~595.
   => Stage C/D need a NEW `_transformer_lm3` (third encoder block + 7 more
      weight args) registered at all those dispatch sites. Mechanical, mirrors
      the existing pattern, but it is CODE not a flag.
   => RISK: 2 blocks = 96 eqns / 95 eliminable. A 3rd block pushes it to ~140
      vertices, so episodes get longer and 1000 episodes gets more expensive.
      MEASURE per-episode wall on a 2-episode smoke BEFORE launching 1000.

## 2026-08-15: Stage C prerequisite PARALLELISED (independent of pool choice)
Agent launched to (a) add a 3-encoder-block `TransformerLM3` target to
examples.py as a NEW name (2-block target untouched -- other results depend on
it), registered at ALL dispatch sites, and (b) MEASURE eqn/vertex count,
per-episode wall and peak memory on a 2-episode PPO smoke, extrapolate to 1000
episodes, and give an affordable/not-affordable verdict. GAZ too if it runs.
Told explicitly: every launcher must set ALPHAGRAD_EXTEND_CHUNK != 0 (the 29x
flat trap), the endgame cache key is O(V^4) so a bigger graph is SUPERLINEAR,
gate must still pass, do not take gpu19 (61233 is there).
Stage A at 18/20 when this was launched.

## STAGE A COMPLETE — POOL HYPOTHESIS REFUTED (18 runs, all rc=0)
Mean within-step R2 across seeds (bar = the *_sizes / +extents range):
  target      mean    last    sumtok   attn    BAR
  ln_factor  -0.131  -0.237  -0.229   -0.131   0.600
  n_diag     -0.137  -0.225  -0.225   -0.151   0.477
  n_paired    0.066  -0.003  -0.025    0.078   0.498
  n_comp      0.121   0.041   0.060    0.120   0.557
  ln_stored   0.110   0.034   0.056    0.116   0.671
  stat_ln_j   0.970   0.972   0.971    0.973   (control, PASSES)
ALL FOUR POOLS FAIL. Ranking attn ~= mean > sumtok > last.
=> The MEAN WAS NOT THE BOTTLENECK. #161 (my diagnosis) is REFUTED by a
   4-way 5-seed A/B. The readout barely matters; the info is not there.
=> This IS the post-cleanup reconfirmation the owner deferred #94 for, and it
   comes back POSITIVE FOR EXTENTS (the *_sizes arms clear the bar 0.429-0.741).
=> Best two pools for Stage B = attn and mean (tied).
Section (d) end-to-end nn256 rollout A/B FAILED (4x rc=1) -- Part 1s MICRObench
(extend_bench.json, the 29x numbers) succeeded and stands; the end-to-end
confirmation did NOT run. Note in final report.

## STAGE B RUNNING + DAG-AGNOSTICISM RESOLVED (PASS)
Job 61243 on pgi15-gpu17: 8-arm factorial {attn,mean} x {mp 0,1} x {extents 0,1},
5 seeds, SEED AS OUTER LOOP so every arm shares the same seed set (paired).
Preflight all green: XLA flags accepted, 8/8 traces OK, audits complete.

**DAG-AGNOSTICISM: CLEAN PASS FOR EVERY ARM incl. the message-passing layers.**
Proof is stronger than "flags vanish on another graph": rebuilt at V=137,
NSTEP=53, F=201, T=99991 and ALL 195 LEAF SHAPES ARE IDENTICAL. The two decode4
flags are confirmed coincidences -- (71,8) is OP_TYPE_VOCAB_SIZE (a hardcoded
jaxpr-primitive table) and (256,96) is 3*embd_dim, proven because the flag
DISAPPEARS in the extents arms where DIN becomes 108.
=> Owners "prefer the DAG-agnostic one" criterion does not discriminate: ALL
   candidates qualify, so the pick is decided purely on accuracy.

Still to come from Stage B: within-step R2 per arm (train beside test),
steps-to-threshold, main effects, and the MP-on-top-of-extents interaction.

## NODE MAP (do not collide)
  gpu19 = 61233 decode4 (stage A, finishing//hung in section (d))
  gpu17 = 61243 stage B factorial
  gpu16 = tlm3_ agent (3-layer target + cost measurement)

## 2026-08-15 STAGE C COSTING — MEASURED, plus a BLOCKER
Target sizes: TransformerLM (2 blocks) = 96 eqns / 95 elim.
             TransformerLM3 (3 blocks) = **136 eqns / 135 elim** (+42%).

**Measured PPO wall on the 2-BLOCK baseline** (tlm3_out/ppo_tlm2.log):
  sec/ep = 278.5 (ep0, includes compile), 87.0 (ep1), 96.6 (ep2)
  => steady state ~90-97 s/episode; peak GPU 8870 MiB
  => **1000 episodes on the 2-BLOCK target alone is ~25.5 HOURS.**
  The 3-block graph is +42% in vertices AND the endgame cache key is O(V^4),
  so that component alone scales (136/96)^4 = 4.0x. Stage C on 3 blocks is
  plausibly 1.5-3 DAYS, and Stage D (GAZ) is historically slower per episode.

**BLOCKER — ALL HEALTH METRICS ARE NaN on the 2-block baseline:**
  [health ep0..2] ppo=nan value=nan ent=nan ratio/max_log=nan kl/approx=nan
                  mu_quality=nan
  Also `ent:nan` in the progress bar. Measurements themselves ARE real
  (best 8.38e+04 ns, means 1.02e+05 ns), so the env/measure path works.
  A prior agent claimed this is PRE-EXISTING and reproduced it on the S1 build,
  attributing it to zero-variance rewards in a tiny smoke. NOT VERIFIED.
  **DO NOT launch a 1000-episode (>=25 h) campaign until this is explained.**
  If the loss really is NaN in a full run, days of compute produce nothing.

## STAGE B SEED 0 — MP IS THE LEVER (my prediction REFUTED, again)
I predicted "MP cannot rescue what a lossy readout never had". WRONG.
within-step R2, seed 0, pool=mean:
  arm                        ln_factor n_diag n_paired n_comp ln_stored
  BAR (*_sizes)                0.600   0.477   0.498   0.557   0.671
  noMP noEXT @3000            -0.153  -0.107   0.064   0.143   0.119
  noMP +EXT  @3000             0.443   0.398   0.564   0.582   0.721
  +MP  noEXT @1000             0.146   0.187   0.287   0.314   0.266
  +MP  +EXT  @1000             0.619   0.582   0.605   0.665   0.783  <== CLEARS BAR
Notes: MP+EXT clears ALL FIVE at step 1000 (1/3 through training) while
EXT-alone is still short on ln_factor/n_diag at 3000. TRAIN-TEST GAP COLLAPSES
(0.61/-0.15 -> 0.86/0.44 -> 0.73/0.62) = information reaching the representation,
not capacity. noEXT arm REPLICATES decode4 (-0.153/-0.107/0.064/0.143/0.119 vs
-0.131/-0.151/0.078/0.120/0.116) so the harness is trustworthy. Controls
0.969-0.980. MP costs ~1.3x (57 vs 43 min/run); 5-seed factorial ~8h total.
=> EMERGING RECOMMENDATION for stages C/D: **pool + MESSAGE PASSING + EXTENTS**.
=> #158 (EdgeCondLayer MP) VINDICATED. #94 (extents) CONFIRMED NECESSARY.
=> HOLD until 3-seed checkpoint before committing.

## ALSO: transfer eval (decode4 summary) — TLM may be METRIC-LIMITED
Same trained params on a DIFFERENT graph score FAR better than in-distribution:
  transfer sumtok: ln_factor .529 n_diag .683 n_paired .633 n_comp .677 ln_stored .670
  transfer attn:   ln_factor .289 n_diag .365 n_paired .383 n_comp .489 ln_stored .492
vs TLM in-distribution ~ -0.13..0.12 for the same arms.
CAVEAT: NOT apples-to-apples -- within-step R2 depends on how much targets VARY
within a step, and the CONTROL moves the wrong way (stat_ln_i .86-.97 on TLM vs
.41-.78 on transfer). Likely reading: TLM faces within a step are nearly
identical, so there is little within-step variance for ANY representation to
explain, and only the direct causal input (extents) captures the remainder.
=> The final report must say the TLM face result may be METRIC-LIMITED rather
   than representation-limited. Also the POOL RANKING INVERTS on transfer
   (sumtok best there, near-worst on TLM) -- params transfer, the winner does not.

## STAGE B SEED 0 COMPLETE — full 8-arm factorial
Mean over the five DECISION targets, TEST within-step R2 (train in parens):
  attn/mp1/ex1  0.743 (0.920)  <- best
  mean/mp1/ex1  0.686 (0.909)
  mean/mp0/ex1  0.542 (0.882)
  attn/mp0/ex1  0.526 (0.866)
  attn/mp1/ex0  0.217 (0.775)
  mean/mp1/ex0  0.209 (0.789)
  mean/mp0/ex0  0.013 (0.690)  <- decode4 failure REPRODUCED
  attn/mp0/ex0  0.005 (0.665)
MAIN EFFECTS: EXTENTS +0.513 | MP +0.192 | POOL +0.010 (noise)
INTERACTION: MP adds ON TOP of extents (+0.217 attn / +0.144 mean at ex=1,
vs +0.212 / +0.196 at ex=0) => COMPLEMENTARY, not substitutes.
Controls 0.964-0.980 across all 8 arms. n_compressed/is_lowrank DEGENERATE.
CAVEAT (agents own, keep in report): ln_maxdim IS algebraically max(f_ext), so
its ~0.96 in extents arms is TAUTOLOGICAL and excluded. The five decision
targets are NOT closed forms of extents (ln_stored depends on actual sparse
storage, not logical dims), so the headline stands.

=> RECOMMENDATION FORMING: **MP + EXTENTS**, pool = MEAN.
   Pool effect is noise, and mean has ZERO params vs attns query+projection --
   cheaper, fewer weights to transfer, and MP already costs 1.3x. Take attn ONLY
   if the 3-seed checkpoint shows it genuinely ahead.
   Round 1 running; 3-seed checkpoint pending.

## STAGE B: capacity control + DAG-agnosticism on the LEADING arm
PARAM COUNTS PROVE THE GAINS ARE INFORMATION, NOT CAPACITY:
  attn/mp1/ex1  378,595 params -> 0.743
  attn/mp1/ex0  SAME +94k MP params -> only 0.217
  attn/mp0/ex1  just +12 input dims (~3k params) -> 0.526
=> +94k params buys 0.217; +3k buys 0.526. More capacity alone moves the metric
   the WRONG way. This is the control that makes the extents/MP result solid.
DAG-AGNOSTIC: 202 leaf shapes ALL IDENTICAL at V=96 and V=137. The three MP
flags are the same 3*embd_dim coincidence, resolved by the alt-shape test.
=> The recommended config transfers with NO per-DAG finetuning (owners criterion),
   proven positively rather than by absence of counter-evidence.

## NaN BLOCKER RESOLVED — it was a DEGENERATE 3-EPISODE SMOKE
Verified by RUNNING LONGER (tlm3_out/nan_d6.log, 12 episodes), not asserted:
  [health ep5]  ppo=-0.08913 value=0.3551 ent=4.007 ratio/max_log=0.117
                kl=0.000288 mu_quality=0.1154 sec/ep=101.3
  [health ep11] ppo=-0.09372 value=0.2603 ent=4.131 ratio/max_log=0.0553
                kl=8.68e-05 mu_quality=0.1181 sec/ep=117.9
All finite and sensible. The 3-episode smoke had no spread for some statistic
to normalise against. => STAGES C/D ARE UNBLOCKED.

OTHER SIGNALS FROM THE SAME LOG:
 - truncated: 0, failures: 0  => the token-budget fix works at this scale.
 - face_key_seg_mismatch 8-11 AND face_dropped 8-11 per episode out of ~250-280
   chunks (~4% of faces SILENTLY DROPPED). Not a blocker but it IS silent
   attrition in the face path -- PUT IT IN THE FINAL REPORT.
 - sec/ep 101-123 => **1000 episodes ~= 28-34 HOURS per arm.**
   Two arms (PPO + GAZ) sequentially = 2.5-3 days; GAZ historically slower still.

## 2-SEED STAGE B (structure stable)
EXTENTS +0.485 | MP +0.217 | POOL +0.007 (noise). MP-on-extents +0.209/+0.175.
Top two arms overlap on 3/5 targets; pool tiebreak needs seed 3, but the
RECOMMENDATION IS UNCHANGED EITHER WAY: extents + MP, pool = mean (zero params).

## COST CORRECTED — my earlier 87-97 s/ep was ROLLOUT-ONLY (no gradient update)
A 3-episode smoke measures ONLY PopArt warm-up rollouts, which run NO update.
Trained steady state (Blackwell RTX PRO 6000, 1 GPU/arm, ep4+):
  TLM2 (V=95)  112.6 s/ep -> 1000 eps = **31.3 h**   peak 8890 MiB
  TLM3 (V=135) 434.4 s/ep -> 1000 eps = **120.6 h = 5.0 DAYS**  peak 17086 MiB
  ratio 3.86x vs O(V^4) prediction (136/96)^4 = 4.02x => V^4 MODEL CONFIRMED.
VERDICT: Stage C as specified NOT AFFORDABLE (5 d PPO alone, ~10 d with GAZ).

TransformerLM3 committed `09e7a3c`. FIVE dispatch sites, not four -- the fifth is
OUTSIDE examples.py at **ppo.py:4305** (dataset gate, == -> .startswith).
2-block target verified BIT-IDENTICAL. Gate PASS bit-identical (job 61238).
Counts: TLM2 96 eqn/95 elim (matches reference), TLM3 136/135.

## NaN ROOT-CAUSED — a REPORTING artifact, not a training bug
ppo.py ~8267 passes HARDCODED NaN placeholders during PopArt warm-up (no
gradient step). host_logs `warmup=True` branch deliberately DROPS them before
wandb, but the stdout `[health ep%d]` print sits OUTSIDE that branch and prints
the placeholders verbatim. ALPHAGRAD_HEALTH_EPISODES defaults 3 and
--popart-init-episodes defaults 3, so the health window is EXACTLY consumed by
warm-up and never shows a trained episode.
Verified 5 arms: entropy-weight 0 (not the cause), advantage-norm none (PopArt
not the cause), PRE-REFACTOR ca43369 via git archive (pre-existing, VERIFIED not
asserted), JAX_DEBUG_NANS (NaN LITERALS, not computed).
Training is healthy: ppo -0.269 -> -0.094, ent 4.0-4.2, kl 3e-3 -> 9e-5.
ONE-LINE FIX available (honour warmup=True in the health print, or default
HEALTH_EPISODES above popart-init-episodes) -- not yet applied.

## THE LEAD THAT MAY FLIP THE VERDICT (job 61248 RUNNING)
`cb.face_enum` = **86% of TLM2 host time** (76.6/89.1 s) and **80% of TLM3s**
(205.9/257.9 s). `face_enum(ext/cold)=0/0` proves ALPHAGRAD_FACE_ENUM_CACHE
(DEFAULT-OFF) is NOT ENGAGED -- this is the RECORDED 78% host fix, never flipped.
61248 = paired OFF/ON A/B, same GPU, same job. If it delivers ~78%, TLM3 falls to
~1.5-2 days and Stage C runs AS SPECIFIED.
FALLBACKS if not: (b) 250 eps on TLM3 ~30 h; (c) 1000 eps on TLM2 (31 h, known-good)
with 3-block as a short confirmation.

## TWO SIDE FINDINGS
- decode4 Part-1s END-TO-END nn256 A/B NEVER RAN: all four ab_*.log died at
  argparse (`--face-actions now requires --unified-face-head`). ONLY the
  extend_bench MICRObench produced data. The end-to-end chunk-0/256 numbers do
  not exist -- do not cite them.
- az_gumbel CLI DIFFERS from ppo: `--wandb` is store_true and the name flag is
  `--wandb-name`, so `--wandb disabled --name X` is a HARD argparse error.

## CACHE A/B HALF-DONE (61248) — OFF arm only
TLM3 cache OFF, trained eps: 434.4 444.1 474.4 483.6 501.6 s/ep
  => mean ~470 s AND **RISING ~15% over 5 episodes**. 1000 eps = 130+ h and
     getting worse. Drift is consistent with the O(V^4) endgame cache key
     growing with the plan prefix. ON arm has NOT run yet -- A/B incomplete.
TWO CAVEATS FROM THE SAME LOG (put in report):
 - `[popart-init] quality: mu=0 sigma=0 norm_var=0.0000 <-- CONSTANT, left COLD`
   and mu_quality DECAYS 0.083 -> 0.021. The quality channel is INERT here.
 - `peak_memory is being SUBSTITUTED with the deterministic memory_analysis()
   estimate because ResourceMonitor reported a zero device peak (structural on
   CPU backends)`. So memory readings are the STATIC estimate, not the runtime
   peak -- DIFFERENT QUANTITIES. Verify the measure path before any campaign
   memory claim.
VERDICT UNCHANGED PENDING THE ON ARM: 3-block @1000 eps NOT AFFORDABLE.

## CACHE A/B COMPLETE — 1.61x, NOT the recorded 78%
TLM3 trained steady state:
  cache OFF: 434.4 444.1 474.4 483.6 501.6  => mean ~470 s/ep
  cache ON : 276.4 270.3 273.6 310.9 328.1  => mean ~292 s/ep
  => **1.61x (38% off)**. BOTH arms still DRIFT UP (ON 270->328 +21%,
     OFF 434->501 +15%), so the rising component is NOT face enumeration.
  => TLM3 @1000 eps = ~81 h = **3.4 days PER ARM**, ~7 days for PPO+GAZ.
     STILL NOT AFFORDABLE.

## *** SCOPE DECISION (mine, flag prominently in the final report) ***
RUN **1000 EPISODES ON THE 2-BLOCK TARGET, CACHE ON**, both arms.
  est. 112.6 s/ep x (292/470) = ~70 s/ep => ~19-20 h per arm, ~40 h for both.
REASONING: the 1000-episode COUNT is the variable the owner deliberately raised
(r5 ran 250), and it is what determines whether we see convergence. It also keeps
the run DIRECTLY COMPARABLE to the r5 baseline on the same target. Cutting to 250
episodes on 3-block would break comparability with every prior run AND risk
stopping before convergence -- defeating the point of asking for 1000.
TransformerLM3 is built, committed (09e7a3c), gate-verified and costed, so it
remains available as a SHORTER CONFIRMATION run if budget allows afterwards.
ALSO: set ALPHAGRAD_FACE_ENUM_CACHE=1 for the campaign (1.61x, recorded as
numerically invisible) and ALPHAGRAD_EXTEND_CHUNK=256 (the 29x trap).

## 2026-08-15 COMPRESS `is_last` GATE REMOVED (commit 5ec1562) — READ IF YOU TOUCH env.py
The gate that emitted COMPRESS only for the newest vertex of a prefix is GONE.
Evidence (all measured, jobs 61257/61258/61260/61265/61267):
 - graphax raises NOTHING for a mid-plan COMPRESS: 0/46 exceptions across all 23
   NeuralNetwork positions and 11 TransformerLM positions, raw AND hook-wrapped.
   graphax's nominal-shape asserts are exact-AD-only; apply_compress drops the
   axis POINTER, not the logical size. The premise in the old comment is STALE
   (the guard it referred to was deleted from apply_compress on 2026-07-15).
 - The gate was DISCARDING the action: terminal measurement of a mid-plan
   COMPRESS = cos 1.000000 vs exact AD (i.e. no approximation at all) while the
   honest application scores 0.653 (NN) / 0.9957 (TLM).
 - Frequency on the campaign target: 4-episode TLM2 PPO run, 316 DISTINCT
   COMPRESS decisions, **316/316** honored at the deciding step and dropped at
   every later step; 2489 of 3182 COMPRESS decodes silently dropped.
CONSEQUENCES FOR YOU:
 - **tests/golden/policy_gate_golden.json was RE-RECORDED** (deliberate semantic
   change). Step 0 was bit-identical; from step 1 the head reads the COMPRESS in
   the prefix it used to be blind to (chunk_eqns_sha unchanged, chunk_tokens_sha
   changed). If your gate suddenly fails, re-pull, do not re-record again.
 - **ALPHAGRAD_TOKENS_MID_COMPRESS is deleted and setting it is a HARD ERROR.**
 - Both prefix caches lost their COMPRESS carve-outs: `_INCR_STREAM_STATS
   [nostore]` is now always 0 and `_FACE_ENUM_STATS[compress]` never
   increments (keys kept so ppo.py's cache line still formats). Expect the
   stream/face-enum caches to ENGAGE on approximation episodes for the first
   time — the v40 ext=1/431 degeneracy is gone. Re-measure host time; the 1.61x
   face-enum A/B was taken with the cache disengaged by COMPRESS.
 - 5 face_prefix_cache tests that CRASHED at HEAD inside the carve-out itself
   (`np.frombuffer(face_key[-1][0])` TypeError on the dense face_key form) now
   pass. `_FACE_LIVE_STATE` access in that file was made getattr-defensive so
   the file is green on a tree that does not define it yet.
 - env.py + ppo.py carried the face-enum telemetry counters (`calls/elims/build/
   compress`, `cb.face_enum_key`) that were uncommitted in the shared worktree;
   they rode along in 5ec1562. Nothing of them was changed or lost.
STILL FAILING, PRE-EXISTING, REPRODUCED ON A CLEAN HEAD SNAPSHOT (git archive):
 - delta_obs_emission_test[True]: `assert clipped` is vacuous since
   MAX_DELTA_TOKENS went 1024 -> 32768 in aa0774c.
 - live_face_prefix_test::test_prefix_replay_matches_the_measurements_graph:
   `lfs.chunk(...)` returns more than 4 values now (live-state rework).
 - per_face_apply_test::test_delta_budget...: only when run in the SAME process
   as policy_regression_gate_test, which leaves ALPHAGRAD_MAX_DELTA_TOKENS=1024
   in the environment its subprocess inherits. Also fails at HEAD.

## 2026-08-15 HOST-PERF: face enumeration was 70-75% of an episode; FIXED (fdb2bc0)
Profiled per-phase (ALPHAGRAD_PROFILE=1, --num-envs 4, steady episode):
  TLM  (96 eqn/95 elim, MAX_FACES=2496): 89.4 s/ep, envstep 88.0,
       cb.face_enum 31.2 + cb.face_tolist 29.3 = 69% of it, 18240 elims.
  TLM3 (136/135, MAX_FACES=5104):       317.5 s/ep, envstep 314.0,
       cb.face_enum 130.3 + cb.face_tolist 104.4 = 75% of it, 36720 elims.
The 2->3 block 3.55x is NOT the O(V^4) endgame key (faces.live_chunk is 4%
and grows 3.2x). It is O(T^2) prefix REPLAY x a MAX_FACES that itself doubles
(2496 -> 5104, the derived max_v |anc|x|desc| bound). cb.face_enum grows 4.18x
against 1.42x more decisions.
FIX: the env now owns one live IncrementalJaxpr per env, advanced by ONE
elimination per step (elims=540, restart=0), and the wires stay NUMPY.
  TLM   91.4 -> 18.1 s/ep (5.0x)     TLM3  317.5 -> 54.5 s/ep (5.8x)
Gate PASS bit-identical; enumerations identical to the cold rebuild.
=> STAGE C/D COSTING CHANGES: 1000 episodes on TLM3 is now ~15 h, not ~88 h.
CHUNK SWEEP (window 32768, Blackwell): peak mem is FLAT 2051 MB at every chunk
(it is the window buffer, not the blocking), so chunk is purely a time knob.
chunk=0 is 453 ms fwd / 2855 ms bwd FLAT. Optimum 256-512 for BOTH forward and
the DIFFERENTIATED loss form -- the code comment predicting a much larger loss
chunk is WRONG (bwd: 32->1028, 128->211, 256->181, 512->179, 1024->208 ms at
the measured delta distribution). KEEP ALPHAGRAD_EXTEND_CHUNK=256; optionally
ALPHAGRAD_LOSS_EXTEND_CHUNK=512.
MEASURED delta distribution (both targets, 380/540 samples): median 0, mean
75-114, p95 537-642, p99 1001-1411, max 1173-2833 -- i.e. 0.23-0.35% of the
32768 budget. The 25737 that sized MAX_DELTA_TOKENS does not occur here.

## SHAPES RERUN (decode6, 27/27 rc=0) — SHAPES DO NOT REPLACE EXTENTS
Within-step R2 TEST, 3-seed median, mean over the 5 decision targets, pool=mean:
  no-shapes no-extents (control)   -0.009  (train 0.702)
  SHAPES, NO EXTENTS  <-- the ask   0.104  (train 0.582)
  no-shapes +EXTENTS  <-- the bar   0.528  (train 0.875)
  shapes + extents                  0.603  (train 0.855)
  shapes no-extents +MP             0.291
  shapes + extents + MP             0.738  <-- best overall
Shapes-no-extents CLEARS 0 OF 5 targets (shortfalls 0.34-0.61) and NEVER reaches
0.6 on any decision target in any seed, with or without MP.
TRAIN BESIDE TEST RULES OUT UNDER-TRAINING: 0.582/0.104 is the same OVERFITTING
signature as the control 0.702/-0.009. Shapes LOWER the train fit while barely
moving test = longer noisier stream, not new usable information.
All four pools rerun on shapes data: mean .104 attn .069 last .020 sumtok .019 --
same ordering, all fail. POOL STILL DOES NOT MATTER.
HYGIENE: datasets verified BYTE-IDENTICAL vs todays code; TRUNCATION ZERO (not a
clipped-budget artifact); controls 0.968-0.982; no-shapes arms reproduce decode5
exactly (-0.009 vs -0.010, 0.528 vs 0.529).
COST: 2.1x wall; stream 105,594 vs 43,678 tokens; worst delta 64,535 and 9/192
trajectories EXCEED the current 32768 budget (which now RAISES), so production
shapes would need 65536 + 2x compute to buy ~1/5 of what extents buy.
=> **#94 EXTENTS CONFIRMED NECESSARY. The owners preferred alternative (state
   shapes per equation instead) is REFUTED.** Best config stays pool + MP + EXTENTS.

## 2026-08-15 LEAN PALIMPSA-CENTRIC POLICY REWRITE (owner-requested, minimal)
Governing rule: EVERY learned parameter sits downstream of palimpsa, and EVERY
path from palimpsa to the loss is differentiated. Nothing the encoder produces
is stored, detached, or precomputed outside the gradient.

**THE DEFECT, MEASURED BEFORE THE CHANGE** (`lean_probe.py`, gate graph, seed 13,
job 61309):
  gradient reaching palimpsa's BASE ENCODE, production topology = **0.000000e+00**
  gradient AVAILABLE through the same encode if moved inside = 1.182467e+03
`base_identity_stream` ran once per episode in `ppo.main` (ppo.py:8205) and its
rows were a CONSTANT inside `filter_grad`: `VertexIdentityPool` trained, the
palimpsa pass that WROTE its input did not. The only encoder gradient in the
build was `--grad-window` K=1's single step delta.

**AFTER** (job 61311): PROD == INSIDE = 8.896708e-01 = **2.85x the K=1 delta
path**. Pinned by `tests/palimpsa_base_grad_test.py` (6/6 pass): the LEGACY
topology is reconstructed IN THE TEST and asserted EXACTLY 0.0, the shipped one
asserted > 0 and > 0 per palimpsa LAYER (a last-layer-only path would pass a
norm test), plus base+dynamic additivity and scatter purity.

**WHAT CHANGED**
 - `carry_stream.init_carry` now returns the BASE MEMORY (owned base rows
   scattered into their OWN vertex's slot). The dynamic accumulation starts
   from `zero_memory` and the two are ADDED in `heads(base_mem=...)`. Legal
   because (sum, count) memories add: pooling the union of the rows is the sum
   of the sums over the sum of the counts. `base_memory` is called INSIDE
   `_dynamic_loss_fn`, once per loss call, outside the per-sample vmap.
 - DELETED: `base_identity_stream`, `VertexIdentityPool`, `Agent.identity`,
   `ctx_proj`, the `[identity || dynamic]` concat (slots 2E -> E),
   `_face_pool`, `_endpoint_ctx`, and `ctx_i`/`ctx_j` from the face head
   (in_dim 3E -> E).
 - ONE parameter-free scatter (`vertex_memory.scatter` = segment_sum + counts),
   keyed by VERTEX for the slots and by FACE for the face latent. `_face_replay`
   is now one searchsorted + one scatter for all F faces instead of a cumsum
   span per face.
 - extents still NOT in the face-head input (owner: "and extents for now").
 - `--grad-window` KEPT and its DEFAULT UNCHANGED at K=1.
 - `az_gumbel` + `common/sampled_az` ported; AZ's loss also recomputes the base
   memory inside its own gradient.

**PARAMS** 159,733 -> 108,885 (**-50,848, -31.8%**); 109 -> 100 leaves.
  vertex_policy 58,368 -> 14,848 | face head 6,206 -> 4,158 |
  identity_pool 3,200 -> 0 | ctx_proj 2,080 -> 0 | encoder 60,124 UNCHANGED.

**GATE RE-DERIVED, NOT FORCED.** Old golden kept alongside at
`tests/golden/policy_gate_golden_pre_lean.json`. Against it the lean build
FAILS with a real semantic diff (different vertex picked at step 0, 1 vs 2 live
faces, different face logp/entropy) -- that is the honest outcome of a
deliberate semantics change. The re-recorded golden PASSES bit-identical,
5 steps / **8** live face decisions (was 7), so the tripwire got sharper.
The gate harness's `ALPHAGRAD_MAX_DELTA_TOKENS` went 1024 -> 4096: the lean
policy samples a different ORDER whose delta is 1053 tokens and the budget
RAISED (correctly -- a dropped delta desyncs the recurrence). Padding-only; the
golden stores live prefixes, so nothing recorded depends on it.

**DAG-AGNOSTICISM: PASS both before and after**, by the positive test (rebuild
at another size and compare every leaf shape), not by absence of a flag:
  before 109/109 leaf shapes identical at V=96 and V=137
  after  100/100 leaf shapes identical at V=96 and V=137
Shape-coincidence flags drop 2 -> 1: only `op_embedding (71,8)` remains, and 71
is OP_TYPE_VOCAB_SIZE (a hardcoded jaxpr-primitive table, bounded and
DAG-independent by construction). The old `3*embd_dim = 96 == V` false positive
is GONE, because the face head is E wide now.

**EXPECTED OUTCOME, STATED HONESTLY.** The VERTEX side is what this helps:
participation + scatter reached DV_FILL within-step R2 0.93 in 7-10 gradient
steps in the supervised probe, and the representation it reads is now trained
end to end instead of on a one-step horizon. The FACE side is expected to stay
~0 without extents -- every tokens-only face target measured ~0 (best 0.104
with shapes explicitly tokenised, against a 0.528 bar) and that was WITH full
end-to-end gradient, so fixing the gradient does not rescue it. This is the
clean minimal baseline that things get added back to: Stage B measured
EXTENTS +0.513 and MP +0.192, complementary, on top of exactly this.

Backups of every patched file: `lean_backup/`. Scratch: `lean_probe.py`,
`lean_audit.py`, `lean_{before,after,gate,pytest,smoke}.sbatch`, `lean_out/`.

### LEAN REWRITE: measured results (jobs 61309-61319, all on pgi15-gpu18)

**SMOKE — IT TRAINS** (Helmholtz, 10 episodes, job 61316, rc=0, peak 2831 MiB):
  ep3 ppo -0.5093 value 0.5913 ent 2.185 kl 8.9e-3
  ep5 ppo -0.3865 value 0.2363 ent 1.959 kl 3.4e-4
  ep7 ppo -0.5460 value 0.1040 ent 1.781 kl 3.9e-4
  ep9 ppo -0.6549 value 0.0555 ent 1.899 kl 1.1e-4    3.1 s/ep steady
The critic descends monotonically and KL converges two orders of magnitude.
ep0-2 `nan` is the KNOWN pre-existing PopArt warm-up REPORTING placeholder
(health print sits outside the `warmup=True` branch), not a training bug.

**--grad-window K, TransformerLM V=95, 6 episodes per arm, steady sec/ep at ep5:**
  K     peak MiB    sec/ep    vs K=1
   1      8971       30.4      1.00x  (DEFAULT, UNCHANGED)
   2      9037       34.3      1.13x
   4     13197       42.9      1.41x
   8     13401       60.0      1.97x
  16     21785       88.7      2.92x
ALL FIVE ARMS rc=0 -- **no affordability ceiling was reached** on a Blackwell
RTX PRO 6000. Every arm trains (K=16 ep5: ppo -0.0399 value 0.2425 ent 4.395
kl 1.1e-3). Memory is 2.4x and wall 2.9x from K=1 to K=16, sub-linear in K
because the rollout and the measurement dominate the episode, not the loss.
The pre-change build measured 8890 MiB on the same target, so putting the base
encode inside the gradient costs about 1% of peak (8971 vs 8890) -- TLM's base
stream is only 870 tokens.
=> K IS THE OWNER'S CHOICE. K=4 is the obvious value-for-money point (+41%
   wall, +47% memory, 4x the palimpsa horizon); K=16 is affordable but at 2.9x
   wall it would put a 1000-episode arm out of reach.

**TEST SUITE, BEFORE vs AFTER** (17 files, one process each, jobs 61310/61317):
  BEFORE: 2 pre-existing failures --
    carry_stream_base_attribution::test_advance_still_credits_the_owning_vertex
    delta_buffer_equivalence::test_face_chunks_and_counts_unchanged
  AFTER: the SAME two, and no others. Everything else passes, plus the new
    tests/palimpsa_base_grad_test.py 6/6.
  ONE test failed at first and it was a TRUE POSITIVE, not a regression:
    test_sampled_az_gradflow::test_gradient_reaches_..._vertex_head_...
    Its fixture built the carry with `base_owners=None`, so every base row
    went to the GLOBAL slot, no VERTEX slot was occupied, and the pointer
    masked all V logits to -inf -- the vertex CE had no gradient to give and
    the assertion was being satisfied by the FACE term, which used to read the
    pointer's contexts as ctx_i/ctx_j. Removing ctx removed the cover. Fixed
    by owning the base rows (a0be392); 3/3 pass, gate still bit-identical.

**STILL OPEN / UNCHANGED BY THIS WORK**: `face_key_seg_mismatch` and
`face_dropped` 8-11 per episode out of ~290 chunks (~3% of faces silently
dropped) is still present on TLM, pre-existing, and still belongs in the final
report.

## 2026-08-16 PPO BACKWARD MEMORY: two fixes landed (5407eec, c35a07d)
Jobs 61320 (gpu18, all arms + gates) / 61322 (gpu17, TLM window-only arm).
Peak MiB (nvidia-smi 2 s poll) / sec/ep (mean of the last 3 health lines;
ep0-2 are the PopArt warm-up rollouts inside episode 0 and print NaN):

  TransformerLM (wikitext2, seq 64)     K=1          K=4          K=16
    BEFORE  (no remat, window 32768)   8967 / 30.2  13189 / 44.0  21751 / 93.3
    window 4096 only                   4873 / 28.3   9095 / 38.2   9461 / 73.8
    AFTER   (remat + window 4096)      4905 / 27.9   5007 / 37.9   7447 / 74.7
  VmappedNeuralNetwork/mnist (NN256)
    BEFORE                             3371 /  7.2   5511 / 12.8  12033 / 36.5
    window 4096 only                   2347 /  6.4   2437 / 10.9   3325 / 28.4
    AFTER                              2349 /  6.5   2451 / 11.2   3367 / 29.6

The BEFORE curve reproduces the earlier K table to <0.2% (8971/13197/21785),
so the harness is trustworthy. K=16 AFTER (7447) is cheaper than K=1 BEFORE.
On NN256 the window WAS the entire K-linear term and the remat is a no-op
(+2-4% wall); on the TLM it is worth a further -45% at K=4. Both kept.

MEMORY IS NO LONGER THE BINDING CONSTRAINT ON K. Extrapolating the AFTER
slope (~203 MiB per unit K past K=4) even K=95 (the whole 2-block episode)
lands ~23 GiB on a 96 GB Blackwell. What now limits K is TRACE/COMPILE of the
UNROLLED PYTHON loop: at K=16 a 6-episode run is ~57 min wall of which ~45 is
compile. If K>16 is ever wanted, convert the loop to a lax.scan first.

CLOSED: the palimpsa Pallas kernel's own chunk_size is NOT a lever. A counter
on palimpsa()/palimpsa_attention() over a full production NN256 episode
records **0 traces** -- encode_extend re-implements the recurrence with
associative_scan/scan and every self.encode call site is behind an
`if precomputed is None` the incremental path never takes, so
PalimpsaMixer.chunk_size is dead code in training. Even hypothetically it is
a bad dial (standalone sweep, H=4 d=32): at T=32768 backward goes 108 -> 613
ms from chunk 8 -> 128 for only 320 -> 153 MiB, and 8 vs 16 saves nothing.
16 is already at the knee. Do not touch it. (bwd_kernel_probe.py,
bwd_kernel_chunksweep.py, bwd_out/.)
