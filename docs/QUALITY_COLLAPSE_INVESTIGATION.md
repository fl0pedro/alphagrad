# Why quality collapses in the PPO TLM campaigns — investigation dossier

Date: 2026-08-18. Author: investigation agent (read-mostly), commissioned after the
LR-family falsification (v57/v59/v60: three LR schedules, same collapse).
Repo: `/Users/assmuth/dsnn/alphagrad` @ `d5b666a` (branch `hostperf-caches`).
All numbers below are reproducible from the artifacts in
`/Users/assmuth/dsnn/collapse_invest/` (wandb exports + analysis scripts, listed in §9).

## 0. The phenomenon, precisely

Four TLM PPO runs, all `--reward-mode mult`, rev-pinned order (`ALPHAGRAD_FORCE_REV_ORDER=1`),
identity-biased init (`ALPHAGRAD_FACE_NONE_BIAS=6`), quality = `loss_drop`
(200-step Adam walk). Identity-plan quality on this target is 0.8853.

| run | job | envs | LR schedule | wandb run | outcome |
|---|---|---|---|---|---|
| v57 | 61485 | 1  | constant 3e-4 | `dll-streetview/dsnn-vertex/it05ku34` | **held 0.885 through ep164** (cancelled); entropy 0.25→1.38, no learning |
| v58b| 61498 | 16 | constant 3e-4 | `ud3cla83` (full metrics) | collapsed; median-q < 0.05 in 50% of eps by ep30-40, 100% by ep60 |
| v59 | 61507 | 16 | piecewise warmup 20% | `3faj3e36` | collapsing at cancel (ep49: q 0.70, entropy 0.17→1.13) at ≤ half peak LR |
| v60 | 61515 | 16 | multiplicative warmup | `ygm8n2jy` | collapsed by ep63 (q 0.28), then **committed**: entropy fell 1.18→0.05, q pinned 0.0 for 440 eps |

Configs verified identical across the four runs except the LR schedule
(wandb `run.config`, exported by `collapse_invest/an3_configs.py`):
`gate_tau=0.5, gate_w=40, anti_degen_penalty=2, anti_degen_tau=0.05, entropy_weight=0.05,
lr=3e-4, gae_lambda=0.95, ppo_epochs=2, discount=0.99 (ppo.py:3274 default),
advantage_norm=popart, popart_init_episodes=3, terminal_rewards_only=true`.
Launch configs: `/Users/assmuth/dsnn/fq_v58_tlm_env16.sbatch` (+ `fq_v57_tlm_fullT.sbatch`,
`fq_v59_tlm_warmup.sbatch`, `fq_v60_tlm_multwarmup.sbatch`; v59/v60 add `--lean-logging`,
so v58b is the only fully-instrumented collapsed run).

Correction to the brief: the collapsed state is **not** "q settles 0.05-0.19".
Per-episode medians settle at **exactly 0.0** (`measure/quality/median_ep` in `ud3cla83`,
rows ep40+); `mean_quality` oscillates −0.19..+0.36 because it mixes exact-0 plans,
occasional −1.0 (diverged walks) and occasional surviving 0.79-0.88 plans.
q = 0.0 exactly is the signature of a **zero/dead gradient** (walk does not move,
L1 == L0), i.e. plans that destroy the gradient rather than merely damage it.

## 1. The reward code (ground truth for every hypothesis)

`_apply_mult_gate`, `src/alphagrad/approx/ppo.py:499` (called at ppo.py:6732 in the
loss path and ppo.py:8888 in the PopArt warm start; cost weights with the quality
slot zeroed built at ppo.py:5173-5176):

```
ppo.py:538   fid = jnp.where(terminal, jnp.clip(fid_raw, 0.0, 1.0), 0.0)  # (E, T)
ppo.py:540   g = jnp.clip((fid - gate_tau) / denom, 0.0, 1.0)
ppo.py:545   cheapness = jnp.maximum(0.0, gate_w - weighted_cost)
             gated = g * cheapness
ppo.py:552   degen = (fid < anti_degen_tau) & terminal
ppo.py:554   shaped = -(anti_degen_penalty - fid_basin * anti_degen_penalty)
             gated = jnp.where(degen, shaped, gated)
```

The quality producer, `env.py` `_loss_drop_quality` (src/alphagrad/approx/env.py:2331):
diverged walk → `return -1.0` (env.py:2441); otherwise `clip(drop, -1, 1)` (env.py:2449).
The additive floor `_apply_quality_gate` (env.py:1986, armed by
`ALPHAGRAD_QUALITY_GATE_MIN=0.05` in every sbatch) clamps latency/mem of q<0.05 plans
to the same-order exact reference — it removes the cost *bribe* for destruction but
adds no gradient.

**The reward surface this defines, with the run's actual constants** (identity plan:
lat≈1.6e5 ns, mem≈5.54e7 B → weighted symlog cost ≈ 29.8, cheapness ≈ 10.2):

| terminal quality q | reward |
|---|---|
| 0.885 (identity) | g=0.77 → **≈ +7.9** |
| 0.5 … 1.0 | 0 → +10.2·g(q), smooth |
| **0.05 ≤ q < 0.5** | **exactly 0, flat** (g clips to 0) |
| **0 ≤ q < 0.05** | −2·(1−q) ∈ [−2.0, −1.9]: total escape slope 0.1 over the whole band, then a +1.9 **discontinuity** at q=0.05 |
| −1 (diverged) | fid clipped to 0 at ppo.py:538 → **identical −2.0** |

Three structural facts, straight from the code:
(a) a 0.45-wide dead-flat zero plateau;
(b) an anti-degen "slope out of the basin" worth only 0.1 reward units, confined to q<0.05;
(c) DIVERGED (−1.0 from env.py:2441) and zero-work (drop=0.0) are **conflated** by the
ppo.py:538 clip — both score −2.0 exactly.

## 2. H1 — g(q) zero-plateau absorber. **CONFIRMED** (as the structural basin)

Code: §1. Data (all from `ud3cla83`, exported to `collapse_invest/v58b_history.csv`;
windows computed by `an6_basin.py`):

```
ep   0- 10: median-q +0.885 | eps(median<0.05)   0% | eps(worst<=-0.99) 30%
ep  10- 20: median-q +0.880 |                    0% |                   60%
ep  20- 30: median-q +0.545 |                   10% |                   80%
ep  30- 40: median-q +0.221 |                   50% |                   90%
ep  40- 60: median-q +0.029 |                   95% |                   95%
ep  60- 90: median-q +0.000 |                  100% |                   77%
ep 150-192: median-q +0.000 |                  100% |                   38%
```

- The majority of the 16 envs crosses into the q<0.05 basin between **ep30 and ep40**;
  by ep60 every episode's median is in it. Catastrophic plans (worst ≤ −0.99, diverged
  walks) appear in 30% of episodes **already at ep0-10** — 16-env exploration finds the
  cliff immediately despite the identity init.
- The final state is the *penalty floor of the basin* (q = 0.0 exactly → reward −2 flat),
  not the mid-plateau; the plateau [0.05, 0.5) is transited, not occupied (only 5/192
  episode medians ever sit in it). The plateau's role is that **the way back out is
  gradient-free**: from q=0 the policy would have to jump ≥0.5 in one step to see any
  positive reward; nothing between −1.9 and 0 rewards partial repair.
- The `[0,1]` clip conflation (c) is confirmed in code but is a secondary detail: it
  removes the one distinction (diverged vs zero) the basin interior *could* have expressed
  beyond the 0.1-unit ramp.
- Env-side confirmation the basin was entered for real: 47 `quality gate CLAMP` prints in
  `/Users/assmuth/dsnn/v58_tlm_61498.log` (pattern `quality gate CLAMP`, printed 1st +
  every 50th per actor, e.g. line 133 `q=-0.0959 < 0.05`).

**v60 demonstrates the basin is absorbing and internally learnable** (`ygm8n2jy`,
`collapse_invest/v60_history.csv`): entropy/approx_head rises to 1.18 by ep63 (exploring),
then falls to **0.05-0.08 for the remaining ~440 episodes** while median q stays exactly 0.
The policy did not diffuse into the basin — it **committed** to a deterministic
destructive mode. The one thing the basin interior rewards ("prefer reliable q=0 zero-work
plans, −2.0, over diverged plans, also −2.0 but adjacent to −1.9…−2.0 ramp noise; and note
z(0) > z(−2) once PopArt re-centers, §3") is exactly what a policy that cannot see
per-face destructiveness (§4) *can* learn. v60's final pareto front is all ep11-32
identity plans (`v60_tlm_61515.log`, tail: `Ep 11 | ... Quality(loss_drop): 0.8853`) —
everything after ep~30 was worthless to the archive.

## 3. H2 — PopArt distortion. **REFUTED as initiator, CONFIRMED as ratchet**

Checks against the brief's specific claims:

- **#89 pre-ART frame fix is in HEAD**: ppo.py:6878-6894 (comment block naming #89;
  degenerate-step neutral value carried old-frame → new-frame). The ART compensation is
  performed in the same update that moves μ/σ: `_popart_update` at ppo.py:6859-6862
  followed immediately by `_popart_rescale_heads` at ppo.py:6866-6868 (definition
  ppo.py:610, output-preserving `σ'·head'+μ' == σ·head+μ`). Nothing missing here.
- **"A window where cost-z dominated quality-z" is structurally impossible in mult mode**:
  head reward weights collapse to one-hot on the quality head (ppo.py:5176-5180,
  `head_reward_weights_np[HEAD_NAMES.index("quality")] = 1.0`); the latency/mem heads
  receive all-zero targets (their gated channel rows are zeroed at ppo.py:557-558) — in
  `ud3cla83`, `popart/sigma_latency` and `popart/sigma_mem` sit pinned at the 0.1 floor
  with μ≈0 for the whole run, and their advantage weight is 0. No cost-z window exists.
  This part of H2 is refuted.
- **Warm start**: `popart/mu_quality` = 4.63, `sigma_quality` = 1.96 at ep5 (ud3cla83).
  The brief's "sigma = walk noise" is wrong in an instructive way: walk noise is ~1e-3
  (§5), and 48 near-identical warmup plans (reward ≈ +7.9) with γ=0.99 over ~95 steps
  give MC returns G_t = 0.99^(T-t)·7.9 spanning 3.0…7.9 — mean ≈ 4.9, sd ≈ 1.4.
  μ=4.63/σ=1.96 is the **discounting spread across timesteps**, approximately correct
  stats, not a distortion.
- **The signs were right when it mattered**: during entry (ep5-40) a zero-reward terminal
  had z = (0−μ)/σ = **−2.4 … −1.8**, a diverged one −3.4 … −2.7 (an2_main.py output).
  Normalization was telling the policy, correctly and strongly, that the basin is bad
  while it walked in anyway. PopArt did not make harmful actions look advantageous
  in any window we can find. That part of H2 is refuted.
- **The ratchet is real and measured**: μ tracks the population. v58b:
  μ_quality 4.63 (ep5) → 2.64 (ep80) → 0.27 (ep191), so z(0): −2.4 → −0.98 → −0.12.
  v60 ran long enough to complete the inversion: μ = −0.95, σ = 0.92 by ep444, i.e.
  **z(0) = +1.03: a zero-quality plan scores a full σ above average**, because the −2
  diverged/degen mass drags μ below the zero-work plans. Once the basin is the
  population, PopArt (correctly, by its own contract) re-centers on it: escape pressure
  decays to nothing and the basin's internal ordering (zero-work ≻ diverged) becomes the
  dominant normalized signal. That is an amplifier/absorber, not an initiator.

## 4. H3 — the model cannot see what it is approximating. **CONFIRMED (representation deficit), with one nuance**

- Code: the face head input is the face-keyed scatter latent alone.
  `UnifiedFacePolicy._repr` (src/alphagrad/approx/unified_face_policy.py:213-224) returns
  `face_latent` (E=32-wide) or zeros; `face_sizes` exists as a parameter
  (unified_face_policy.py:228, :249, :293, :338) but `grep -n face_sizes
  src/alphagrad/approx/ppo.py` shows **ppo.py never passes it** (only the import at
  ppo.py:114 and the `unified_face_head` flags at ppo.py:3705/4935 touch the class).
  The class docstring itself says "``face_sizes`` is deferred"
  (unified_face_policy.py:64).
- Online probe, THIS representation, in-run (job 61458,
  `/Users/assmuth/dsnn/probe_gpu_61458.log`, pattern `[probe] arm=lean`): within-step R²
  for the 5 face targets `ln_factor 0.03-0.09, n_diag 0.03-0.09, n_comp 0.05-0.17,
  ln_stored 0.09-0.18, n_paired ≈0.00-0.15` while the controls `stat_ln_i/stat_ln_j`
  score 0.38-0.75 (e.g. log lines 34-63). The head is near-blind to the quantities that
  determine whether an approximation is destructive.
- Offline campaign (`docs/CAMPAIGN_STATE.md`, "STAGE A COMPLETE"
  and "STAGE B SEED 0" sections): all four lean pooling arms score −0.24…+0.12 (bar
  0.48-0.67); `+extents` clears the bar with main effect ≈ +0.5; MP+EXT clears all five
  targets at 1/3 of training. "#94 (extents) CONFIRMED NECESSARY" is recorded there
  verbatim (lines 201, 447).
- **Nuance the brief's strong form gets wrong**: quality outcomes were *not*
  policy-independent noise at the episode level. Pre-collapse (ep0-60, ud3cla83):
  `corr(approx_applied/quant, mean_quality) = −0.71`, `compress −0.68`, `fraction −0.59`
  (an2_main.py). A coarse, learnable "do fewer approximations" signal existed, and it is
  expressible by the policy (it is exactly what `FACE_NONE_BIAS=6` biases). The policy
  still moved the other way (`approx_prob/none` 0.98 → 0.22 by ep90). So the
  representation deficit makes *fine* credit (which face, which op) unlearnable — which
  is what makes the basin's "destroy everything reliably" mode the only stable learned
  behavior — but it does not by itself force the initial walk into the basin.

## 5. H4 — quality measurement noise / walk validity. **REFUTED (as a driver)**

- Noise floor from near-identical plans: ep0 (16 envs, identity-biased,
  ~20 stray approximations across all 16): `measure/quality` best/median/worst =
  0.885328 / 0.885324 / 0.884034 (ud3cla83 row 0) — spread ≤ 1.3e-3, best-median 5e-6.
  Against a plan-induced range of 0.885 → 0 → −1, channel SNR is ~10³. The channel is
  clean.
- #126 (PRNG split arity, trainer-vs-actor W0): the walk fingerprint is printed per
  process (env.py `loss-drop walk armed ... fingerprint(probe+W0)=`). In BOTH v58b and
  v60 every measuring process reports the **same** fingerprint `8e19f02936b07dc0`
  (v58_tlm_61498.log lines 94/99; v60_tlm_61515.log lines 93/97); no trainer-side walk
  line exists (quality is measured on the `--ray-measure 3` actors). All quality numbers
  in these campaigns come from one consistent (probe, W0) pair; episode-to-episode
  loss_drop comparability is intact. #126 remains a real latent hazard for configs where
  the trainer also measures, but it did not bite here.
- The walk itself behaves as designed: identity 0.8853 stable over hundreds of
  measurements across 4 runs; destroyed gradients score exactly 0; blow-ups −1.

## 6. H5 — entropy bonus as the only coherent gradient. **REFUTED as stated; residual role OPEN**

- Actual weight: `--entropy-weight` default **0.05** (ppo.py:3272), not 1e-3; none of the
  four sbatches overrides it.
- Magnitudes (ud3cla83, an5_final.py): `0.05·H(approx_head)` vs `|ppo loss|` per window:
  ratio 0.14 (ep1-10), 0.45-0.78 (ep20-40, the entry window), 0.30-0.38 after. The
  entropy term is material but **never dominant**, including post-plateau — the
  brief's H5 prediction ("post-plateau the entropy term dominates") is refuted.
- Two independent falsifiers of "entropy is the driver":
  (1) v60 spent 440 episodes collapsed at entropy **0.05-0.08** — the collapsed state is
  a low-entropy commitment, not entropy-driven diffusion;
  (2) GAZ has no entropy bonus at all and collapsed faster (§7).
- What remains OPEN: during ep10-40 the entropy bonus is the only term that *rewards*
  leaving the identity init, and with `--terminal-rewards-only`, γλ = 0.9405 over ~95
  steps attenuates the terminal advantage by (γλ)^(T-t) (≈0.003 at t=0, ≈0.05 at t=45)
  while the entropy gradient acts undiscounted at every step; explained variance is
  0.43 → ~0.05-0.15 (ud3cla83 `explained variance`), so the critic does not shortcut
  the attenuation. This *per-step* imbalance is consistent with the observed entry in
  PPO (and echoes the documented v55-era "PPO never left uniform" GAE math), but v57
  (1 env, same entropy weight, held 165 eps) and GAZ (no entropy, collapsed) show it is
  neither necessary nor sufficient. Consistent-with, not demonstrated.

## 7. The cross-learner control: GAZ collapses too ⇒ the mechanism is NOT policy-gradient-specific

GAZ NN256, same mult scalar by construction (az_gumbel.py:327-352: `mult_gate_scalar`
with identical τ=0.5/W=40/P=2/τ_d=0.05, "PPO _apply_mult_gate parity"), search-based
(Gumbel AZ), no entropy bonus, but driving the **same UnifiedFacePolicy** face
representation (az_gumbel.py comment "through the SAME UnifiedFacePolicy PPO trains").

From the persistent jsonl (fields `raw`=[latency, xla_peak, flops, cos], `popart_mu`,
`popart_sigma`; analysis `an4_gaz.py`):

- `run_61531/updates.jsonl` (depth-0, 122 rows): sampled quality 0.999 at ep1 →
  intermittent by ep17 → **0 from ep21 onward**; overall 90.2% of episodes q<0.05,
  0% in [0.05,0.5), 9.8% ≥0.5 (all early).
- `run_61532/updates.jsonl` (deep20, 333 rows at reading, job still RUNNING): 0.999 →
  0.047 by ep27 → **0 from ep40 onward**; 91.6% q<0.05. GAZ's popart μ(cos) decays
  0.999 → 0.010 — the same ratchet signature as §3.
- Matched measurement counts (GAZ measures 1 plan/episode, n_meas==ep; PPO 16/ep):
  PPO median-q is still 0.83 at 300 measurements and dies by ~800; GAZ sampled-q is dead
  by **~40 measurements**. GAZ collapses *faster* per measurement, not slower.

Interpretation discipline: this control kills every PPO-optimizer-specific root cause
(PopArt-window distortion, entropy bonus, ratio/GAE pathologies) as *the* mechanism.
It does **not** by itself separate "reward shape" from "shared blind representation":
GAZ's search is guided by value/policy nets on the same latents, so a blind prior +
flat basin fails the same way. The two surviving hypotheses (H1 basin, H3 blindness)
are exactly the two things PPO and GAZ share. Confounds to note honestly: GAZ arms are
NN256 (not TLM) and quality=cos (not loss_drop) — the collapse reproduces across
target AND quality-metric, which strengthens the reward-shape reading.

## 8. H6 — other findings

- **GAZ depth-0 arm is broken**: job 61531 ended at ep122 with
  `AssertionError: vertex 16: bucketed draw decided 2 faces at width 2 but the
  authoritative enumeration has 0 -- face_count_fn and face_keys_of disagree`
  (az_gumbel.py:1317 `_draw_face_sequence`; log
  `/Users/assmuth/dsnn/gaz_nn256/gaznn256_61531.log`, tail). Slurm shows COMPLETED
  because the wrapper swallowed RC=1 (`T1=... RC=1` in the log). Fix before trusting
  any depth-0 numbers.
- **Face enumeration drift feeding wrong/absent latents exists but is small**:
  `face_dropped`≈29-32/ep ≈ 1.1% of ~2750 face events (`[health epN] live-faces` lines,
  v58_tlm_61498.log lines 41/76…), `face_key_seg_mismatch` equal, `failures: 0`,
  `truncated: 0`. Too small to explain a 100%-of-envs collapse.
- Sentinel/degen machinery quiet: `collapse/count_this_ep` = 0 for all 192 episodes
  (ud3cla83); only 2 SENTINEL mentions in the whole v58 log.
- Cheapness clamp footnote: a plan with q ≥ 0.5 but a failed cost measurement
  (sentinel −1e10 → symlog 23/channel → weighted 46 > W=40) also lands at reward 0
  via ppo.py:545. Rate is negligible here (see previous bullet) but the clamp is a
  second, independent zero-conflation to remember when reshaping the reward.

## 9. Reproducibility

`/Users/assmuth/dsnn/collapse_invest/`:
`export_v58b.py` (wandb → CSV: `v57_history.csv`, `v58b_history.csv`, `v59_history.csv`,
`v60_history.csv`, `v58b_keys.json`), `an2_main.py` (trajectories, H1/H2/H3 tables,
`v58b_extract.csv`), `an3_configs.py` (run configs), `an4_gaz.py` (GAZ jsonl analysis,
`gaz_61531_quality.npy`, `gaz_61532_quality.npy`), `an5_final.py` (v57/v59/v60
trajectories, matched-n_meas PPO-vs-GAZ, entropy-vs-pg magnitudes), `an6_basin.py`
(basin occupancy windows). Run on pgi15-cpu2 via
`srun -p pgi15-cpu -w pgi15-cpu2 ... uv run --no-sync python -u <script>` from the repo.

## 10. DECISION — what the evidence supports

Causal account, at the confidence the data supports:

> The mult reward defines a surface where everything between "destroyed" (−2.0) and
> "half-preserved" (0.5) carries **zero or near-zero gradient** (total escape slope 0.1
> units, then a +1.9 discontinuity, then a 0.45-wide flat). Sixteen-env exploration
> finds catastrophic plans within the first ten episodes. Fine-grained credit ("which
> face/op was destructive") is unlearnable because the face head is blind to the
> contraction it is approximating (probe R² 0.03-0.18 vs bar 0.48-0.67), so the only
> stable learned behavior inside the basin is the reliably-reachable q=0 mode — v60
> commits to it at entropy 0.05. Adaptive normalization then re-centers on the basin
> (z(0): −2.4 → +1.0 in v60), erasing the escape signal. The same surface + the same
> blind representation collapse a search-based learner (GAZ) even faster, at both a
> different target and a different quality metric. LR schedules are irrelevant
> (v57/v59/v60), and measurement noise (≤1.3e-3), PopArt bookkeeping (#89 present,
> ART correct), and the entropy bonus (v60 commits at H≈0.05; GAZ has none) are all
> excluded as root causes.

Ranked interventions:

1. **Reshape g / remove the flat basin** (smooth-g or Lagrangian — both belong to the
   same family "no zero-gradient band, destruction still strictly dominated").
   *Supported by*: §2 (basin entered and absorbing in every 16-env run), §7 (reproduces
   across learners ⇒ reward-level), §3 (any fixed shape beats the μ-ratchet only if the
   basin has slope). *Predicted effect*: collapse stops being absorbing; policy retains
   a path back to identity; does NOT by itself produce approximation wins (that needs 3).
   *Cheapest falsifying test — zero GPU*: recompute the counterfactual reward for the
   3,024 already-measured v58b plans (quality quantiles are in the CSV; exact per-plan
   values in the run's pareto/measure records) under smooth g (e.g. g=q, or
   −P·(1−q) extended to τ) and check the advantage ordering identity > partial > zero >
   diverged is monotone and non-degenerate at every episode. Then one 100-episode
   v58-config rerun with the reshaped gate (collapse historically visible by ep40;
   ~4h on one node) — if it still collapses with a monotone surface, H1 is falsified
   as the binding constraint and H3 is promoted.
   Between the two variants: the **Lagrangian constraint** additionally keeps the
   quality price adaptive (immune to the §3 ratchet by construction) and is the better
   long-term bet; the smooth-g reshape is the cheaper first test of the same claim.
2. **Feed the face head what it approximates** (`face_sizes`/extents + message passing;
   #94/#158). *Supported by*: §4 — necessary for any *positive* result (learning which
   approximations are safe), already CONFIRMED NECESSARY by the 5-seed factorial
   (`docs/CAMPAIGN_STATE.md`). *Not supported as*: an anti-collapse fix on its own — the
   coarse "approximate less" signal (r = −0.71) was visible to the current policy and
   did not prevent collapse. Do it with (1), not instead of it.
   *Cheapest test*: the Stage-B MP+EXT checkpoint arms already exist; wire `face_sizes`
   through the ppo.py call sites (unified_face_policy.py:293/:338 already accept it)
   and rerun the 61458-style online probe (~1.5h GPU) — R² on the 5 face targets must
   clear the 0.48-0.67 bar in-run before any 500-episode spend.
3. **Freeze or floor the quality-head PopArt stats once basin occupancy passes ~50%**
   (or exclude sub-τ_d terminals from μ/σ updates). *Supported by*: §3 ratchet (μ
   4.63→−0.95, z(0) → +1.0). Secondary: it slows absorption but cannot restore a
   gradient the reward doesn't emit. *Cheapest test*: offline — replay v58b's return
   stream through `_popart_update` (ppo.py:579) with and without the freeze and compare
   z(identity) late-run.
4. **Not supported as levers**: LR schedules (three-way falsification), entropy-weight
   reduction alone (v57 confounded — 1 env AND 2 weak updates/ep; GAZ counterexample;
   though halving 0.05 alongside (1) is cheap and harmless), walk/measurement fixes
   (§5: channel is clean), #126 arity fix for these campaigns (fingerprints identical;
   still worth fixing on principle).
5. **Prerequisite hygiene before the next GAZ control**: fix the 61531
   `face_count_fn`/`face_keys_of` disagreement (§8) so the depth-0 arm can serve as a
   clean comparator; keep 61532 running untouched.

What the evidence cannot yet distinguish: whether (1) alone prevents collapse with the
representation still blind (the policy can express "approximate less" globally), or
whether (1)+(2) are jointly required for the policy to *stay* out of the basin while
actually using approximations. The 100-episode reshaped-gate rerun answers the first;
its failure mode directly quantifies the second.

## Appendix A — overnight endings (status as of 2026-08-18 ~09:00)

- **v60 / job 61515: COMPLETED** normally (sacct: COMPLETED 0:0, elapsed 08:38:56, end
  2026-08-18T04:55:20). 508 wandb rows (`ygm8n2jy`, state finished). Final state:
  median q = 0.0, entropy 0.05, mean latency ≈1.35e5 ns (plateau; the early-run "latency
  drifting worse" did not persist — full-run mean settles 1.31-1.36e5 vs 1.62e5 at
  ep63). Final pareto (log tail): top-10 all ep11-32 identity plans, q 0.8853 —
  8.6 GPU-hours produced nothing after ep~32.
- **GAZ depth-0 / job 61531: CRASHED at ep122** (sacct COMPLETED 0:0 is misleading —
  the batch wrapper logged `RC=1`; end 2026-08-17T23:59:54, elapsed 1:44:45). Cause: the
  §8 face-enumeration assertion at az_gumbel.py:1317. 122 valid rows in
  `run_61531/updates.jsonl`; quality was already 0 from ep21, so the crash does not
  censor the collapse finding.
- **GAZ deep20 / job 61532: RUNNING** (gpu8, 8h+; 333 rows at reading). Untouched, as
  required.
- v57 (61485) and v59 (61507) were operator-cancelled mid-run (sacct CANCELLED+ 0:0),
  which is why their wandb states read "crashed".

## 11. Intervention 1 implemented: Lagrangian quality-constrained reward (2026-08-18)

Implementation (commit f332f13, branch hostperf-caches): `--reward-mode
lagrangian` in `src/alphagrad/approx/ppo.py`. ADDITIVE composition (owner
decision 2026-08-18 -- NOT built on the mult scalar): the latency/mem
channels flow exactly as `--reward-mode additive` computes them; the quality
slot is replaced by a stationary violation channel

    v = max(0, tau_q - q_eff),   q_eff = clip(q, -0.5, 1.0)

stored NEGATED on the terminal step (`_apply_lagrangian_channels`). The
q_eff clip kills the section-1(c) conflation: DIVERGED (-1.0 sentinel) maps
to q_eff = -0.5 and carries violation tau+0.5 = 1.25, strictly worse than a
zero-work plan (0.75). lambda enters ONLY as the quality slot of the
advantage-scalarization preference (`traj.preference` at the
`norm_adv = sum(norm_adv_components * traj.preference)` site); value
targets and PopArt statistics are lambda-free by construction -- pinned by
`tests/lagrangian_reward_test.py` (7 tests: violation values, no-flat-band
monotonicity, cost-channel bitwise passthrough, lambda-invariance of value
targets, dual-ascent clip, basin-freeze trigger/non-trigger). Dual ascent:
`lam <- clip(lam + 0.05 * mean_violation, 0.1, 10)`, once per episode,
host-side, after the PPO update. Section-10(3) guard: `--popart-basin-freeze`
(default on) holds the quality channel's (m1, m2, w) for an episode when
>50% of the batch sits in the basin (terminal q_eff <= 0.05, the section-2
occupancy measure -- covers both the historically observed q = 0.0 zero-work
mode and diverged plans). Corrected 2026-08-18: the first cut used
violation > 0.9*(tau+0.5), which needs q < -0.375 and could never fire on
the v58-v60 basin (q = 0.0 exactly).

Offline falsifier -- zero GPU, the section-10 cheapest test -- against the
3,024 measured v58b plans (192 eps x 16 envs; the wandb export carries
per-episode best/median/worst/mean quantiles per channel, not per-plan
tuples, so per-plan checks use the 576 quantile samples + plans anchored to
measured cost ranges). Script + outputs:
`/Users/assmuth/dsnn/collapse_invest/lagrangian_falsifier/`
(`falsify_lagrangian.py`, `falsifier_report.txt`, `falsifier_verdict.json`,
`lambda_trajectory_v58b.csv`). Scalarization under test:
`score = -log1p(lat) - log1p(mem) - lam * v(q)`, lam in {0.1,0.5,1,2,5,10}.

VERDICT: PASS on all four checks.

- (i) NO flat region: finite-difference slope == lam exactly, everywhere on
  the operative band [-0.5, tau), at every lam; 0 flat segments (the mult
  surface had a 0.45-wide flat + a +1.9 discontinuity). The band (-1, -0.5]
  saturates BY DESIGN (maximal violation); 12/576 measured quantile
  qualities fall in (-1, -0.5) and tie with diverged there.
- (ii) diverged strictly below every plan measured at q > -0.5 at every lam:
  zero-work minus diverged = 0.5*lam (+0.05 .. +5.0); the worst measured
  non-diverged plan (q = -0.4705) clears diverged by +0.003 (lam 0.1) to
  +0.30 (lam 10).
- (iii) identity NOT dominant when a real latency win exists: 22 episodes
  held median q >= tau; best measured latency there 9.12e4 ns vs identity
  1.60e5 ns. A constraint-satisfying cheaper plan beats identity by +0.562
  (the latency symlog term) at EVERY lam -- feasible plans tie on violation
  (0) and the cost channels decide. Cost work below tau still pays down to
  q > tau - dlat/lam: q > 0.19 at lam=1, q > 0.47 at lam=2, q > 0.69 at
  lam=10 -- lam prices quality loss, it does not forbid approximation.
- (iv) dual-ascent replay of the v58b episode sequence (violation estimated
  from the quantiles, weights 0.25/0.5/0.25): lambda 1.00 -> 1.08 (ep10) ->
  1.42 (ep30) -> 1.72 (ep40, basin entry) -> 2.40 (ep60) -> 7.05 (ep191),
  monotone non-decreasing -- the price of destruction RISES as the basin
  fills, the anti-ratchet by construction (contrast section 3: mu_quality
  fell 4.63 -> 0.27 over the same window). Advantage ordering at ep35
  (lam = 1.55, mean measured costs of ep30-40): identity -29.72 = survivor
  (q=0.79) -29.72 > partial (q=0.4) -30.27 > zero-work -30.89 > diverged
  -31.66. Quality-preserving plans rank strictly above destroyed ones; the
  feasible tie at matched cost is the intended constraint semantics.

Caveats, stated honestly: the falsifier proves the reward SURFACE has the
claimed shape on real measured data; it cannot prove the policy escapes the
basin (that is the 100-episode v58-config rerun of section 10), and the
representation blindness of section 4 is untouched (intervention 2).

Launcher prepared (NOT submitted):
`/Users/assmuth/dsnn/fq_v61_tlm_lagrangian.sbatch` -- v60 clone, constant
LR (mult-warmup flags dropped), `--reward-mode lagrangian --lag-tau 0.75
--lag-eta 0.05 --lag-init 1.0 --lag-min 0.1 --lag-max 10
--popart-basin-freeze`, 500 eps / 16 envs / --grad-window 0 / TLM pins,
wandb name v61-tlm-lagrangian, placeholder comment for the probe-flag
workstream. New wandb keys: lagrangian/lambda, lagrangian/mean_violation,
lagrangian/frac_violating, lagrangian/popart_frozen.

## 12. v62/v63 — the anti-none global-credit slide: CONFIRMED (with a corrected mechanism), its igniter, and the fix spec (2026-08-25)

v61's fix (lambda=10 prices destruction out to q > 0.69) held the cost story:
no plan profits from destruction any more. Both follow-up runs still slid into
an absorber. This section pins the mechanism from the two full records —
v62 (job 61844, wandb `w6sh91ya`, crashed ~ep122) and v63 (job 61866, wandb
`as9s5yrl`, COMPLETED, 500 eps) — plus a CPU falsifier with per-plan
instrumentation. Analysis artifacts:
`/Users/assmuth/dsnn/collapse_invest/v63_anti_none/` (wandb exports
`v6{2,3}_full.csv`, throwaway `ppo_instr.py` built by `make_instr.py` — NO
src/ change — per-episode `dumps_armA/advdump_*.npz`, `analyze_dumps.py`).

### 12.1 The phenomenon, aligned across both runs

Two phases, identical in both runs, shifted ~9 episodes:

| phase | v62 | v63 |
|---|---|---|
| slow drift (hinge active, H<0.3) | ep0–49 | ep0–57 |
| H crosses the 0.3 floor | ep49→50 (0.229→0.327) | ep60→61 (0.180→0.344) |
| runaway (hinge OFF) | ep50–~90 | ep61–~73 |
| none prob 0.98 → | 0.15 (ep80), 0.02 (ep100) | 0.016 (ep70), 0.000 (ep74+) |
| terminal absorber | uniform-ish incl. skip 0.22 | quant/diag/compress ~0.28/0.28/0.17, skip ~0.27, none 0 |

v63 detail (log + export): none 0.975 (ep51) → 0.964/0.949/0.926/0.829
(ep55-58) → 0.54 (ep60) → 0.11 (ep64) → 0.016 (ep67) → 0.000 (ep74, exactly —
the shared OP_NONE bias is at the clamp). frac_violating at onset (ep55-58):
0.06–0.38; saturation (0.875+) only from ep63. mean_raw_q 0.75→−0.12.
lambda: 10.13 (onset) → 10.35 (ep67) → 20.0 (pinned, ep~350+) with **zero
recovery**: mean_raw_q ≈ 0 for 430 further episodes while scalarized_return
falls monotonically −58 → −114.5 — the policy sits in an absorber where its
own optimized objective keeps worsening, because with every plan destroyed
the advantage contrast is ~0 and PG is dead (the v61 endgame, reached by a
different road).

Sampled-approx census (v63): applied+skipped ≈ 40-60/batch pre-slide →
489 (ep61) → 2814 (ep70). Op-prob gains ep53→ep67: quant 52×, compress 41×,
diag 36× — proportional within 1.4× — skip 2.4× (pinned at 0.003 while the
others reach 0.30-0.38). Final ordering quant > compress ≈ diag ≫ skip is
exactly the harm-ordering (one skip deterministically DCEs the TLM graph;
diag/compress sometimes destroy; quant rarely).

Post-collapse per-decision entropy is 0.03–0.09 nats while the op MARGINALS
are near-uniform: the head is (near-)deterministic PER FACE with different
ops on different faces — a face-deterministic destroyed-plan absorber, not a
high-temperature one. The entropy floor (still active there, penalty ~0.62)
cannot lift it: the raw logits sit far beyond the ±15 tanh clamp where the
clamp's gradient attenuation (~50×) neuters the hinge — the "restoring
gradient survives saturation" design failed in exactly the state it was
built for.

### 12.2 Mechanism: three forces on ONE shared parameter

The face head's OP_NONE logit is one shared bias entry per slot
(`agent_factory.py::apply_face_none_bias`, `unified_face_head` final linear):
every face's none-vs-rest decision moves together. Three forces act on it:

**(F1) The entropy-floor hinge — the IGNITER (ALT-3, confirmed as such).**
`entropy_floor/penalty` was **continuously ~0.6–0.72 from ep0** in BOTH runs
(H ≈ 0.03–0.06 ≪ floor 0.3; hinge gradient dP/dH = −2·10·(0.3−H) ≈ −5.4 —
~1000× v63's 0.005 face bonus, ~100× v62's 0.05). It pushes H up = none down,
and its per-parameter bite GROWS as saturation lifts (dH/dθ ∝ p(1−p)·…), so
the drift accelerates: each op's prob doubles over ~50 eps, then triples in
3. The penalty hits exactly 0 at the crossing (v62 ep50, v63 ep61) — the
hinge cannot explain anything past H = 0.3 (p_none ≈ 0.93). Note the floor
was mis-targeted from the start: H = 0.3 per face decision means ~7% approx
per decision ≈ 8–18 approx ops per TLM plan — structurally inside the
violation regime. The igniter is a config bug, not bad luck.

**(F2) The entropy bonus — v62's tail only.** v63 cut the face bonus 10×
(0.05→0.005) and slid anyway, ~9 eps later and FASTER through the runaway —
falsifying the §11-era attribution of the slide to the bonus. The bonus only
explains v62's late walk to full uniform (skip 0.01→0.22 by ep99); v63's
absorber keeps skip low until after none dies, then skip relaxes up to ~0.27
once every plan is destroyed anyway and skip is no longer differentially
punished.

**(F3) Advantage-mediated anti-none — the RUNAWAY (H-NONE, corrected).**
Terminal-only rewards + GAE(0.99, 0.95) give plan-global, tail-weighted
credit (step T−k carries 0.9405^k of the terminal advantage). PopArt stats at
onset: sigma_quality ≈ 0.127–0.129, mu_quality ≈ −0.036 → one destroyed plan
(violation 0.75) scores z ≈ −5.6, ×lambda 10 ≈ **−56**, against healthy
plans' +0.3 × 10 ≈ **+3** (their violation channel is exactly 0, so healthy
contrast exists only through −mu). One violator ≈ 19 healthy plans.

The naive statement of H-NONE — "Σ_batch (advantage × N_none) < 0 ⇒ anti-none
gradient" — is NOT the gradient. For a plan-constant advantage the softmax
score function obeys E[Σ ∇log π] = 0: the per-plan drift on the shared none
bias is A_p · (N_none,p − N_p·p̄_none), a covariance, and at epoch 0 its
expectation is CORRECTIVE (a plan violates because it sampled more approx
than the mean, so its none-count deviation is negative and A_p·dev > 0
supports none). What actually breaks the symmetry is **PPO's
negative-advantage clipping asymmetry over epochs** (--ppo-epochs 2 × 4
minibatches): for A < 0 the surrogate min(rA, clip(r)A) is UNCLIPPED as the
ratio rises, while the healthy plans' positive-advantage terms saturate at
1+ε. The epoch-2 updates keep pushing every step of a violating plan's JOINT
log-prob down without bound; ~(1−p_none) ≈ 0.02 lands per none decision × 
~200 decisions ≈ the ~1 × 3 landed on the sampled approx logits, and the
relative winners are the ~60 UNSAMPLED approx variants — which is exactly the
observed proportional quant/diag/compress rise with skip (sampled, punished,
deterministic harm) pinned.

The signature is in the export: `kl/approx` (joint-ratio KL) sits at 0.35–0.5
all through the healthy phase, rises with violation frequency
(0.69→1.45, ep58–60), then explodes through the runaway — 3.3, 6.0, 8.5,
**109 (ep65)**, 9.7 — with `ratio/max_log` 10→35 and the ppo surrogate loss
spiking to +60 (ep66). Updates of that size are only reachable through the
unclipped negative branch.

The loop: hinge-driven drift → more approx per plan → violation frequency up
→ more −56 plans per batch → unclipped negative pressure on the shared none
bias → more approx → … lambda's dual ascent (+0.0015/ep pre-slide) is a slow
follower — an amplifier, never the trigger.

### 12.3 No cost payoff anywhere in the slide (ALT-1)

During the slide latency means WORSEN monotonically: 1.60e5 → 2.22e5 ns
(ep57→70); memory flat at 5.54e7. The transition pays cost, it does not
collect it. (The post-collapse absorber IS cheaper — 1.37e5 ns, −14% — but at
lambda ≥ 11 its violation price is z·λ ≈ −56…−108 against a latency gain of
~+2; the absorber persists because PG contrast is dead, not because the
trade pays.) The z-space asymmetry also rules it out numerically: |quality
advantage| ≈ 56 vs |cost advantage| ≲ 2. v61 is the control: at lambda = 1
destruction WAS profitable (falsifier fact iii: pays down to q > 0.19) and
the policy collapsed INTO the cost-optimal absorber (100% SKIP). At
lambda = 10 the same machine collapses into a cost-WORSE absorber — the
slide's driver is not cost.

### 12.4 v62 kills the endpoint-read coupling (ALT-2)

v62 has no `--face-endpoint-read` and slid ~9 episodes EARLIER with the same
two-phase shape, same hinge-crossing structure, same KL blowup (0.47 → 11.9),
same absorber. The v63 endpoint read changed nothing material about the
slide. REFUTED as a cause.

### 12.5 CPU falsifier: the slide REPRODUCES, and the per-plan data picks the mechanism

Job 61936 (pgi15-cpu2), Helmholtz (6 vertices, 5 steps, face bound 9, ~10
face decisions/plan), the proven v63 smoke config scaled to 16 envs ×
4 minibatches × 200 eps with the campaign pins (FACE_NONE_BIAS=6,
FORCE_REV_ORDER=1, QUALITY_GATE_MIN=0.05, lag-init 10, floor 0.3/10, face
bonus 0.005, quality = Jacobian cosine). Instrumented via the throwaway
`ppo_instr.py` (per-episode npz: per-(env,step) scalarized + per-channel
normalized advantages, raw rewards, face actions). Numbers below are the
first 60 trained episodes (run still extending the record in
`dumps_armA/` + `armA_61936.log`).

**The slide reproduces, compressed.** Valid-face none: 0.99 (ep1) → 0.95
(ep7) → 0.85 (ep12) → 0.77 (ep46) → 0.70 (mean of ep55–59; single episodes
down to 0.60). H crosses the 0.3
floor at ~ep7-8 (hinge phase compressed to ~7 eps by the small alphabet) and
keeps rising after the hinge zeroes — same two-phase shape as TLM. quant is
the main gainer (harmless on this target, so corrective PG never opposes
it); skip stays pinned ≤ 0.02 with P1_true_skip −3…−9 on every violating
episode — the per-op corrective signal works where causality is
deterministic, exactly as on TLM.

**P1, decided.** The naive statistic Σ_p A_p·N_none,p flips sign with the
batch mean (−2690…+108) and does not track the slide. The actual epoch-0
score-function drift on the shared none bias, Σ_p A_p·(N_none,p −
N_p·p̄_none), is **positive (none-SUPPORTIVE) even in violating episodes**
(+0.3…+10.5) — the corrective covariance argument is confirmed in vivo. Yet
none falls. The discriminating measurement: **Δnone(t→t+1) = −0.015 to
−0.021 after an episode containing a violator vs −0.001 after a clean
episode (−0.0149 vs −0.0008 over 60 eps, ~19×; corr(frac_viol, Δnone) ≈
−0.13)**. The anti-none
drift is violation-DRIVEN but not epoch-0-PG-driven — the only channel left
is the epoch-2 negative-advantage pressure (v62/v63's kl/approx 0.4→109 is
the same channel at TLM scale). H-NONE's substance is confirmed; its
mechanism is the clipping asymmetry, not the raw advantage×count sum.

**P2, refined.** The none decline is NOT position-uniform: over ep55–59 the
LAST step-tercile's none rate averages 0.39 while the first tercile holds
0.85. Tail-weighted plan-global credit lands where GAE puts it — late
decisions first — on top of the global shared-bias shift. (On TLM the same
gradient concentrates on the shared bias; the exported aggregate cannot
resolve position, but the repro says the tail leads.)

**A third, gentler pressure appears late:** healthy plans' quality-channel
tail advantage `qual[ok]` drifts slightly negative (−0.02…−0.04 by
ep47–58) — critic optimism (mu ratchet, §3) turns even clean plans'
quality credit mildly negative, adding wholesale-negative pressure on
sampled actions. Same family, smaller than the violator kicks.

The full TLM-style terminal absorber (frac_violating→1, none→0) has not
fired by ep59 on this target — quant absorbs the redistributed mass and
quant is harmless on Helmholtz, so the violation-density feedback loop is
weak. The load-bearing claims (drift reproduces; violation-coupled anti-none
kicks; epoch-0 PG corrective; skip pinned; late-first) do not depend on it.

### 12.6 Verdict table

| claim | verdict | decisive evidence |
|---|---|---|
| P1 Σ(adv×N_none) < 0 at onset | REFUTED as stated, CONFIRMED as corrected | the naive sum sign-flips with the batch mean and does not track the slide; epoch-0 drift on the none bias is POSITIVE even in violating episodes; the anti-none channel is the epoch-2 unclipped negative branch (repro: Δnone ~19× larger after violator episodes; TLM: kl/approx 0.4→109, ppo loss +60) |
| P2 simultaneous global none fall | CONFIRMED, with a tail-first refinement | OP_NONE is ONE shared bias/slot (structural globality); none hits 0.000 exactly across all faces on TLM; repro tercile split shows LATE faces lose none first (0.39 vs 0.85) — GAE tail-weighting rides on the global shift |
| P3 proportional gains except punished ops | CONFIRMED | quant 52× / compress 41× / diag 36× vs skip 2.4×; final order = harm order |
| P4 higher lambda accelerates | UNTESTED (no contrast) | v62/v63 identical lag config, both ignited at λ≈10.10–10.14; λ is a slow follower (dual ascent), rises only AFTER violations; v61 (λ=1) shows λ selects WHICH absorber, consistent with advantage-scale mechanics |
| P5 onset at first strong violators (frac 0.1–0.4), not saturation | HALF-CONFIRMED | onset frac 0.06–0.38, saturation only 5+ eps later ✓; but strong violators existed from ep6–9 (frac 0.25, q 0.63) with NO slide for ~46 eps ✗ — the trigger is the hinge-driven approx-prob level, not the first violators |
| ALT-1 cost advantage pays for destruction | REFUTED (λ=10) | latency worsens 1.60→2.22e5 during the slide; |A_qual| ≈ 19–28× |A_cost|; true at λ=1 (v61) only |
| ALT-2 endpoint-read coupling | REFUTED | v62 (no read) slid earlier, same shape |
| ALT-3 entropy floor hinge | CONFIRMED as IGNITER, refuted as runaway | penalty ~0.6–0.72 continuously ep0→crossing, exactly 0 after; floor 0.3 targets p_none≈0.93 ≈ 8–18 approx/plan = inside the violation regime; cannot explain none 0.83→0.016 post-crossing |
| overall H-NONE | **CONFIRMED with corrected mechanism** | anti-none global credit is real and drives the runaway, but it enters through PPO's negative-advantage epoch asymmetry on the shared OP_NONE bias, ignited by the always-on entropy floor — not through the naive adv×count sum |

### 12.7 Fix spec (NOT implemented)

1. **Causal mask on the quality-channel advantage.** At the scalarization
   (`norm_adv = Σ_c norm_adv_components[...,c] · pref[...,c]`,
   ppo.py ~line 7895): replace the quality slot's constant weight λ by
   λ·m(e,t), where m(e,t) = 1 iff step t of env e contains a causal action —
   any valid face with skip = 1 or sampled op ≠ OP_NONE (from
   traj.face_skip / face_op_type / face_valid, all already in the batch).
   Under rev-pin, `none` cannot cause a violation, so the violating plan's
   −56 lands only on its 2–10 causal actions and never on the shared none
   bias. Free-order generalization: mask to {approx actions} ∪ {vertex
   choices} — the vertex head stays inside the quality credit because order
   changes which faces exist, but an all-none exact plan keeps quality
   advantage 0 on every face decision. Cost channels stay unmasked (every
   action shapes cost).
2. **|z| winsorize ≈ 3 per channel** (ppo.py ~line 7859:
   `norm_adv_components = clip(advantages/new_sigma, ±3)`): bounds a
   destroyed plan at −30 (λ=10) instead of −56 and, with (1), caps the
   per-episode unclipped negative drift.
3. **Fix the igniter**: `--face-entropy-floor 0.05` (matches the identity
   init H≈0.03–0.06; 0.3 structurally demands ~8–18 approx ops/plan), or
   hinge on the batch op-MARGINAL entropy instead of the per-decision mean.

Predicted dynamics under (1)+(2): violating plans stop suppressing none (their
quality credit lands on causal actions only — which is also a *better*
credit assignment for learning which ops are safe); the positive feedback
loop is broken; quant learning survives on its own merits (its cost advantage
is its own: bf16 pulldown −5.8…−7.7% latency at cos 0.99997); skip stays
priced out by its own causal punishment. Without (3) the hinge still drags
p_none toward 0.93, but the masked credit turns the resulting violations
into per-op corrective signal instead of anti-none fuel — the slide should
flatten into noisy op-level selection pressure.

**Cheapest online falsifiers** (one 4-GPU TLM run each, 150 eps):
- Igniter test: v63 config, ONLY `--face-entropy-floor 0.05`. Floor-as-igniter
  predicts no slide in 150 eps; if it still slides, F3 self-ignites and the
  mask is the load-bearing fix.
- Fix test: v63 config + mask + winsorize (floor untouched): predicts
  frac_violating stays < 0.2, none ≥ 0.9, quant applied/batch keeps its
  ep50-55 growth, and kl/approx never leaves O(0.5).

### 12.8 Implementation + falsifier + launch record (2026-08-25)

Fix implemented as commit `6d917fc` (`--lag-causal-mask`, `--adv-winsorize`, floor-help igniter note;
tests/credit_fix_test.py 8/8; flag-off smoke bit-identical to 83a4ced). CPU no-slide falsifier
(job 61964, Helmholtz slide-repro config + full fix + floor 0.05, 60 eps): **PASS** -- none held
0.993-1.0 the entire run (unfixed 61936 baseline: 0.70 by ep60), frac_violating 0 except one
recovered 0.062 blip, mean_raw_q ~1.0, entropy stable ~0.10 with no hinge drag, lambda decayed
9.99->9.95 (first in-the-wild exercise of --lag-target decay), mask_frac 0.025-0.038.
Log: collapse_invest/v63_anti_none/armFix_61964.log.

Online arms: **v64b** = full fix production candidate (job 61983, gpu16, 500 eps, wandb 38oyqf4g,
predictions: frac_violating<0.2, none>=0.9, kl/approx O(0.5), quant learns without slide).
Igniter-only arm moved to CPU (repro_armB, job 61984: floor 0.05, NO mask/winsorize -- prediction:
slide delayed/absent vs 61936; if it slides late, mask+winsorize are load-bearing, not just belt).

### 12.9 v64b post-mortem: H-ZNEUT **REFUTED**, and the corrected diagnosis — the critic is the bottleneck (2026-08-26)

#### 12.9.1 What v64b actually did

v64b (job 61983, wandb `38oyqf4g`, log `v64b_tlm_61983.log`) ran the full
sec-12.7 credit fix (`--lag-causal-mask`, `--adv-winsorize 3`,
`--face-entropy-floor 0.05`, `--popart-basin-freeze`, λ init 10) and ran to
completion — 500/500 episodes — **inside the terminal absorber**: λ pinned at
`--lag-max` 20, mean_violation 0.75 (the maximum reachable at τ=0.75 with
q clipped at 0), mean_raw_q 0.000, PopArt frozen. The credit fix did not
prevent the collapse; it changed its *shape* from v62/v63's runaway into a
~25-episode drift.

The history (`collapse_invest/v64b_zneut/v64b_full.csv`, ep0–120):

| ep | λ | frac_viol | mean_raw_q | approx_prob/none | H_face | mask_frac | σ_q | value loss | kl/approx |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 10.00 | 0.062 | 0.807 | 0.982 | 0.028 | 0.052 | 0.158 | 0.503 | 0.45 |
| 16 | 10.02 | 0.000 | 0.880 | 0.981 | 0.046 | 0.064 | 0.149 | 0.019 | 0.54 |
| 30 | 10.07 | 0.188 | 0.730 | 0.980 | 0.050 | 0.070 | 0.142 | 0.268 | 0.56 |
| 48 | 10.12 | 0.125 | 0.803 | 0.966 | 0.080 | 0.106 | 0.133 | 0.163 | 0.84 |
| 56 | 10.15 | 0.312 | 0.807 | 0.955 | 0.117 | 0.149 | 0.129 | 0.028 | 1.18 |
| 64 | 10.18 | 0.062 | 0.750 | **0.709** | 0.503 | 0.475 | 0.126 | 0.315 | 6.06 |
| 72 | 10.25 | 0.375 | 0.534 | **0.419** | 0.807 | 0.730 | 0.125 | 0.507 | 10.53 |
| 80 | 10.44 | 1.000 | −0.110 | **0.053** | 0.987 | 0.869 | 0.132 | 0.674 | 10.51 |
| 96 | 11.01 | 1.000 | 0.000 | 0.000 | 0.011 | 0.020 | 0.132 | 0.800 | 0.14 |
| 120 | 11.89 | 1.000 | 0.000 | 0.000 | 0.005 | 0.016 | 0.132 | 0.691 | 0.15 |

Read the two right-hand columns together with `none`. The ordering is
unambiguous: **H_face rises first** (0.028 → 0.05 by ep30 → 0.117 by ep56 →
0.50 by ep64), `none` follows it down, and only then do violations become
universal. By ep96 the run has settled into a *deterministic* destructive
policy (H_face 0.005, none 0.000, q 0.000) — the absorber is not a noisy
plateau, it is a committed plan.

#### 12.9.2 H-ZNEUT as stated, and its falsifier

H-ZNEUT (drafted from the v64b shape before the history was pulled): under
PopArt the quality channel's advantage term is `A_raw/σ_q` and λ multiplies
that *relative* signal. As violations become prevalent the quality head's
return distribution should move — µ_q tracking down, σ_q widening — so the
same absolute violation reads as an ever-smaller z. The penalty would then
fade exactly when the constraint must bind, and no λ could set an absolute
price on a relative signal.

That is a quantitative claim about σ_q, and it is directly measurable. The
falsifier compares the one clean **recovery** window (ep20–30, n=11, where a
violation spike was pushed back to frac_violating 0) against the **drift**
window (ep55–75, n=21, where the policy slid): if H-ZNEUT holds, the
effective per-unit violation price must be materially LOWER in the drift
window.

#### 12.9.3 Evidence: the price went UP while the policy slid

`/Users/assmuth/dsnn/collapse_invest/v64b_zneut/v64b_zneut_summary.txt`
(analysis CSV alongside it; windows as above, rec = ep20–30, drift = ep55–75):

| quantity | recovery | drift | rec/drift |
|---|---|---|---|
| µ_q | −0.03828 | −0.03819 | 1.002 |
| **σ_q** | **0.14461** | **0.12649** | **1.143** |
| λ | 10.0506 | 10.1975 | 0.986 |
| **λ/σ_q** | **69.51** | **80.63** | **0.862** |
| z at violation 0.25 | −1.4642 | −1.6747 | 0.874 |
| **per-unit price @0.25** | **58.87** | **68.31** | **0.862** |
| per-unit price @0.75 (winsorized) | 40.20 | 40.79 | 0.986 |
| per-unit price @0.75 (unwinsorized) | 65.96 | 76.53 | 0.862 |
| winsorize clip frac, quality | 0.0098 | 0.0219 | 0.445 |
| frac_violating | 0.1477 | 0.3066 | 0.482 |
| mean_violation | 0.0809 | 0.1748 | 0.463 |
| mean_raw_q | 0.7589 | 0.6285 | 1.207 |
| mask_fraction | 0.0692 | 0.4542 | 0.152 |

Every leg of H-ZNEUT fails:

- **σ_q did not widen — it SHRANK**, 0.1446 → 0.1265 (and monotonically
  across the whole run, 0.158 at ep3 → 0.125 at ep72). µ_q is flat to four
  decimals. The distribution the mechanism needs simply did not move.
- **The per-unit price ROSE**, 58.9 → 68.3 (+16%) at a 0.25 violation, and
  65.96 → 76.53 at 0.75 before winsorization. λ/σ_q rose 69.5 → 80.6.
- The only price that is flat is the *winsorized* one at large violations
  (40.20 → 40.79), because |z| there is past the clip — but the clip is
  binding on **1–2% of quality entries** (0.98% → 2.19%), so winsorization is
  not neutralizing the channel either.

**Verdict: H-ZNEUT is REFUTED.** The penalty stayed fully priced —
*increasingly* priced — and the policy slid anyway. Any explanation that
routes through "the penalty got cheap" is dead.

#### 12.9.4 The corrected diagnosis: the critic is the bottleneck

The penalty's *price* was never the problem; its *reliability per step* was.
The quality term entering the policy gradient is
`λ · m(e,t) · (G_q(e,t) − V_q(e,t)) / σ_q`, and on the overwhelming majority
of steps the true quality advantage is ~0 (no violation to explain), so what
that expression carries is **the value net's error, amplified by 1/σ_q ≈ 7.9
and by λ ≈ 10 — roughly 80× per raw unit**. A quality head whose error has
random sign therefore injects a large, sign-random force on exactly the
parameters λ was supposed to steer, and it averages out over the batch
instead of steering. Four coupled loops make this worse rather than
self-correcting:

1. **Noise amplification by the normalizer.** σ_q is small (0.13) *because*
   the channel is sparse-terminal and mostly zero — so the very sparsity that
   makes the true signal rare is what multiplies the critic's error by ~8.
2. **Moving frame.** PopArt rescales the heads (ART) and renormalizes the
   targets (POP) every episode; the critic is chasing a target whose units
   move under it.
3. **Stale optimizer moments.** Adam's second-moment estimates for the value
   head were accumulated in the *previous* frame; after a rescale they are
   mis-scaled for the new one, so the effective critic learning rate is wrong
   exactly when the frame moves most.
4. **Policy-dependent σ and a µ ratchet.** σ_q and µ_q are statistics of the
   *policy's own* returns, so the per-channel exchange rate between quality
   and cost drifts as a function of the thing being optimized — a
   nonstationary objective, not a fixed one.

The corroborating observation from the v64b history is the *density*
asymmetry. With `mask_fraction` 0.05–0.07 through ep3–30, the quality term
reaches ~6% of steps; the entropy bonus (`--face-entropy-weight 0.005`)
reaches every face slot of every step and always points the same way (raise
H). A sparse, sign-random force loses to a dense, coherent one regardless of
its nominal per-unit price — which is exactly the observed ordering, H_face
rising *before* `none` falls and long before violations become universal.
This vindicates the residual role left open for H5 in sec 6: the entropy
bonus is not the *initiator*, but once the quality term is noise-dominated it
is the only coherent gradient on the face head.

Note what this does NOT claim. It does not claim the critic is badly
implemented (sec-H2's ART/POP carry and the #89 neutral-target fix are
verified); it claims the *objective the critic has to track is nonstationary
and the normalizer amplifies whatever error remains*. The remedy is therefore
not a better critic — it is an objective that does not move.

#### 12.9.5 Remedy under test: a fully STATIC objective

Owner-directed (2026-08-26). Remove every adaptive statistic from the
objective and make the exchange rate a constant chosen in advance:

- `--advantage-norm none` — no PopArt, no batch z-score. No moving frame, no
  head rescale, no stale-moment mismatch, no policy-dependent σ.
- **symlog on the cost channels** (these arms DROP `--no-symlog`) — a FIXED,
  policy-independent magnitude compression that replaces PopArt's job for the
  ~1e5…1e10 cost scales.
- **the violation channel RAW** — bounded by construction in `[−(τ+0.5), 0]`,
  so it needs no compression, and symlogging it would discount the absolute
  price λ is meant to set (symlog(0.75) = 0.56, a 25% discount exactly at the
  constraint bound).
- **λ frozen** via `--lag-eta 0` — no dual ascent. The exchange rate between
  a unit of violation and a unit of symlog-cost is a predetermined constant
  (10–16, i.e. quality ~10–16× the cost weights) for the entire run.

Under this objective a given plan scores the same at ep 5 and ep 500, and the
quality term's scale is a constant instead of `λ/σ_q`. If the critic-noise
diagnosis is right, the quality-channel value error should be materially
smaller and better behaved, and `none` should hold.

#### 12.9.6 The v65 raw-violation-advantage arm, retained as a control

`--lag-raw-viol-adv` was built as the H-ZNEUT fix: it computes the QUALITY
channel's advantage on the RAW scale — `(raw violation-channel return) −
(raw-scale value prediction)`, clipped at ±2.0 for a transiently wrong
critic, NOT winsorized — bypassing PopArt z-normalization for that channel
only, while the cost channels keep PopArt+winsorize and the value loss stays
in normalized space. It composes with `--lag-causal-mask`; default off is
bit-identical. Telemetry: `lagrangian/raw_adv_mean|min` (violating envs only)
and `raw_adv=` on the `[lagrangian]` stdout line.

H-ZNEUT being refuted does not make the flag useless — it makes it the right
*control*. It removes the 1/σ_q amplification (loop 1) and the
policy-dependent exchange rate for the quality channel (loop 4) while
*keeping* PopArt on the costs and dual ascent on λ. Run against the static
arms it isolates the PopArt layer specifically. It is therefore shipped and
launched as arm D of the sec-12.10 battery rather than as "the fix".

#### 12.9.7 Battery design

Four arms, 250 episodes each, one per GPU node, identical except for the
objective layer under test:

| arm | normalizer | cost transform | violation channel | λ | causal mask |
|---|---|---|---|---|---|
| A `v66a-static-lam10` | none | symlog | raw | frozen 10 | yes |
| B `v66b-static-lam16` | none | symlog | raw | frozen 16 | yes |
| C `v66c-static-lam13-nomask` | none | symlog | raw | frozen 13 | **no** |
| D `v65-tlm-rawviol` | popart (+winsorize 3) | none (`--no-symlog`) | raw advantage | dual ascent from 10 | yes |

A vs D is the whole-PopArt-layer contrast. A vs B is the price sensitivity of
the static objective. A vs C asks whether the sec-12.7 causal mask is still
load-bearing once the scales stop moving. Per-channel critic telemetry
(`value_loss/latency|mem|quality`, added for this battery — v64b logged only a
summed `value loss`) is what makes the diagnosis measurable across arms
rather than inferred. Registered predictions in sec 12.10.

### 12.10 The static-objective battery: implementation, launch record, registered predictions (2026-08-26)

#### 12.10.1 What was implemented

Three mechanisms, all off by default, shipped in commit `0e601ae`:

1. **Per-channel symlog exemption** (`_set_no_symlog_indices`, a pre-trace
   setter for `_NO_SYMLOG_MASK` — the machinery sec 12's header comment left
   in place for exactly this). Under `--reward-mode lagrangian` *without*
   `--no-symlog`, the violation slot (`REWARD_INDEX["cosine_sim"]`, which the
   lagrangian rewrite has already replaced with `−violation`) is exempted and
   the latency/memory channels are symlog'd. Prints
   `[cfg] lagrangian + symlog: cost channels symlog'd, violation channel RAW`.
   `--no-symlog` still short-circuits everything to the identity, so every
   PopArt arm — v65 included — is untouched.

   **The three-sites check** (project memory: "symlog vs PopArt: 3 sites must
   agree"). The sites are (a) the reward transform `_symlog_rewards`, (b) the
   value target `_value_target`, (c) the GAE's value decode
   (`get_advantages` = `make_get_advantages(use_symlog=True)` symexps the head
   output; `_GAE_POPART` = `use_symlog=False` does not, and is selected only
   when `advantage_norm == "popart"`). Only (a) is per-channel; (b) and (c)
   are a *uniform* encode/decode pair around the value head. That is what
   makes the exemption safe: whatever per-channel reward space (a) chooses,
   (b) symlogs it and (c) symexps it back, so GAE, the value loss and the
   advantages all live in the space (a) defined, per channel, with no further
   changes. The `--advantage-norm none` branch's degenerate-step neutral
   target (`inverse_reward_normalization_fn(traj.value)`) inverts (b) exactly
   and stays a ~0-loss substitution channel-wise. Pinned by
   `tests/static_objective_test.py`.

2. **`--lag-eta 0` freezes λ.** No new code: `_lag_dual_ascent` reduces to
   `clip(λ, lag_min, lag_max)`, which is bitwise identity for any λ inside
   the bounds (all three arms: 10/13/16 inside [2, 20]). The `[lagrangian]`
   stdout line and the `lagrangian/*` wandb keys keep printing off the frozen
   float, and `popart_frozen` is structurally 0 under `--advantage-norm none`
   (the freeze test requires `advantage_norm == "popart"`). Pinned by a
   bitwise no-op test over the λ × violation × target grid, with a positive-η
   control so the test cannot pass against a broken updater.

3. **Per-channel critic telemetry.** `_per_channel_value_loss` returns
   `(total, per_channel)` where `total` is *bitwise* the pre-existing
   `mean(sum(sq, −1))` value loss, and the `(NUM_VALUE_HEADS,)` vector is
   threaded out through the metrics tuple (now 12 entries) to
   `value_loss/latency`, `value_loss/mem`, `value_loss/quality`. v64b logged
   only a single summed `value loss`, which cannot distinguish "the critic is
   noisy on the channel λ prices" from "the critic is noisy on a cost
   channel" — the whole point of the battery. The keys join the warm-up drop
   list alongside the other loss-derived keys, so PopArt warm-up episodes
   leave a clean gap rather than a NaN point.

`--lag-raw-viol-adv` (sec 12.9.6) ships in the same commit for arm D.

#### 12.10.2 Shipment gate (job 62068, pgi15-cpu2)

One CPU job on pgi15-cpu2 ran the whole gate serially: a flag-off
Helmholtz smoke on the *pristine* HEAD build, then the install, then the same
smoke on the new build, then the five unit suites, then the two new-mode
smokes. Every episode's full `log_dict` was captured through
`ALPHAGRAD_UPDATE_JSONL` and compared entry by entry.

- **Flag-off bit-identity: PASS.** All 248 shared metric entries over 3
  episodes compare bitwise equal between HEAD (`8bd84d5`) and the new build
  under a v64b-shaped flag set (`--advantage-norm popart --no-symlog
  --lag-causal-mask --adv-winsorize 3 --popart-basin-freeze`). The only
  difference is the three *added* `value_loss/<channel>` keys on the two
  gradient episodes — no value changed anywhere.
- **Unit suites: 51/51 green.** credit_fix 12 (incl. the four raw-adv cases:
  bitwise (µ,σ)-invariance with a 1/σ control, quality-slot-only, mask
  composition, boundedness), lagrangian_reward 12, endpoint_read 7,
  edge_mem 12, static_objective 8.
- **static_objective_test.py (new, 8 cases)** pins: the symlog exemption hits
  exactly the violation slot and leaves latency/mem bitwise symlog'd;
  resetting to `()` restores the all-symlog default bitwise; `--no-symlog`
  overrides the mask entirely; the end-to-end
  `_apply_lagrangian_channels → _symlog_rewards` pipeline puts exactly
  `−violation` on the terminal step, 0 elsewhere, bounded by `τ+0.5`;
  `--lag-eta 0` is a bitwise no-op over the λ × violation × target grid with
  a positive-η control; `_per_channel_value_loss` totals bitwise and
  decomposes exactly; and it follows the `--no-symlog` switch like the summed
  loss does.
- **Static-mode smoke (arm-A flag set): PASS.** Prints
  `[cfg] lagrangian + symlog: cost channels symlog'd, violation channel RAW`;
  λ frozen **bitwise** at 10.0 across episodes (the arm-D smoke's λ moves
  10.0009 → 10.0027 in the same two episodes — the freeze is η doing the
  freezing); `value_loss/{latency,mem,quality}` present on every gradient
  episode (ep0: 0.01547 / 2.3816 / 0.004286) and absent on the warm-up row.
- **Raw-viol smoke (arm-D flag set): PASS.** `lagrangian/raw_adv_mean|min`
  emitted; `[lagrangian] … raw_adv=0.003 / −0.013` on stdout; the static arm
  correctly reports `raw_adv=nan` (flag off).

**One reporting nuance, recorded so nobody mis-reads the panels.** The
existing scalar `value loss` key is `--value-weight × Σ_channels MSE`
(default 0.5), while the new `value_loss/<channel>` keys are the **raw**
per-head MSE. In the smoke: per-channel sum 2.40139 vs `value loss` 1.20069,
exactly the factor 2. `--value-weight` is identical across all four arms, so
cross-arm comparison is unaffected; do not compare the two keys directly
within a run.

The final build differs from the tested build only in three documentation
edits (the `_raw_viol_override` docstring, the `--lag-raw-viol-adv` help
text, and one comment) that replace the refuted H-ZNEUT framing with the
sec-12.9 verdict. That was proved mechanically before commit: both files
have identical ASTs once docstrings and >40-character text constants are
masked, and the two fast suites (20 cases) were re-run green on the final
build.

#### 12.10.3 The arms as launched

Code shipped as commit `0e601ae`; all four jobs stamp
`ag=0e601ae gx=4ea0bf8` in their logs. Launched 2026-08-26 14:11 CEST,
250 episodes each, 4 GPUs per node, one arm per node, all four RUNNING and
syncing to wandb `dll-streetview/dsnn-vertex`.

| arm | job | node | wandb run | launcher | log |
|---|---|---|---|---|---|
| A `v66a-static-lam10` | 62072 | pgi15-gpu15 | `318ktrgq` | `fq_v66a_static_lam10.sbatch` | `v66a_tlm_62072.log` |
| B `v66b-static-lam16` | 62073 | pgi15-gpu17 | `0olsxsjl` | `fq_v66b_static_lam16.sbatch` | `v66b_tlm_62073.log` |
| C `v66c-static-lam13-nomask` | 62074 | pgi15-gpu18 | `s1537jdd` | `fq_v66c_static_lam13_nomask.sbatch` | `v66c_tlm_62074.log` |
| D `v65-tlm-rawviol` | 62075 | pgi15-gpu16 | `8sht6x1m` | `fq_v65_tlm_rawviol.sbatch` | `v65_tlm_62075.log` |

Config confirmed live from the logs: A/B/C each print
`[cfg] lagrangian + symlog: cost channels symlog'd, violation channel RAW
(symlog-exempt, bounded [-(tau+0.5), 0])`, and D prints
`[cfg] symlog DISABLED; PopArt alone scales the channels` — i.e. the
exemption is active on exactly the three static arms and the PopArt control
is untouched by it. Everything outside the objective layer is v64b's stack
verbatim (rev-pin, face bound 2538, `--measure-grad --seed-vertices`,
`--quality-metric loss_drop`, `--face-endpoint-read`, var probe,
`--popart-init-episodes 3`, seed 250197). `--popart-init-episodes 3` is
retained on the static arms even though PopArt statistics are never consumed
under `--advantage-norm none`: it keeps the random-plan warm-up census and
the `scalarized_return` frame on the same footing as arm D, at a cost of
3 episodes out of 250.

#### 12.10.4 What the static arms actually price — CORRECTED against the launched runs' own census

**This subsection was rewritten within the hour after launch.** Its first
version estimated the static arms' realized quality:cost pull at 5.0–8.1×
against v64b's 10×, by taking the symlog'd cost channel's advantage scale to
be the *relative* spread σ_lat/|µ_lat| ≈ 0.28. That was wrong: symlog
preserves a channel's relative spread but not its absolute footing, and the
advantage under `--advantage-norm none` carries the absolute scale. The four
launched jobs each print a `[popart-init]` census of their own MC returns at
ep0, which measures the quantity directly.

Measured at ep0 (v66a/b/c identical to 3 significant figures; v65 = the
PopArt/no-symlog control):

| channel | v66a/b/c µ | v66a/b/c σ | v65 µ | v65 σ | rel. spread (static / v65) |
|---|---|---|---|---|---|
| latency | −7.743 | **2.119** (symlog) | −102883 | 28944 (raw) | 0.274 / 0.281 |
| mem | −11.545 | **3.158** (symlog) | −3.588e7 | 9.817e6 (raw) | 0.2735 / 0.2736 |
| quality | −0.04156 | **0.15852** (raw) | −0.04156 | 0.15852 (raw) | — |

Two things are confirmed and one is falsified.

**Confirmed — the exemption works exactly as specified.** The quality channel
reads µ = −0.0415611, σ = 0.15852 on *all four arms*, bitwise: the static
arms' violation channel is in precisely the same raw space as the PopArt
control's, while their cost channels sit at symlog scale (−7.7, −11.5) where
v65's sit at raw scale (−1.0e5, −3.6e7). **Confirmed — symlog preserved the
relative spread**: σ/|µ| is 0.274 vs 0.281 for latency and 0.2735 vs 0.2736
for memory, i.e. the same underlying plan-to-plan variability, re-expressed.

**Falsified — the pricing.** Under PopArt every channel is divided by its own
σ, so v64b's realized quality:cost pull was λ : 1 = **10 : 1** on both cost
channels. Under `--advantage-norm none` the channels keep their absolute
scales, so the realized pull is λ·σ_q/σ_cost:

| λ | quality : latency | quality : mem |
|---|---|---|
| 10 (v66a) | **0.75 : 1** | **0.50 : 1** |
| 13 (v66c) | 0.97 : 1 | 0.65 : 1 |
| 16 (v66b) | 1.20 : 1 | 0.80 : 1 |
| v64b / v65 (PopArt) | 10 : 1 | 10 : 1 |

So the launched static arms price quality **8–13× more weakly relative to
cost than v64b did**, not 1.2–2× more weakly as first estimated. The nominal
"quality 10–16× the cost weights" is true of the CLI numbers and false of the
objective: symlog compresses the cost channels' dynamic range but leaves them
~50–70× above the bounded violation channel in absolute magnitude, and λ ≤ 16
does not close that gap. Matching v64b's realized 10 : 1 would need
**λ ≈ 134** (against latency) or **λ ≈ 199** (against memory) — equivalently,
`--lambda-cmp`/`--lambda-mem` ≈ 0.05–0.08 at λ = 10.

**Consequence for the battery, and the early read.** The arms as launched
confound the objective's *stationarity* (what the battery is meant to test)
with a large drop in the quality *price* (what it is not). If the static arms
slide, mispricing is the leading explanation and the critic-noise hypothesis
is NOT thereby falsified. This is cheap to detect early, so the arms were left
running rather than killed: **if `lagrangian/frac_violating` climbs and
`mean_raw_q` falls below τ = 0.75 within roughly the first 20 episodes on
v66a/b/c while v65 stays near its ep0 level, the mispricing dominates** — kill
the three static arms and relaunch at λ ∈ {130, 170, 210} (or equivalently
scale the cost weights down by ~13×). If instead the static arms hold `none`
and quality through the ep60–90 window where v64b's H_face crossed its floor,
they are holding at a *weaker* quality price than the arm that collapsed,
which is a strictly stronger result for the static objective than the battery
was designed to produce.

Predictions (i)–(iv) below were registered before this correction; they stand
as written, read through this caveat.

#### 12.10.5 Registered predictions

Registered before any arm produced an episode.

**(i) The critic-noise reading.** If value-net noise at the penalty's scale is
what randomizes the quality advantage, then the static arms — which remove the
1/σ_q amplification and the moving frame — should show a **materially lower
variance of `value_loss/quality`** than v65 (episode-to-episode variance over
a matched window, and lower relative to `value_loss/latency|mem` within the
same arm), and should **hold `approx_prob/none` ≥ 0.9** with violation spikes
that **recover** (the ep24-style snap-back v64b managed exactly once) rather
than accumulate. Falsified if the static arms slide with `value_loss/quality`
variance indistinguishable from v65's.

**(ii) Price sensitivity.** v66b (λ=16) should be **strictly more
conservative** than v66a (λ=10): lower `frac_violating`, higher `mean_raw_q`,
lower `approx_applied/fraction`, and later (or no) onset of any drift. If
v66a slides and v66b does not, the binding variable is the realized price and
sec 12.10.4's λ≈20 follow-up is indicated. If **both** hold, the static
objective is robust across the whole 5–8× realized band and the price is not
the binding variable.

**(iii) Is the causal mask still load-bearing?** v66c (λ=13, **no**
`--lag-causal-mask`) sits between A and B in price, so a monotone
interpolation of A and B is the null. If v66c slides while A and B hold, the
sec-12.7 mask is still load-bearing *even under static scales* — i.e. the
global-credit path onto the shared `OP_NONE` bias survives the removal of
PopArt. If v66c tracks the A/B interpolation, the mask's value was specific
to the PopArt regime and it can be retired.

**(iv) v65 as the PopArt control.** v65 is **expected to slide like v64b**
(drift into the absorber within ~100 episodes), because it fixes only the
quality channel's normalizer while leaving PopArt on the costs, the moving
frame, dual ascent, and the stale-moment coupling. If it does **not** slide,
the discriminating variable is not the normalizer at all but
**dual-ascent-vs-frozen-λ** (v65 is the only arm whose λ still moves) or the
raw-advantage clip acting as a de-facto bound the static arms lack — and the
next contrast is v65 with `--lag-eta 0` against v65 as launched.

**Cross-arm reading rule, fixed in advance.** The headline outcome is the
joint pattern, not any single arm: (A,B hold; C,D slide) ⇒ critic noise +
mask both real, static objective adopted; (A,B,C hold; D slides) ⇒ PopArt was
the whole story and the mask is retirable; (all slide) ⇒ read sec 12.10.4
first (realized price) before concluding the static objective failed;
(all hold, D included) ⇒ 250 eps is too short a window and the battery must
be re-run at 500.

#### 12.10.6 Reading the battery

First checkpoint is ~ep60–90, the window in which v64b's H_face crossed its
floor and `none` began to fall. The keys that decide each prediction:
`value_loss/quality` (and its variance) against `value_loss/latency|mem`;
`approx_prob/none`; `entropy_floor/H_face`; `lagrangian/frac_violating` and
`mean_raw_q`; `kl/approx` (v64b's slide took it from O(0.5) to 10.5);
`lagrangian/mask_fraction` on A/B/D. λ is constant by construction on A/B/C —
if `lagrangian/lambda` ever moves on a static arm, the frozen-λ mechanism is
broken and the arm is void.

## 13. The entropy path audited end to end: H-SIGN **REFUTED**, the metric is population-diluted, and the "collapse to determinism" is mostly an artifact (2026-08-26)

Commissioned to test **H-SIGN** — "somewhere in the entropy path a sign or a
normalization is wrong, so the term does not do what its name says" — against
the characteristic `entropy/approx_head` shape every face run reproduces:
~0.03 at init, a slow rise over tens of episodes, then a fall to near zero.
Read-only audit of `hostperf-caches` @ `0e601ae`; every claim below is either a
quoted `file:line` or a number produced by running the **real** functions
(`unified_face_head.py` loaded by path; `_face_entropy_floor_penalty`,
`_split_entropy_bonus`, `_mask_vertex_logits`, `utils.entropy` `exec`'d
straight out of the source text via `ast`, so nothing is a transcription).
Trajectory numbers are from `collapse_invest/v64b_zneut/v64b_full.csv`
(`38oyqf4g`, ep0–122) and `collapse_invest/v63_anti_none/v6{2,3}_full.csv`.

**Verdicts.** bonus sign **CORRECT** · floor sign **CORRECT** · split sign
**CORRECT** · metric-vs-loss consistency **CORRECT** · ve-head inertness
**CORRECT** · arity/padding denominator **BUG** · floor arithmetic **BUG**
(in sec 12.2's stated numbers, not in the code). No sign is inverted anywhere;
**H-SIGN is refuted**. What is wrong is the *population* the reported
quantity averages over, and the consequences are large enough to change the
reading of sec 12.1 and sec 12.9.

### 13.1 Every site, with its sign

`entropies` (per sample) and `face_ents` (per sample) come out of the vmapped
`evaluate_action_dynamic` at `ppo.py:7346-7358` (slots 1 and 11).

| # | quantity | formula | site | sign in loss | intended | actual gradient |
|---|----------|---------|------|--------------|----------|-----------------|
| 1 | per-face entropy `e` | `e_skip + 1{¬skip}·Σ_s(e_op + branch-masked e_i,e_j / e_ax,e_fn / e_dt)` | `unified_face_head.py:247-294` | — | one face's decision entropy | — |
| 2 | per-face arity `ar` | `gate_face + Σ_s active·1{op≠OP_NONE}` ∈ [1,4] | `unified_face_head.py:250,295` | — | "number of decisions" | **counts only non-NONE slots; the numerator counts skip + all three op softmaxes** |
| 3 | per-step face sums | `(Σ_f lp, Σ_f e, Σ_f ar)` over F slots, invalid faces gated to exactly 0 | `ppo.py:2900-2913, 2955-2957` | — | one env step | valid faces only ✓ |
| 4 | per-sample `face_entropy` | `f_ent / max(f_arity, 1)` | `ppo.py:3140` | — | arity-normalised H | **0/1 = 0 on a step with no live face** |
| 5 | joint per-sample entropy | `vertex_ent + ent_sub/max(sub_len,1) + face_entropy` | `ppo.py:3100,3141` | — | per-head normalisation | ✓ |
| 6 | `entropy_loss` | `mean_B(entropies)` | `ppo.py:7501` | `−` (via 8) | bonus | ✓ |
| 7 | `H_face` (THE metric) | `jnp.mean(face_ents)` — batch mean over ALL samples | `ppo.py:7653, 7662, 7678` | `−` (via 8), `+` (via 9) | mean per-face entropy | **mean over steps, face-less steps enter as 0** |
| 8 | `_ent_bonus` | `w·(H_tot − H_face) + w_f·H_face`; `w·H_tot` if unsplit | `ppo.py:807-830, 7661` | `total_loss … − _ent_bonus` (`ppo.py:7664-7668`) | raise entropy | raises it ✓ |
| 9 | floor hinge | `w_fl·relu(floor − H_face)²` | `ppo.py:683-693, 7674-7680` | `total_loss = total_loss + …` | raise H when below floor | raises it ✓ |
| 10 | logit clamp | `C·tanh(z/C)`, `C = --face-logit-clamp` (default **15**) | `unified_face_head.py:88-93, 214-222`; installed pre-trace `ppo.py:4900-4903` | — | keep dH/dθ alive at saturation | ✓ (attenuation cost noted in 12.1) |
| 11 | logged panel | `entropy_components[5]` → `entropy/approx_head`, `entropy_floor/H_face`, `entropy_floor/penalty = w·max(0,floor−H)²` | `ppo.py:7651-7654, 9271-9286` | — | report what the loss hinges on | **identical expression, same minibatch, reduced by `mean` over (epochs, minibatches) at `ppo.py:8632-8635`** ✓ |

Two structural notes that matter later. (i) `arity` and the entropy numerator
count *different things*: a face that emits three `OP_NONE`s contributes three
op-softmax entropies to the numerator and **1** to the denominator; each
approximation it emits adds 1 to the denominator and its own field entropies to
the numerator. (ii) The face head is the **only** action head
`_scale_output_heads` does not touch (`ppo.py:4443-4505` scales
`vertex_policy`, `pref_proj`, `micro_action_policy`; `face_path_policy.head`
is absent), so it is **not** given the ×0.1 near-uniform init the other heads
get — on top of `apply_face_none_bias`'s +6 on each slot's `OP_NONE` and −6 on
`SKIP` (`common/agent_factory.py:87-111`; v64b and all four v65/v66 arms log
`[factory] face-head IDENTITY INIT: OP_NONE bias +6.0, SKIP bias -6.0`).

### 13.2 Numerical proofs (CPU, real functions, `jax.grad`)

400 Adam steps on the real 94-logit head, scoring stored actions, with the
real loss terms assembled exactly as `ppo.py:7664-7680` assembles them:

```
(a) bonus w=0.05, no split, floor OFF : H 0.2054 -> 4.8520  RISES  (correct)
(a') split wf=0.005,        floor OFF : H 0.2054 -> 4.8520  RISES  (correct)
(b) bonus 0, floor 0.3 w=10, H<floor  : H 0.2054 -> 0.3800  RISES  (correct)
(c) H=2.1479 > floor 0.3: penalty=0.000e+00, max|grad_on-grad_off| = 0.000e+00
(d) d(bonus)/dH_total = +0.050000 == --entropy-weight
    d(bonus)/dH_face  = -0.045000 == (face_w - w); NET face coefficient
                      = +0.005000 == --face-entropy-weight, exactly
    unsplit (None): d/dH_total=+0.05, d/dH_face=0 (face rides the joint term)
(e) ve head, one legal vertex: H = -0.0e+00, max|dH/dlogits| = 0.0e+00, finite
    (3 legal vertices: H = 0.9643, max|dH/dlogits| = 1.37e-01 — the test is live)
```

(a)–(d) close the sign question: the bonus is a bonus, the hinge pushes up,
the hinge is **bitwise** inert above the floor, and the split re-weights the
face component only, preserving direction. (e) closes the second-silent-force
question: with `FORCE_REV_ORDER`'s one-legal-vertex mask
(`_mask_vertex_logits`, `ppo.py:4508-4528`) the ve-head entropy is exactly
zero **and its gradient is exactly zero** — no NaN, no hidden pull. Every
episode of v64b and of the four live arms logs `ve_head=0 macro_vertex=0`,
consistent.

### 13.3 What `entropy/approx_head` actually measures

Sampling the real head with the real init (`+6/−6`, all-legal masks) and
forming exactly `Σ_f e / Σ_f ar`:

| none bias | p_none/slot | **H reported** | E[arity] | p_skip | p_approx/slot |
|---|---|---|---|---|---|
| 0 | 0.250 | 3.104 | 3.239 | 0.002 | 0.748 |
| 2 | 0.711 | 2.712 | 1.875 | 0.002 | 0.292 |
| 4 | 0.948 | 1.066 | 1.167 | 0.002 | 0.056 |
| **6** | 0.993 | **0.232** | 1.024 | 0.002 | 0.008 |
| 8 | 0.999 | 0.051 | 1.003 | 0.002 | 0.001 |
| 10 | 1.000 | 0.023 | 1.000 | 0.002 | 0.000 |

Per-face decomposition at the shipped init: an unskipped all-NONE face carries
`ent = 0.1724, arity = 1` (three op softmaxes + the skip Bernoulli); a
**skipped** face carries `ent = 0.0173, arity = 1` (all slot terms are gated
off by `active`). So the reported number is neither "nats per decision" nor an
entropy over the 94-logit alphabet: it is `(skip + up to 3 op softmaxes +
whatever fields the chosen ops consume) / (1 + #non-NONE slots)`. Between
`p_none = 0.99` and `p_none = 0.25` the raw numerator moves 42× while the
reported ratio moves 13× — the metric is a **compressive (≈3.2×) transform**
of the head's true uncertainty, monotone but not affine, and its scale shifts
with the action distribution itself.

**The padding-denominator trap, one level up.** `approx_prob/*` was fixed in
2026-08 to divide by *valid faces* (`ppo.py:8695-8745`, and the comment there
records the earlier bug: "none 0.9919 measured vs 0.99148 predicted from
padding alone"). `entropy/approx_head` was **not** given the same treatment:
it is `jnp.mean(face_ents)` over the flat (env × step) batch, and a step whose
vertex has **no live face** contributes `0/max(0,1) = 0` — a structural zero,
indistinguishable from a deterministic head. The same run logs the size of
that population every episode as `faces/mean_valid` (`ppo.py:9735`), and it is
**not** stationary.

### 13.4 De-diluting v64b, v63 and v62

`faces/mean_valid` is the mean live-face count per env step; the face-bearing
fraction `f` is `mean_valid / k` for a per-face-bearing-step count `k` that is
unlogged but bounded (below). `H_rep / mean_valid` is therefore the diluted
metric divided by its own population, up to the constant `k`.

v64b (`38oyqf4g`), the endgame:

| ep | H_rep | faces/mean_valid | H_rep÷mean_valid | approx_prob/none | skip | applied |
|---|---|---|---|---|---|---|
| 3 | 0.0283 | 1.212 | 0.023 | 0.982 | 0.005 | 40 |
| 60 | 0.1736 | 1.230 | 0.141 | 0.932 | 0.003 | 151 |
| 84 | 0.9285 | 1.195 | 0.777 | 0.002 | 0.023 | 2368 |
| 88 | 0.5949 | 0.785 | 0.758 | 0.000 | 0.061 | 1557 |
| 90 | 0.5038 | 0.663 | 0.760 | 0.000 | 0.086 | 1265 |
| 92 | 0.2736 | 0.386 | 0.710 | 0.000 | 0.140 | 694 |
| 93 | 0.1403 | 0.190 | 0.738 | 0.000 | 0.142 | 327 |
| 96 | 0.0105 | 0.020 | 0.516 | 0.000 | 0.516 | 13 |
| 110 | 0.0054 | 0.016 | 0.329 | 0.000 | 0.640 | 10 |
| 122 | 0.0026 | 0.013 | 0.196 | 0.000 | 0.800 | 8 |

Between ep84 and ep122 the reported entropy falls **357×** while the face
population falls **91×**. The quotient falls 4×. The "decrease to near-zero
determinism as the policy commits to destruction" is, to first order, **the
disappearance of the decisions being measured**: by ep96 the policy's own
skips have DCE'd the graph to 0.02 live faces per step (so at most 2% of
steps carry any face at all), and the head's per-face entropy on the survivors
is **at least** 0.2–0.5 nats (the quotient is a lower bound: k ≥ 1) — **4–10× ABOVE the 0.05 floor it is simultaneously being
penalised for missing** (`entropy_floor/penalty` is 0.014–0.022 across ep96–122).

v63 makes the point over 420 episodes: from ep80 to ep500 `H_rep` wanders
between 0.019 and 0.14 (7×) while `H_rep ÷ mean_valid` is pinned in
**0.66–0.96** — flat. Over the same 420 episodes `entropy_floor/penalty` sits
at **0.5–0.79 continuously**, i.e. the hinge fires at ~full strength
(`dP/dH = −2·10·(0.3−0.04) ≈ −5.2`) against a head whose per-face entropy is
**at least** 0.66–0.96 nats, i.e. **≥2.2× above the 0.3 floor it is
supposedly restoring**. v62 is the
same: ep90→ep121 `H_rep` 1.361→0.156 (8.7×) with `mean_valid` 1.220→0.141
(8.6×) and the quotient constant at ~1.1, and the penalty re-igniting to
0.10–0.35 at ep119–121 on a head ≥3.5× above the floor.

So the shape decomposes as:

* **the fall to ~0 is an ARTIFACT** (population, not policy) in all three runs;
* **the slow rise is REAL** — `mean_valid` is flat at 1.15–1.24 from ep3 to
  ep84 in v64b, so the ep3→ep84 rise 0.028→0.93 is a genuine 33× rise in
  per-face entropy, matching `approx_prob/none` 0.982→0.002 decision-for-decision;
* **the low init is REAL but is not "94 logits near uniform"** — see below.

**The init level.** For the observed ep3 marginals (`p_approx/slot = 0.0129`,
`p_skip = 0.0054`) the head model's *minimum* reported H is **0.222** (zero
per-face logit spread) and 0.326 at the spread that reproduces the observed
approx rate; no `(none-bias, spread)` pair in a 25×25 grid reproduces
`H = 0.028` **and** the marginals (best fit misses the approx rate by 6.4×).
Dilution cannot close it either: the uniform ceiling of the metric is ~3.1
nats, so ep84's `H_rep = 0.93` forces `f ≥ 0.30`, hence `k ≤ 4.0`, hence
`f(ep3) ≥ 0.30` and a de-diluted init of **at most 0.09** — still 2.5× under
the model floor. The residual is **Jensen**: the reported quantity is a
*mean of per-face entropies* while `approx_prob/*` is the *marginal over
faces*, and a head that is decisive per face but disagrees across faces has a
low mean and a spread marginal (exactly what sec 12.1 measured post-collapse:
"0.03–0.09 nats while the op MARGINALS are near-uniform"). That it is already
true at **init** is explained by 13.1(ii): the face head never receives the
×0.1 output scaling that makes the other heads near-uniform, so its 94 logits
carry full orthogonal-init magnitude from step 0.

### 13.5 The floor-target arithmetic, corrected

Sec 12.2 states "H = 0.3 per face decision means ~7% approx per decision ≈
8–18 approx ops per TLM plan". The 0.3 is in the units of the metric the hinge
reads, not of one op softmax. Solving `Σe/Σar = floor` on the real head:

| floor | none bias | p_none/slot | p_approx/slot | approx/face | approx per plan (150 / 160 / 250 faces) |
|---|---|---|---|---|---|
| 0.30 | 5.66 | **0.9897** | 0.0108 | 0.032 | **4.9 / 5.2 / 8.1** |
| 0.05 | 8.01 | 0.9990 | 0.0009 | 0.003 | 0.40 / 0.43 / 0.67 |

For comparison, `H = 0.3` for a *single* 4-way op softmax gives
`p_none = 0.938` — the number sec 12.2 quotes. The metric is ~3.4× that
single softmax (three slots plus the skip Bernoulli, over arity ≈ 1), so the
true `p_none` at the 0.3 floor is **0.990, not 0.938**. Sec 12.2's *conclusion*
survives — 4.9–8.1 approx ops per plan is still structurally inside the
violation regime, so floor 0.3 was still mis-targeted and still an igniter —
but its stated `p_none` and its 8–18 range were derived from a different
quantity than the one the hinge reads, and the corrected range is lower.

Worse, the floor's meaning is **not stable within a run**: because `H_face` is
population-diluted, the same 0.05 corresponds to a per-face entropy of ~0.05 at
`mean_valid = 1.2` and to ~3.2 nats at `mean_valid = 0.016`. A floor is a
static gate on a quantity whose scale moves 60× — and it moves *because of the
policy's own destruction*, which is a feedback path, not a measurement.

### 13.6 Verdicts

| item | verdict | evidence |
|---|---|---|
| bonus sign | **CORRECT** | 13.2(a)(a'); `−_ent_bonus` at `ppo.py:7664-7668` |
| floor sign | **CORRECT** | 13.2(b)(c); `+ w·relu(floor−H)²` at `ppo.py:7677` |
| split sign | **CORRECT** | 13.2(d): net face coefficient == `--face-entropy-weight` exactly |
| metric vs loss | **CORRECT** | same `jnp.mean(face_ents)` expression feeds hinge, bonus and panel (`ppo.py:7653/7662/7678`), same (epoch,minibatch) mean; `entropy_floor/H_face == entropy/approx_head` byte-for-byte in 739/739 populated rows across v62/v63/v64b |
| ve-head inertness | **CORRECT** | 13.2(e): H and dH/dθ both exactly 0 under the rev-pinned mask |
| **arity / padding denominator** | **BUG** | (i) `mean(face_ents)` averages face-less steps in as structural zeros — 91× population swing inside v64b, 7× reported-metric swing at constant de-diluted entropy across 420 v63 episodes; (ii) `arity` counts only non-NONE slots while the numerator counts skip + all op softmaxes, making the metric a 3.2×-compressive, action-dependent transform |
| **floor arithmetic** | **BUG (dossier, not code)** | 13.5: `p_none(H=0.3) = 0.990` not 0.938; 4.9–8.1 approx/plan not 8–18; and the target is non-stationary in `mean_valid` |

**What this does NOT explain.** The anti-none slide itself. Sec 12.2's F1/F3
mechanism is untouched: the hinge really did fire continuously from ep0 in
v62/v63 (13.4 confirms the penalty trace), and the rise from ep3 to ep84 in
v64b is a real 33× rise in per-face entropy. H-SIGN was a reasonable
hypothesis for "neither exploratory nor exact, drifts then commits" and it is
**refuted**: the drift is real and the commitment is mostly the metric.

### 13.7 Minimal fix, and the four running arms

**Fix (≈10 lines, `ppo.py`).** Return the per-sample pair `(f_ent, f_arity)`
in slot 11 of `evaluate_action_dynamic` instead of their ratio, and in the
loss aggregate as a **ratio of sums over the batch**:

```
H_face_batch = jnp.sum(face_ents_raw) / jnp.maximum(jnp.sum(face_arities), 1.0)
```

Use that single quantity for `_entropy_components[5]` (`ppo.py:7653`) and for
the floor hinge (`ppo.py:7678`). **Leave `total_entropy` / the bonus alone**
(`ppo.py:3141`, `7662`): the bonus is a per-sample objective term and a step
with no decision correctly buys no exploration, so changing it would alter the
PPO objective rather than a diagnostic. Ship behind
`--face-entropy-agg {mean,valid-weighted}` defaulting to `mean` so the
flag-off path stays bit-identical, per the sec 12.10 shipment gate. Log
`faces/frac_steps_with_faces` alongside `faces/mean_valid` to pin `k` (the
one number this audit had to bound rather than read).

**Do the four arms (62072/62073/62074/62075) need relaunching? NO.**

1. No sign is wrong, so the objective they are optimising is the intended one.
2. All four run `--face-entropy-floor 0.05 --face-entropy-floor-weight 10
   --face-logit-clamp 15 --face-entropy-weight 0.005`,
   `ALPHAGRAD_FACE_NONE_BIAS=6` — identical to v64b. Floor 0.05 sits *at* the
   init level (v64b crossed it at ~ep28 and the penalty was exactly 0 from
   ep31 to ep95), so the hinge is inert through the entire window the battery
   is being read on. It only re-arms in the endgame absorber, where the run is
   already decided.
3. The defect is in a **diagnostic** and can be corrected **post hoc from keys
   already being logged**: `entropy/approx_head ÷ faces/mean_valid`.

**Amend the sec 12.10.6 reading rule accordingly**: `entropy_floor/H_face` is
only comparable across episodes at constant `faces/mean_valid`. Read the
quotient, and treat any drop in `H_face` that tracks a drop in
`faces/mean_valid` as a graph-destruction signal, **not** an entropy collapse.
As of 2026-08-26 ~18:40 all four arms are at `approx_head` ≈ 0.85–1.05 after
~140 episodes (62072: 1.029, 62073: 0.853, 62074: 1.001, 62075: 0.965), i.e.
already past the crossing and in the v62-shaped high-entropy regime — the
`mean_valid` quotient (wandb only; the stdout `[entropy]` line does not carry
it) is what says whether that is exploration or an emptied graph.

**Also worth recording**: the first three logged episodes of every run report
`approx_head=nan macro_vertex=nan ve_head=nan` (warm-up rows, before any
update contributes metrics). Not a defect, but it means `_step` 0–2 are not
data.

## 14. Mechanism synthesis for the whole campaign: the two absorbers, the instrument bug that makes one of them pay, and what is now dead (2026-08-26)

Commissioned as the third of three parallel pieces of work on the v57→v66c
record. The factual base is taken as given and is **not** re-derived here:

* `docs/run_analysis/params_matrix.md` (+ `.csv`) — what every run was
  configured to do, the consecutive-run delta list, the never-set defaults and
  the twelve launcher-vs-runtime contradictions C1–C12.
* `docs/run_analysis/how_runs_ran.md` (+ `run_analysis/figs/*.png`,
  `run_analysis/data/*.csv`) — landmarks, phase timings, live-arm status.
* Sections 0–13 of this dossier, and `docs/CLEAN_DESIGN_AUDIT.md`.

Everything new below is computed from `run_analysis/data/*.csv` (the full
`scan_history` of all twelve runs), from the STDOUT logs in
`/Users/assmuth/dsnn/`, and from read-only inspection of
`src/alphagrad/approx/env.py` at `0e601ae`. No GPU job was touched; jobs
62072–62075 were running throughout.

Vocabulary, following `how_runs_ran.md` §2:

* **group B** — v58b, v60, v61, v62, v63, v64b: cross `approx_prob/none < 0.5`
  at ep38–71, then lose 88–99 % of their live faces as `approx_prob/skip`
  rises to 0.20–1.00, and never leave that state (up to 420 further episodes).
* **group C** — v65, v66a, v66b, v66c (the four live arms): cross at ep81–112,
  `faces/mean_valid` stays at 1.19–1.24 = at or above their own base,
  `skip` stays at 0.0002–0.004, no entropy collapse — but quality still
  degrades (v66a −0.021, v66c −0.105, v66b +0.233, v65 +0.594).

---

### 14.1 The finding that reframes everything: the anti-destruction gate's floor is measured with a different instrument, and under-floors by 13 %

`ALPHAGRAD_QUALITY_GATE_MIN=0.05` is set in all twelve launchers. Its job
(`_apply_quality_gate`, `env.py:1986`) is to make destruction cost-neutral: a
plan whose measured quality falls under 0.05 has its latency and memory
**floored at what its own elimination order would cost done exactly**, so
"destroy the graph" cannot win the cost channels. The docstring is explicit:
*"destruction pays what exact computation pays, so it gains nothing"*.

It does not. The gate prints its own before→after on every 50th clamp, and the
numbers say the floor is 13–15 % below what an honest exact plan is charged in
the same run:

```
v63  (61866)  quality gate CLAMP #1   : q=0.0000 < 0.05; lat  65.8 -> 138.5us
              quality gate CLAMP #50  : q=0.0000 < 0.05; lat  73.5 -> 137.7us
              quality gate CLAMP #2050: q=0.0000 < 0.05; lat 102.8 -> 133.6us
v64b (61983)  quality gate CLAMP #1   : q=0.0000 < 0.05; lat  67.9 -> 137.9us
              quality gate CLAMP #2100: q=0.0000 < 0.05; lat  75.2 -> 136.5us
              quality gate CLAMP #2550: q=0.0000 < 0.05; lat  82.9 -> 137.5us
```

Against `mean_latency_ns` of **155–160 µs** for the identity/exact plans of the
same runs (ep3–10 means: v63 159.0, v64b 158.3, v66a 155.1, v65 157.6 µs).

Two things follow, and both are load-bearing.

**(a) A DCE'd graph really is ~2.3× cheaper.** The pre-clamp readings are
53–103 µs against 156 µs. One `SKIP` on the TLM graph removes the work (the
project-memory "skip cliff": one SKIP DCEs the graph, 37 µs, cos 0). The gate
is the only thing standing between the policy and that prize.

**(b) The floor is measured with a different instrument than the thing it
floors.** `_measure_exec_cost` (`env.py:1912-1952`) times the reference as the
**median of 3 laps of 20 back-to-back executions** — a throughput timing that
amortises per-call dispatch. The plan's own latency comes from the campaign
measurement path — `--num-data-points 5 × --reps-per-point 4`, each at
`--latency-inner-reps 5` (C11 records that the standing project rule is 50),
reduced by `_aggregate_samples(..., want_top_quartile=True)` = a plain median
(`env.py:2686-2705`). The `_measure_exec_cost` docstring says the protocol is
*"shared by the global rev reference and the per-order floor so the clamp
compares like with like"* — which is true **between the two references** and
false **between the floor and the plan being floored**.

Net effect, present in every one of the twelve runs: **a plan that destroys the
gradient is charged 132–139 µs where an honest exact plan is charged 155–160 µs
— a guaranteed −13 % latency and −1.4 % memory bonus for destruction.**

This is the correct location of the "destruction pays" claim, which §12.3
refuted at the level of the *reward design* (and was right to: at λ=10 the
latency of the drift state gets *worse*). It pays at the level of the
*instrument*, and only for the one destruction mode that makes the graph
cheap — `SKIP` — not for the approximation-heavy mode.

That asymmetry is the hinge of the whole campaign, and it is visible in the
clamp prints of the live arms. v66a's late clamps read:

```
v66a (62072)  quality gate CLAMP #400: q=-0.2505 < 0.05; lat 181.4 -> 181.4us
              quality gate CLAMP #450: q= 0.0000 < 0.05; lat 179.4 -> 179.4us
              quality gate CLAMP #400: q= 0.0000 < 0.05; lat 189.8 -> 189.8us
              quality gate CLAMP #450: q=-1.0000 < 0.05; lat 202.5 -> 202.5us
```

`X -> X`: the floor never binds, because v66a's destroyed plans are destroyed
by ~180 approximations and are **more expensive** than the floor. Group B's
destroyed plans are destroyed by `SKIP` and are cheaper than the floor, so they
collect the bonus every time.

**Falsifier / fix.** Measure the floor with the same protocol as the plan (or
raise `--latency-inner-reps` to the project-standard 50, at which the two
protocols converge). Until then, treat every clamped-plan cost number in
v57–v66 as carrying a −13 % artefact, and treat the group-B absorber's
"136 µs" as **the floor's own value, not a measurement of anything the policy
achieved**.

---

### 14.2 Q1 — why group C does not destroy the graph

#### 14.2.1 Group C is not a different mode; it is group B's drift window, held open

The two groups are not distinguished by the state they occupy. Group B passes
straight through the group-C state:

| window | none | skip | faces/mean_valid | applied/batch | mean_q (sd over eps) | lat |
|---|---|---|---|---|---|---|
| v63 ep64–77 (post-cross, pre-skip) | 0.43→0.00 | 0.002–0.014 | 1.119 | 2319 | +0.016 (0.261) | 205.6 µs |
| v64b ep71–87 (post-cross, pre-skip) | 0.47→0.01 | 0.004–0.013 | 1.117 | 1873 | +0.121 (0.271) | 208.2 µs |
| v66a ep100–199 (trailing) | 0.000 | 0.0004 | 1.242 | 2895 | −0.021 (0.126) | 179.1 µs |
| v66c ep100–199 (trailing) | 0.000 | 0.0012 | 1.229 | 2826 | −0.105 (0.076) | 300.5 µs |

Same census, same face population, same approximation density, same
already-destroyed quality. The difference is what happens **next**: in group B
`skip` leaves its floor within 6–14 episodes of `none` reaching ~0 (v63 ep78,
v64b ep88), the graph is DCE'd 3–5 episodes later (v63 ep83, v64b ep93), and
from that point the run is over. In group C, 100+ episodes past the same point,
`skip` has not moved.

#### 14.2.2 The skip rise is an *active* learned change, and so is its absence

Because `OP_NONE` and the other ops share one softmax per slot, `none → 0`
mechanically renormalises every other op upward. That alone predicts
`p_skip → p_skip · (1−p_none^after)/(1−p_none^before)`. Measured against it:

| run | window | none | skip | skip predicted by renormalisation alone | observed / predicted |
|---|---|---|---|---|---|
| v63 | ep58→82 | 0.964→0.000 | 0.0016→0.1441 | 0.0457 | **3.15×** |
| v64b | ep77→95 | 0.163→0.000 | 0.0037→0.3651 | 0.0045 | **81.8×** |
| v66a | ep89→131 | 0.190→0.000 | 0.00053→0.00053 | 0.00065 | **0.81×** |
| v66b | ep88→142 | 0.127→0.121 | 0.00159→0.00111 | 0.00161 | **0.69×** |
| v66c | ep87→141 | 0.293→0.017 | 0.00161→0.00172 | 0.00223 | **0.77×** |
| v65 | ep106→172 | 0.751→0.462 | 0.00426→0.00377 | 0.00920 | **0.41×** |

Group B raised the `SKIP` logit by 3–82× beyond renormalisation. Group C
**lowered** it (0.41–0.81×). Both are learned; neither is an artefact of the
`none` collapse. So the question is well posed: what makes the per-op
corrective signal on `SKIP` survive in one group and not the other?

#### 14.2.3 The two forces on SKIP, with numbers

A `SKIP` action buys a cost bonus and pays a quality penalty.

*Bonus* (from §14.1): the plan is floored at 136 µs where an honest plan is
charged 157 µs. Under PopArt (group B, σ_lat ≈ 28 944 ns) that is
21 000/28 944 = **+0.72 σ** of latency advantage, at head weight 1. Under the
static arms' symlog scale (σ_lat = 2.119) it is ln(157/136) = **+0.14**.

*Penalty*: the violation channel moves from the plan's current `v` to its cap
0.75 (q clipped at 0), scaled by λ and by the channel's normaliser:

* group B (PopArt): `λ/σ_q · (0.75 − v)` = 10/0.1375 · (0.75−v) = **72.7·(0.75−v)**
* group C (raw scale): `λ · (0.75 − v)` = **10–16 · (0.75−v)**

Both penalties **vanish identically when the batch's violation saturates at the
cap** — and only then does the bonus decide. Whether the batch saturates is
directly measurable:

| run | window | `lagrangian/mean_violation` | sd across episodes | `frac_violating` |
|---|---|---|---|---|
| v61 | ep100–140 | **0.7500** | **0.0000** | 1.000 |
| v64b | ep100–140 | **0.7500** | **0.0000** | 1.000 |
| v63 | ep100–140 | 0.7289 | 0.0332 | 0.971 |
| v66a | ep160–200 | 0.7347 | **0.0984** | 0.983 |
| v66b | ep160–200 | 0.4231 | **0.1275** | 0.845 |
| v66c | ep160–200 | 0.8152 | **0.0682** | 0.990 |
| v65 | ep160–200 | 0.1902 | **0.0858** | 0.389 |

Group B's absorbers are *pinned at the cap with zero variance* — the quality
channel emits exactly no contrast, so `SKIP` is unpunished and collects +0.72 σ
of floored latency for free. Group C sits near the cap **in the mean** but with
a live spread of 0.07–0.13, so a `SKIP` still moves a plan measurably further
into violation and is still punished at 10–16 × that move.

The same reading in the quality channel itself (rolling 15-episode sd of
`mean_quality`):

```
ep      60     80    100    120    140    160    180    195
v64b  0.076  0.293  0.098  0.000  0.000  0.000  0.010  0.013
v63   0.077  0.136  0.104  0.035  0.023  0.034  0.039  0.052
v66a  0.075  0.102  0.147  0.156  0.210  0.113  0.111  0.150
v66b  0.075  0.090  0.174  0.189  0.199  0.221  0.162    —
v66c  0.075  0.089  0.218  0.158  0.209  0.243  0.104  0.079
v65   0.070  0.111  0.088  0.121  0.202  0.132  0.105  0.141
```

Group B's contrast dies (v64b to *exactly* 0.000 for ~60 consecutive episodes);
group C's does not, and is not trending down after 115 episodes of drift.

Note the ordering: v64b's contrast is still 0.098 at ep100, i.e. **after** the
faces are gone (ep93). Contrast death is the *consequence* of graph
destruction, not its cause. The cause is the loss of the per-op corrective
signal on `SKIP` while contrast still existed.

#### 14.2.4 The single factor that separates the two groups

Six candidate factors, tested against the 12-run membership:

| candidate factor | in group B | in group C | verdict |
|---|---|---|---|
| symlog on the cost channels | no | v66a/b/c only (**v65 has `--no-symlog`**) | **not necessary** |
| `--advantage-norm none` | no | v66a/b/c only (**v65 is PopArt**) | **not necessary** |
| frozen λ (`--lag-eta 0`) | no | v66a/b/c only (**v65 does dual ascent**) | **not necessary** |
| `--lag-causal-mask` | v64b **yes**; v61/62/63 no | v65/66a/66b yes; **v66c no** | **neither necessary nor sufficient** |
| `--adv-winsorize 3` | v64b **yes**; v63 no | v65 yes; v66a/b/c no | **neither** |
| `--face-entropy-floor 0.05` | v64b **yes**; v62/63 = 0.3; v61 = 0 | all four | **necessary-looking, not sufficient** |
| **quality advantage computed on the RAW scale** (not divided by σ_q) | **none of the six** | **all four** | **perfect separation, 12/12** |

v65 gets the raw scale from `--lag-raw-viol-adv`; v66a/b/c get it from
`--advantage-norm none`. No other flag partitions the campaign this way.

**And the contrast is causal, not merely correlational.** Per the params
matrix §5, v64b → v65 changed `--episodes 500→250` and added
`--lag-raw-viol-adv`, *and nothing else* — same SHA family (`6d917fc` →
`0e601ae`, flag-off bit-identical on 248 metrics per §12.10.2), same node
(pgi15-gpu16), same seed 250197, same 16 envs. The two histories are identical
to three decimals for the first ~25 episodes and then separate:

```
        v64b                                  v65
ep  3   none 0.982 q +0.807 H 0.028 kl 0.45 | none 0.982 q +0.807 H 0.028 kl 0.45
ep 20   none 0.980 q +0.824 H 0.042 kl 0.52 | none 0.981 q +0.824 H 0.041 kl 0.51
ep 40   none 0.972 q +0.673 H 0.057 kl 0.66 | none 0.975 q +0.669 H 0.051 kl 0.60
ep 60   none 0.932 q +0.733 H 0.174 kl 1.73 | none 0.976 q +0.844 H 0.056 kl 0.58
ep 70   none 0.531 q +0.642 H 0.712 kl 15.5 | none 0.972 q +0.885 H 0.077 kl 0.74
ep 90   none 0.000 q -0.106 H 0.504 kl 1.46 | none 0.952 q +0.846 H 0.113 kl 1.15
ep120   none 0.000 q  0.000 H 0.005 kl 0.15 | none 0.390 q +0.381 H 0.897 kl 9.44
ep190   none 0.000 q  0.000 H 0.014 kl 0.16 | none 0.454 q +0.802 H 0.800 kl 9.22
```

At ep60 the flag has already cut the per-face entropy rise 3.1× (0.174 vs
0.056) and the joint-ratio KL 3.0× (1.73 vs 0.58). Crossing moves 71 → 112
(+58 %); absorption at ep93 becomes no absorption at ep203.

**One flag moved a run from group B to group C.** That is the campaign's
cleanest single result and it is a *scale/noise* result, not a *pricing*
result — which matters, because §12.9.3 already refuted the pricing story
(H-ZNEUT) from the other direction.

#### 14.2.5 What the raw scale actually changes: magnitude, not signal-to-noise

Dividing the quality advantage by σ_q scales the true violation signal and the
critic's error by the same factor, so it cannot change that channel's SNR. What
it changes is the **magnitude of the quality term relative to everything else in
the loss** — the cost channels, the entropy bonus, and the PPO ratio at which
the negative-advantage branch stops being clipped (the §12.2 F3 mechanism).

Measured directly from the `diag/estim_return_raw_quality` and
`diag/value_raw_quality` series over each run's own pre-crossing window
(ep10 → first `none<0.5`):

| run | \|A_raw\| mean | per-unit coefficient | typical per-step \|quality term\| |
|---|---|---|---|
| v62 | 0.0109 | λ/σ_q = 62.9 | **0.682** |
| v63 | 0.0107 | λ/σ_q = 72.7 | **0.776** |
| v64b | 0.0116 | λ/σ_q = 73.5 | **0.855** |
| v65 | 0.0112 | λ = 10.2 | **0.114** |
| v66a | 0.0098 | λ = 10.0 | **0.098** |
| v66b | 0.0100 | λ = 16.0 | **0.160** |
| v66c | 0.0098 | λ = 13.0 | **0.127** |

A clean 5–8× separation, with no overlap, and it matches group membership
exactly. (v61 sits at 0.169 with λ=1.53 and is still group B — it is the
λ-too-low case §12.3 already diagnosed: at λ=1 destruction was *priced* to pay,
and v61 went straight to 100 % SKIP by ep35. It fails for a different, already
closed reason.)

The corresponding critic-error claim in F3 needs a units correction. The raw
per-head MSE `value_loss/quality` reads 0.0035 on v66a and 0.451 on v65
pre-crossing — 128×. But v65's quality head lives in PopArt-normalised space,
so its MSE carries a 1/σ_q² = 1/0.128² = 61× unit factor. Converting:

```
v65  0.4506 × σ_q²(0.01648) = 0.00743  (raw units)
v66a 0.00352                = 0.00352  (raw units)
                       ratio = 2.11×
```

Over the full ep10–190 window the ratio is 1.7×. **The genuine improvement in
critic accuracy on the quality channel is ~2×, not 150×** — the headline
figure is ~61× unit conversion times ~2× real. §12.9.4's diagnosis survives in
direction but its magnitude was overstated by the units.

#### 14.2.6 Q1 answer

> Group C avoids graph destruction because its quality-channel advantage is
> computed on the raw, un-normalised scale, which shrinks the per-step quality
> term 5–8× (0.68–0.86 → 0.098–0.160). That slows the anti-`OP_NONE` drift
> enough (crossing 64–71 → 81–112) that the batch never reaches the
> zero-variance saturated state (`mean_violation` 0.7500, sd 0.0000) in which
> the quality penalty on `SKIP` vanishes identically. While a live violation
> spread survives, `SKIP` keeps paying 10–16 × its own deterministic damage,
> and the head keeps its logit actively suppressed (0.41–0.81× of pure
> renormalisation). Group B saturated, `SKIP` became free, and the quality
> gate's mis-measured floor then *paid* +0.72 σ of latency advantage for taking
> it — which is why the group-B absorber is not merely gradient-free but a
> genuine local optimum, stable for up to 420 episodes.

Not the discriminator: symlog, `--advantage-norm none`, frozen λ, the causal
mask, the winsorize (each present in group C members and absent in others, or
present in group B members). The entropy floor at 0.05 is common to all four
group-C arms and to v64b, so it is necessary-looking but demonstrably not
sufficient.

---

### 14.3 Q2 — what degrades quality in group C, with an intact graph and a quiet critic

#### 14.3.1 It is cheap-but-numerous, by two orders of magnitude

`approx_applied/total` is per 16-env batch; divide by 16 for per plan.

| run | window | applied/batch | **per plan** | quant : diag : compress (op prob) | quality |
|---|---|---|---|---|---|
| all | ep3–10 | 42–44 | **2.6–2.8** | — | +0.80…+0.84 |
| v65 | trailing 20 | 782 | **49** | 0.102 : 0.278 : 0.165 (none 0.452) | +0.594 |
| v66b | trailing 20 | 2475 | **155** | 0.389 : 0.387 : 0.224 | +0.233 |
| v66c | trailing 20 | 2826 | **177** | 0.339 : 0.337 : **0.323** | −0.105 |
| v66a | trailing 20 | 2895 | **181** | 0.389 : 0.387 : 0.224 | −0.021 |
| v63 | trailing 20 | 132 | 8 (graph gone) | 0.287 : 0.281 : 0.167 (skip 0.264) | −0.001 |

The graph carries ~115 live faces per plan (FACE_LATENT §1). At 155–181
applied ops per plan, essentially every face is approximated in every slot that
will take one — `approx_applied/fraction` is 0.60–0.65 against 0.78 at ep3–10
when there were 2.7 ops per plan. This is not a handful of catastrophic
choices; it is saturation coverage of cheap individually-survivable
approximations whose composition destroys the gradient.

Within-arm, over each arm's own drift window, more approximations is
monotonically worse quality:

```
corr(mean_quality, approx_applied/total)   v66c −0.738  v66b −0.612  v66a −0.415  v65 −0.231
```

#### 14.3.2 No arm is winning the cost channel — the density is bad for latency too

`corr(approx_applied/total, mean_latency_ns)` over the same windows is
**+0.86 (v66c), +0.48 (v66b), +0.51 (v65), +0.32 (v66a)** — every additional
approximation makes the plan *slower*. Bucketing every episode of the live arms
by approximation count and normalising latency to that run's ep3–10 mean:

| arm | 20–60 ops | 200–1 000 ops | >1 000 ops |
|---|---|---|---|
| v66a | 1.005 × (q 0.79) | 1.022 × (q 0.64) | **1.356 ×** (q −0.04) |
| v66b | 0.998 × (q 0.79) | 1.035 × (q 0.69) | **1.106 ×** (q 0.17) |
| v66c | 1.000 × (q 0.79) | 1.006 × (q 0.69) | **1.385 ×** (q 0.12) |
| v65 | 1.005 × (q 0.79) | 1.107 × (q 0.54) | — |

There is no bucket, in any live arm, where approximation is cheaper than the
exact plan. **Group C loses on both axes simultaneously; it is not a trade.**
Trailing-20 latency is 179.1 / 179.6 / 300.5 / 171.3 µs against a 155–160 µs
identity — 15 % to 92 % *worse*.

Group B's trailing 133–136 µs is not a counter-example: §14.1 shows it is the
quality gate's own floor value, printed by the clamp itself.

v66c's 300 µs deserves naming, because it is a clean within-arm demonstration.
Between ep152 and ep194 its compress probability goes 0.055 → 0.324 and its
applied-compress count 134 → 912, while diag applications collapse 40 → 26:

```
ep152  lat 171.5us  q +0.486  comp_p 0.069  applied_compress  134
ep164  lat 276.7us  q -0.027  comp_p 0.203  applied_compress  688
ep188  lat 302.0us  q -0.150  comp_p 0.324  applied_compress  881
ep194  lat 294.8us  q -0.149  comp_p 0.324  applied_compress  912
```

Compress is the expensive op on this graph, and v66c walked into it and lost
0.6 of quality and 72 % of latency in 40 episodes. Note that v66c had, at
ep140–152, a genuinely respectable state (q +0.37…+0.49 at 171 µs) and left it.

#### 14.3.3 Q2 answer

> Group C degrades because it converges on **saturation-density approximation**
> — 155–181 ops per plan against 2.7 at initialisation — in a regime where each
> additional op subtracts quality *and* adds latency. The op mix is
> near-identical between v66a and v66b (quant 0.389 / diag 0.387 / compress
> 0.224); what differs is the *count*, which λ controls: λ=16 realises 155
> ops/plan and q +0.233, λ=10 realises 181 ops/plan and q −0.021. v66c is the
> outlier in *mix*, not count, and its compress share is what carries its 300 µs.
> No arm beats the identity plan on latency at any approximation density, so
> nothing here is a quality-for-speed trade: it is a loss on both axes. The
> critic being 2× (not 150×) more accurate did not help, because the thing the
> head cannot do is tell *which* of 115 faces is safe to approximate — the §4
> representation deficit (probe R² 0.03–0.18 against a 0.48–0.67 bar), which
> none of v61–v66 touched.

**And the win it is failing to find is real and was on the table from episode
zero.** v58b is the only 16-env run that logged per-plan order statistics
(because C8 dropped `--lean-logging` from that one launcher). At its **ep0**,
with ~20 stray approximations across all 16 plans and `measure/quality/worst_ep`
= 0.884 (i.e. *every* plan at identity quality), `measure/latency_ns/best_ep`
= **91.2 µs** against a median of 157.5 µs. The all-time best reaches 76.2 µs
by ep~50. v57 corroborates without any min-order-statistic bias — it ran
`--num-envs 1`, so each episode is one plan measured once, and 14 of its 162
episodes have `q ≥ 0.5` **and** latency < 0.9 × base:

```
ep 41   101 µs   q 0.885   1 approximation
ep117    96 µs   q 0.882  14 approximations
ep155   108 µs   q 0.885  23 approximations
ep162   123 µs   q 0.885  63 approximations
```

Against a v57 latency distribution of mean 155.9, sd 15.3 µs, so 96 µs is 4 σ
below and 101 µs is 3.6 σ below; two such points in 162 episodes is not the
measurement noise. **Every other run in the campaign has zero such episodes.**
(v61 has one, at 520 applied ops, and is not the same phenomenon.)

So a ~35–40 % latency win at *no measurable quality cost* exists on this target
and is reachable with **1–15 approximations per plan** — and both instruments
that could see it found it by random exploration inside the first fifty
episodes. Every learner in the campaign then walked away from it, in the
direction of 155–2 900 approximations per batch, where both axes are worse.

Caveats stated honestly: the v58b number is a min over 16 plans and is
downward-biased by measurement noise (a 10 % sd gives ≈ −18 % for a min of 16;
the observed gap is −42 %, so selection accounts for at most half of it); the
v57 numbers are unpaired single measurements on a machine whose state moves
~18–20 % (project memory, paired-measurement rule), which is why the 4 σ
framing and the two independent instruments matter more than any single point.

---

### 14.4 Q3 — is the monotone delay a mechanism improvement or just weaker forcing?

The crossings are 38 (v58b), 64 (v61, v62, v63), 71 (v60, v64b), 81/82/83
(v66c/b/a), 112 (v65). Normalising against the actual per-step pressure toward
approximation, measured in each run's own pre-crossing window:

| step | what changed | dense pro-approximation force | quality restraint (§14.2.5) | crossing | bought |
|---|---|---|---|---|---|
| v62 → v63 | face entropy bonus 0.05 → **0.005** (10× cut) | floor penalty 0.404 → 0.604 (both hinge hard) | 0.682 → 0.776 | 64 → 64 | **0 episodes** |
| v63 → v64b | floor 0.3 → **0.05** (+ mask + winsorize) | penalty 0.604 → **0.0002** (hinge goes inert) | 0.776 → 0.855 | 64 → 71 | **7 episodes** |
| v64b → v66a/b/c | static objective (5 simultaneous changes, C6) | penalty 0.0001 (unchanged, inert) | 0.855 → 0.098–0.160 | 71 → 81–83 | **10–12 episodes** |
| v64b → v65 | `--lag-raw-viol-adv` **only** | penalty 0.0001 (unchanged, inert) | 0.855 → 0.114 | 71 → 112 | **41 episodes** |

Read this in two halves, because they answer differently.

**The first half is weaker forcing, and that is all it is.** The 0.3 entropy
floor fired continuously at penalty 0.40–0.60 from ep0 in v62/v63 (§12.2 F1,
§13.4); dropping it to 0.05 makes the hinge bitwise inert (penalty 2e-4) and
bought 7 episodes. Cutting the face entropy bonus 10× bought nothing at all.
So the igniter fix delayed the crossing by ~11 % and changed nothing else —
v64b absorbed exactly like v63, 10 episodes later. **We bought time, not
immunity, from ALT-3.** Say it plainly: the entropy-floor fix is worth 7
episodes out of 500.

**The second half is not weaker forcing.** From v64b onward the dense
pro-approximation force is *bitwise identical* across v64b, v65, v66a, v66b,
v66c — same `--face-entropy-weight 0.005`, same inert floor, same
`FACE_NONE_BIAS=6`, same clamp. The restraint got **weaker**, not stronger: the
static arms' realised quality:cost pull is 0.75:1 to 1.20:1 against v64b's 10:1
(§12.10.4, an 8–13× reduction), and v65's raw-advantage path cut the quality
term 7.5×. Under any pricing account, weaker restraint at constant forcing must
cross *earlier*. It crossed **10 to 41 episodes later**, and one of the four
arms stopped destroying the graph altogether.

> **Q3 answer.** The delay is mechanism up to v62→v63→v64b only in the trivial
> sense that the igniter was real and its fix was worth 7 episodes. From v64b
> onward the forcing is constant and the restraint is weaker, so the additional
> 10–41 episodes cannot be "we forced it less" — they are the removal of a term
> that was destabilising the policy rather than steering it. That is a genuine
> mechanism change, and it is corroborated by the qualitative outcome (no graph
> destruction) rather than only by the timing.

The honest limit: v65 is 91 episodes past its crossing and v66a/b/c are 116
past theirs, against group B's 6–22 episodes from crossing to skip onset. That
is 5–19× the group-B horizon, which is evidence, not proof. §14.2.3 gives the
concrete thing that would end it: if any live arm's `mean_violation` spread
falls below ~0.02 across episodes, the `SKIP` penalty vanishes and the +0.72 σ /
+0.14 floor bonus takes over. v66a is the closest (mean 0.735, sd 0.098).

---

### 14.5 Q4 — re-reading the mult era in the light of C2

C2 records that for the whole mult block (v57–v60) the training head weights
were `latency=+0, mem=+0, quality=+1` while the display line said `+1,+1,+1`,
and C4 that PopArt consequently left both cost channels cold
(`mu=0 sigma=0 norm_var=0`). C1 records that v57 additionally ran 1 minibatch
instead of 4.

What this does **not** mean is that cost was invisible: `_apply_mult_gate`
folds cost into the *scalar on the quality head* through
`cheapness = max(0, gate_w − weighted_cost)`. What it means is that cost had
**no value head of its own, hence no credit assignment**, and entered only as a
multiplicative modifier of the quality term. With the run's constants (§1:
weighted symlog cost ≈ 29.8, W = 40 → cheapness ≈ 10.2), a 40 % latency win
moves symlog(lat) by ln(1.6) = 0.47, i.e. cheapness 10.2 → 10.67, reward
7.9 → 8.2: **+4 %**. Losing quality below 0.5 costs **100 %**.

| §12/§0 conclusion | status after C2 | why |
|---|---|---|
| **"g(q) = 0 flat basin"** (§2, H1 CONFIRMED) | **SURVIVES as a description, DEMOTED as the binding constraint** | The surface analysis in §1 already priced cost inside the gate, so C2 does not invalidate the arithmetic; it sharpens it — with no cost value head, the only channel carrying credit was the one with the 0.45-wide flat. But §11's falsifier removed the flat band entirely (finite-difference slope = λ everywhere) and v61–v64b collapsed anyway, so the basin was a real feature of one reward mode, not the campaign's cause. |
| **"destruction pays"** (§12.3 REFUTED at λ=10) | **REFUTED as a reward-design claim, CONFIRMED as an instrument bug** | §14.1: every q<0.05 plan is floored at 132–139 µs where honest plans are charged 155–160 µs. Destruction pays −13 %, in all twelve runs, because the floor is timed by a different protocol. The v66 arms then *falsify* the reward-design version from the other side: they price cost 8–13× more strongly relative to quality than v64b did and still never take the cheap destructive action, because their approximation-destroyed plans are *above* the floor. |
| **"SKIP cliff"** (one SKIP DCEs the graph) | **SURVIVES structurally, RE-RANKED as the amplifier, not the initiator** | The DCE is real (pre-clamp readings of 53–103 µs against 156 µs). But §14.2.2 shows the skip rise is a *learned* 3–82× move above renormalisation that happens only *after* `none` reaches ~0 and only when the violation channel has saturated. It converts a policy drift into an irreversible absorber — by deleting the decisions themselves, so that no gradient can exist afterwards — rather than starting anything. |
| **PopArt as the ratchet** (§3) | **SURVIVES for the mult block only** | §3's μ 4.63 → −0.95 inversion is a mult-mode measurement. In the lagrangian block σ_q *shrank* monotonically and the price *rose* (§12.9.3), and the four raw-scale arms show that removing the normaliser from the quality channel helps for a reason unrelated to any ratchet. |
| **v57 "held 0.885, no learning"** (§0) | **RE-READ: v57 is the campaign's only positive result** | v57 is the run with 1 env, 1 minibatch (C1) and zero cost head weight (C2) — i.e. the least-optimised configuration in the campaign — and it is the only run that ever produced plans at 96–108 µs with q ≈ 0.88 (§14.3.3). It did not learn; it *sampled* the win, repeatedly, and had no machinery that could reward it. |

---

### 14.6 Q5 — one account for both groups, and the dead list

#### 14.6.1 The account

1. **The head cannot see what it is approximating** (§4, unchanged, untouched
   by every intervention v61–v66: face-probe R² 0.03–0.18 against a 0.48–0.67
   bar; `face_sizes` still never passed). Per-face credit is therefore
   unlearnable, and the only expressible policies are approximately global:
   "approximate more" / "approximate less".
2. **`OP_NONE` is a single shared bias per slot** (§12.2), so "approximate
   less" is one parameter, and every force in the loss lands on it together.
3. **A dense, coherent force pushes that parameter one way.** In v62/v63 it was
   the mis-targeted 0.3 entropy floor firing at penalty 0.40–0.60 from ep0
   (ALT-3, worth 7 episodes when fixed). From v64b onward the floor is inert and
   what remains is the 0.005 face entropy bonus, which reaches every face slot
   of every step and always points the same way — plus, per §12.2 F3, PPO's
   unclipped negative-advantage branch, which pushes every sampled action of a
   violating plan down and hence pushes the *unsampled* alternatives up. `none`
   is the most-sampled action at init, so it loses first, mechanically.
4. **The force that should oppose it is sparse and, when amplified, acts as
   noise.** The quality term reaches only the masked ~11–16 % of steps and, on
   most of those, carries the critic's error rather than a true violation
   signal. Multiplying that by λ/σ_q ≈ 63–74 (group B) makes the per-step
   quality term 0.68–0.86 — large, sign-varying, and landing on a shared
   parameter. On the raw scale it is 0.098–0.160 and the deterministic
   signals inside it (above all `SKIP` → certain violation) survive.
5. **Both groups therefore cross.** Every run in the campaign that ran past
   ep112 lost `approx_prob/none` — 10 out of 10. The crossing is universal and
   objective-independent; only its date moves (38 → 112).
6. **After crossing, the outcome is decided by whether the violation channel
   saturates.** Group B's 5–8× larger quality term drives every plan to
   `q = 0.000` within ~20 episodes, `mean_violation` pins at its 0.75 cap with
   *zero* variance, and the quality penalty on `SKIP` becomes identically zero.
   Group C's smaller term leaves a spread of 0.07–0.13 and `SKIP` stays priced.
7. **Once `SKIP` is free, the quality gate pays for it.** §14.1: the floor is
   under-measured by 13 %, so a DCE'd plan collects +0.72 σ of latency
   advantage. Group B's absorber is a genuine local optimum, which is why it
   survives 420 episodes and why nothing in the reward can dislodge it.
8. **Once `SKIP` has run, there is nothing left to learn from.**
   `faces/mean_valid` falls to 0.011–0.147 — the decisions the head acts on no
   longer exist, so both the contrast and the entropy metric go to zero
   (§13: the entropy fall is 91 % population, not policy).
9. **Group C's own absorber is different and cheaper to escape, but it is still
   an absorber**: saturation-density approximation, 155–181 ops/plan, where
   quality is ≈0 and latency is 15–92 % *worse* than exact. v66a's `kl/approx`
   has fallen to 0.001–0.02 — its policy has effectively stopped moving.
10. **The region that actually wins was sampled at ep0 and never revisited**
    (§14.3.3: 1–15 ops/plan, 96–108 µs, q 0.88). Nothing in any of the twelve
    configurations creates gradient pressure back toward low density once
    `none`'s shared logit has been driven to its clamp.

#### 14.6.2 DEAD

| explanation | status | killed by |
|---|---|---|
| **Entropy bonus as the driver** (H5 strong form) | **DEAD** since §6; re-killed here | `--face-entropy-weight 0.005` is bitwise identical in v63/v64b (destroyed) and v66a/b/c (intact); the 10× cut v62→v63 bought 0 episodes |
| **Entropy floor as *the* igniter** (ALT-3 general form) | **DEAD as general; SURVIVES for v62/v63 only** | v64b/v65/v66 run the floor at 0.05 with penalty 1–2e-4 (bitwise inert) and cross anyway, at ep71–112 |
| **PopArt neutralisation / the penalty got cheap** (H-ZNEUT) | **DEAD** (§12.9.3) and now dead from the other side too | σ_q *shrank*, price *rose* 16 %; and the arms with an 8–13× **weaker** realised quality price did **better** |
| **Reward shape / the flat basin as the binding constraint** | **DEAD** | §11's falsifier removed the flat band (slope = λ everywhere); v61–v64b collapsed regardless |
| **Cost bribe / "destruction pays" as a reward-design property** | **DEAD** | §12.3 at λ=10; and group C prices cost 8–13× more strongly than v64b yet never takes the cheap destructive action |
| **The causal mask as load-bearing for survival** (registered prediction iii) | **DEAD as a discriminator** | v64b has the mask and is group B; v66c lacks it and is group C |
| **Critic noise as a *sufficient* explanation** (§12.9.4) | **DOWNGRADED** | Direction confirmed (removing the amplification changes group membership) but v66a/b/c hold their graph with a critic only ~2× better in raw units, and they still lose all their quality |
| **The 150× critic-quality improvement** (F3 as stated) | **CORRECTED to ~2×** | 61× of it is the σ_q² unit conversion between normalised and raw value-head spaces |
| **"The policy collapsed to determinism"** | **DEAD** (§13) | 91 % of the entropy fall is the disappearance of the decisions being measured |
| **PPO-specific pathologies as the root cause** | **DEAD** (§7) | GAZ, no entropy bonus, search-based, collapses faster per measurement |
| **Winsorize / λ dual-ascent / symlog / `--advantage-norm none` individually** | **DEAD as discriminators** | each is present in some group-C arm and absent from another (§14.2.4) |

#### 14.6.3 What remains unexplained

1. **Why `none` falls in *every* configuration.** 10 of 10 runs that reached
   ep112 crossed, across two reward modes, three normalisers, four λ values,
   two entropy-floor settings and with/without the causal mask. Four candidate
   residual drivers remain unseparated: the 0.005 entropy bonus; PPO's
   negative-advantage epoch asymmetry on a shared bias; the fact that the face
   head is the only action head `_scale_output_heads` never touches (§13.1(ii),
   so it starts with full-magnitude orthogonal-init logits); and the terminal-
   only reward with (γλ)^95 = 0.0079 credit at step 0. None of the twelve runs
   varies any of these.
2. **Why group C loses quality at all** given an intact graph, live contrast,
   an inert floor and a 2×-better critic. The inference is §4's representation
   deficit (the head cannot rank 115 faces by safety, so it converges to
   "approximate all of them"), but nothing in this record *demonstrates* it —
   the decisive test is the online face probe against the 0.48–0.67 bar with
   the per-face latent wired through, which has never been run in a GPU
   campaign arm.
3. **Whether v65 and v66a/b/c eventually slide.** They are 91 and 116 episodes
   past their crossings against a group-B horizon of 6–22. Evidence, not proof.
4. **The size of the real Pareto set.** We have two instruments showing
   96–108 µs at q ≈ 0.88 and one showing 76–91 µs as a min-order-statistic. We
   have never measured one of those plans a second time, paired against its own
   exact reference, on the same GPU state — which is exactly what the project's
   paired-measurement rule exists for. Until that is done the size of the prize
   is bracketed, not known.

---

### 14.7 Q6 — the radical-simplification design, element by element

Predictions are stated against this analysis only; each names the evidence.

| element | prediction | why |
|---|---|---|
| **Additive `λ_lat·symlog(lat) + λ_mem·symlog(mem) + λ_q·q`** | **HELPS** (stationarity) but the λ_q values are the whole ballgame | It keeps the one property that separates group C — a quality advantage on the raw scale — and drops the moving frame. But see the next row: the proposed λ_q sweep lands entirely inside the band that has already failed. |
| **λ_q sweep {1, 4, 16}** | **1 and 4 are predicted to fail outright; 16 reproduces v66b** | §12.10.4's arithmetic, redone for raw q: realised quality:cost pull = λ_q·σ_q/σ_cost with σ_q ≈ 0.159 and σ_lat ≈ 2.12 (symlog). λ_q=1 → **0.075 : 1**; λ_q=4 → 0.30 : 1; λ_q=16 → **1.20 : 1** — which is exactly v66b, the arm that ended at q +0.233 with 155 ops/plan and latency 12 % worse than exact. Matching v64b's realised 10:1 needs **λ_q ≈ 134**. If the sweep is to be informative it must be {16, 130, 400}, or `--lambda-cmp/--lambda-mem` must come down ~13×. This is the single most consequential number in the design. |
| **γ = 1, GAE λ = 1** | **HELPS** | It removes the (γλ)^95 = 0.0079 attenuation that gives step 0 0.8 % of the terminal advantage (§6, and the v55-era "PPO never left uniform" result). It is the only proposed element that attacks a documented, quantified defect. Cost: credit becomes perfectly plan-global, so per-face resolution comes *only* from the representation — which raises the stakes on the face-latent row below. |
| **No PopArt** | **HELPS** | This is the group-C property, established at 12/12 separation and by the v64b→v65 single-flag same-seed contrast (§14.2.4). Highest-confidence element in the design. |
| **No entropy terms at all** | **HELPS on the crossing, HARMFUL on exploration, net unknown** | Removes the dense coherent anti-`none` force (§14.6.1 step 3) — the one that v62/v63 proved can drive the whole slide by itself. But combined with random init (below) there is then *nothing* maintaining coverage, and the only reason any run ever sampled the 1–15-op win was stochasticity at low density. |
| **No gate** | **HARMFUL at λ_q ≤ 4; SAFE at λ_q ≥ 16 — and it removes a real bug** | Removing the gate deletes the mis-measured floor (good: kills the −13 % destruction bonus, §14.1) but also uncaps the full `SKIP` prize: pre-clamp readings are 53–103 µs against 156 µs, i.e. symlog(157/70) ≈ **+0.81** for one action. Against `λ_q · Δq ≈ λ_q · 0.885`, `SKIP` wins outright at λ_q = 1 (0.81 vs 0.89 per unit, and cheaper still at 53 µs), is marginal at λ_q = 4, and is priced out at λ_q = 16. **Do not run λ_q ∈ {1, 4} without the gate.** If the gate is kept instead, fix its measurement protocol first. |
| **No `OP_NONE` bias, standard init** | **HARMFUL, and it is the element most likely to make the run uninterpretable** | `CLEAN_DESIGN_AUDIT` §4(i): at bias 0 the head emits ~372 approximation ops per plan and P(any of 16 plans is exact) ≈ 10⁻²¹³. That is *twice the density of the state group C converged to after 200 episodes*, and §14.3.2 shows that region is worse than exact on both axes with q ≈ 0. Episode 0 therefore starts inside the terminal absorber, with the batch-quality contrast that PG needs already at ~0. The 40 %-win region (1–15 ops/plan) is ~370 decisions away and will not be sampled. |
| **Standard init on the face head** | **HELPS (independently)** | §13.1(ii): the face head is the only action head `_scale_output_heads` never touches, so it starts with full-magnitude orthogonal logits — decisive per face while near-uniform in the marginal. Giving it the ×0.1 the other heads get is a genuine fix, and it is *separable* from removing the +6 bias. Do that one; think twice about the bias. |
| **Quant PULLUP (vs the campaign's `GRAPHAX_QUANT_PULLDOWN=1`)** | **CONFOUNDED — do not change it in the same run as the objective** | Quant is 34–40 % of the op mass and 2 258 of v66a's 2 895 applied ops. PULLUP changes the quality-per-op *and* the latency-per-op of the dominant action, so it changes the target function, not the learner. Any result would be unattributable. Run it as its own A/B on a fixed policy. |
| **Order pinned to reverse** | **NEUTRAL for this question** | It is the campaign's constant, keeps the 156 µs identity reference clean, and keeps the `ve` head's entropy and gradient at exactly 0 (§13.2(e)). It also means approximation is the *only* lever, whose entire measured range on TLM is −40 % … +92 %. |
| **Face head reads palimpsa's per-face latent instead of the pooled chunk mean** | **HELPS, and is the only element addressing the unexplained item (2)** | §4 + FACE_LATENT §9 (addressing confirmed; width does not substitute) + `CLEAN_DESIGN_AUDIT` §4(ii) (chunk-mean size-R² negative on CV at every width). Necessary for any *positive* result; §10(2) records that it is not by itself an anti-collapse fix. |
| **One full TLM run + NN256 for the sweep** | **NN256 is CONFOUNDED for the cost channels** | Project memory, measurement-resolution limit: at nn256/batch16 the memory channel has no signal and the latency floor is ~10 %, so only quant is measurable. A λ_q sweep there measures the quality channel alone and cannot tell you whether the trade exists. Sweep λ_q on TLM if the sweep is about the trade. |

#### The one measurement that would most cheaply falsify the whole design

**Drop `--lean-logging`, and at ep30 read the per-plan joint of (approximation
count, quality, latency): is there any plan, in any episode so far, with
≤ 20 approximations, `q ≥ 0.8`, and latency ≤ 0.9 × the exact-rev cost measured
by the same protocol?**

Why this one, and not a loss or entropy curve:

* It is the *only* region in twelve runs that has ever contained a win (§14.3.3:
  v57 ep41/117/155 at 96–108 µs and q 0.88 with 1–23 ops; v58b ep0
  `best_ep` 91.2 µs at `worst_ep` quality 0.884). Everything the campaign
  learned to do lives 10–100× further out in density and is worse on both axes.
* Under the proposed random init, `CLEAN_DESIGN_AUDIT` §4(i) predicts that
  region has probability ≈ 10⁻²¹³ of being sampled. So the measurement is a
  direct test of the design's most dangerous element, and it resolves in 30
  episodes rather than 250.
* If the answer is "no plan, ever", then the reward can be perfect and the
  critic silent and it will still have nothing to select — and the fix is the
  init and the representation, not the objective. If the answer is "yes, and
  the advantage ordering puts it above the batch mean", the objective is doing
  its job and the run is worth its 36 hours.
* It costs one launcher flag and one wandb panel. v58b already proved the keys
  exist (`measure/{quality,latency_ns}/{best,median,worst}_ep`); C8 records that
  they were lost for the other eleven runs by an accidental flag flip.

Second-cheapest, as a companion on the same panel: the **across-plan spread of
the quality channel within the batch** at ep ≤ 30. If it is ~0 (every plan
equally destroyed), no policy gradient exists at all, and §14.2.3 shows that is
precisely the state from which `SKIP` becomes free.

---

## 15. Pointer: the unbiased Pareto re-measurement (2026-08-27)

This dossier stops at §14's synthesis, which was written from **campaign
telemetry**. Everything §14 asserts about latency has since been re-measured
**paired, warm, on pinned code**, and the results — including two corrections to
§14 and one refutation of a session hypothesis — live in a separate file:

> **[`UNBIASED_PARETO_AND_MEASUREMENT.md`](UNBIASED_PARETO_AND_MEASUREMENT.md)**

Read it before quoting any cost number from §0–§14. In brief:

* **§14.3.3's "the win is real and was on the table from episode zero" is
  CONFIRMED, paired.** The archived winners reproduce at latency ratio
  **0.522–0.646** against an identity-vs-itself drift floor of
  **1.0007 ± 0.0008 (n=6)**. The competing "cold-measurement artifact"
  hypothesis is **REFUTED**: the cold effect is −7.2 % on measurement #1 only,
  and its sign *hides* winners rather than inventing them.
* **The win is ONE SKIP, in elimination steps 13–24.** Five independent runs
  (v63, v64b, v65, v66a, v66c) archived the *same* one-face plan at ratio
  0.578–0.581, quality 0.9258 against identity 0.9260. Adding 20–43 further
  approximations buys **zero** extra latency and costs 0.25 of quality.
* **§14.1's gate-floor instrument bug is FIXED** (`1c1e480`).
* **§14.6.2's DEAD list gains an entry: DIAG.** On TLM, DIAG applies **0 of 103**
  requested rules — every diag result in the campaign is an identity plan wearing
  a diag label. COMPRESS applies but is net *negative* on latency at every budget.
  QUANT applies 100 % for a real but small −5.0 % / −1.8 %.
* **Two protocol bugs the campaign ran with**: all 264 archived Pareto points
  were unreplayable (0-based action index vs 1-based vertex id, fixed `907c231`,
  recovery 0/42 → 42/42), and `ALPHAGRAD_NEW_SLOT_JOIN=1` — the default — emits a
  face form graphax rejects, so res-slot plans died at elimination and were
  *silently dropped from the gradient*. Worked around, root cause unfixed.
* **One caveat is still open** (job 62415): three of the winning faces skip the
  path to a broadcast `(1,128)` parameter-shaped operand, so the win may be
  "stop differentiating a parameter", which a 200-step Adam walk cannot see.
