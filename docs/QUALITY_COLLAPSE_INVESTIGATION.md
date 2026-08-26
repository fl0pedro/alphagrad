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
- Offline campaign (`/Users/assmuth/dsnn/alphagrad/CAMPAIGN_STATE.md`, "STAGE A COMPLETE"
  and "STAGE B SEED 0" sections): all four lean pooling arms score −0.24…+0.12 (bar
  0.48-0.67); `+extents` clears the bar with main effect ≈ +0.5; MP+EXT clears all five
  targets at 1/3 of training. "#94 (extents) CONFIRMED NECESSARY" is recorded there
  verbatim (lines 194, 440).
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
   (CAMPAIGN_STATE.md). *Not supported as*: an anti-collapse fix on its own — the
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

#### 12.10.4 What the static arms actually price (read this before reading the results)

The arms' *nominal* weights are quality λ ∈ {10, 13, 16} against
`--lambda-cmp 1 --lambda-mem 1`. The *realized* pull of a channel on the
policy is that weight times the channel's own advantage spread, and the two
differ because the static objective no longer divides each channel by its σ.
Estimating both from v64b's own history (ep32: σ_lat 2.88e4, σ_mem 9.80e6,
σ_q 0.141, µ_lat −1.02e5, µ_mem −3.56e7):

| | v64b (PopArt) | static arms |
|---|---|---|
| cost channel advantage scale | ÷σ ⇒ O(1) | symlog ⇒ σ_lat/\|µ_lat\| ≈ **0.28** |
| quality channel advantage scale | ÷σ_q ⇒ O(1) | raw ⇒ σ_q ≈ **0.141** |
| realized quality : cost pull | λ ≈ **10×** | λ·0.141/0.28 = **5.0×** (λ=10), **6.5×** (13), **8.1×** (16) |

So the battery does not merely repeat v64b's price — it **brackets** it from
below: v66b (λ=16, ≈8×) sits closest to v64b's realized ≈10×, v66a (λ=10,
≈5×) is deliberately at half of it, v66c (λ=13) in between. This is a
registered caveat, not a defect: if all three static arms slide, "quality was
priced lower than v64b in realized terms" is a live alternative to the
critic-noise reading and the *right* follow-up is λ ≈ 20 rather than
abandoning the static objective. Prediction (ii) below is the test that
separates them. (Both scales are first-order estimates from v64b's policy;
the arms log `value_loss/*` and the raw channel spreads so the realized
numbers can be recomputed in place.)

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
