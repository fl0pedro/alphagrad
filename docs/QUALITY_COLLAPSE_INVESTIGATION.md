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
