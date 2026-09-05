# Approximation campaign — parameter matrix (v57 → v66)

> **Contamination note (2026-09-03).** Every run in this matrix, and in `params_matrix.csv` (which cannot carry this note), ran with discount 0.99 and GAE-lambda 0.95 inherited from `ppo.py` defaults; nobody chose those values, and the first eliminations of each episode received about 0.31 % of the terminal reward. All of these runs (v57 through v66) also ran under the -13 % timer bug fixed in alphagrad `1c1e480f`. The configuration record stands; conclusions drawn from these runs are recorded only and superseded by the campaign in the dsnn-3qm map. See `.scratch/trustworthy-approx-search/findings/55-gamma-lambda-rot.md` (dsnn superproject).

Descriptive record of **what was configured** for every run in the alphagrad
approximation campaign. No interpretation of outcomes.

Sources, in order of authority:

1. the run's own log (`v5*_tlm_*.log` / `v6*_tlm_*.log`) — `ag=`/`gx=` SHAs, the
   `[cpu_approx_worker … ] env:` dump, and the `[cfg]`/`[alphagrad]`/`[lr]`/
   `[popart-init]`/`[grad-window]`/`variant=`/`head weights` banner lines;
2. the sbatch launcher in `/Users/assmuth/dsnn/`;
3. the argparse block of `src/alphagrad/approx/ppo.py` **at that run's SHA**
   (`git show <sha>:src/alphagrad/approx/ppo.py`), for flags not passed;
4. `sacct` for node / state / wall time.

Provenance suffix used in `params_matrix.csv` and in the tables below:

| suffix | meaning |
|---|---|
| `\|E` | explicitly set in the launcher (env export or CLI flag) |
| `\|D` | argparse / code **default** — the flag was **not** passed |
| `\|L` | observed in the log (runtime-derived / resolved value) |
| `\|X` | the flag did not exist at that SHA |

Machine-readable companion: `params_matrix.csv` (18 rows × 174 columns).

---

## 1. Run identity

| run | job | SHA (alphagrad) | SHA (graphax) | node | state | elapsed | wall limit | eps req. | last ep seen | wandb id |
|---|---|---|---|---|---|---|---|---|---|---|
| v57  | 61485 | `29fa59e` | `4ea0bf8` | pgi15-gpu15 | CANCELLED (user) | 00:41:28 | 36 h | 500 | 161 | `it05ku34` |
| v58  | 61494 | `cb8c76d` | `4ea0bf8` | pgi15-gpu15 | CANCELLED (user) | 00:16:56 | 36 h | 500 | 3 | `cdbu0s1c` |
| v58b | 61498 | `cb8c76d` | `4ea0bf8` | pgi15-gpu15 | CANCELLED (user) | 03:55:31 | 36 h | 500 | 189 | `ud3cla83` |
| v59  | 61507 | `8e3ff8a` | `4ea0bf8` | pgi15-gpu15 | CANCELLED (user) | 01:22:42 | 36 h | 500 | 46 | `3faj3e36` |
| v60  | 61515 | `f0cee7d` | `4ea0bf8` | pgi15-gpu15 | COMPLETED | 08:38:56 | 36 h | 500 | 499 | (not echoed) |
| v61  | 61610 | `0519f3d` | `4ea0bf8` | pgi15-gpu16 | **TIMEOUT** | 1-12:00:16 | 36 h | 500 | 476 | `auw0vzlm` |
| v62  | 61844 | `519cb66` | `4ea0bf8` | pgi15-gpu16 | CANCELLED (user) | 05:43:55 | 36 h | 500 | 118 | `w6sh91ya` |
| v63  | 61866 | `1f58649` | `4ea0bf8` | pgi15-gpu16 | COMPLETED | 13:33:38 | 36 h | 500 | 499 | `as9s5yrl` |
| v64b | 61983 | `6d917fc` | `4ea0bf8` | pgi15-gpu16 | COMPLETED | 12:44:18 | 36 h | 500 | 499 | `38oyqf4g` |
| v65  | 62075 | `0e601ae` | `4ea0bf8` | pgi15-gpu16 | RUNNING | 09:18:57+ | 36 h | 250 | 198 | `8sht6x1m` |
| v66a | 62072 | `0e601ae` | `4ea0bf8` | pgi15-gpu15 | RUNNING | 09:18:58+ | 36 h | 250 | 194 | `318ktrgq` |
| v66b | 62073 | `0e601ae` | `4ea0bf8` | pgi15-gpu17 | RUNNING | 09:18:57+ | 36 h | 250 | 187 | `0olsxsjl` |
| v66c | 62074 | `0e601ae` | `4ea0bf8` | pgi15-gpu18 | RUNNING | 09:18:57+ | 36 h | 250 | 194 | `s1537jdd` |

Secondary block:

| run | job | SHA | node | state | elapsed | notes |
|---|---|---|---|---|---|---|
| gaz-nn256 | 61531 | `d5b666a` | pgi15-gpu18 | COMPLETED | 01:44:45 | `az_gumbel`, `VmappedNeuralNetwork` / mnist, `--rollout-depth 0` |
| gaz-d20   | 61532 | `d5b666a` | pgi15-gpu8  | COMPLETED | 11:57:32 | same + `GAZ_DEEPEN=1`, `--rollout-depth 20` |
| armA   | 61936 | (not echoed) | pgi15-cpu2  | CANCELLED (user) | 02:04:44 | `ppo_instr.py` (instrumented copy), Helmholtz, floor 0.3 |
| armFix | 61964 | (not echoed) | **pgi15-gpu12** | COMPLETED | 02:05:23 | real `ppo.py`, Helmholtz, floor 0.05 + mask + winsorize 3 |
| armB   | 61984 | (not echoed) | pgi15-cpu2  | FAILED (0:9) | 03:17:32 | `ppo_instr.py`, Helmholtz, floor 0.05 |

`gx=4ea0bf8` for **every** TLM run and both GAZ runs — graphax is a constant
across the whole campaign.

---

## 2. Environment — constant across all twelve TLM launchers

The `export` block of `fq_v57 … fq_v66c` is **byte-identical** (verified by
`diff` of the sorted export lines). It is therefore a controlled variable, not
a per-run parameter:

```
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_FLAGS=--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0
GRAPHAX_ALLOW_PARTIAL_ORDER=1   GRAPHAX_PLANNER_EXACT=1   GRAPHAX_DEMAND_EMIT=1
GRAPHAX_QUANT_PULLDOWN=1
ALPHAGRAD_TLM_SEQ=32   ALPHAGRAD_TLM_DMODEL=128   ALPHAGRAD_TLM_VOCAB=1024
ALPHAGRAD_QUALITY_GATE_MIN=0.05   ALPHAGRAD_FACE_NONE_BIAS=6
ALPHAGRAD_FORCE_REV_ORDER=1       ALPHAGRAD_MAX_FACES=2538
ALPHAGRAD_POLICY=palimpsa         ALPHAGRAD_MAX_DELTA_TOKENS=32768
ALPHAGRAD_MAX_EQNS=512            ALPHAGRAD_ACTOR_PROF_EVERY=20
ALPHAGRAD_INCREMENTAL_TOKENS=1    ALPHAGRAD_DEBUG_APPROX_PROB=1
ALPHAGRAD_CLEAR_JIT_CACHES_EVERY=0 ALPHAGRAD_DEBUG_MEM=1
ALPHAGRAD_SKIP_COST_ANALYSIS=1    ALPHAGRAD_DEBUG_MEASURE=1
ALPHAGRAD_DEBUG_DEGEN=1           ALPHAGRAD_PROFILE=1
ALPHAGRAD_EXTEND_CHUNK=128        ALPHAGRAD_EXTEND_UNROLL=32
ALPHAGRAD_MULS_SENTINEL_CAP=5e12  ALPHAGRAD_SKIP_COUNT_OPS=1
ALPHAGRAD_DIRECT_MEASURE=1        ALPHAGRAD_FACE_ENUM_CACHE=1
ALPHAGRAD_UNIFIED_FACE_ENUM=1     ALPHAGRAD_BATCHED_CALLBACK=1
JAX_COMPILATION_CACHE_DIR=$HOME/dsnn/.jax_compile_cache
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=2
PYTHONDONTWRITEBYTECODE=1   RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID
CUDA_VISIBLE_DEVICES=0,1,2,3
```

Trainer-injected env seen in the actor dump but **not** in any launcher (the
trainer re-exports them from CLI flags): `ALPHAGRAD_MEASURE_ACTOR=1`,
`ALPHAGRAD_QUALITY_METRIC=loss_drop`, `ALPHAGRAD_VOCAB_SIZE=512`,
`ALPHAGRAD_WALK_STEPS=200`, `ALPHAGRAD_WALK_LR=0.001`,
`ALPHAGRAD_WALK_PROBE_SEED=20260807`, `ALPHAGRAD_WALK_NOISE_STD=0.0`.

Target / graph facts, identical in every TLM log:

```
Total vertices 96, valid 95, rollout_length 95, max_rules 16
dynamic-substeps: max_substeps=16, max_axis_size=1024, allow_compress=True
face width: derived bound 2538 (in force: 2538)
base token stream: 870 tokens (per-step delta budget 32768)
quality channel = loss_drop (200 Adam steps, lr 0.001, probe seed 20260807, noise 0)
FORCE REV ORDER: vertex choice pinned to reverse; only approximations are learned
[factory] face-head IDENTITY INIT: OP_NONE +6.0, SKIP -6.0, P(approx/face) ~ 0.007
```

---

## 3. CLI — constant across all twelve TLM launchers

```
--example TransformerLM --exec-on-gpu --seed 250197
--measure-latency --latency-inner-reps 5 --num-data-points 5 --reps-per-point 4
--dataset wikitext2 --hidden-dim 256 --num-layers 3 --vocab-size 512
--incremental-encode --cmp-type latency --mem-type peak_memory
--terminal-rewards-only --minibatches 4 --popart-init-episodes 3
--rewards cmp mem acc --lambda-cmp 1 --lambda-mem 1
--set-pointer --face-actions --unified-face-head --live-faces
--ray-measure 3 --ray-measure-timeout 600
--measure-grad --seed-vertices --quality-metric loss_drop --grad-window 0
--wandb online --wandb-entity dll-streetview --wandb-project dsnn-vertex
```

---

## 4. The variable axis, run by run

| param | v57 | v58/v58b | v59 | v60 | v61 | v62 | v63 | v64b | v65 | v66a | v66b | v66c |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `--episodes` | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | **250** | 250 | 250 | 250 |
| `--num-envs` | **1** | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| minibatches *effective* | **1** (clamped) | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| `--reward-mode` | mult | mult | mult | mult | **lagrangian** | lagrangian | lagrangian | lagrangian | lagrangian | lagrangian | lagrangian | lagrangian |
| `--advantage-norm` | popart | popart | popart | popart | popart | popart | popart | popart | popart | **none** | none | none |
| `--no-symlog` | on | on | on | on | on | on | on | on | on | **off (dropped)** | off | off |
| symlog *effective* | off | off | off | off | off | off | off | off | off | **on (costs), viol RAW** | on | on |
| `--lambda-acc` | 2 `\|E` | 2 `\|E` | 2 `\|E` | 2 `\|E` | 2 `\|D` | 2 `\|D` | 2 `\|D` | 2 `\|D` | 2 `\|D` | 2 `\|D` | 2 `\|D` | 2 `\|D` |
| lambda_acc *effective* | 1.0 | 1.0 | 1.0 | 1.0 | ignored | ignored | ignored | ignored | ignored | ignored | ignored | ignored |
| `--lag-init` (λ₀) | – | – | – | – | 1.0 | 10 | 10 | 10 | 10 | 10 | **16** | **13** |
| `--lag-eta` | – | – | – | – | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | **0** | 0 | 0 |
| `--lag-min` / `--lag-max` | – | – | – | – | 0.1 / 10 | 2 / 20 | 2 / 20 | 2 / 20 | 2 / 20 | 2 / 20 | 2 / 20 | 2 / 20 |
| `--lag-tau` | – | – | – | – | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| `--lag-target` | `\|X` | `\|X` | `\|X` | `\|X` | **`\|X`** | 0.02 `\|E`=def | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| `--lag-causal-mask` | `\|X` | `\|X` | `\|X` | `\|X` | `\|X` | `\|X` | `\|X` | **on** | on | on | on | **off** |
| `--lag-raw-viol-adv` | `\|X` | `\|X` | `\|X` | `\|X` | `\|X` | `\|X` | `\|X` | `\|X` | **on** | off | off | off |
| `--adv-winsorize` | `\|X` | `\|X` | `\|X` | `\|X` | `\|X` | `\|X` | 0.0 `\|D` | **3** | 3 | **0.0 `\|D`** | 0.0 | 0.0 |
| `--popart-basin-freeze` | `\|X` | `\|X` | `\|X` | `\|X` | on `\|E` | on `\|E` | on `\|E` | on `\|E` | on `\|E` | **True `\|D` (inert)** | True `\|D` | True `\|D` |
| `--face-entropy-floor` | `\|X` | `\|X` | `\|X` | `\|X` | 0.0 `\|D` | **0.3** | 0.3 | **0.05** | 0.05 | 0.05 | 0.05 | 0.05 |
| `--face-entropy-floor-weight` | `\|X` | `\|X` | `\|X` | `\|X` | 10 `\|D` | 10 `\|E`=def | 10 | 10 | 10 | 10 | 10 | 10 |
| `--face-logit-clamp` | `\|X` | `\|X` | `\|X` | `\|X` | `\|X` | 15 `\|E`=def | 15 | 15 | 15 | 15 | 15 | 15 |
| `--face-entropy-weight` | `\|X` | `\|X` | `\|X` | `\|X` | `\|X` | None `\|D` | **0.005** | 0.005 | 0.005 | 0.005 | 0.005 | 0.005 |
| `--face-endpoint-read` | `\|X` | `\|X` | `\|X` | `\|X` | `\|X` | `\|X` | **on** | on | on | on | on | on |
| `--face-edge-mem` | `\|X` | `\|X` | `\|X` | `\|X` | `\|X` | `\|X` | off `\|D` | off `\|D` | off `\|D` | off `\|D` | off `\|D` | off `\|D` |
| `--var-probe` | `\|X` | `\|X` | `\|X` | `\|X` | **on** | on | on | on | on | on | on | on |
| `--var-probe-steps` | `\|X` | `\|X` | `\|X` | `\|X` | **`\|X`** (full replay) | 16 `\|E`=def | 16 | 16 | 16 | 16 | 16 | 16 |
| `--lr-warmup-frac` | 0.0 `\|D` | 0.0 `\|D` | **0.2** | 0.2 | 0.0 `\|D` | 0.0 `\|D` | 0.0 `\|D` | 0.0 `\|D` | 0.0 `\|D` | 0.0 `\|D` | 0.0 `\|D` | 0.0 `\|D` |
| `--lr-warmup-mult` | `\|X` | `\|X` | `\|X` | **on** | off `\|D` | off `\|D` | off `\|D` | off `\|D` | off `\|D` | off `\|D` | off `\|D` | off `\|D` |
| `--lean-logging` | on | **off (v58b)** | on | on | on | on | on | on | on | on | on | on |
| head w (lat, mem, qual) | 0,0,1 | 0,0,1 | 0,0,1 | 0,0,1 | **1,1,1** | **1,1,10** | 1,1,10 | 1,1,10 | 1,1,10 | 1,1,10 | **1,1,16** | **1,1,13** |

---

## 5. What changed between consecutive runs

Exact parameter deltas. Everything not listed is unchanged.

### v57 (61485) → v58 (61494)
* code: `29fa59e` → `cb8c76d` — *popart-init warm-starts CONSTANT-with-nonzero-mean
  channels from the reference measurement instead of leaving them cold.*
* `--num-envs` **1 → 16** (and with it, the minibatch clamp lifts: effective
  minibatches 1 → 4, envs/minibatch 1 → 4).
* `--name` `v57-tlm-revapprox-fullT` → `v58-tlm-fullT-env16`.
* Nothing else. Env block identical.

### v58 (61494) → v58b (61498)
* Same SHA, same node. The sbatch file was **overwritten at 14:37:43**, i.e. at
  61498's launch, so 61494's exact command line is unrecoverable.
* Recoverable difference: `--name` → `v58b-tlm-fullT-env16`.
* File-level difference vs v57: `--lean-logging` is **absent** in the surviving
  v58 file (it is present in v57 and returns in v59). Whether 61494 also lacked
  it cannot be established.

### v58b (61498) → v59 (61507)
* code: `cb8c76d` → `8e3ff8a` — *adds `--lr-warmup-frac` (default 0 = old
  schedule bit-identical).*
* **`--lr-warmup-frac 0.2`** added → `[lr] linear warmup 800 steps (20% of 4000),
  then cosine to 3.00e-05`.
* `--lean-logging` restored.
* `--name` → `v59-tlm-fullT-warmup`.

### v59 (61507) → v60 (61515)
* code: `8e3ff8a` → `f0cee7d` — *`--lr-warmup-mult` (7bfc472) plus the fix that
  binds the warmup step count at schedule definition (job 61514 died on it).*
* **`--lr-warmup-mult`** added → `[lr] MULT warmup 800 steps (20% of 4000):
  linear envelope × full cosine`.
* `--name` → `v60-tlm-fullT-multwarmup`.
* Note 61514 was the same launcher and crashed; 61515 is the surviving job.

### v60 (61515) → v61 (61610)
Largest single step of the campaign.
* code: `f0cee7d` → `0519f3d` — *lagrangian reward (f332f13), var_probe module
  (57de347) + PPO wiring (6167978), basin-freeze predicate → occupancy (0519f3d).*
* `--reward-mode` **mult → lagrangian**.
* `--lambda-acc 2` **removed** (falls to default 2.0, and is ignored in
  lagrangian mode either way).
* **added** `--lag-tau 0.75 --lag-eta 0.05 --lag-init 1.0 --lag-min 0.1 --lag-max 10`.
* **added** `--popart-basin-freeze`.
* **added** `--var-probe --var-probe-lr 1e-3` (full per-episode oracle replay —
  `--var-probe-steps` does not exist yet at this SHA).
* **removed** `--lr-warmup-frac 0.2` and `--lr-warmup-mult` → back to the plain
  cosine schedule.
* node moves gpu15 → gpu16 (all later TLM runs except the v66 fan-out).
* Consequence in the log: head weights go from `0,0,1` to `1,1,1` — the cost
  heads enter the training objective for the first time.

### v61 (61610) → v62 (61844)
* code: `0519f3d` → `519cb66` — *`--lag-target` decay-on-satisfied (fdd5d45),
  face-head entropy floor + tanh logit clamp (739b658), `--var-probe-steps`
  (ea11343), tanh tolerance fix (9a0661c), vp-target hygiene (519cb66).*
* `--lag-init` **1.0 → 10**, `--lag-min` **0.1 → 2**, `--lag-max` **10 → 20**.
* **added** `--lag-target 0.02` (equal to the new default).
* **added** `--face-entropy-floor 0.3 --face-entropy-floor-weight 10` (weight
  equals the default).
* **added** `--face-logit-clamp 15` (equals the default).
* **added** `--var-probe-steps 16` (equals the default).
* Head weights go `1,1,1` → `1,1,10` (quality head weight **is** `--lag-init`).

### v62 (61844) → v63 (61866)
* code: `519cb66` → `1f58649` — *`--face-endpoint-read` (0605ad0), edge-keyed
  memory `--face-edge-mem` (e16a255, left off), `--face-entropy-weight` split
  (1f58649).*
* **added** `--face-entropy-weight 0.005` — the face head's entropy bonus is
  split out of the global `--entropy-weight 0.05`.
* **added** `--face-endpoint-read`.
* Everything else identical.

### v63 (61866) → v64b (61983)
* code: `1f58649` → `6d917fc` — *sec-12.7 credit-layer fix: `--lag-causal-mask`
  and `--adv-winsorize`.*
* `--face-entropy-floor` **0.3 → 0.05**.
* **added** `--lag-causal-mask`.
* **added** `--adv-winsorize 3`.
* (`fq_v64_tlm_joint.sbatch` and `fq_v64a_tlm_igniter.sbatch` exist but were
  never submitted as GPU campaign runs; v64b is the only v64 on the GPU.)

### v64b (61983) → v65 (62075)
* code: `6d917fc` → `0e601ae` — *static-objective mode: per-channel symlog
  exemption, `--lag-eta 0`, per-channel critic telemetry, `--lag-raw-viol-adv`.*
* `--episodes` **500 → 250**.
* **added** `--lag-raw-viol-adv` (raw-scale quality advantage).
* Nothing else — v65 is the PopArt control arm, i.e. v64b + one flag.

### v65 (62075) → v66a (62072)
Same SHA `0e601ae`; the four current arms all launched at 14:11 on 2026-08-26.
* `--advantage-norm` **popart → none**.
* `--no-symlog` **removed** → symlog switches **on** for the cost channels
  (`[cfg] lagrangian + symlog: cost channels symlog'd, violation channel RAW`).
* `--lag-eta` **0.05 → 0** (dual ascent becomes a strict no-op; λ frozen at
  `--lag-init`).
* `--popart-basin-freeze` **removed** from the CLI (still `True` by default,
  but inert — see §7).
* `--adv-winsorize 3` **removed** → 0.0.
* `--lag-raw-viol-adv` **removed**.
* `--lag-causal-mask` retained.
* node gpu16 → gpu15.

### v66a (62072) → v66b (62073)
* `--lag-init` **10 → 16**. Node gpu15 → gpu17. Nothing else.

### v66a (62072) → v66c (62074)
* `--lag-init` **10 → 13**.
* `--lag-causal-mask` **removed**. Node gpu15 → gpu18.

---

## 6. CRITICAL — defaults that were never set but shape learning

Every default value below is **constant over the whole SHA range
`29fa59e … 0e601ae`**: diffing the extracted argparse table between all nine
campaign SHAs shows only *added* flags, never a *changed* default. So each entry
applies to every run in the campaign unless a "first exists at" note says
otherwise.

### 6.1 Never passed in any launcher, and load-bearing

| flag | default | applies to | why it matters |
|---|---|---|---|
| `--discount` | **0.99** | v57–v66, all arms | Discounting over a 95-step, terminal-reward-only rollout: the first elimination sees `0.99^95 ≈ 0.386` of the terminal signal. |
| `--gae-lambda` | **0.95** | v57–v66, all arms | Combined with γ=0.99, `(γλ)^95 ≈ 0.0079` — the credit reaching step 0 is ~0.8 % of the terminal advantage. |
| `--entropy-weight` | **0.05** | v57–v66, all arms | The global entropy bonus. From v63 on, `--face-entropy-weight 0.005` splits the face head out of this term (`_split_entropy_bonus`, ppo.py:807); before v63 the face head sat inside the 0.05 bonus. |
| `--value-weight` | **0.5** | all | Critic loss weight; never tuned across the campaign. |
| `--ppo-epochs` | **2** | all | Two epochs per update, negative-advantage asymmetry across them is the sec-12 mechanism. |
| `--ppo-clip-eps` | **0.2** | all | |
| `--lr` | **3e-4** | all | Cosine to `--lr-decay-min-mult 0.1` × 3e-4 = 3.0e-5 (confirmed by the v59 `[lr]` banner). |
| `--lr-decay-min-mult` | **0.1** | all | |
| `--max-grad-norm` | **0.5** | all | |
| `--adam-b1` / `--adam-eps` | **0.9 / 1e-7** | all | |
| `--popart-beta` | **1e-2** | v57–v65 (PopArt on) | PopArt EMA rate. |
| `--popart-sigma-min` | **0.1** | v57–v65 | σ floor; directly sets the 1/σ_q amplification the sec-12.9 note is about. |
| `--popart-init-temperature` | **10.0** | v57–v65 | |
| `--anti-degen-penalty` / `--anti-degen-tau` | **2.0 / 0.05** | all | Degeneracy penalty, never touched. |
| `--gate-tau` / `--gate-w` / `--gate-fidelity` | **0.5 / 40.0 / `cos`** | **v57–v60 only** (mult mode) | These *are* the multiplicative reward gate `g(q)` for the entire mult-mode block. Never set, never varied. |
| `--head-init-scale` | **0.1** | all | Output-head init scale. |
| `--loss-mode` | **`multi_head`** | all | Required by lagrangian mode (ppo.py raises otherwise). |
| `--embd-dim` / `--num-heads` / `--op-embd-dim` / `--set-pointer-blocks` | **32 / 2 / 8 / 2** | all | Policy encoder capacity — never varied; note the FACE_LATENT_INFO_LOSS work concluded E=32. |
| `--max-substeps` / `--max-axis-size` / `--max-rules` | **16 / 1024 / 16** | all | Confirmed in every log's `dynamic-substeps:` and `variant=` banner. |
| `--allow-compress` / `--dynamic-substeps` | **True / True** | all | |
| `--lambda-frob` | **0.0** | all | The frob-residual channel is off everywhere. |
| `--face-edge-mem` | **off** | v63–v66 (exists from `1f58649`) | Implemented and gated, never enabled in a GPU campaign run. |
| `--unified-head` / `--per-face` / `--preference-conditioned` | **off** | all | |

### 6.2 Explicitly passed but at exactly the argparse default (no-ops in the diff)

These read as tuned knobs in the launchers but change nothing relative to not
passing them: `--lag-target 0.02` (v62+), `--face-entropy-floor-weight 10`
(v62+), `--face-logit-clamp 15` (v62+), `--var-probe-steps 16` (v62+),
`--var-probe-lr 1e-3` (v61+), `--num-data-points 5`, `--reps-per-point 4`,
`--ray-measure-timeout 600`, `--popart-init-episodes 3`, `--lambda-cmp 1`,
`--lambda-mem 1`, `--mem-type peak_memory`, `--seed 250197`.

### 6.3 Defaults that took over when a flag was *dropped* mid-campaign

| flag | dropped at | default then in force |
|---|---|---|
| `--lambda-acc 2` | v61 (with the move to lagrangian) | 2.0 — **and ignored** in lagrangian mode |
| `--lr-warmup-frac 0.2`, `--lr-warmup-mult` | v61 | 0.0 / off — plain cosine |
| `--adv-winsorize 3` | v66a | 0.0 (no winsorization) |
| `--lag-raw-viol-adv` | v66a | off |
| `--popart-basin-freeze` | v66a | **True** (BooleanOptionalAction default) |
| `--no-symlog` | v66a | symlog **on** |
| `--lag-causal-mask` | v66c | off |
| `--lean-logging` | v58b only | off |

### 6.4 Flags that did not yet exist (so their behaviour was absent, not defaulted)

| flag | first exists at | runs without it |
|---|---|---|
| `--lr-warmup-frac` | `8e3ff8a` (v59) | v57, v58/v58b |
| `--lr-warmup-mult` | `f0cee7d` (v60) | v57–v59 |
| `--lag-*` family, `--popart-basin-freeze`, `--var-probe`, `--var-probe-lr` | `0519f3d` (v61) | v57–v60 |
| `--lag-target`, `--face-entropy-floor(-weight)`, `--face-logit-clamp`, `--var-probe-steps` | `519cb66` (v62) | v57–v61 — **v61 ran with no λ decay, no entropy floor and no logit clamp** |
| `--face-endpoint-read`, `--face-edge-mem`, `--face-entropy-weight` | `1f58649` (v63) | v57–v62 |
| `--adv-winsorize`, `--lag-causal-mask` | `6d917fc` (v64b) | v57–v63 |
| `--lag-raw-viol-adv` | `0e601ae` (v65) | v57–v64b |

---

## 7. Contradictions between launcher and log (silent divergences)

**C1 — v57 (61485): `--minibatches 4` was silently clamped to 1.**
Launcher asks `--num-envs 1 --minibatches 4`. Log:

```
[grad-window 0] --minibatches=4 > --num-envs=1: the full-horizon path batches
whole TRAJECTORIES, so minibatches is clamped to 1. Updates per epoch are the
minibatch count, i.e. this run does 1 instead of 4.
```

v57 therefore performed **1 update per epoch**; every later run performed 4.
v57 is not comparable to v58+ on update count.

**C2 — v57–v60: `--lambda-acc 2` was overwritten to 1.0, and the cost heads
were weighted 0.** `--reward-mode mult` replaces the head-weight vector with a
one-hot on the quality head (`ppo.py:6058-6061`). The log shows the
contradiction on two adjacent lines:

```
reward weights (display): latency_ns=+1, peak_memory=+1, quality=+1
head weights (training):  latency=+0,   mem=+0,          quality=+1
```

So for the whole mult block, `--lambda-cmp 1 --lambda-mem 1 --lambda-acc 2`
had **no effect on training** — latency and memory entered only through the
cheapness term inside the gate, not through the value/advantage heads.

**C3 — v61–v66: `--lambda-acc` is ignored outright.** ppo.py:6062-6081, comment
in-source: *"--lambda-acc is ignored in this mode"*. The quality head weight is
`args.lag_init`. Hence head weights `1,1,10` for v62–v66a, `1,1,16` for v66b,
`1,1,13` for v66c, `1,1,1` for v61.

**C4 — v57 (and v58–v60): PopArt left the cost channels COLD.**

```
[popart-init]   latency: mu=0 sigma=0 norm_var=0.0000  <-- CONSTANT in the sample, left COLD
[popart-init]   mem:     mu=0 sigma=0 norm_var=0.0000  <-- CONSTANT in the sample, left COLD
[popart-init]   quality: mu=5.10439 sigma=1.39631 norm_var=1.0000
```

Under `FORCE_REV_ORDER` + `FACE_NONE_BIAS=6` every warm-up plan is the identical
exact-rev plan, so latency/mem sample as constants. `cb8c76d` (v58) was written
to fix exactly this, but the banner is unchanged in v58/v59/v60 logs — the cold
channels persist for the whole mult block. From v61 (lagrangian) the channels
initialise properly (`latency: mu=-102105 sigma=28868.4`).

**C5 — v66a/b/c: `--popart-basin-freeze` is still `True` despite being removed
from the launcher.** The launcher comment says *"No `--adv-winsorize`, no
`--popart-basin-freeze`, no `--lag-raw-viol-adv`"*, but the flag is declared
`action=argparse.BooleanOptionalAction, default=True` (ppo.py:3453-3454) — not
passing it leaves it **on**. It happens to be harmless because the freeze is
gated on `args.popart_basin_freeze and args.advantage_norm == "popart"`
(ppo.py:10652-10653) and these arms pass `--advantage-norm none`. Turning it off
would have required `--no-popart-basin-freeze`.

**C6 — v66a/b/c: dropping `--no-symlog` changed the reward transform in the same
step as the advantage-norm change.** v65 log: `[cfg] symlog DISABLED; PopArt
alone scales the channels.` v66a/b/c log: `[cfg] lagrangian + symlog: cost
channels symlog'd, violation channel RAW (symlog-exempt, bounded
[-(tau+0.5), 0]).` The v66 arms therefore differ from v65 in **five**
simultaneous ways (advantage-norm, symlog, lag-eta, winsorize, raw-viol-adv),
not one.

**C7 — v66a/b/c still pass `--popart-init-episodes 3` with `--advantage-norm
none`.** The three warm-up rollouts still run; PopArt statistics are then unused.

**C8 — v58 (61494) CLI is unrecoverable.** `fq_v58_tlm_env16.sbatch` was
rewritten at `2026-08-17 14:37`, the launch time of 61498. The surviving file
also **lacks `--lean-logging`**, which v57 had and v59 restored — a one-run flip
that was almost certainly unintended.

**C9 — armFix (61964) did not run where its script says.** `repro_fix.sbatch`
carries `#SBATCH -p pgi15-cpu` / `#SBATCH -w pgi15-cpu2`; sacct reports
`Partition=pgi15, NodeList=pgi15-gpu12`. It was submitted with command-line
overrides. Compute stayed CPU-only (`JAX_PLATFORMS=cpu`), but the arm ran on a
GPU node while its siblings armA/armB ran on `pgi15-cpu2` — so it is not a
same-hardware comparison against them. armA/armB also run
`collapse_invest/v63_anti_none/ppo_instr.py` (an instrumented copy generated by
`make_instr.py`) while armFix runs the real `src/alphagrad/approx/ppo.py`.

**C10 — armB (61984) terminated `FAILED` with ExitCode `0:9` (SIGKILL) at
ep ~105/200**, whereas armA was user-cancelled at ep ~64/200. The two arms are
not equally long.

**C11 — `--latency-inner-reps 5` in every campaign run.** The standing project
rule is 50. Every TLM run and both GAZ runs used 5.

**C12 — v61 (61610) is the only run to hit the wall clock** (`TIMEOUT`,
`1-12:00:16` against a 36 h limit) at ep ~476 of 500. Its 500-episode request
was never satisfiable at the v61 per-episode cost (`--var-probe` full replay,
before `--var-probe-steps 16` existed).

### Non-contradictions worth recording

* The `[cpu_approx_worker … ] env:` dump filters to `ALPHAGRAD_*` and
  `JAX_COMPILATION_*` only (`cpu_approx_worker.py:171`). The absence of
  `GRAPHAX_PLANNER_EXACT`, `GRAPHAX_DEMAND_EMIT`, `GRAPHAX_QUANT_PULLDOWN`,
  `GRAPHAX_ALLOW_PARTIAL_ORDER` and `XLA_FLAGS` from that line is a print
  filter, **not** a failed export.
* `ALPHAGRAD_MAX_FACES=2538` in the launcher matches the log's derived bound
  exactly (`face width: derived bound 2538 … in force: 2538`) in every run.
* `ALPHAGRAD_MAX_DELTA_TOKENS=32768` matches the log's `per-step delta budget
  32768` and the `[grad-window 0] … delta window 32768` line in every run.
