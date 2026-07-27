# `approx` — plan of record

Living document. Update the checkboxes as work lands; cite it in commits.
Branches: alphagrad `palimpsapprox-statetok` · graphax `core-v2`.

---

## 0. Context

Goal: an RL policy (PPO + Gumbel-AlphaZero) that searches a cross-country
**vertex-elimination order** *and* per-path **approximations** of the intermediate
Jacobians (edges), scored by measured latency / peak memory / accuracy and
trained on NN, Transformer, and ALIF-SNN targets.

The machinery mostly exists and trains end-to-end, but it is spread over
**49 Python files / ~20k LOC** in `src/alphagrad/approx/` with two parallel
trainers that have *different* feature sets, the approximation action space is
still **per-vertex** instead of per-path, and the last full run **collapsed to the
degenerate optimum** (flops→0, cosine→0). This plan states what exists, what is
missing, and the order to fix it.

---

## 1. Status answers (asked 2026-07-25)

### 1a. "What happened to the previous clunky version?"

It was never deleted — it is **still the more featureful one**. There are two
trainers side by side:

| | `ppo.py` (what we run) | `ppo_ray.py` (the "clunky" one) |
|---|---|---|
| process model | single-process, jit+vmap | Ray actors |
| PopArt | ✗ (lives in `common/popart.py`, unused here) | ✓ |
| Pareto archive + hypervolume | ✗ | ✓ (`common/pareto_archive.py`) |
| winsorized measurement | ✗ | ✓ (also in `env.py`, `az_gumbel.py`) |
| currently exercised | ✓ (all recent runs) | ✗ |

So the "mess" is real and specific: **we migrated to the lean trainer and left
PopArt / Pareto / winsorization behind on the old one.** Several features the
spec asks for are not missing — they are in the file we stopped running.

`az_gumbel.py` (GAZ) also has PopArt + hypervolume + winsorization, and is far
less exercised than `ppo.py`.

### 1b. "What is the status of the stashed versions?"

Nothing is lost; nothing is urgent. Full inventory:

**pgi15 `~/dsnn/alphagrad`** — detached at `1129c4f`, working tree clean.
Branch `palimpsapprox-statetok` still holds **12 commits that exist nowhere else**
(top: `a4d74b2` "restore live per-edge masking — no best-effort crutch", plus the
`train_lean.py` lean-trainer line `dbd6b61` → `61fe13b`, and `e13ac89`
MicroPPOAgent→ApproxAgent rename). **Action: review + push or explicitly drop.**
- stash@{0} WIP on `palimpsapprox-statetok` (`81175a9` CSV-export warmup)
- stash@{1} "approx wip pre-cvar-switch 20260616"

**pgi15 `~/dsnn/graphax`** — `core-v2` @ `eb6ad3e`, **0 unpushed commits**
(already upstream), but **3 uncommitted tracked files**: `dtype_compute.py`,
`micro_actions.py`, `tensor.py` (+139/−31). This is the in-progress
**asymmetric zero-point** quantization work, which we deliberately **replaced with
sign-flip half-range**. Almost certainly superseded — **diff before discarding.**
- 2 stashes (`30441f6` ADALIF_SNN_SEQ WIP; "local core.py grad stopgap 20260616")

**pgi15 `~/dsnn/gxf`** (worktree used by runs) — detached `68fb9d7`, clean.

**local `alphagrad`** — 2 stashes: `stash@{0}` WIP on the old `approx` branch
(`run_nns.sh`, `ppo.py` +95); `stash@{1}` trivial (3 lines, `main`).
Old branch `approx` diverged from the current line at `38bca38`:
**approx-only 110 commits, statetok-only 353.** It is the ancestor of today's work,
not a competitor — no merge intended.

**local `graphax`** — 14 stashes, all old bases (`proxy-init`, pre-`core-v2`
merges), incl. `stash@{12}` "before gemini" (base `d09cd54`, mostly tests).
Historical; nothing here is needed by the current line.

> Verdict: **one real decision** (the 12 pgi15 alphagrad commits), **one diff-then-discard**
> (3 pgi15 graphax files). Everything else is archaeology and can stay stashed.

---

## 1c. Six-agent forensic synthesis (2026-07-26)

Six independent read-only agents (2 per target, identical rubric, no cross-hints),
plus my own spot-verification of every anchor claim. **Both twins agreed on every
material fact for all three targets** — the findings below are high-confidence.

### Target N — `ppo.py` (mainline, the only line that runs and trains)

**The collapse is the objective, verified end to end:**
- `HEAD_REWARD_INDICES = (flops, peak_memory, frob_residual)` (ppo.py:135-142) —
  **cosine_sim is not in the training signal at all**; `HEAD_NAMES` labels the
  frob head "acc" (verified).
- The frob penalty is bounded at −1; and after the per-channel z-score
  (ppo.py:4813-4818) the raw-magnitude argument is moot anyway: it is a
  scale-free **2-vs-1 vote for zeroing** — and once the batch is uniformly
  degenerate, std(frob-adv)→0 and the lone opposing channel **vanishes**
  (a one-way ratchet).
- The counterweight the comments promise — "pre-training calibration (lines
  ~5545-5640)" — **does not exist**; the file ends at 5530 (verified).
- Second backdoor: `_quality_metrics` returns **(cos=1.0, frob=0.0) = perfect**
  for empty/mismatched/zero-size Jacobians (env.py:894-897), reachable via
  terminal-vertex COMPRESS. A shape-destroying plan beats every honest one.
- Best/top-N heaps use raw display-weighted sums → zero-compute plans dominate
  `best_return` by ~10 orders of magnitude; only *perfect* cosine is filtered
  (the wrong direction).
- Solid parts (verified): ratio-1 invariant holds (stored oracle masks +
  rewritten-action scoring both sides); factored quant head fully wired;
  weight re-init per episode; per-vertex oracle bridge works.
- Caveats: oracle replay uses `rules=()` (structural only — violates
  `advance()`'s own contract; crash/over-mask risk once approximations land);
  quality measured on eval sample 0 only; `op_marginal/end` actually logs
  QUANT; token-truncation counters never polled; env-var ordering bug at
  ppo.py:3577→3581 (latent — our sbatches export first); instrumentation
  merges eval samples into the wrong arg slots → dynamic vertex features
  silently zero (swallowed exception); ~30-40% of the file is dead for our
  invocation; "Elimination order" wandb table logged forever-empty.

### Target C — `ppo_ray.py`/`ppo_ray_worker.py` (the "clunky" line)

**Verdict (both agents): cannot run against today's tree.** Five independent
hard failures: (1) 8-vs-10 channel skew — `PPORayWorker.__init__` indexes
channel 9 of an 8-vector → IndexError (root cause: clobber commit `761c6a6`
replaced the 10-channel env; restores brought back masking but not channels);
(2) FactoredQuantHead migration not ported (deleted `quant_dtype_head.proj`
AttributeError + `MicroAction` rebuilt without required `quant_scale_sign` →
TypeError); (3) **100% sentinel measurements** — `evaluate` passes `point_idx=`
which this env's `_callback` doesn't accept → TypeError → blanket except →
−1e10 every call, silently; (4) per-channel best argmax unguarded — a failed
env's zeroed sums pin `best_per_channel/latency_ns` at 0 forever; (5) the
advertised async pipeline is impossible (no `--p3o` flag → SystemExit; its
learner method `train_on_trajs` was removed but is still called).

**But it is the sole home of genuinely working designs** (all 10-channel-era —
porting = adaptation, not copy):
- **PopArt, real**: update → output-preserving head rescale → normalized
  targets → per-channel σ-scaled advantages (+ per-channel σ floors, robust
  MAD-winsorized EMA).
- **Dynamic sentinel + truly-neutral-μ failed-row substitution** (failed
  measurements get advantage ≡ 0 instead of poisoning the batch).
- **`ALPHAGRAD_REWARD_MODE=mult`** — multiplicative cosine-gate reward +
  shaped anti-degeneracy penalty: **a ready-made countermeasure for exactly
  our collapse**.
- Per-component KL + KL early-stop; entropy-coef anneal passed as runtime arg;
- Pareto archive + hypervolume + per-episode `pareto/*` wandb keys +
  front JSON dumps (driver-side, genuinely fed);
- `--measure-queue` per-(env,point) fan-out with P60 aggregation (the only
  place `--percentile-keep` does anything); `ALPHAGRAD_LOCAL_TOKENIZE`
  fast path; failed-terminal aggregation exclusion; cost-head aux (unproven).

**Correction to §3-P1:** *winsorized measurement is live NOWHERE.* The
`--latency-winsor`/`--num-data-points`/... knobs are forwarded and then
**silently swallowed** by `from_jaxpr(**_compat)` (env.py:1511-1518). The
"[6:8] of 10" latency band is not a top-quartile mean and applies only under
`--measure-latency`. The real winsorized implementation exists only in the
fat-era env (STASH1 evidence patch).

### Target S — pgi15-lean (the 12 unpushed commits + stashes)

- `train_lean.py` is a **rollout+measure+log driver with NO update step** (the
  docstring says so; no optimizer exists). It is NOT a competing trainer.
- Its base (`9a35aab`) had an ImportError-dead oracle (dangling
  `diag_row_to_pair` import — verified); `a4d74b2` fixed it by inlining. Our
  mainline fixed the same thing independently.
- **Worth cherry-picking:** ADALIF_SNN / ADALIF_SNN_SEQ + EncoderDecoder
  examples with argnums fixes (fixes silent differentiate-wrt-inputs), the
  `_widen`/`ALPHAGRAD_EXAMPLE_WIDTH` eval-width fix (fixes an all-zero-reward
  incident), `tools/log_to_csv.py`, and the **commit-SHA wandb-config logging**
  (the only line that has it; needs the hardcoded-path caveat fixed).
- **Superseded:** the graphax zero-point quant diff (we shipped sign-flip
  half-range instead); the MicroPPOAgent→ApproxAgent rename; STASH0 (the raw
  slimming diff — provenance only).
- **Donor patch:** STASH1 holds the quality-metric fixes our mainline lacks:
  degenerate → **(0.0, 1.0) worst**, `_align_jac` transpose alignment,
  `jnp.real(cos)`, latency-floor sentinel.

### Shared rot (present in every line — fix once, in mainline)

1. Degenerate quality = perfect (env.py:894-897).
2. Winsorized measurement nowhere; quality from eval sample 0 only.
3. Oracle replay ignores applied rules (`rules=()`).
4. Token truncation silent (counters never reach wandb in any runnable line).
5. compile-cache no-op single-process + documented ~9 MB/call cost_analysis and
   ~18 MB/call ResourceMonitor leaks → the 55403 end-of-run OUT_OF_MEMORY.
6. No git-SHA logging (except train_lean).
7. Phantom comments: calibration, Stage-C shaping, "mirrors mu0_ray*" (files
   don't exist), "(3,) mask" (it's 4), stale 3-op docstrings, `--allow-compress`
   help says off-by-default while default=True.

### Consolidated verdict

**Base everything on `ppo.py` (N)** — the only runnable trainer, with ratio-1
and the factored quant head already right. **Port from C** (adapting 10→8
channels): PopArt stack, neutral-μ sentinel handling, per-component KL +
early-stop, Pareto/hypervolume driver block, entropy anneal, and the mult-mode
cosine gate as the collapse countermeasure. **Cherry-pick from S**: SHA
logging, examples/argnums/width fixes, log_to_csv; port STASH1's quality
fixes into env.py. **Retire**: the async pipeline, the dead elif blocks, the
zero-point diff (superseded), and — after the ports — `ppo_ray*` itself.

**Phase A, sharpened by the forensics** (order):
A1. Reward fix in ppo.py: put cosine into the trained signal (4th value head
    or replace-frob — decide), fix the "acc"-label lie, and port the mult-mode
    cosine gate as an option (`--reward-mode mult`).
A2. env.py quality fixes from STASH1: degenerate → (0.0, 1.0), `_align_jac`,
    `jnp.real`, latency-floor sentinel.
A3. Collapse guard on best/top-N (reject flops==0 / latency==0 / cos==0 rows)
    + per-channel argmax guard.
A4. Oracle replay: advance with the actually-applied rules (both the ppo.py
    bridge and any lean driver).
A5. Re-run approx (exact 55409 stays valid — VE-only has no degenerate route;
    it only reorders).

---

## 1d. Trim (2026-07-26) — ppo.py 5718 -> 3949 lines

The audits' dead-code findings were acted on rather than filed. Removed with
44 local tests green before and after (commit `1e891bb`):
the whole legacy rule-policy family (RuleDecoder / AutoregRulePolicy /
SparsityRatio* / SingleRulePolicy, ~890 lines) plus its loss body, rollout
branch and Agent methods; BC warm-start; the cache-encoding path;
MLPVertexPolicy; 8 orphaned CLI flags and the use_pointer/use_autoreg
plumbing. `_HEAD_PATH_MARKERS` was repointed from the deleted `rule_policy.*`
paths to the live micro-action heads (the per-head LR ramp and freeze masks
had been matching parameters that no longer existed — a silent no-op).

**Why it was 5.7k:** three generations of experiment (rule-decoder ->
sparsity-ratio -> micro-actions) layered without deletion. The essential
algorithm is ~3-4k: it is genuinely ~2x the simple `alphagrad/ppo` example
because it adds a per-vertex sub-episode of typed approximation actions,
live-structure masking via a host oracle, a factored 7-way quant head, and
an 8-channel measured reward with a 4-head value function. Further real
reduction means MOVING code out (argparser -> args_ppo.py, logging -> its
own module), not deleting.

---

## 2. What is already DONE

### Core loop
- [x] Vertex-elimination pointer policy with availability masking
- [x] Palimpsa linear-attention encoder as the trunk
- [x] Tokenized-jaxpr observation (one-shot `extract_jaxpr`)
- [x] Dynamic sub-episode of micro-actions per vertex (`max_substeps`)
- [x] End-to-end training on nn256 / MNIST, GPU, 1000 episodes
- [x] `--exact` (VE-only) baseline — **fixed 2026-07-25** (`1129c4f`): it set a
      local var but not `args.variant`, so "exact" silently ran full approx
- [x] 8 reward channels: muls_adds_fmas, flops, latency_ns, max_io_sum,
      bytes_accessed, peak_memory, cosine_sim, frob_residual
- [x] Measurement via compiled grad-of-loss (XLA fusion), `peak_bytes_in_use`,
      `perf_counter`, cost/memory analysis; `uv run --no-sync`

### Approximations (graphax)
- [x] `Diag(i, j, factor)`, `Compress(axes, kind)`, `Quant(dtype, scale_sign)`
- [x] Quantization redesign: **sign-flip half-range** (not zero-point shift),
      scaling to the *target dtype's* max, sign folded into `scalar_mult`
- [x] Full ml_dtypes catalog (float4/6/8, int/uint 2..64) + `verify_hardware_compat`
      availability scan + contraction-compat matrix; 64-bit dropped unless x64
- [x] `QUANT_DTYPE_ATTRS` (kind, bits, exp, mantissa, bias, finite, unsigned_zero)
- [x] **Factored semantic quant head** (sign/kind/bits/exp/mantissa/bias/finite/uz)
- [x] int4 `abs` crash fix — general to all sub-byte dtypes (graphax `48efdac`)

### Masking (the requirement set)
- [x] Per-edge legality is decided **against the live operand**, not vertex ndims
- [x] `LiveVertexMaskOracle` — exact per-vertex `pair_valid` / `compress_valid`,
      replayed host-side, bridged into the jitted rollout via `pure_callback`
- [x] Fully-masked op ⇒ op removed from the categorical (no dead branches)
- [x] **Head skipping** — 0 options ⇒ head dropped; 1 option ⇒ forced, 0 logp/entropy
- [x] Quant masked to hardware-available dtypes
- [x] **Per-FACE legality masks** `face_masks()` (alphagrad `2168b5f`) — same screens
      as `vertex_mask` but not AND-ed over faces

### Infrastructure / hygiene
- [x] Local alphagrad unit testing on macOS (graphax venv + distrax/optax/tqdm/wandb
      + `jax_memory_monitor` stub) — see `local-alphagrad-testing` memory
- [x] `GRAPHAX_DISPATCH_STATS` telemetry (graphax `68fb9d7`)
- [x] Forced-densify investigation **closed**: correctness-safe by default;
      reproducer `diag_conflict_repro_test.py`; composed-dense fix deferred
      (14–16% of structured contractions but <600 MB on nn256)
- [x] Incremental tokenizer de-risked: bounded-vocab **incremental** id-safety
      pinned (graphax `c4f76f8`)

---

## 3. What is NOT done

### P0 — blocks every result
- [x] **Reward collapse.** FIXED: cosine_sim is now a trained value head
      (`--lambda-acc`, default 2.0), `--reward-mode mult` ports the cosine-gate
      + anti-degeneracy penalty, PopArt replaces the z-score ratchet, the
      degenerate-Jacobian backdoor now scores WORST, and collapsed rows are
      excluded from best/top-N. Commits 0a81d0a, 8fd8397, 61e7027.
- [~] **Memory stability.** Root-caused and mitigated: exact is host-memory
      heavy, so it must run SOLO with `--mem=0` (co-scheduling under one shared
      `--mem` is what OOM-killed it). Awaiting confirmation from the current
      runs. Original symptom: 55403 ended `OUT_OF_MEMORY`; the first exact run was
      OOM-killed at ep 65 sharing `--mem=200G`; exact solo then got requeued.
- [~] **Exact-vs-approx comparison IN FLIGHT** — jobs 55445 (exact, pgi15-gpu16)
      and 55449 (approx `--per-face`, pgi14-gpu12), plus 55450 (ADALIF-SNN).
      Run history (each failure real, each found only by running):
      * 55447 approx WITHOUT `--per-face` -> `TRANSFORM DID NOT FIT` (a Diag the
        per-vertex mask admitted did not fit the live edge) — exactly the class
        `--per-face` removes by construction.
      * 55449 approx WITH `--per-face` -> `TypePromotionError (float4_e2m1fn,
        float32)`: the quant hardware scan only probed same-dtype `dot`, so
        narrow types passed and then died against an un-quantized float32
        partner. Fixed in graphax 4715c00 (scan now probes float32 mixing;
        narrow types masked out unless GRAPHAX_QUANT_ALLOW_NARROW=1, which
        needs the partner-cast).
      * 55450 ADALIF-SNN -> `ptxas exited with error code 139` (compiler
        segfault, register spilling). TOOLCHAIN issue, not our code; needs XLA
        flags or a smaller SNN before it can be re-tried.
      * 55452 approx `--per-face` on the fixed graphax — in flight.

### P1 — spec compliance (action space + measurement)
- [x] **Per-path/per-face approximation.** `--per-face` wraps each vertex's
      rules in the per-face callable graphax invokes with that face's LIVE
      operand, so a rule lands only where legal on that path. Commit 02a4fd4.
- [x] **Skip operation** — implemented as the natural consequence of
      per-face application: a face on which no rule is legal is left
      exact. Pinned by a test (an all-illegal rule set is a byte-identical
      no-op, not a raise). Commit 02a4fd4.
- [x] **Incremental tokenization into the encoder** — done in two layers:
      (a) `env.incremental_token_delta` emits per-elimination deltas and
      `MAX_TOKENS` is configurable (commit bbbe68f); (b) `approx/face_stream.py`
      `InterleavedFaceDriver` gives the spec's ORDERING — path tokens emitted
      before the skip/approx decision for that face, chosen action emitted
      before the next substep sees it, palimpsa carry extended by only the new
      tokens (commit eeb162b).
- [x] **Winsorized measurement protocol** — 20 runs = 5 samples × 4 reps, looped
      (default 50), weights re-initialized once per episode and shared across envs.
      **CORRECTED by the 2026-07-26 forensics: live NOWHERE** — the knobs are
      forwarded and silently swallowed by `from_jaxpr(**_compat)`; the real
      implementation survives only in the STASH1 evidence patch (fat-era env).
      This is a rebuild-from-donor, not a port.
- [x] **PopArt in `ppo.py`** (exists in `common/popart.py`; genuinely
      updated+applied in `ppo_ray_worker.py` — but that line cannot construct
      against today's 8-channel env, so PopArt currently runs nowhere).

### P2 — logging (the 2026-07-25 spec)
- [x] Measurements — best / mean / median / worst, **all-time and current episode**,
      for: max_io, fmas, flops, bytes_accessed, xla_peak_memory, peak_memory,
      latency, cossim, frob_norm, **sparsity/compression (logical/physical size)**
- [x] **Collapse guard** on "best" (reject latency 0, flops 0, cosine 0, …)
- [x] Normalization — PopArt μ, σ, ev, advantage stats, final normalized value,
      scalarized reward
- [x] PPO — KL, grad norm
- [x] Pareto — hypervolume + 3 scatter plots (latency×cossim, memory×cossim,
      latency×memory) with episode number to show front movement
      (`common/pareto_archive.py` exists; not wired to `ppo.py`)
- [x] Entropy — mean, macro (vertex), micro (approximation) *(partially present)*
- [x] **Log alphagrad + graphax commit SHAs** alongside the dsnn/thesis SHA
      (`commit`/`git_sha`: 0 hits today)

### P3 — scope
- [x] GAZ brought to parity — `az_gumbel` imports and initializes again (measure_grad is now a scalar-output CONTRACT, not a hard raise; the 10-channel-era objective names resolve through an alias table). Commit d037765.
- [x] Transformer target; ALIF-SNN target (scan unrolled **and** not
      unrolled) — the EXAMPLES now exist and resolve
      (ADALIF_SNN / ADALIF_SNN_SEQ / EncoderDecoder, commit c8da4f6);
      and an ADALIF_SNN training run is launched (job 55450, --per-face).
- [x] Contraction-coupling: post inherits pre's forced quant, one quant per turn (`couple_quant_rules`, 02a4fd4)
- [x] Consolidate: `ppo.py` trimmed 5718 -> 3949 (commit 1e891bb);
      `measure_worker.py.bak_dbg` deleted; `ppo_ray{,_worker,_actors}.py` carry a
      DEPRECATED / DOES-NOT-RUN banner naming the five verified failures and
      pointing at COMPONENTS.md + ppo.py.

---

## 4. Order of work

**Phase A — make results trustworthy (P0).**
Collapse guard + accuracy constraint; fix memory topology (exact solo, `--mem=0`);
then re-run exact vs approx. *Nothing else matters until a run means something.*

**Phase B — logging (P2).** Cheap, independent, and it is how we will *see*
Phase A working (collapse guard, Pareto movement, PopArt stats, commit SHAs).

**Phase C — spec compliance (P1).** Winsorized measurement + PopArt into `ppo.py`;
then per-face stages 2–5; then the skip op; then incremental tokenization
(also fixes the 4096-token truncation).

**Phase D — scope (P3).** GAZ, Transformer, ALIF-SNN, folder consolidation.

Housekeeping (do early, cheap): review/push the 12 pgi15 alphagrad commits; diff
the 3 pgi15 graphax files then discard.

---

## 5. Verification

- Unit: `PYTHONPATH="graphax/src:alphagrad/src" ALPHAGRAD_DISABLE_RESOURCE_MONITOR=1 \
  graphax/.venv/bin/python -m pytest alphagrad/tests/<...> -q` (35 green today)
- graphax: `pytest tests/misc/test_face_transforms.py tests/misc/test_incremental_tokenizer.py \
  tests/core/sparse_tensor/ -q`
- Collapse guard: a trajectory with cosine 0 / flops 0 must **never** appear as best.
- Per-face: same order + two different face-transform sets ⇒ **different** measured
  cosine/flops (the check that would have caught the per-vertex dead end).
- Ratio invariant: sample == evaluate ⇒ PPO ratio 1.
- End-to-end: exact vs approx on nn256/MNIST, wandb online, 0 `DID NOT FIT`,
  0 sentinel measure-exceptions, and both runs reaching 1000 episodes.

---

## 6. Per-path loop — clarified semantics (2026-07-26, from the author)

The face/path sub-episode, once wired, is:

1. Path tokens (the local pass through the eliminated vertex) are emitted and
   encoded.
2. **One SKIP decision, at the start of the path only.** Skipping means the
   contraction of this path is not performed; emit `approx SKIP` and move to
   the next path. There is NO per-substep skip.
3. If not skipped, loop **`dynamic-substeps` times**: approximate PRE, then
   POST, then NEW — three head invocations in sequence with **no encoder call
   between the three** — then emit `approx: {diag,compress,quant}(*args)` plus
   the updated path accumulation and re-encode before the next substep.

Deltas against the existing (dormant) `face_stream.AutoregressiveFaceDriver`:

* it currently offers skip-or-approximate at EVERY substep — must become a
  single gate at path start;
* it currently calls `encoder.extend` once per SLOT (pre/post/new) — must
  batch the triple and extend once per substep;
* its SKIP comment says "the edge is deleted" — wrong on both counts today
  (the driver leaves the contraction to proceed); under the clarified
  semantics SKIP must actually suppress the contraction of that path.

---

## 7. Per-path decisions — P1 design (2026-07-27)

Engine prerequisites are DONE: `graphax.SKIP_FACE` (true path-skip: contraction
dropped, `approx SKIP` tokenized, all-skip ⇒ zeros-with-structure via jacve,
gxf 00a60fa) and the face_transforms prefix-cache guard (same commit). The
learned face heads described in the 07-22 README do NOT exist in this tree —
P1 is a build, not a wiring job.

### Scope (dynamic-substeps = 1, per the spec's "for now we only do this once")

Per chosen vertex, for each face k < MAX_FACES (masked by `k < n_faces`):
1. ONE skip gate at path start (APPROX_PLAN §6). Skip ⇒ the env installs
   `SKIP_FACE` for that face key.
2. Else THREE slot decisions (pre / post / new), each a single micro action
   (op ∈ {DIAG, COMPRESS, QUANT, END=no-approx}) sampled from the existing
   head zoo under the face's oracle masks.

### Masking / ratio-1 contract

* Sampling masks: `LiveVertexMaskOracle.face_masks(v, MAX_FACES)` —
  (F,N,N) pair + (F,N) compress, probed on each face's live contraction in
  canonical visit order (same order `faces_of` / the env enumerate). The SAME
  masks serve all three slots at sample time — an approximation of slot-level
  legality (lhs/rhs are join intermediates, unknowable pre-elimination).
* Apply-time truth: each non-skip slot's rule is wrapped in
  `make_live_masked_hook`, so a rule illegal on ITS operand is skipped there
  (never raises), counted in per_face_stats.
* Ratio-1 needs only that the loss re-masks with the STORED sampling masks —
  store (face_pair_valid, face_comp_valid, face_valid) in the trajectory,
  exactly like the vertex path stores its oracle masks.

### Wire format

`StepAction` grows `face_rows (MAX_FACES, 3, 3) int32` (slot-major spec rows,
same encoding as `sparsity_specs` rows: DIAG/COMPRESS_SENTINEL/QUANT_SENTINEL/
-1=none) + `face_skip (MAX_FACES,) int32`. `EnvState.face_specs_hist` mirrors
it per step for oracle replay + incremental-token reproduction. `_callback`
builds `face_transforms={key_k: SKIP_FACE | (hook(pre), hook(post), hook(new))}`
with keys enumerated at elimination time (`faces_of` order == oracle order).

### Build order

P1a. env: wire format + `_callback` application + hand-built-action test
     (no policy). Flag: `--face-actions` (default off, campaign-neutral).
P1b. heads: `FaceSkipHead` (binary, from v_context + face mask-summary
     features) + per-slot single-step decision reusing OpType/Axis/Prime/
     Kind/FactoredQuant heads; sample/evaluate mirror.
P1c. ppo: rollout sampling (face_masks via the existing pure_callback oracle
     bridge, extended to return face masks), trajectory fields, loss
     log-prob/entropy, per-face KL later.
P1d. smoke: exact arm unaffected (flag off); approx smoke with --face-actions
     on Helmholtz; verify per_face_stats applied>0, SKIP tokens in stream,
     ratio finite, distinct rewards across seeds.

Phase 3b (autoregressive re-encode between faces via incremental extend +
vertex memory) layers ON TOP of P1 — P1 decides all faces from the
pre-elimination encoding.

### 7b. Phase 3b design — autoregressive re-encode (carry-storage route)

The palimpsa recurrent state is small enough to STORE PER STEP
(layers x heads x d x d for M and I ≈ ~128 KB at embd 128 → ~24 MB per
16-env x 12-step rollout), which dissolves the loss-side reconstruction
problem: the trajectory stores each step's PRE-step carry, the loss extends
it by that step's DELTA tokens only — same stored-context ratio-1 pattern as
the oracle masks and axis_state. Build order: (1) expose (init_state,
extend) on the mainline PalimpsaEncoder (port from incremental_encoder.py,
which is written against the deprecated MicroPPOAgent); (2) env returns
per-step delta tokens + length (the incremental stream cache already knows
them); (3) rollout carries (palimpsa state, vertex memory) through the scan,
pointer/value read from vertex_memory (from_vertex_memory — written, tested,
unwired); (4) FacePathPolicy conditions on the carry AFTER each face's
emission (APPROX_PLAN §6 ordering: one extend per substep, none between the
pre/post/new triple).

STATUS: BUILT (--incremental-encode, default OFF). Deviations from the
sketch, discovered during the build:

* (2) needed NO env changes: pad id 0 never occurs in a real stream, so
  stream length is `sum(tokens != 0)` and the delta window is gather-sliced
  (`jnp.take` mode="fill") from the already-stored append-only buffer at
  the carried `pos`. Delta ownership is the rollout's own `elim_order` —
  the delta visible at step t was emitted by step t-1's elimination.
* The mainline PalimpsaMixer's relational forget-gate counts same/earlier/
  later over the WHOLE stream — acausal, unrepresentable in a carry. The
  incremental path redefines the features CAUSALLY from a carried
  cumulative eqn-id histogram (MAX_EQNS buckets, ALPHAGRAD_MAX_EQNS=4096).
  Rollout and loss share the computation exactly; ratio-1 is between those
  two, never against the full path (which stays byte-identical when the
  flag is off).
* Vertex-memory ids: base-stream eqn ids map to vertex slots positionally
  (`eqn < total_v`, else global slot); delta rows land in the owner
  vertex's slot; structural (eqn -1) rows in the global slot
  (vertex_memory.update_ids).
* Gradient truncates at the stored pre-step carry (flows through the last
  delta + heads only) — the accepted cost of the carry-storage route.
* next_value does a second extend per step (kept self-contained instead of
  threading heads across scan iterations; deltas are O(hundreds) tokens).
* (4) holds trivially at max_substeps=1: the face heads consume
  vertex_contexts from the carry-derived heads_from_memory, and the step's
  emission lands in the NEXT step's delta. Multi-substep re-extends are
  still future work, gated on max_substeps>1 becoming real.

Tests: tests/incremental_encode_test.py (chunked-extend bitwise, prefix
property under nonzero rel_gate, vmem associativity, precomputed-hook
sample/evaluate parity).
