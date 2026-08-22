# Face-latent information loss: why the face probe decodes at baseline while the vertex probe is near-perfect

**Evidence base.** v61 (job 61610, `/Users/assmuth/dsnn/v61_tlm_61610.log`, TLM wikitext,
`--var-probe`, FORCE REV ORDER, face head identity-init P(approx)≈0.007): vertex probe
ndim 1.00/0.75(base), shape-exact 0.98-1.00, size-R² 0.96-0.99; face probe ndim 0.48/0.47
(lhs), 0.77/0.76 (rhs), 0.65/0.64 (res), dtype at baseline, size-R² 0.01-0.02. The face
probe *did* train — loss 16.04 → ≈3.0 over the first ~25 episodes — and then converged to
exactly the majority-class predictor, loss frozen at 3.228 to four digits for the rest of
the run while the vertex loss kept falling (0.105 → 0.095).

All offline numbers below come from one instrumented CPU episode of the same target
(reverse order, all-exact — v61's operating point) captured with the production
`LiveFaceStream.chunk` path and a *keyed* variant of the oracle target builder:
`/Users/assmuth/dsnn/probe_invest/` (scripts `capture_tlm.py`, `analyze_stage1.py`,
`analyze_stage1b.py`, `analyze_markov.py`, `analyze_stage2.py`; artifacts
`data/tlm_capture.pkl`, `data/stage2_results.txt`, logs `data/run_all_61664.log`,
`data/run_fix_61665.log`, `data/run_fix2_61666.log`). 115 faces / 95 steps; the capture
reproduces the online target marginals exactly (ndim majorities 0.47/0.76/0.64 = the
online baselines), so it is measuring the same population. Code line numbers are as of
alphagrad `9a0661c` (branch hostperf-caches).

## 1. The two read paths as built

**Vertex path — an accumulated, attribution-anchored read.** Every palimpsa row of the
870-token base stream is scattered into its owning vertex's slot
(`carry_stream.init_carry`, src/alphagrad/approx/common/carry_stream.py:77; owners from
`IncrementalPathTokenizer.last_owner_ids`), every step delta's rows into all
*participating* slots (`carry_stream.advance`, carry_stream.py:217;
`participation_mask`, ppo.py:2322). `heads_from_memory` (ppo.py:1956) reads
`(base+dyn sums)/counts` and runs the content-based Set pointer (2 SetBlocks,
set_pointer.py:171). Probe input: `vertex_contexts[vertex_idx]` (ppo.py:2080, packed
ppo.py:6939). Probe targets: **static** avals of the vertex's equation in the original
jaxpr (`vertex_var_table`, var_probe.py:264) — constant per vertex, every episode.

**Face path — a transient, chunk-local mean.** Face f's chunk = face f-1's approximation
tail + face f's contraction (live_faces.py:627-637), mean ~77 tokens (max 163-183).
Rollout: `_face_encode` (ppo.py:2272) extends a side carry and takes the parameter-free
single-segment `scatter_mean` of the chunk's palimpsa rows → one E=32 vector. Loss/probe:
`_face_replay` (ppo.py:2428) re-encodes the stored emission window and pools by
`searchsorted(cumsum(face_counts))` (ppo.py:2515). The head input is `face_latents[f]`
and nothing else (`UnifiedFacePolicy._repr`, unified_face_policy.py:213; the endpoint
contexts were deleted 2026-08-15 — ppo.py:2341). Probe targets: **live** stored-form
shapes/dtypes of (lhs, rhs, res) from an oracle prefix replay (`face_var_targets_host`,
var_probe.py:180).

Two corrections to the working sketch: (a) the chunk rows ARE post-recurrence — full
3-layer palimpsa outputs, carry-dependent (`_extend_sequential`, ppo.py:1650) — the face
path does not read raw embeddings; (b) the vertex probe input is post-Set-pointer, not a
raw slot row.

## 2. Hypotheses, each with its decisive test

### H-A Key misalignment (#74 family) — REFUTED as the cause; one real hygiene bug found
- Offline, a keyed recorder (hooks tagged with the face key) against the production
  positional parse: **115/115 rows aligned (100%)**, emission order == enumeration order
  in all 32 multi-face steps (`run_all_61664.log [H-A]`).
- The live drop channel is real but tiny and *counted*: capture 3/118 chunks dropped
  (`face_dropped=face_key_seg_mismatch=3`), v61 online 29-32 per ~1133-1295 chunks
  (~2.5%). Since the by-key mapping fix (live_faces.py:598-608) a dropped face yields an
  **empty chunk → zero latent** for that face only; its target stays valid, so 2.5% of
  probe rows train a zero latent against a real target — a noise floor, not a shift.
- Residual hygiene defects (fix regardless, they bite once approximations are real):
  (i) `face_var_targets_host` parses hook events positionally ("new face at each lhs",
  var_probe.py:249-256) — after a mid-elimination oracle failure the tail rows
  misassign; key it by face key like live_faces keys chunks. (ii) The vp-target oracle
  replay is built from per-vertex specs only (`_oracle_replay`; under --live-faces those
  are all-exact), so **face approximations are invisible to the target graph** — targets
  drift as the face head starts approximating (irrelevant at v61's 0.7% approx rate,
  not at a trained v63's).

### H-B Missing content — the chunk does not carry the values; the STREAM does
- A chunk is `path <central> & <pred> & <succ>` + name-only equations
  (`out = F : a _ b`); shape integers ride only in first-sight `fns` definitions
  (graphax jaxpr.py:1170-1240) and approx payloads. Digit-token audit (fixed mapping,
  `run_fix2_61666.log`): all of a target's non-trivial dims appear in the face's OWN
  chunk for only **13-16%** of faces — but in the stream-so-far (base header + earlier
  `fns`) for **100%** of faces.
- So the owner's premise is confirmed: the stream carries the information; the chunk
  alone mostly does not. Decoding live shapes from the chunk-mean therefore relies on the
  RECURRENCE having bound shape facts to variable names — which works partially at init
  (below) and is what training destroys.
- The online "mild dims lift" is not latent signal at all: the shape-GRU is
  teacher-forced (var_probe.py:330-346), and a latent-free (position, prev-bucket)
  majority table alone scores **0.69 / 0.80 / 0.74** (lhs/rhs/res) — identical to the
  online 0.69 / 0.80 / 0.74 (`analyze_markov.py`). Discount this metric entirely.

### H-C Pooling washout — REFUTED as primary
Same frozen random-init palimpsa rows, same ridge protocol, four parameter-free poolings
(`stage2_results.txt`; train / 3-fold-CV, majority in parens):

| pooling | lhs ndim | rhs ndim | res ndim | lhs szR² | res szR² |
|---|---|---|---|---|---|
| mean (production) | 0.82/0.60 (0.47) | 0.86/0.73 (0.76) | 0.89/0.68 (0.64) | 0.45 | 0.55 |
| sum | 0.85/0.55 | 0.87/0.73 | 0.90/0.64 | 0.42 | 0.56 |
| max | 0.81/0.59 | 0.87/0.71 | 0.87/0.73 | 0.45 | 0.50 |
| mean‖log-count | 0.83/0.59 | 0.85/0.73 | 0.88/0.68 | 0.45 | 0.58 |

All four are statistically indistinguishable: switching pooling buys nothing. (raw-
embedding mean is comparable too — the info at this read-point is token-content-level.)

### H-D Read-point — the real lever, in two parts

**(1) The chunk-local read DOES carry signal at init.** Plain ridge on the production
chunk-mean (32-dim, n=115) decodes ndim at 0.82-0.89 train vs 0.47-0.64 majority and
size at R² 0.45-0.55 train. A linear probe finds face structure in exactly the tensor
the head reads — *at initialization*.

**(2) Training drains it.** v61's own timeline (one `[vprobe]` line ≈ one episode):
lhs ndim rises to **0.64 by ~ep17** (base 0.47), szR² to **0.18**; both decay smoothly —
0.56, 0.52, 0.51, 0.50 — to exact baseline by ~ep60 and freeze there for the remaining
~100 episodes, probe loss constant at 3.228 to four digits (gradient ≈ 0: the head sits
at the marginal-entropy optimum of an uninformative input). The face-stream pipeline is
unchanged late in the run (health lines: chunks/tok_total/failure counters stable), so
the decay is representational, not mechanical. Meanwhile the vertex probe holds 1.00
throughout — its read is anchored: slot contents are dominated by the base stream's
attribution-routed rows and its targets are static, so under FORCE REV ORDER it is a
memorizable identity task (control: random-init palimpsa + ridge already reaches ndim
0.86-1.00 train on the vertex read, `stage2_results.txt` vertex-control block).

Nothing in the loss rewards keeping face-latent variance: at identity init the face head
receives ~no advantage signal (every decision is NONE), no other consumer reads
`face_latents`, and the palimpsa weights are being pulled hard by the vertex/value paths.
The chunk-mean is a transient functional of a moving encoder; whatever face information
it carries at init is free to be optimized away — and measurably is.

**Read-point comparison** (same targets, same protocol): the face's own ENDPOINT slot
rows — already stored per face (`face_endpoints`, `_ends`, live_faces.py:439-443;
participation already routes every face delta into those slots) — decode size better
than the chunk-mean, and the concatenation dominates everything:

| read-point | lhs ndim | rhs ndim | res ndim | lhs szR² | rhs szR² | res szR² |
|---|---|---|---|---|---|---|
| chunk mean (as built) | 0.82 | 0.86 | 0.89 | 0.45 | 0.48 | 0.55 |
| endpoint slot rows | 0.88 | 0.95 | 0.97 | 0.60 | 0.80 | 0.74 |
| mean ‖ endpoint slots | **0.99** | **1.00** | **1.00** | **0.90** | **0.95** | **0.94** |

(train numbers; at n=115 the CV columns are noise — the online task is the train regime:
same faces every episode.) The prior probe round agrees from the live side: job 61458's
lean arm decoded live targets at R² 0.03-0.18 while the same protocol's endpoint-derived
controls scored 0.33-0.78 *during training* — the slot read stayed informative under
exactly the optimization pressure that killed the chunk read
(docs/QUALITY_COLLAPSE_INVESTIGATION.md §4).

Cross-checks: #88 `residual_state` exists only in gfn_ray_worker.py / alpha0.py — the PPO
face path never touches it. The face scatter reads no vmem, so #156's attribution changes
cannot move the face latent; conversely the proposed fix below *consumes* the
participation-attributed slots, so #156 improves it for free.

## 3. Root causes, ranked

1. **Training-induced collapse of the transient chunk-local read** (evidence: strong —
   online rise-then-decay to exact marginals with frozen loss; offline init-time
   decodability at the same read-point; healthy late-run stream telemetry; vertex-side
   immunity explained by its anchored, static-target task). This is "the gap between
   palimpsa→vertex and palimpsa→face": the vertex head reads palimpsa's *persistent*
   memory, the face head reads a 77-token *transient* of it that nothing anchors.
2. **Chunk content is name-only** (evidence: strong, mechanical — 13-16% of targets'
   dims present in the chunk vs 100% in the stream): even a perfect chunk-mean must rely
   on the recurrence for shape facts, lowering the ceiling of any chunk-only read.
3. **Teacher-forcing artifact** (exact match 0.69/0.80/0.74): the one metric that looked
   like surviving signal is target autocorrelation, not representation.
4. **Target hygiene** (real, currently small): positional target parse; face-blind
   oracle replay for targets; 2.5% zero-latent faces. None explains the baseline
   convergence today; (b) will corrupt the probe precisely when the face head starts
   approximating.
5. **Key misalignment / pooling choice: refuted** (100% alignment; all poolings equal).

## 4. Minimal stream-pure fix

**Change WHERE the face read happens — add the face's own endpoint-slot reads to the
face latent.** Concretely: in `_face_loop` / `_face_replay`, gather the two
participation-attributed slot rows for the face's stored endpoints
(`read(base+dyn)[face_endpoints-1]`, zero row for endpoint 0) and hand the head
`[chunk_mean ‖ slot_i ‖ slot_j]` (3E=96) — or, to keep the head width, the parameter-free
mean of the three. Everything needed is already plumbed: `face_endpoints` is stored per
face, the slots already receive every face delta via `participation_mask`, and
`_vmem.read` is the identical primitive the vertex path uses. This is pure data routing
in the token stream's own representation — no side-channel features (no #94
`face_sizes`), no graph encoder. It attacks cause 1 (the slot read is the anchored,
collapse-resistant representation — the online 61458 controls prove it survives
training) and cause 2 (the slots aggregate the rows where the operands' defining tokens
actually live — the stream-so-far, which carries 100% of the values).

Cheap accompanying hygiene (same patch or v63 follow-up): key `face_var_targets_host`
rows by face key; thread the prefix's face wires into the vp-target oracle replay; log
`std(face_latents[valid])` once per episode as collapse telemetry.

## 5. Falsifiable predictions for a fixed v63

With the endpoint-slot read concatenated (probe input 96-wide), same probe protocol:
- `probe/face/*/ndim_acc` starts ≥ 0.60 (lhs) / ≥ 0.85 (rhs) / ≥ 0.80 (res) and — the
  actual test — does NOT decay to its majority baseline: still ≥ baseline+0.10 at ep150
  (v61 was back at baseline by ~ep60).
- `probe/face/*/size_r2` ≥ 0.30 somewhere in ep0-30 and ≥ 0.15 at ep150 (v61: peak 0.18,
  then 0.01).
- `probe/face/*/shape_dim_acc` stays ≈ 0.69/0.80/0.74 whether or not the fix works —
  teacher-forcing floor; treat `shape_exact` and `ndim` as the real shape metrics.
- If the concatenated probe ALSO decays to baseline while `probe/vertex/*` holds ≈1.0,
  the collapse is upstream in the shared rows and the next lever is palimpsa row-collapse
  telemetry (attention-entropy diagnostic), not the read-point.

## 6. Implementation (v63)

The section-4 fix is implemented in commit `0605ad0` behind
`--face-endpoint-read` (default OFF; the disabled path is bit-identical to
v62 -- pinned in `tests/endpoint_read_test.py`). When on, both
`_face_loop` (rollout) and `_face_replay` (loss) hand the head
`[chunk_mean || slot_i || slot_j]` (3E = 96): the two endpoint slot rows are
`read(base+dyn)[face_endpoints-1]` with a zero row for endpoint 0, gathered
from the memory synced through the previous elimination's delta (the state
that exists when the step's faces are decided). `UnifiedFaceHead` widens to
`in_dim = 3E`; the gradient flows through the endpoint rows into palimpsa
(the read is a POLICY input -- that is what anchors it), while the var probe
decodes the SAME concatenation through its own stop-gradient-isolated heads
(`VarProbes(face_in_dim=3E)`), so the section-5 predictions are measured on
exactly the tensor the head reads. Launcher:
`~/dsnn/fq_v63_tlm_endpoint.sbatch` (v62 config + the flag). az_gumbel does
not support the flag yet and fails loudly if handed an endpoint-read policy.
