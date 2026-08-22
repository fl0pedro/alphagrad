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

## 7. Numeric-lexicon ablation — REFUTED

**Hypothesis tested** (owner's): shape dims/sizes enter the token stream as
bare digit-symbol token ids; the embedding treats them as unrelated symbols,
so magnitude/order/product structure is unlearnable — and THIS, not
routing/anchoring, is what caps the face probes (sizes-are-products worst,
supervised ceiling == probe plateau, vertex 1.00 = static-target
memorization). Decisive prediction if TRUE: a numeric channel at the
embedding layer lifts chunk-mean size-R² from ~0.45-0.55 to >0.85 and ndim
to ~1.0; if it stays ~baseline, the hypothesis is falsified.

**Where dims become tokens** (graphax `4ea0bf8`,
src/graphax/jaxpr.py): `IncrementalPathTokenizer._format_shape`
(jaxpr.py:1315-1324) spells a shape as `< d0 * d1 * ... >` with each dim
rendered MSB-first in base-10 by `int_to_base` (jaxpr.py:755); input shapes
enter the base header at jaxpr.py:1256 (`base_tokens`), and numeric op
params (reshape `new_sizes`, dot dims, scalar literals, dtype tags) enter
first-sight `fns` definitions via `tokenize_value` (jaxpr.py:887-925)
called from `_emit_op_params` (jaxpr.py:1165). Every digit symbol lands in
the token-id range `[len(vocab), len(vocab)+10)` = [230, 240) via
`_emit_symbols` (jaxpr.py:884); name atoms occupy [240, vocab_size), with
`ALPHAGRAD_INCR_TOKEN_VOCAB` (=512 here) bounding the name alphabet.

**Protocol** (`probe_invest/lexicon/analyze_lexicon.py`, job 61852, outputs
`lexicon_results.txt` / `lexicon_reprs.npz` / `run_61852.log`): same
capture, same replay, same ridge probes as stage 2 (§2 H-C/H-D). Ground
truth: every maximal digit-token run in every stream (base, per-step
emissions, per-face chunks) is parsed back to its numeric value v. Each
digit occurrence is remapped to a synthetic id unique to its
(digit-symbol, v) pair and the frozen `eqx.nn.Embedding` table is extended
so the new row implements the arm — (a) original digit row (baseline,
reproduces stage 2 bit-for-bit); (b) digit row + s·P·phi(v); (c) s·P·phi(v)
only (no symbol identity); (d) digit row + s·MLP(phi(v)) (random 2-layer
MLP), with phi(v) = [log1p v, sin/cos(w·log1p v), w ∈ {0.5,1,2,4,8,16}]
column-standardized and s matching the numeric component's std to the
symbol table's. This is exactly the candidate production change
(embedding-layer-only symbol_embed + numeric-enc for value tokens) injected
BEFORE the recurrence, so it reaches both reads through the full pipeline.

**Lexicon census first** — the premise is structurally thinner than
assumed: the whole episode stream contains **1126 digit-token occurrences
but only 8 distinct numbers** (19 distinct (digit,value) pairs), and only
**2 distinct dims ever appear in shape context**: 128 (23 runs) and
1024 (1 run). Most face-target dims never enter the stream as numbers at
all — the sizes the probe must predict are PRODUCTS created by graph
structure (contractions), never spelled as tokens (§2 H-B said 13-16% of
targets' dims appear in the own chunk; the census sharpens it: as *shape
dims* almost nothing appears anywhere).

**Results** (train/cv, majority in parens; n=115 faces):

| read | slot | arm | ndim | size-R² | dtype | dim0-bucket |
|---|---|---|---|---|---|---|
| chunk mean | lhs | a | 0.82/0.60 (0.47) | **0.45**/−0.64 | 0.94/0.89 (0.91) | 0.78/0.50 (0.48) |
| chunk mean | lhs | b | 0.86/0.67 | **0.50**/−0.47 | 0.93/0.89 | 0.79/0.50 |
| chunk mean | lhs | c | 0.86/0.60 | 0.45/−0.80 | 0.93/0.90 | 0.73/0.49 |
| chunk mean | lhs | d | 0.86/0.63 | 0.47/−0.59 | 0.94/0.87 | 0.80/0.53 |
| chunk mean | rhs | a | 0.86/0.73 (0.76) | **0.48**/−0.25 | 0.99/0.99 (0.99) | 0.96/0.87 (0.89) |
| chunk mean | rhs | b | 0.85/0.73 | **0.44**/−0.43 | 0.99/0.99 | 0.96/0.90 |
| chunk mean | res | a | 0.89/0.68 (0.64) | **0.55**/−0.95 | 1.00/0.99 (0.99) | 0.93/0.79 (0.77) |
| chunk mean | res | b | 0.88/0.60 | **0.54**/−1.18 | 1.00/0.99 | 0.93/0.84 |
| mean‖ep_slot | lhs | a | 0.99/0.48 | 0.90/− | 1.00/0.64 | 1.00/0.41 |
| mean‖ep_slot | lhs | b | 1.00/0.45 | 0.92/− | 1.00/0.68 | 1.00/0.39 |
| mean‖ep_slot | rhs | a/b | 1.00 / 1.00 | 0.95 / 0.95 | 1.00 / 1.00 | 1.00 / 1.00 |
| mean‖ep_slot | res | a/b | 1.00 / 1.00 | 0.94 / 0.92 | 1.00 / 1.00 | 1.00 / 1.00 |

(arms c and d track a/b within ±0.05 everywhere; full table in
`lexicon_results.txt`.) The decisive number — chunk-mean size-R² — moves
0.45→0.50 (lhs), 0.48→0.44 (rhs), 0.55→0.54 (res): **noise, nowhere near
the predicted >0.85**; ndim stays ~0.85, not ~1.0. The endpoint-slot
concatenation was already at 0.90-0.95 under symbols alone and does not
move either. Vertex-side control: ctx-read ndim/szR² is unchanged by the
numeric channel (lhs 0.86→0.89, rhs szR² 0.82→0.88, res 0.45→0.49 train —
same within-protocol jitter), confirming no lift is needed where targets
are static.

**Frequency-tracking check** (cv preds, chunk-mean read, faces binned by
stream frequency of the target's largest dim): the memorization account
predicts accuracy tracks per-symbol frequency under (a) but NOT under (b).
Observed: identical tracking in both arms — ndim-acc 0.61→0.73 (a) vs
0.60→0.71 (b) from rare-dim to frequent-dim bin; dim0/|szerr| likewise
arm-independent. The frequency signature is a property of the targets and
the read, not of the symbol lexicon.

**Verdict: REFUTED.** Giving the embedding layer full magnitude information
(linear, Fourier, or MLP-encoded; with or without symbol identity) changes
nothing at either read-point. The face-probe cap is not a
numeric-lexicon/embedding problem: with a 2-symbol dim lexicon there is no
magnitude structure to compose, and the sizes the probe misses are products
that never appear in the stream as tokens — the bottleneck stays where §3
put it: (1) the transient, unanchored chunk-mean read that training drains,
and (2) name-only chunk content whose shape facts must ride the recurrence.
The §4/§6 endpoint-slot read (`--face-endpoint-read`, commit `0605ad0`)
remains the live lever; a v64 that instead shipped
symbol_embed+MLP(numeric) at the embedding layer is predicted to be a null
experiment (face size-R² still ~0.0-0.2 online, ndim decaying to majority
as in v61). Mechanically, if a numeric channel is ever wanted for
richer-dim targets (ViT-scale shape diversity), this ablation validates the
embedding-layer-only route: extend the table above id 512 keyed by
(digit, value) pairs — it composes with `ALPHAGRAD_INCR_TOKEN_VOCAB` (pair
ids must sit above the name range) and is untouched by the two-name-
universes constraint (digit ids are shared by both universes; only names
diverge).

## 8. Edge-keyed memory — H1 (endpoint read degrades on intermediates) CONFIRMED, H2 (edge-keyed fix) CONFIRMED

**Hypotheses tested** (owner's): **H1** — the §4/§6 endpoint-vertex read
under-specifies faces whose lhs/rhs are ACCUMULATED INTERMEDIATE Jacobians
(not primitives): `vmem_i` entangles ALL edges incident to vertex i, so the
binding "which events belong to THIS edge" fails as fill-in accumulates.
**H2** — an EDGE-KEYED memory (the same parameter-free scatter/segment-mean
primitive as vmem, keyed by canonical edge id) restores decodability for
intermediate operands.

**Edge identity.** The canonical edge id is a pair in the face-key space
(`_stable_var_index` positions, graphax core.py:896-908): a face with key
(i, j) at central v has lhs edge (i, vidx(v)) (in-edge Jacobian, `pre_val`),
rhs edge (vidx(v), j) (`post_val`), and res edge (i, j) — **the face key IS
the res-edge id** (`faces_of`, core.py:1154-1215). An edge is INTERMEDIATE
iff it was a PRIOR face's res edge in the episode (created/updated by
fill-in), else PRIMITIVE (an original elemental partial).

**The rev capture is stratum-degenerate.** Classifying the 115-face §2
capture (`probe_invest/edgemem/classify.py`, job 61853): **1 both-primitive
/ 114 one-intermediate / 0 both-intermediate** — under FORCE REV ORDER a res
edge (i, j) always has j past the central, so no prior elimination can have
written an edge ENDING at a not-yet-eliminated vertex: every lhs is
primitive, nearly every rhs intermediate. All §2-§7 face numbers were
measured on this one stratum. Extension (`capture_ext.py`, job 61854): three
episodes with seeded RANDOM orders, same protocol otherwise — 938 faces =
164 bp / 311 oi / 463 bi (929 with keyed truth + chunk; 3 dropped
faces/episode as in §1).

**Protocol** (`analyze_edgemem.py`, job 61857, outputs
`edgemem_results.txt` / `edgemem_faces.pkl`): same frozen random-init
palimpsa + ridge probes as stage 2. emem write = for each emitted face,
its emission rows (per-face spans from `last_face_segments`) accumulate
into bucket[res edge = face key]; reads are strictly-prior-step (online-
causal), vmem reads post-step as in stage 2. Arms: **A** chunk-mean, **B**
chunk‖vmem_i‖vmem_j (v63 read), **C** chunk‖emem_lhs‖emem_rhs, **D** =
B∪C (all five), plus emem-only. One global ridge per arm/slot/target
(fit on the pooled 929), metrics per stratum; size-R² uses the stratum's
own mean (within-stratum SST). Harness validation: replaying the rev
capture reproduces stage 2 — arm A ndim 0.84/0.86/0.89, szR² 0.46/0.47/0.54
(§2: 0.82/0.86/0.89, 0.45/0.48/0.55); arm B 1.00 ndim, szR² 0.90/0.95/0.93
(§2 mean‖ep_slot: 0.99-1.00, 0.90-0.95).

**Stratified results** (random-order pool, n=929; train/cv, majority in
parens; full table in `edgemem_results.txt`):

| slot | arm | both-prim (n=158) | one-inter (n=308) | both-inter (n=463) |
|---|---|---|---|---|
| lhs ndim | A | 0.78/0.72 (0.53) | 0.54/0.52 (0.51) | 0.52/0.45 (0.43) |
| lhs ndim | B | **0.85**/0.74 | 0.70/0.56 | **0.61**/0.49 |
| lhs ndim | C | 0.79/0.78 | 0.73/0.67 | **0.72**/0.55 |
| lhs ndim | D | 0.89/0.78 | 0.83/0.69 | **0.77**/0.57 |
| lhs szR² | B | **0.39**/−0.17 | 0.27/0.05 | **0.19**/−0.06 |
| lhs szR² | C | 0.33/0.22 | 0.60/0.49 | **0.44**/0.20 |
| lhs szR² | D | 0.52/0.10 | 0.68/0.46 | **0.53**/0.23 |
| rhs ndim | B | 0.74/0.58 (0.47) | 0.70/0.55 (0.45) | 0.76/0.68 (0.56) |
| rhs ndim | C | 0.68/0.65 | 0.70/0.60 | **0.89**/0.75 |
| rhs ndim | D | 0.82/0.64 | 0.82/0.66 | **0.93**/0.84 |
| rhs szR² | B | 0.32/−0.07 | 0.52/0.32 | 0.37/0.19 |
| rhs szR² | C | 0.25/0.18 | 0.58/0.45 | **0.60**/0.37 |
| rhs szR² | D | 0.49/0.18 | 0.68/0.48 | **0.67**/0.40 |
| res szR² | B | 0.17/−0.26 (0.53) | 0.44/0.24 (0.43) | 0.44/0.33 (0.37) |
| res szR² | C | 0.05/−0.07 | 0.43/0.27 | 0.49/0.28 |
| res szR² | D | 0.21/−0.35 | 0.55/0.21 | **0.59**/0.36 |

**H1: CONFIRMED.** The endpoint read (B) drops materially from its primitive
stratum to the intermediate ones — lhs ndim 0.85 → 0.61 train (0.74 → 0.49
cv; majority-margin 0.32 → 0.18), lhs szR² 0.39 → 0.19 — and monotonically
with depth (step terciles, lhs: ndim 1.00→1.00 on rev but 0.85 → 0.71 →
0.59 on random orders while C holds 0.76/0.75/0.71). The chunk-mean drops
too, as predicted (lhs ndim 0.78 → 0.52). rhs/res margins shrink more
mildly (B rhs margin 0.27 → 0.20). Note the rev capture hid all of this:
its one populated stratum sits where B scores 0.90-0.95 szR².

**H2: CONFIRMED** (operand slots). On the both-intermediate stratum C beats
B everywhere it structurally can: lhs ndim 0.72 vs 0.61 (szR² 0.44 vs
0.19), rhs ndim 0.89 vs 0.76 (szR² 0.60 vs 0.37); measured as
majority-margin recovered relative to B's own both-primitive stratum, C-on-
both-inter reaches 91% (lhs ndim), >100% (lhs szR², rhs ndim, rhs szR²) —
the ≥90% criterion is met on all four decisive numbers. D (B∪C) dominates
or ties every cell, including res (szR² 0.59/0.36 vs B 0.44/0.33). Sanity
checks: emem-only is exactly at majority on both-primitive faces (no
writes: coverage 0 by construction, 1.00/1.00 on intermediates) yet reaches
0.83/0.73 ndim, 0.57/0.41 szR² on both-inter rhs — the edge-keyed rows
alone carry most of the intermediate signal. The res slot is the one place
C cannot act (the res edge is being created NOW; its first write is this
face's own emission) — there the information must come through the
operands, which is what D does.

**Practicality** (per episode): distinct edge keys 206-286 on random orders
(rev: 101); tokens/key mean 129-169, median 61-77, max 2631; emission↔key
alignment failures 0 of 285 steps; every intermediate operand's bucket had
≥1 write. **Edge keys are recoverable ONLINE from the delta stream**: the
tokenizer emits each face's edge identity in its `path` header —
`_emit_face_header` (graphax jaxpr.py:1438-1445) writes
`path <central> & <pred> & <succ>` with pred = `fr.in_edge`, succ =
`fr.out_edge` — and `_emit_step_paths` / `last_face_segments`
(jaxpr.py:1489-1497 / 1499-1545) hand back per-face token spans keyed to
faces. alphagrad already reconstructs `(vidx[fr.in_edge],
vidx[fr.out_edge])` per emitted face online (`LiveFaceStream._emitted`,
live_faces.py:378-408) — the attribution the write path needs is the face
key live_faces has been keying chunks by since §1.

**Minimal production design** (`--face-edge-mem`, composable with
`--face-endpoint-read`):
- *Write path*: the identical `vertex_memory.scatter` primitive
  (vertex_memory.py:72-110 — "two keyings of one operation") over an
  edge-slot table: host side assigns a dense slot to each NEW res-edge key
  as faces are emitted (the live_faces host loop already walks per-face
  segments with their keys); capacity K = MAX_FACES (each face writes
  exactly one res edge, so distinct keys ≤ faces eliminated). State:
  `(K, E)` sums + `(K,)` counts next to vmem's `(V+2, E)`.
- *Read*: per face, host resolves lhs edge (i, vidx(v)) and rhs edge
  (vidx(v), j) to slots (−1 → zero row, exactly the endpoint-0 convention);
  device gathers two rows. Head/probe input
  `[chunk ‖ emem_lhs ‖ emem_rhs]` = 3E = 96, or with the v63 endpoint read
  `[chunk ‖ vmem_i ‖ vmem_j ‖ emem_lhs ‖ emem_rhs]` = 5E = 160
  (`UnifiedFaceHead.in_dim` widens; `VarProbes(face_in_dim=...)` reads the
  same tensor). Like §6, the read is a POLICY input, so gradient flows
  through the emem rows — the same anchoring that kept the 61458 endpoint
  controls alive.
- *Predicted online probe numbers* (next campaign): under FORCE REV ORDER,
  ≈ no change (lhs all-primitive — the regime v63 already covers). On
  random/learned orders, at init on intermediate-operand faces: ndim ≥ 0.72
  (lhs) / ≥ 0.85 (rhs) vs the endpoint read's 0.61 / 0.76, operand size-R²
  ≥ 0.4 vs ≤ 0.2 (lhs); with both flags (arm D) rhs ndim ≥ 0.9. If the
  trained probe still decays to majority with emem in the loop, the §5
  fallback stands: the collapse is upstream in the shared rows.

Artifacts: `probe_invest/edgemem/` (`classify.py`, `capture_ext.py`,
`analyze_edgemem.py`, `env.sh`, sbatch files; `data/cap_rand{1,2,3}.pkl`,
`data/classify_*.pkl`, `data/vidx_map.json`, `data/edgemem_results.txt`,
`data/edgemem_faces.pkl`, logs `data/run_cap_61854.log`,
`data/run_ana_61857.log`).

**Implementation (v64).** The production design above is implemented in
commit `e16a255` behind `--face-edge-mem` (default OFF; the disabled
path is bit-identical -- pinned in `tests/edge_mem_test.py`, and the v63
endpoint suite still passes 7/7). Write: the SAME `_vmem.scatter`,
riding `carry_stream.advance`'s single encode fold (new
`edge_mem`/`edge_ids` arms), keyed by a host-assigned res-edge slot
table (`face_driver.EdgeSlotTable`: K = ALPHAGRAD_MAX_FACES, assign on
first emission, evict-oldest past K; per-episode telemetry `edgemem/*`
incl. `evictions` and `nonzero_reads`). The span attribution is
`last_face_segments`' own tiling, recovered on device from the stored
chunk counts plus a new approx-echo prefix-length wire
(`live_faces.chunk_ex` -> `ppo._edge_write_ids`) -- exactly the spans
the offline arm pooled. Read: `_face_loop` (rollout) and
`_face_replay` (loss, off the re-derived POST-delta memory, gradient
through both write and read) gather the face's lhs/rhs operand-edge rows
(-1 -> zero row), and the head + var probe input becomes
`[chunk || (vmem_i||vmem_j)? || (emem_lhs||emem_rhs)?]` = E / 3E / 5E =
32 / 96 / 160 at E=32, composing with `--face-endpoint-read`.
Rollout==replay logp (both flag combinations), slot-assignment determinism
+ eviction telemetry, zero rows for never-written operands, and the
write->read binding (a later face's emem_lhs equals the scattered rows of
its lhs edge's creation event) are unit-pinned. CPU gate (job 61862,
Helmholtz 2-ep, free orders): RC=0 for edge-mem alone, both flags
(160-wide), the flag-off regression, and the --grad-window 1 anchor path;
`edgemem/nonzero_reads` = 2-3 per episode > 0, evictions 0. Launcher:
`~/dsnn/fq_v64_tlm_joint.sbatch` (v63 + the flag, 160-wide), NOT
submitted: under ALPHAGRAD_FORCE_REV_ORDER=1 the emem half is predicted
inert (the stratum degeneracy above), so v64 is only meaningful once the
rev pin is lifted or orders are randomized -- the launcher keeps the pin
with that note inline.

## 10. Fast-weight query falsifier — could a palimpsa DECODER replace the keyed slots?

(§9 is reserved for the in-flight width sweep of arms B/C,
`probe_invest/width/`; this section is independent of it.)

**Hypothesis tested** (owner's): palimpsa's per-layer fast-weight state is
an associative matrix memory — the live recurrence (`_extend_sequential`,
src/alphagrad/approx/ppo.py:1846-1866) updates, per layer l and head h,

    M_l = v k^T           + exp(-gt·g) · M_l          (numerator)
    I_l = beta ⊙ k^2 + (1-exp(-gt·g)) · Ip + exp(-gt·g) · I_l   (precision)

with retrieval `v_hat = (M/I) q` (`out = einsum("hdn,hn->hd", mu, q·d^-0.5)`),
beta = sigmoid(bias_proj)·softplus(b_scale) the Bayesian importance gate
(palimpsa_encoder.py:75-90, Palimpsa-D bounded form) and q,k L2-normalised.
In theory a large enough S retains the whole graph evolution, and what §1-§8
diagnosed as missing is only an EXTRACTION mechanism — a decoder issuing
edge-shaped queries against S — so the §8 edge-keyed slot archive would be a
convenience, not a necessity. Counter-consideration: clean random-access
retrieval from linear-attention state is bounded by key orthogonality
(~d_k = E/2 per head, 6 (layer,head) matrices at L=3,H=2), against 206-286
distinct edges per random-order episode.

**Protocol** (`probe_invest/squery/`: `squery_replay.py`,
`squery_analyze.py`, `run_squery.sbatch`; job 61867, outputs
`data/sq_E{32,64,128,256}.pkl`, `data/squery_results.txt`,
`data/run_61867.log`). Same frozen random-init palimpsa, same captures
(rev + 3 random orders), same ridge/3-fold protocol, targets and strata as
§8, via `edgemem_faces.pkl` (join 1044/1044 faces, header-parse failures 0;
harness validation: chunk-mean on the rev capture reproduces §2 ndim
0.83/0.86/0.89). Per face, at its decision step, the SIDE carry's (M, I) is
recorded after the face's chunk — the state palimpsa actually has when the
face is decided — and probed with three query arms (all
`chunk ‖ v_hat(lhs) ‖ v_hat(rhs) ‖ v_hat(res)`, retrieval = the model's own
read-out contraction):

- **Sa** hand-built: q = the key palimpsa itself would form for the edge —
  the model's own key projection over the edge's `path`-header atoms
  (lhs = pred+central, rhs = central+succ, res = pred+succ), run through the
  3-layer stack from an empty carry, mean-pooled, re-L2-normalised. A
  query-projection variant (SaQ) tracks Sa within noise everywhere.
- **Sb** learned ("decoder" arm): a small MLP (edge header embedding → q per
  layer/head, L2-normalised) trained JOINTLY with the linear probe heads on
  the train folds — extraction is learned, storage is palimpsa's.
- **S-shuf** control: same queries against a random OTHER face's S.

Width sweep E ∈ {32, 64, 128, 256} (d_k = 16…128), fresh random init at
each width under the same `derive_agent_keys(250197)` discipline.

**Interpretation rule — CV only.** The retrieval features are 10E wide
(320…2560 at n=929): from E=64 up every S-arm ridge interpolates (train
1.00 across the board), and Sb's joint fit reaches train 1.00 even at E=32.
The §8 train-column convention is meaningless here; every number below and
the verdict are OUT-OF-FOLD (3-fold CV), where C@32's floor is also
evaluated. C@32 cv on both-intermediate: lhs ndim 0.55 (maj 0.43) szR²
0.20, rhs ndim 0.75 (maj 0.56) szR² 0.37.

**Results** (both-intermediate stratum, n=463; cv, maj lhs 0.43 / rhs 0.56
/ res 0.37; full arm×stratum×E tables incl. one-inter/both-prim in
`squery_results.txt`):

| arm | E | lhs ndim | lhs szR² | rhs ndim | rhs szR² | res szR² |
|---|---|---|---|---|---|---|
| A chunk | 32 | 0.45 | −0.07 | 0.59 | −0.07 | 0.15 |
| C@32 (slot floor) | 32 | 0.55 | 0.20 | 0.75 | 0.37 | 0.28 |
| Sa hand-built | 32 | 0.51 | −0.89 | 0.68 | −0.53 | −0.03 |
| Sa hand-built | 128 | 0.54 | −1.15 | 0.81 | −0.36 | −0.12 |
| Sa hand-built | 256 | 0.54 | −2.15 | 0.83 | −1.09 | −1.10 |
| Sb learned | 32 | 0.66 | −0.14 | 0.87 | 0.18 | 0.29 |
| Sb learned | 64 | 0.69 | −0.09 | 0.89 | 0.39 | 0.47 |
| Sb learned | 128 | 0.75 | 0.15 | 0.92 | 0.53 | 0.56 |
| Sb learned | 256 | **0.70** | **0.24** | **0.90** | **0.41** | **0.51** |
| S-shuf control | 32-256 | 0.40-0.45 | ≤−1.0 | 0.31-0.60 | ≤−0.6 | ≤−0.4 |

Verdict block (majority-margin recovered vs C@32, the §8 four decisive
numbers, cv): Sb@32 fails (lhs szR² −70%), Sb@64 fails (lhs szR² −44%),
Sb@128 3-of-4 (lhs szR² 74%; the other three 142-270%), **Sb@256 clears
all four: lhs ndim 231%, lhs szR² 120%, rhs ndim 181%, rhs szR² 109%**.
Sa (hand-built) fails size-R² at EVERY width (cv negative throughout, down
to −2.15 at 256) while its rhs ndim does scale (0.68→0.83, above C's 0.75
from E=128); the shuffled-S control sits at/below majority everywhere with
deeply negative szR² — the retrieved signal is real state content bound to
THIS face's step, not chunk leakage or query-feature leakage.

**Verdicts.**
- **EXTRACTABLE — at E=256 with LEARNED queries** (criterion: ≥90% of C@32
  on all four both-intermediate cv numbers; met at E=256, near-met at
  E=128). The owner's storage claim survives its falsification attempt:
  the fast-weight state of a random-init palimpsa DOES retain
  operand-level structure for accumulated intermediates that no §1-§8 read
  point reaches, including res-size (Sb res szR² 0.51 vs C's 0.28 — the
  one slot emem structurally cannot serve). The decoder theory is viable
  and the design conversation changes accordingly.
- **Extraction NEEDS learning.** Palimpsa's own key/query projections do
  not address edges at random init: Sa never decodes size at any width
  (interference: composite multi-token addresses against d_k-bounded
  superposition), so "S + a reader" is not enough — the reader must be a
  trained query map, exactly the pathway class PPO training drained in §3.
  Any production decoder therefore needs the same anchoring the §4/§6/§8
  slot reads get for free (probe-loss or policy-gradient through the
  query map), plus ~8× encoder width.
- **Capacity picture:** the E=32 production width (d_k=16 against 206-286
  edges/episode) is where size extraction fails hardest (Sb szR² lhs
  −0.14); crossing d_k ≈ 64-128 (E=128-256) is what buys size decodability
  — consistent with the key-orthogonality bound, softened by the 6
  (layer,head) stores and a tolerant linear read.
- **Per-layer:** no layer owns the operand info — per-layer Sa reads track
  each other within noise at every width (E=32 both-inter rhs ndim
  0.66/0.67/0.71 for L0/L1/L2) and none decodes size alone; the signal is
  distributed across all three (M, I) pairs.

**Caveats.** (i) Random-init palimpsa is the storage-quality floor —
trained fast weights could store better; but §H-D cuts the other way too
(v61 training measurably DRAINED face-relevant content from the shared
rows), so the trained state could equally be worse for these targets:
extraction viability at init does not promise it under the full training
pull. (ii) The probe-trained-encoder arm was skipped — no decode5/probe
checkpoint exists on disk (decode5_out holds logs/audits only). (iii) At
E≥64 every train column is interpolation (1.00); conclusions here are
cv-only, n=463 in the decisive stratum.

**Answer to the owner's question — could a palimpsa decoder replace the
keyed slots?** In principle yes, but not at the production operating
point: with learned edge-queries and E=256 (8× the shipped width, ~64×
the per-step state: (M,I) = 2·3·2·d_k² floats), retrieval from palimpsa's
own fast weights matches or beats the edge-keyed archive on every
decisive both-intermediate number, res-size included — storage is not the
binding constraint, extraction is, and extraction is learnable. But the
slot archive reaches that floor TODAY at E=32, parameter-free, with a
read that is anchored by construction and already implemented
(`--face-edge-mem`, §8). The pragmatic split: keep the keyed slots as the
production mechanism at current width; treat the learned-query decoder as
the scale-up path (it subsumes the slots' function only once E≥128-256
AND the query map is anchored against the §3 collapse — e.g. trained
under the probe loss, not the PPO surrogate alone).

Artifacts: `probe_invest/squery/` (`squery_replay.py` — state capture +
hand-built keys via the model's own projections; `squery_analyze.py` —
arms, ridge + joint decoder training, verdict block; `run_squery.sbatch`;
`data/sq_E*.pkl`, `data/squery_results.txt`, `data/run_61867.log`).
