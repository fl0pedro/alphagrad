# CLEAN-DESIGN AUDIT — every prior the PPO trainer adds beyond the owner's minimal design

Read-only audit at `alphagrad` HEAD `0e601ae` (code) / `f7ef2c0` (docs), branch
`hostperf-caches`. Launchers audited: `~/dsnn/fq_v64b_tlm_creditfix.sbatch` and
`~/dsnn/fq_v66a_static_lam10.sbatch`. Nothing under `src/` was modified; the four
GPU arms 62072–62075 were not touched.

## 0. The reference design

From the owner's drawing + spec:

- ENV → jaxpr → palimpsa → two heads: a **vertex-elimination head** (pointer
  network) and an **approximation head** that is a plain 94-output MLP; loop back
  to ENV per step.
- Per step: choose a vertex to eliminate (**VE1, VE2, … are real learned
  decisions**), then per face choose among {skip, diag, reduce/compress, quant,
  none}; the drawing's "3×" is the three operand slots.
- Reward: **terminal only**, at the end of the whole elimination sequence.
- Init: **start random** — explicitly not biased toward perfect quality /
  identity / all-NONE.
- Memory: **no vertex memory and no edge memory feeding anything except the
  pointer network** at the vertex-elimination head.
- The approx head is just the MLP on whatever palimpsa emits for that face.

Verdict vocabulary: **CONFLICTS** = contradicts one of the five constraints
above; **EXTRA** = added machinery the clean design neither asks for nor forbids;
**REQUIRED** = the code or graphax is incorrect / throws without it.

---

## 1. AUDIT TABLE

### (a) Initialization priors

| # | Item | Where | Bias injected | Verdict |
|---|---|---|---|---|
| a1 | `ALPHAGRAD_FACE_NONE_BIAS=6` | `common/agent_factory.py:87-112`; v66a:64, v64b:45 | +6 on each of 3 slots' `OP_NONE` logit, −6 on `SKIP`. Measured (§13.3): p_none/slot 0.993, p_approx/slot 0.008, E[arity] 1.024. Every run logs `[factory] face-head IDENTITY INIT`. | **CONFLICTS** — this is the identity/all-NONE prior the owner explicitly rejects |
| a2 | `init_linear_weights` | `common/init.py:11-47` | orthogonal(√2) weights + **zero biases** on every `eqx.nn.Linear`, face MLP included. Sets the logit scale and removes any learned constant offset at init. | **EXTRA-BUT-COMPATIBLE** — this *is* the random init |
| a3 | `_scale_output_heads` + `--head-init-scale` (default 0.1; **neither launcher passes it**) | `ppo.py:4443-4505` | Scales `vertex_policy.k_proj` ×0.1 (near-uniform pointer), zeroes `pref_proj` ×0.0, scales `micro_action_policy.head` ×0.1. **`face_path_policy.head` is absent from the function** (§13.1(ii)) — and under `--live-faces` `micro_action_policy` is `None` (`ppo.py:4374-4381`), so the *only* live approximation head is the one head that is never rescaled. Its 94 logits carry full orthogonal magnitude from step 0. | **CONFLICTS** — asymmetric "random": pointer near-uniform, face head arbitrary. See CODE-CHANGE #1 |
| a4 | `--face-logit-clamp 15` (code default 15; v66a:160, v64b:141 pass 15) | `unified_face_head.py:88-93, 216-222`; set at `ppo.py:4902-4903` | Every face logit → `15·tanh(z/15)`. Near-identity for \|z\|≪15; ~50× gradient attenuation beyond (§12.1). A bounded reparameterization added only to keep the entropy-floor hinge alive. | **CONFLICTS** (exists solely to serve item f3) |
| a5 | `--popart-init-episodes 3` + `--popart-init-temperature 10.0` | `ppo.py:3578-3589`, `ppo.py:10391-10470`; v66a:156, v64b:137 | 3 warm-start rollouts of random-but-valid plans, vertex pointer softmax at T=10, to seed PopArt (µ,σ). Under `--advantage-norm none` the stats are computed but unused for normalization (they still fund the ep0 census). | **EXTRA** under `none`; **CONFLICTS** under `popart` |
| a6 | `pref_proj` zeroed at init | `ppo.py:4465-4470` | Preference-conditioning path forced to produce identical step-0 policies. Inert: `--preference-conditioned` is not passed. | **INERT** |

### (b) Action-space / legality priors

| # | Item | Where | Bias injected | Verdict |
|---|---|---|---|---|
| b1 | `ALPHAGRAD_FORCE_REV_ORDER=1` | `common/masks.py:147-151, 183-191`; v66a:68, v64b:49 | `vertex_avail_at_step` keeps **only the highest-index available vertex**, so the pointer has exactly one legal action every step. Consequence (§13.2e): ve-head entropy is `-0.0e+00` and `max|dH/dlogits| = 0.0e+00` — the pointer has taken **zero gradient across the entire v57–v66 campaign**; every episode logs `ve_head=0 macro_vertex=0`. | **CONFLICTS** — directly contradicts VE1/VE2 being learned |
| b2 | `_compute_op_legality` | `heads.py` (`def _compute_op_legality`) | DIAG legal iff ≥1 (i,j) pair survives; COMPRESS iff ≥1 eligible axis; QUANT iff ≥1 dtype survives the hardware mask; **`end_legal` is a hard 1.0 and the override does `.at[OP_END].set(1.0)` — NONE is never maskable**. | **REQUIRED** (graphax throws on an ill-fitting transform), with a noted structural asymmetry: NONE always has mass |
| b3 | `_approx_allowed` | `heads.py` (`def _approx_allowed`) | A variant mask leaving only END also closes the SKIP gate (a skip *is* an approximation). Inert here — no variant override is passed. | **REQUIRED** / inert |
| b4 | `j_mask_given_i` | `unified_face_head.py:183-193` | j ≠ i, and `gcd(N_i,N_j)>1` — coprime pairs admit only factor 1, a no-op DIAG. Removes a region of the action space. | **REQUIRED** (no-op filter) |
| b5 | Branch masking | `unified_face_head.py:254-296` | Only the fields the chosen op consumes contribute log-prob/entropy; a padding face and every slot behind `skip==1` are forced to a canonical no-op contributing exactly zero. | **REQUIRED** (also what keeps the PPO ratio 1 at epoch 0) |
| b6 | DIAG **factor is not an action** | `unified_face_policy.py:192-206` | `factor = gcd(N_i,N_j)` — the largest legal factor, i.e. the smallest blocks — is hardcoded. The 94-slot layout has no factor field. Docstring: "the clamp, not the head, decided it". | **CONFLICTS-EXTRA** — a decision the drawing implies is learned has been deleted. See CODE-CHANGE #4 |
| b7 | Substeps | `unified_face_head.py:64` (`FACE_SLOTS = 3`); `unified_face_policy.py` docstring | The drawing's "3×" **is** `FACE_SLOTS = 3` (pre/post/new). `max_substeps` is 1 by construction under the unified face head; `--max-substeps` (default `2*MAX_AXES_PER_VERTEX` = 16) is **inert** — `micro_action_policy` is `None` under `--live-faces`. | **MATCHES the drawing** |
| b8 | `ALPHAGRAD_MAX_FACES=2538` | `env.py:1012`; v66a:70, v64b:51 | Static buffer width only. `_face_loop` is a `while_loop` over the *actual* face count; `derived_max_faces = max_v |anc(v)|·|desc(v)|` is **order-independent**, so 2538 stays a valid bound if the rev pin is lifted. | **REQUIRED** (shape bound) |
| b9 | Face enumeration order | graphax `faces_of` — `env.py:2894, 3207`; `common/masks.py:583-587` | Canonical visit order. The head has **no face embedding and no learned index table** (`unified_face_policy.py:227-242` — "a label is not information"), so order enters only through `--live-faces` sequencing: face *f+1*'s chunk reflects face *f*'s approximation. | **EXTRA-BUT-COMPATIBLE** |
| b10 | `--measure-grad` (`--seed-vertices` **REMOVED**, A4) | `common/examples.py` (`grad_target_setup`, `_seed_vertices_requested`) | The measured executable is the **gradient**, now unconditionally: `grad_target_setup` wraps the target in `scalar_loss_fn` *before* tracing whether or not `--measure-grad` is passed, so the traced graph IS the loss graph (nn256: 15 vs 13 eqns) on every path. Without the flag it used to trace the RAW example, and the NN/vision/Encoder examples return PER-ELEMENT losses — so a flag-off run measured a full per-class **Jacobian** and reported it as a gradient. `--measure-grad` survives with two jobs only: it flips the quality default cosine→loss_drop, and it gates `--seed-vertices`. The scalar-output contract check is armed by `from_jaxpr(scalar_target=True)`, which the three production builders pass unconditionally. Pinned by `src/alphagrad/approx/tests/test_jacobian_equals_grad.py` (jacve == jax.grad, ≤1e-5). `--seed-vertices` is gone from every launcher: the tangent-seed `add` and adjoint `reduce_sum` were 2 ordinary eliminable vertices in the pointer's action space and in `derived_max_faces`, and the forward/reverse/cross-country freedom they were meant to buy was **unreachable anyway while b1 (`ALPHAGRAD_FORCE_REV_ORDER=1`) is on** — so the search paid for a degree of freedom b1 had already closed. | **RESOLVED** — bias removed |
| b11 | `--pin-rules-to-exact` | `ppo.py:3977` | Not passed. | **INERT** |

### (c) Reward shaping

| # | Item | Where | Bias injected | Verdict |
|---|---|---|---|---|
| c1 | `ALPHAGRAD_QUALITY_GATE_MIN=0.05` | `env.py:1986-2049`, called at `env.py:4305`; v66a:62, v64b:43 | Below qmin, latency **and** peak memory are floored at the **same-order exact reference** (`max(latency, ref_lat), max(peak, ref_mem)`). Under the rev pin the order is one constant ⇒ one constant floor ⇒ every sub-qmin plan in the batch receives **bitwise-identical cost**, so the cmp and mem advantages are exactly 0 across that whole sub-population. 47 `quality gate CLAMP` prints in `v58_tlm_61498.log`. | **CONFLICTS** — a reward-shaping term. Has an off switch (`=0`, `env.py:1997-2002`) |
| c2 | Lagrangian violation transform | `ppo.py:615-633` (`_lag_violation`), `636-664` (`_apply_lagrangian_channels`) | `q_eff = clip(q_raw, −0.5, 1.0)`; `violation = max(0, tau − q_eff)`; channel stores `−violation`. The `−0.5` clip maps the DIVERGED sentinel (−1.0, `env.py:2441`) to −0.5 ⇒ violation `tau+0.5` = 1.25, strictly worse than zero-work's 0.75. | **CONFLICTS** |
| c3 | `--lag-tau 0.75` | `ppo.py:3424`; v66a:158, v64b:139 | The quality constraint threshold. A hard prior on "how good is good enough". | **CONFLICTS** |
| c4 | `--lag-init 10 --lag-min 2 --lag-max 20 --lag-eta 0 --lag-target 0.02` | `ppo.py:3427-3444`, `_lag_dual_ascent` `ppo.py:666-679`; v66a:158-159 | λ = the exchange rate between one unit of violation and one unit of symlog cost. v66a freezes it at 10 (`--lag-eta 0` makes ascent a strict no-op; lag-min/max 2/20 contain 10 so the clip is identity). §12.10.4: the realized pull at λ=10 is **0.75:1 vs latency and 0.50:1 vs memory**, i.e. 8–13× weaker than v64b's 10:1; matching v64b would need λ ≈ 134–199. | **CONFLICTS** |
| c5 | `--reward-mode lagrangian` | `ppo.py:3405-3423`, dispatch `ppo.py:7940-7958` | Replaces the quality slot with the stationary violation channel. Default is `additive`. | **CONFLICTS** |
| c6 | `--lambda-cmp 1 --lambda-mem 1`; `--lambda-acc` **default 2.0, never passed** | `ppo.py:3369-3378`; `common/agent_factory.py:151-164` | Channel scalarization weights. `--lambda-acc`'s 2.0 default ("so quality balances the two cost heads' combined vote") is a 2:1 quality prior — **but in lagrangian mode it is overwritten by λ** at `ppo.py:10315-10325`, so it is inert in v64b/v66a and becomes live the moment `--reward-mode additive` is used. | **EXTRA** (weights are unavoidable); the silent 2.0 default is a hidden prior |
| c7 | symlog | `ppo.py:497-510`; per-channel exemption `ppo.py:434-450`; applied `ppo.py:7959` | v66a **drops** `--no-symlog`, so cost channels are symlog-compressed and the violation channel is symlog-**exempt** (symlog(0.75)=0.56 would discount the price exactly at the bound). Fixed, policy-independent. v64b passes `--no-symlog` and lets PopArt do it instead. | **EXTRA-BUT-COMPATIBLE** — under `--advantage-norm none` it is the only thing making 1e5-ns latency and O(1) quality commensurable |
| c8 | `--quality-metric loss_drop` | `ppo.py:3624-3636`; `env.py:2331-2449`; v66a:163 | Quality = relative loss drop of a 200-step Adam walk driven by the plan's own gradient on one resident wikitext2 batch. Diverged → −1.0; else `clip(drop, −1, 1)`. Pearson 0.922 vs downstream accuracy (cosine: 0.610, 44× more expensive). | **EXTRA-BUT-COMPATIBLE** — a proxy choice, but the defensible one |
| c9 | `GRAPHAX_QUANT_PULLDOWN=1` | graphax lowering; v66a:69, v64b:50 | Makes bf16-native compute actually pay, i.e. **makes QUANT profitable**. Launcher comment: at fixed rev the cost axis is otherwise flat. Shapes which approximation the agent is rewarded for. | **EXTRA-BUT-COMPATIBLE** — a property of the measured hardware path, not a policy prior, but it does select the winner |
| c10 | `ALPHAGRAD_MULS_SENTINEL_CAP=5e12` | `env.py:3607-3612`; v66a:92, v64b:73 | Would refuse plans above the symbolic-op ceiling before compile and return the truncated reward (excluded from gradient). Launcher sets 5e12, **10× tighter** than the code default 5e13. **BUT the guard is `if not _SKIP_COUNT_OPS and ...` and both launchers set `ALPHAGRAD_SKIP_COUNT_OPS=1` (v66a:93) ⇒ the cap is dead code in every current run.** | **INERT (dead)** — config-hygiene finding |
| c11 | Degenerate-plan advantage zeroing | `ppo.py:8040-8073` | A plan whose *all* cost channels sit at the −1e10 sentinel gets `advantage × 0` and is dropped from the value target. Measurement-failed plans teach nothing. Also catches Ray-measure timeouts and the ~10% of plans Triton cannot compile. | **EXTRA** — necessary to protect PopArt, but it censors the action space |
| c12 | Anti-degeneracy penalty `--anti-degen-penalty 2.0 --anti-degen-tau 0.05` | `ppo.py:558-609, 3391-3396`; applied `ppo.py:7940-7950` | Shaped negative reward on terminal steps with fidelity < τ. **Gated on `reward_mode == "mult"`** — inert under lagrangian and under additive. | **INERT** |
| c13 | Potential-based shaping (Ng et al. 1999) | `ppo.py:6708-6714` | Comment only; no coefficient flag survives, no term is added. | **INERT (dead comment)** |

### (d) Credit / optimization priors

| # | Item | Where | Bias injected | Verdict |
|---|---|---|---|---|
| d1 | `--terminal-rewards-only` | `ppo.py:3676`, `env.py:3501-3503`; v66a:155, v64b:136 | Intermediate steps return a zero reward vector. | **MATCHES the clean design** |
| d2 | `--discount` **default 0.99** and `--gae-lambda` **default 0.95** — *neither launcher passes either* | `ppo.py:3961, 3871` | With a terminal-only reward, GAE attenuates the terminal advantage by `(γλ)^(T−t)` with `γλ = 0.9405` over ~95 steps: **≈0.003 at t=0, ≈0.05 at t=45**. The first half of every episode receives essentially no credit for the only reward that exists. This is the mechanism behind the standing "PPO never left the uniform policy" finding (first elimination gets 0.31% of the terminal signal). | **CONFLICTS** — a severe temporal credit prior silently riding on defaults |
| d3 | PPO ratio clipping + epoch asymmetry: `--ppo-clip-eps 0.2`, `--ppo-epochs 2`, `--minibatches 4` | `ppo.py:3872-3874`, surrogate `ppo.py:7476-7495`; v66a:155 | §12.2 F3: for `A < 0` the surrogate `min(rA, clip(r)A)` is **unclipped as the ratio rises**, while positive-advantage terms saturate at `1+ε`. Epoch-2 updates push every step of a violating plan's *joint* log-prob down without bound. Signature: `kl/approx` 0.35–0.5 healthy → 3.3, 6.0, 8.5, **109 (ep65)**, with `ratio/max_log` 10→35 and the surrogate spiking to +60. | **CONFLICTS-EXTRA** — an asymmetric optimization prior. `--ppo-epochs 1` makes the ratio identically 1 and the clip inert |
| d4 | `--lag-causal-mask` | `ppo.py:3920-3931`, `_causal_quality_mask` `ppo.py:684-732`, applied `ppo.py:8199-8206`; v66a:166, v64b:148 | Multiplies the quality-channel preference by `m(e,t) = 1` iff step *t* took any causal face action. **Its own docstring specifies the free-order causal set as {approx actions} ∪ {vertex choices}, and the code implements only the face half** (lines 723-732 read `face_valid`/`face_skip`/`face_op_type` and nothing else). | **CONFLICTS**, and becomes **incorrect** if b1 is lifted. See CODE-CHANGE #2 |
| d5 | `--adv-winsorize 3` | `ppo.py:734-751, 3932`; **v64b:148 only** (v66a leaves it 0) | Clips each per-channel normalized advantage to ±3 before preference weighting. Bounds a destroyed plan at −30 instead of −56. | **CONFLICTS** (v64b arm only) |
| d6 | `--grad-window 0` | `ppo.py:3268-3285`; v66a:164, v64b:146 | K=0 selects the **full-horizon** path: one `lax.scan` over the whole episode from the base carry, gradient horizon T. The *default* K=1 gives palimpsa gradient from one delta and none from the elimination history. | **REQUIRED** — this is the *least* truncated option, correctly chosen |
| d7 | `--max-grad-norm 0.5`, `--adam-b1 0.9`, `--adam-eps 1e-7` (defaults, unpassed) | `ppo.py:3962-3964` | Global gradient clipping + Adam moments. | **EXTRA-BUT-COMPATIBLE** |
| d8 | LR schedule: `--lr 3e-4`, cosine decay to `--lr-decay-min-mult 0.1`, `--lr-warmup-frac 0.0` | `ppo.py:3850, 3966-3970`, built `ppo.py:6231-6248` | Cosine decay over `episodes × ppo_epochs × minibatches` steps; warmup off. A mild "anneal exploration away" prior. | **EXTRA-BUT-COMPATIBLE** |
| d9 | `--num-envs 16 --minibatches 4` | v66a:155 | 16 plans per episode is the entire population the advantages are computed from. Combined with c1 (which collapses cost contrast within the sub-qmin subset), the effective contrast population can be far below 16. | **EXTRA** |

### (e) Normalization

| # | Item | Where | Bias injected | Verdict |
|---|---|---|---|---|
| e1 | `--advantage-norm` | `ppo.py:3489-3500`, PopArt `ppo.py:865-935`, applied `ppo.py:7979-8140` | v66a: **`none`** — advantages stay in raw symlog units, CLI λ's are the only scaling, semantics stationary across the run. v64b: **`popart`** — per-channel debiased-EMA µ/σ + output-preserving head rescale. §12.9.4 diagnoses PopArt's moving frame as a principal nonstationarity: the quality term carries `λ·m·(G−V)/σ_q` ≈ **80× per raw unit** of critic error. | v66a's `none` **MATCHES**; `popart` **CONFLICTS** |
| e2 | `--popart-beta 1e-2`, `--popart-sigma-min 0.1` | `ppo.py:3590-3592` | σ floor of 0.1 caps the amplification; β sets the frame's drift rate. | **EXTRA** (inert under `none`) |
| e3 | `--popart-basin-freeze` | `ppo.py:833-860, 3453`; **v64b:142 only** | Freezes the quality channel's raw PopArt accumulators when >half the batch sits in the basin (`q_eff ≤ 0.05`). | **CONFLICTS** (v64b only) |
| e4 | `--lag-raw-viol-adv` | `ppo.py:762-800, 3940` | Not passed by either launcher. Built for the refuted H-ZNEUT. | **INERT** |
| e5 | Per-channel value heads + `--value-weight 0.5` | `ppo.py:4337-4340` (three MLPs, `value_dims "64,32"`), `ppo.py:3960` | Three separate critics (latency/mem/quality), each with its own GAE channel; the logged scalar `value loss` is `0.5 × Σ_channels MSE` (§12.10). | **EXTRA-BUT-COMPATIBLE** |
| e6 | `_value_target` symlog encode/decode around the value head | `ppo.py:446-450` | The critic learns a symlog-normalized target. | **EXTRA** |

### (f) Exploration forces

| # | Item | Where | Bias injected | Verdict |
|---|---|---|---|---|
| f1 | `--entropy-weight` **default 0.05, never passed** | `ppo.py:3875`; bonus `_split_entropy_bonus` | Applies to the vertex/ve-head entropy. Under b1 the ve entropy *and its gradient* are exactly 0 (§13.2e), so it is **inert while the order is pinned and immediately live the moment the pin is lifted**. | **CONFLICTS** (a latent exploration force) |
| f2 | `--face-entropy-weight 0.005` | `ppo.py:3876-3890`, `_split_entropy_bonus`; v66a:160, v64b:141 | Entropy bonus on the face head. §13.2(d) proves by `jax.grad` on the real loss that the **net face coefficient equals the flag exactly**. §12.9.4: it is a *dense, coherent* force (every face slot of every step, always "raise H") competing against a *sparse, sign-random* quality term (mask_fraction 0.05–0.07). | **CONFLICTS** |
| f3 | `--face-entropy-floor 0.05 --face-entropy-floor-weight 10` | `ppo.py:3891-3912`, `_face_entropy_floor_penalty`; v66a:159, v64b:140 | Hinge `w·relu(floor − H)²` added to the loss. §12.2 F1 identifies it as **the igniter** of both collapses: at the v62/v63 setting (floor 0.3) `entropy_floor/penalty` was continuously 0.6–0.72 from ep0 with `dP/dH ≈ −5.4` — ~1000× v63's 0.005 bonus. §13.4 shows the floor firing on heads 2.2–10× *above* it because of the denominator bug (f4). | **CONFLICTS** |
| f4 | Arity + population normalization of the face entropy | `ppo.py:3140` (`f_ent / max(f_arity, 1.0)`); arity built `unified_face_head.py:250, 295`; batch metric `ppo.py:7651-7655` (`jnp.mean(face_ents)`) | Two mismatches (§13.1, §13.4): (i) `f_arity = gate_face + Σ_s active·1{op≠NONE} ∈ [1,4]` counts **only non-NONE slots**, while the numerator counts the skip Bernoulli plus all three op softmaxes; (ii) the batch metric averages over **all env steps**, so a step whose vertex has no live face enters as `0/max(0,1) = 0` — a structural zero indistinguishable from a deterministic head. Measured dilution: v64b ep84→ep122 the metric falls **357×** while `faces/mean_valid` falls **91×**; v63's quotient is pinned at 0.66–0.96 for 420 episodes while the raw panel swings 7×. | **CONFLICTS + BUG**. `--face-entropy-agg` from §13.7 **does not exist at 0e601ae**. See CODE-CHANGE #3 |

### (g) Observation / representation

| # | Item | Where | Bias injected | Verdict |
|---|---|---|---|---|
| g1 | `--face-endpoint-read` | `ppo.py:3305-3315`; `unified_face_policy.py:93-104, 227-242`; v66a:166, v64b:148 | Head input becomes `[chunk_mean ‖ vmem_slot_i ‖ vmem_slot_j]` (3E). **This routes the vertex memory into the face head** — precisely what the owner forbids. | **CONFLICTS** |
| g2 | `--face-edge-mem` | `ppo.py:3317-3331`; `unified_face_policy.py:101-102` | Would append two edge-keyed memory rows (+2E, total 5E). **Not passed by either launcher.** | **would CONFLICT**; currently inert |
| g3 | Pointer head read | `set_pointer.py:115-160` (`SetPointerVertexPolicy._score`) | Reads `vmem` (the vertex memory) through permutation-equivariant Set-Transformer blocks; query = occupancy-weighted mean over slots; no parameter depends on V. **Confirmed: this is the legitimate vertex-memory read the clean design allows.** | **MATCHES** |
| g4 | `--set-pointer` (+ `--set-pointer-blocks 2`) | `ppo.py:3563-3572`; v66a:161 | Content-based, V-independent pointer instead of a fixed `Embedding(num_vertices, embd_dim)` id table. | **MATCHES** (it *is* the pointer network) |
| g5 | `--unified-face-head` | `ppo.py:3817-3822`; `unified_face_head.py:211-212` | The head is literally `eqx.nn.MLP(in_dim, 94, hidden=embd_dim, depth=1)`. **Exactly the owner's "plain 94-output MLP"** — provided `in_dim == E`, i.e. provided g1 and g2 are off. | **MATCHES** |
| g6 | `--live-faces` | `ppo.py:3823-3834`; `live_faces.py` | Each face's own token chunk (previous face's approximation tail + this face's contraction) is emitted, palimpsa extends a side carry, the head decides from it. **This is "whatever palimpsa emits for that face".** | **MATCHES** |
| g7 | `--incremental-encode` + `ALPHAGRAD_MAX_DELTA_TOKENS=32768` | `ppo.py:3478`; `env.py:309, 364-385`; v66a:75 | Mandatory O(delta) encoding path. A delta exceeding the bound has tokens **silently dropped** (with telemetry). Job 61350 truncated at 19657 on this exact target at 16384. | **REQUIRED** + a truncation risk to monitor |
| g8 | `--var-probe --var-probe-lr 1e-3 --var-probe-steps 16` | `ppo.py:3292-3300`; probe pack `ppo.py:7684-7700` | Returned as `has_aux` **data**, `stop_gradient`-ed at the loss site *and* inside `Probe.__call__`, with its own optimiser. **Confirmed gradient-isolated ⇒ not a bias on the policy.** | **EXTRA-BUT-COMPATIBLE** (pure measurement) |
| g9 | `face_sizes` never passed | `unified_face_policy.py:142-172, 244, 309` accept it; `ppo.py` never passes it | `_face_feats_1` falls back to the **shared vertex features** when `sizes_f is None`, so every face of a vertex derives its **legality masks** from the vertex's static features rather than its own live contraction. (The head's *input* is unaffected.) | **EXTRA** — a latent correctness gap. See CODE-CHANGE #5 |
| g10 | Architecture: `--vocab-size 512 --num-layers 3 --hidden-dim 256`, `--embd-dim 32` (default), `ALPHAGRAD_POLICY=palimpsa`, positional encoding OFF | `common/agent_factory.py:30-41`; `ppo.py:4276-4306` | Palimpsa's gated exponential-decay carry supplies relative position, so the absolute sinusoidal PE is off by design (and would be arbitrary under append-only). `E=32`; §9 shows width is "the expensive non-fix". | **EXTRA-BUT-COMPATIBLE** |
| g11 | `ALPHAGRAD_MAX_EQNS=512` | `ppo.py:1027` (default 4096); v66a:87 | Token/eqn shape bound. | **REQUIRED** (shape bound) |

### (h) Anything else that shapes the reachable solution

| # | Item | Where | Note |
|---|---|---|---|
| h1 | `ALPHAGRAD_MULS_SENTINEL_CAP` is dead | `env.py:3608` | Guarded by `not _SKIP_COUNT_OPS`; every launcher sets `ALPHAGRAD_SKIP_COUNT_OPS=1`. The 5e12 in the launcher does nothing. |
| h2 | `--ray-measure 3 --ray-measure-timeout 600` | v66a:162 | A timed-out measurement returns the sentinel and is then advantage-zeroed by c11 — i.e. slow plans are silently removed from the gradient, which is a *systematic* censoring correlated with plan cost. |
| h3 | Triton compile failures | (standing finding) | ~10% of plans are deterministically uncompilable; they land in the same c11 zeroing. Correct (never score worst), but it is a hole in the action space. |
| h4 | `GRAPHAX_PLANNER_EXACT=1 GRAPHAX_DEMAND_EMIT=1 GRAPHAX_ALLOW_PARTIAL_ORDER=1` | v66a:41-46 | Lowering choices that define what "exact" costs, and therefore what an approximation is measured against. Environment, not policy prior — but they set the baseline. |
| h5 | `--lean-logging`, `ALPHAGRAD_DEBUG_*`, `ALPHAGRAD_ACTOR_PROF_EVERY`, `ALPHAGRAD_FACE_ENUM_CACHE`, `ALPHAGRAD_EXTEND_CHUNK/UNROLL`, `JAX_COMPILATION_CACHE_DIR` | v66a:78-116 | Telemetry / host-performance only; proven bit-identical. No bias. `EXTEND_CHUNK` must stay ≠ 0 (29× flat penalty trap). |

---

## 2. CLEAN-ROOM CONFIG

`fq_v67_cleanroom.sbatch`, derived from `fq_v66a_static_lam10.sbatch`. Only the
lines that change are shown; everything not listed is carried over verbatim.

### Environment variables

```diff
- export ALPHAGRAD_QUALITY_GATE_MIN=0.05
+ export ALPHAGRAD_QUALITY_GATE_MIN=0          # c1: reward shaping removed (env.py:1997 early-return)
- export ALPHAGRAD_FACE_NONE_BIAS=6
+ export ALPHAGRAD_FACE_NONE_BIAS=0            # a1: identity/all-NONE init prior removed
- export ALPHAGRAD_FORCE_REV_ORDER=1
+ export ALPHAGRAD_FORCE_REV_ORDER=0           # b1: VE1/VE2 become learned decisions
- export ALPHAGRAD_MULS_SENTINEL_CAP=5e12      # h1: dead under SKIP_COUNT_OPS=1; delete for hygiene
  export GRAPHAX_QUANT_PULLDOWN=1              # c9: KEPT — property of the measured path, not a policy prior
  export ALPHAGRAD_MAX_FACES=2538              # b8: KEPT — derived_max_faces is order-independent
  export ALPHAGRAD_MAX_DELTA_TOKENS=32768      # g7: KEPT — watch tokenization/truncated_count under free order
```

### CLI flags

```diff
  CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --no-sync python src/alphagrad/approx/ppo.py \
    --example TransformerLM --episodes 250 --exec-on-gpu --seed 250197 \
    --measure-latency --latency-inner-reps 5 \
    --num-data-points 5 --reps-per-point 4 --dataset wikitext2 --hidden-dim 256 \
    --incremental-encode --cmp-type latency --mem-type peak_memory \
    --terminal-rewards-only --advantage-norm none --num-envs 16 --minibatches 4 \
-   --popart-init-episodes 3 --lean-logging \
+   --popart-init-episodes 0 --lean-logging \            # a5: no warm-start scale prior
-   --rewards cmp mem acc --lambda-cmp 1 --lambda-mem 1 --reward-mode lagrangian \
+   --rewards cmp mem acc --lambda-cmp 1 --lambda-mem 1 --lambda-acc 1 \
+   --reward-mode additive \                             # c5/c2/c3/c4: no violation transform, no clip, no tau, no lambda
+                                                        # c6: --lambda-acc pinned to 1 (silent default is 2.0)
-   --lag-tau 0.75 --lag-eta 0 --lag-init 10 --lag-min 2 --lag-max 20 \
-   --lag-target 0.02 --face-entropy-floor 0.05 --face-entropy-floor-weight 10 \
-   --face-logit-clamp 15 --face-entropy-weight 0.005 \
+   --face-entropy-floor 0 \                             # f3: hinge (the igniter) off
+   --face-entropy-weight 0 --entropy-weight 0 \         # f2/f1: BOTH entropy bonuses zeroed
+   --face-logit-clamp 0 \                               # a4: tanh reparameterization off
+   --discount 1.0 --gae-lambda 1.0 \                    # d2: pure terminal Monte-Carlo credit
+   --ppo-epochs 1 \                                     # d3: ratio == 1, clip inert, no epoch asymmetry
+   --lr-decay-min-mult 1.0 \                            # d8: flat LR (optional de-biasing)
    --num-layers 3 --vocab-size 512 --set-pointer --face-actions \
    --unified-face-head --live-faces --ray-measure 3 --ray-measure-timeout 600 \
    --measure-grad --quality-metric loss_drop \
    --grad-window 0 \
    --var-probe --var-probe-lr 1e-3 --var-probe-steps 16 \
-   --face-endpoint-read --lag-causal-mask \
+                                                        # g1: no vmem into the face head
+                                                        # d4: causal mask is lagrangian-only AND free-order-incorrect
    --name v67-cleanroom --wandb online \
    --wandb-entity dll-streetview --wandb-project dsnn-vertex
```

**Kept deliberately, with reasons:** `--terminal-rewards-only` (d1, matches),
`--set-pointer` (g4), `--unified-face-head` (g5), `--live-faces` (g6),
`--advantage-norm none` (e1, matches), symlog on the cost channels (c7 — under
`none` it is the only thing making 1e5-ns latency and O(1) quality commensurable;
dropping it *and* PopArt makes the objective pure latency), `--grad-window 0`
(d6, least truncated), `--var-probe` (g8, gradient-isolated), `--head-init-scale`
at its 0.1 default (a3 — the near-uniform *pointer* init is what "start random"
means; see CODE-CHANGE #1 for why the face head does not get it).

---

## 3. CODE-CHANGE-REQUIRED

| # | What cannot be turned off by a flag | Minimal change |
|---|---|---|
| **1** | **`ALPHAGRAD_FACE_NONE_BIAS=0` does not give a uniform face head.** `_scale_output_heads` (`ppo.py:4443-4505`) scales `vertex_policy`, zeroes `pref_proj`, and scales `micro_action_policy.head` — but `face_path_policy.head` is absent, and under `--live-faces` `micro_action_policy` is `None`, so the *only live approximation head is the one head that is never rescaled*. Its 94 logits keep full orthogonal(√2) magnitude, so "random" is an arbitrary, input-dependent, sharply non-uniform distribution — not uniform. §13.4 lists this as part of why the observed init entropy is 2.5× below the model floor. | Append to `_scale_output_heads`, mirroring the `UnifiedApproxHead` branch already there: `if getattr(agent, "face_path_policy", None) is not None: agent = scale_module_weight(agent, lambda a: a.face_path_policy.head.proj.layers[-1].weight, scale)` |
| **2** | **`_causal_quality_mask` has no free-order term** (`ppo.py:723-732`). Its own docstring specifies `{approx actions} ∪ {vertex choices}` once the rev pin is lifted, but the code reads only `face_valid`/`face_skip`/`face_op_type`. With `FORCE_REV_ORDER=0` it would zero the quality credit on every step whose *only* decision was the vertex choice — exactly the decisions the clean design wants learned. | Not needed for the clean run (`--lag-causal-mask` is lagrangian-only and is dropped). Required for any future free-order + lagrangian arm: OR the face term with "this step's vertex choice had ≥2 legal options". |
| **3** | **`--face-entropy-agg` does not exist at `0e601ae`.** §13.7's valid-face-weighted ratio-of-sums is proposed but unimplemented; `entropy/approx_head` is still `jnp.mean(face_ents)` over all env steps (`ppo.py:7651-7655`) with face-less steps entering as structural zeros, and `f_arity` (`unified_face_head.py:250, 295`) counts only non-NONE slots while the numerator counts the skip Bernoulli plus three op softmaxes. With both entropy terms zeroed this is telemetry-only — but the panel is **not comparable across episodes** (357× vs 91× in v64b), so it cannot be used to read the clean run. | `H_face_batch = jnp.sum(face_ents_raw) / jnp.maximum(jnp.sum(face_arities), 1.0)` for the metric (and for the hinge, if ever re-enabled). |
| **4** | **The DIAG factor is not in the action space.** `unified_face_policy._rows` (192-206) pins `factor = gcd(N_i,N_j)`; the 94-slot layout has no factor field. If the drawing's per-face choice is meant to include the block size, no flag reaches it. | Would need a factor field in the head layout + `_rows` + `score`/`sample` (a real change to `HEAD_WIDTH`). Recommend **not** doing this for the clean run — it was removed because the env clamped it to a divisor of the gcd anyway. |
| **5** | **`face_sizes` is never passed by `ppo.py`.** `UnifiedFacePolicy.sample/evaluate` accept it, but with `None` `_face_feats_1` falls back to the shared vertex features, so each face's **legality masks** are derived from the vertex's static features rather than its own live contraction. | Thread `LiveVertexMaskOracle.face_features` through the `sample`/`evaluate` call sites. Affects legality only, not the head input — so it does *not* violate the clean design. |
| — | **A4 (LANDED): `--seed-vertices` dropped — seeds are NOT vertices.** Three findings, one change. **(i) They were real vertices.** `seed_loss_fn`'s tangent-seed `add` and adjoint `reduce_sum` entered as ORDINARY ELIMINABLE VERTICES — into `valid_vertices` (`env.py`), into `masks.build_vertex_valid_static`, and into the `derived_max_faces` closure bound (nn256: 13 → 15 eqns). And the freedom they were bought for — forward / reverse / cross-country seeding as part of the search — was **unreachable while b1 (`ALPHAGRAD_FORCE_REV_ORDER=1`) is on**: `common/masks.py:145-151` keeps only the HIGHEST remaining vertex, so the order is pinned to reverse and only approximations are learned. The search was paying action space and face budget for a degree of freedom b1 had already closed. **(ii) The flag was a silent no-op without `--measure-grad`** for the entire campaign — `grad_target_setup` / `grad_target_fn` return before the seed branch when grad mode is off, and said nothing; the shell encoded the rule by hand instead (`run_full_nn256_v2_5seed.sh:305,309`). **(iii) Two leaf conventions contradicted each other**: `_walk_argnums` EXCLUDED 0-d argnums (stepping the seed moves every weight by `t*ones`) while `_leaf_norms` COUNTED the 0-d seed, so `--reject-frozen-grads` (default ON) could sentinel an entire plan on the SEED's gradient. `--measure-grad` is **KEPT** and is *not* redundant: `scalar_loss_fn` wraps the target BEFORE tracing so the traced graph is the loss graph, it flips the quality default cosine→`loss_drop`, and it arms the scalar-output contract check. | Landed: `common/examples.py` `_seed_vertices_requested` (hard error on the incoherent combo + deprecation warning); `env.py` `_leaf_norms` 0-d convention, `_grad_coverage` nan-safe eps and `n_uncountable`; every launcher stripped; pinned by `src/alphagrad/approx/tests/test_seed_vertices_dropped.py`. |
| — | **Entropy: both terms CAN be zeroed by flag.** Verified: `_split_entropy_bonus` returns `entropy_weight*(H_total − H_face) + face_entropy_weight*H_face`, and §13.2(d) proves by `jax.grad` on the real loss that the net face coefficient equals `--face-entropy-weight` exactly. The floor is behind a static `> 0.0` gate. **No code change needed.** | — |
| — | **The quality gate CAN be turned off** by `ALPHAGRAD_QUALITY_GATE_MIN=0` (`env.py:1997-2002` early-returns unchanged costs). **No code change needed.** | — |
| — | **The pointer CAN be fed memory while the face head is denied it** — that is the default. Omit `--face-endpoint-read` and `--face-edge-mem`; `UnifiedFacePolicy._repr` then returns the E-wide face latent alone, while `SetPointerVertexPolicy._score` keeps reading `vmem`. **No code change needed.** | — |

---

## 4. CONSEQUENCE ANALYSIS

Honest, and short. Three of the owner's four constraints are individually
achievable today; two of them put the run, at episode 0, into the state every
prior campaign took 60–90 episodes to *fall* into.

### (i) Random init: what fraction of ep0 plans is catastrophic, and does any advantage contrast survive?

**Rate.** The measured none-bias sweep (§13.3, 400 Adam steps on the real 94-logit
head) gives, at bias 0: `p_none/slot = 0.250`, `p_approx/slot = 0.748`,
`E[arity] = 3.239`, `p_skip = 0.002`. The TLM episode is **115 faces / 95 steps**
(FACE_LATENT §1). So:

- P(a face receives no approximation at all) = `(1−0.002)·0.25³ = 0.0156`.
- E[approximation ops per plan] ≈ `115 × 3.239 ≈ 372`.
- P(any plan in a 16-env batch is exact) ≈ `0.0156^115` ≈ 10⁻²¹³.

For scale: the shipped bias-6 config produces **~20 stray approximations across
all 16 envs** at ep0 (§5); the 0.3 entropy floor corresponds to **4.9–8.1 approx
ops per plan** (§13.5) and §12.2 already calls that "structurally inside the
violation regime". Random init is ~50–75× past the floor.

**Catastrophe fraction.** The empirical anchor is v64b at ep84 — `applied = 2368`
over 16 envs = **148 approx/plan**, `approx_prob/none = 0.0024` — where the
measured outcome was `lagrangian/frac_violating = 0.9375` and
`mean_quality = 0.0948` (`collapse_invest/v64b_zneut/v64b_full.csv`). Random init
sits at 372/plan, 2.5× further along the same axis. Independently, v58b's
basin-occupancy windows (§2, from `collapse_invest/v58b_history.csv` via
`an6_basin.py`) show `eps(worst ≤ −0.99)` — a genuinely diverged Adam walk — at
**30% of episodes already at ep0–10**, with only ~20 stray approximations in the
batch. **Prediction: essentially 100% of ep0 plans below `qmin`, and the diverged
(−1.0) fraction is a large minority, not a tail.**

**Does contrast survive?** Split by channel:

- **Cost (cmp, mem).** With `ALPHAGRAD_QUALITY_GATE_MIN=0.05` on, every sub-qmin
  plan's latency and memory are floored at the *same-order* exact reference. Under
  a rev pin the order is one constant ⇒ one constant floor ⇒ the whole batch gets
  bitwise-identical cost ⇒ the cmp and mem advantages are **exactly zero**. The
  clean config turns the gate off, which restores cost contrast — *and restores it
  as the SKIP cliff*: §12.3 records that at λ=1 destruction was cost-profitable and
  v61 collapsed into the 100%-SKIP cost-optimal absorber. The gate and the contrast
  are the same lever pointing opposite ways; removing the gate is not free.
- **Quality.** `loss_drop` is continuous in principle, but §0 records that the
  settled empirical distribution is `measure/quality/median_ep` **exactly 0.0**
  (the zero/dead-gradient signature, L1 == L0) plus occasional −1.0 (diverged) plus
  a rare survivor. Only **5 of 192** episode medians ever sat in `[0.05, 0.5)`; GAZ
  measured **0%** in that band across 122 and 333 episodes. Under `additive` mode
  that is close to a one-bit "did this plan die" signal.
- The informative regime — approximated *and* alive — is the regime the measured
  distribution says barely exists on this target.

**The symmetric problem, for fairness.** Identity init has the opposite failure:
§5 measures ep0 quality spread ≤ `1.3e-3` across 16 near-exact plans, best−median
`5e-6`. So identity init gives no quality contrast and random init gives no cost
contrast plus a coarse quality contrast. The owner's constraint trades one for the
other rather than fixing either.

**Bottom line.** Random init does not start the agent in a neutral state; on the
measured distribution it starts it inside the absorber's basin. Every run that
reached that state stayed there — v63 held λ pinned at 20 with `mean_raw_q ≈ 0`
for **430 further episodes with zero recovery**, `scalarized_return` falling
monotonically −58 → −114.5. The one change in the clean config that materially
opposes this is `--discount 1.0 --gae-lambda 1.0` (see iii).

### (ii) Face head with chunk-mean-only input

Achievable today — just drop `--face-endpoint-read`. This is exactly the
configuration v62 ran, and v62 slid **~9 episodes earlier** than v63 (§12.4).

What the head **can** see (FACE_LATENT §7, train/cv, majority in parens):
`ndim` for lhs/rhs/res at 0.82/0.86/0.89 train against majorities 0.47/0.76/0.64;
`dtype` at 0.94/0.99/1.00 against majorities 0.91/0.99/0.99 — i.e. **dtype is at
baseline**, it sees nothing there.

What it **cannot** see: **live operand sizes, at any width.**
`size-R²` is 0.45/0.48/0.55 train but **−0.64/−0.25/−0.95 on CV**. §9's width
ablation (both-intermediate stratum, CV, λ tuned per cell) puts chunk-mean at
`0.46 ndim / −0.10 szR²` at E=32 against a 0.43 majority, and widening to E=512
moves lhs size-R² only to +0.03 — "slope +0.021/doubling; reaching C@32's 0.22
extrapolates to E ≈ 10⁵". Verdict quoted: *"the lever is the keying, not the
width. Width is the expensive non-fix."*

The mechanism, from §2 H-B: all of a target's non-trivial dims appear in the
face's **own chunk** for only **13–16%** of faces, versus **100%** in the
stream-so-far. Sharpened in §7's census: the whole episode stream contains 1126
digit-token occurrences but only **8 distinct numbers**, and only **2 distinct
dims ever appear in shape context** (128 and 1024). The sizes the head needs are
*products created by contractions*, never spelled as tokens.

And the little signal that is there at init **drains during training** (§2 H-D(2)):
lhs ndim rises to 0.64 by ~ep17, decays to the 0.47 baseline by ~ep60, and freezes
with probe loss constant at 3.228 to four digits — gradient ≈ 0, the head sitting
at the marginal-entropy optimum of an uninformative input. Nothing in the loss
rewards keeping face-latent variance.

**Consequence.** The 94 outputs include a 6×6 `(i, j)` DIAG pair choice and a
9-way reduce-axis softmax whose correctness depends on exactly the sizes the head
provably cannot decode. Under this constraint the face head is choosing *which*
approximation essentially blind, and can only learn *how much* — which is the
scalar the entropy floor and the none-bias were both crude proxies for.

**One route stays open under the constraint.** Both known fixes (endpoint read,
edge-mem) are memory reads and are therefore forbidden. But §10's alternative —
a **learned-query palimpsa decoder at E=256** — is still "the MLP on what palimpsa
emits for that face", with a learned read replacing the parameter-free mean.
Measured: `Sb@256` clears all four targets (lhs ndim 231%, lhs szR² 120%, rhs ndim
181%, rhs szR² 109% of bar), while the hand-built variant never decodes size at
any width. It is the only fix compatible with the owner's memory constraint, and
§10 is explicit that **extraction needs learning**.

### (iii) Unpinned vertex order (`FORCE_REV_ORDER=0`)

**The pointer has never been trained.** Under the pin, `_mask_vertex_logits` leaves
exactly one legal vertex, so §13.2(e) measures ve-head entropy at `-0.0e+00` and
`max|dH/dlogits| = 0.0e+00`; every episode of v64b and of all four live arms logs
`ve_head=0 macro_vertex=0`. Lifting the pin starts the pointer from its `×0.1`
near-uniform init with no accumulated learning anywhere in v57–v66.

**Search cost, measured.** From FACE_LATENT §8's random-order capture: the episode
face count goes **115 (rev) → ~313 (random orders)** — 938 faces across three
episodes — and distinct edge keys go **101 → 206–286**. `derived_max_faces` is
`max_v |anc(v)|·|desc(v)|`, order-independent, so `MAX_FACES=2538` remains a valid
bound; the per-step `while_loop` and the per-face head calls scale with the actual
count, so expect roughly **2.7× the face work per episode**, plus a longer delta
stream (watch `tokenization/truncated_count`).

**Cost floor.** `_apply_quality_gate`'s own comment (`env.py:2020-2026`) records
that **random orders measure ~500× above rev** — which is *why* the gate had to
move from a global rev reference to a per-order exact floor. That per-order floor
is a compile per sub-qmin plan and can OOM (`env.py:2029-2046`). With the gate off
in the clean config this cost disappears, but so does the protection: a random
order that is 500× slower than rev now competes directly on the cost channel with
a rev-ordered plan that has been destroyed by approximation, and the destroyed
plan wins.

**Prior evidence on whether the search pays.** The repo's own record is negative
and consistent: POMO only **matched** reverse (1.01×) and never beat it; PPO
**never left the uniform policy**, with the diagnosed cause being raw-ns reward
under PopArt at 0.08σ combined with `γλ = 0.99·0.95` over 95 steps, so the first
elimination receives **0.31%** of the terminal signal. Neither dossier reports a
measured wall-clock or sample-complexity cost for order *search* — it has been
pinned off throughout, so this is genuinely un-de-risked.

**The one genuinely favourable change.** `--discount 1.0 --gae-lambda 1.0` removes
the second of those two causes outright: with a terminal-only reward and
`γ = λ = 1`, every step of the episode receives the *same* terminal return, so the
first elimination is credited identically to the last. This is the single most
defensible item in the whole clean-room set, and it is the change most likely to
make order search learnable at all. It is also the change most likely to *increase*
variance, since the credit assignment is now pure Monte Carlo over ~95 steps.

**Constraints (ii) and (iii) compound.** Under the rev pin the face stratum is
degenerate — §8 measures **1 both-primitive / 114 one-intermediate / 0
both-intermediate** — and every §2–§7 face number was measured on that one
stratum. Free orders move the population to **164 / 311 / 463**, i.e. the
both-intermediate stratum becomes the majority, and that is precisely where the
chunk-mean read is measured *worst*: arm A lhs ndim `0.52/0.45` on both-intermediate
versus `0.78/0.72` on both-primitive, with `szR²` negative throughout. Unpinning the
order therefore makes the face head's blindness materially worse at the same time
as it makes the pointer's job harder.

### (iv) Plain 94-MLP approx head

Already satisfied. `UnifiedFaceHead.proj` is literally
`eqx.nn.MLP(in_dim, 94, hidden=embd_dim, depth=1)` (`unified_face_head.py:211-212`),
with `in_dim == E` once `--face-endpoint-read` and `--face-edge-mem` are off. The
only additions on top are (a) legality masks, which are **required** — graphax
fails loudly on an ill-fitting transform; (b) branch masking, which is **required**
for the PPO ratio to be 1 at epoch 0; and (c) the tanh logit clamp, which the clean
config removes. The one silent deviation from "plain" is CODE-CHANGE #1: the head is
the only action head `_scale_output_heads` never touches, so its 94 logits are not
on the same scale as any other head's.

---

## 5. Summary of verdicts

- **CONFLICTS (14):** a1 face-none-bias, a3 unscaled face head, a4 logit clamp,
  b1 force-rev-order, b6 pinned DIAG factor, c1 quality gate, c2 lagrangian
  transform, c3 lag-tau, c4 lambda, c5 reward-mode, d2 γ/λ defaults, d3 PPO epoch
  asymmetry, d4 causal mask, f1/f2/f3/f4 the four entropy terms, g1 endpoint read
  — plus e1/e3/d5 on the v64b arm specifically.
- **REQUIRED (9):** op legality, approx-allowed, j-mask, branch masking, MAX_FACES,
  MAX_EQNS, MAX_DELTA_TOKENS, measure-grad/seed-vertices, grad-window 0.
- **MATCHES the clean design already (6):** terminal-rewards-only, set-pointer,
  unified-face-head, live-faces, `advantage-norm none`, the pointer's vmem read.
- **INERT / dead (6):** muls sentinel cap, anti-degen penalty, potential shaping,
  lag-raw-viol-adv, pin-rules-to-exact, pref_proj zeroing.
- **Un-fixable by flag (5):** the five CODE-CHANGE items in section 3.
