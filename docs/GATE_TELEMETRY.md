# Gate telemetry: the wandb contract for G1-G6 (ticket dsnn-3qm.45)

One row per field. Emitted once per episode by
`alphagrad.approx.common.gate_telemetry.episode_fields`, called from the
per-episode logging block in `ppo.py` (`host_log`, right after the plan-log
drain). The module is read-only: its inputs are copies the trainer already
made (the drained plan records, the critic target/prediction pair the value
loss compares, the face-head entropy the loss reports, the per-env preference
and terminal reward rows); nothing it computes is read by the reward, the
advantage, the value target or the sampler. `tests/gate_telemetry_test.py`
pins that a fake episode emits exactly the names below.

The gate (ticket .6, accepted 2026-09-02): G1 recovery of the sweep winners by
(vertex, primitive); G2 explained variance of the latency head > 0; G3 face-head
entropy near the uniform floor and decreasing; G4 q = 0 fraction -> 0; G5 front
spread at the simplex corners > drift floor; G6 offline contrast >= 1 %.

Conventions:

- **Ratio** = candidate / rev-exact, paired (same actor, back to back, warm),
  dimensionless, < 1 = cheaper. The per-plan reference is ticket .9's plan-log
  fields `ref_latency_ns`, `ref_temp_bytes` and `ref_watermark_bytes`
  (positive units), and the candidate's own side is `candidate_latency_ns`,
  `mem_temp_bytes` and `mem_watermark_bytes`.
- **NOT the reward slot.** Under `--cost-form paired-log` (ticket .9, every
  2026-09-13 arm) reward slots 2 and 5 hold `-(log cost - log ref)`, a
  dimensionless log-difference, NOT `-ns` and `-bytes`. Reading a latency off
  the slot gave a negative number and the positivity guard turned every
  `paired/lat_ratio_*` into NaN between 2026-09-05 and 2026-09-13 (job 65321).
  The module reads the absolute fields and refuses to reconstruct ns from a
  log-difference; a record written under `--cost-form absolute` still falls
  back to the negated slot.
- **Positive units.** The reward vector stores costs negated; every ns / bytes
  value here is positive.
- **Live** = not sentinelled (no cost channel at -1e10). Every count and
  statistic is over live records unless the row says otherwise.
- **NaN** means undefined or absent, never 0. `*/present` flags say whether an
  input existed. A missing input prints one `[gate] ...` line to stderr.
- **Rev-exact** is decided on the wire (reverse order, no rule, no face row),
  not on a label.
- **Corner** = the env's preference put >= 90 % of its mass on one head.
- **G3 floor, DERIVED.** For face f with per-slot legal choice counts n_s
  (1 for None + ordered Diag pairs i != j + reduce axes x reduce fns +
  legal Quant dtypes, each term only when its op is legal under ticket .59's
  bottom-up rule and ticket .40's profile mask), the legal joint outcomes
  number N_f = [skip legal] + prod_s n_s. `uniform_floor_per_face_nats` is the
  mean of log N_f. The trainer's `entropy/approx_head` divides the summed face
  entropy by the arity (one per valid face plus one per non-None slot), so the
  comparable `uniform_floor_nats` is sum_f log N_f / sum_f E[arity_f] with,
  under the uniform law, E[arity_f] = 1 + (prod_s n_s / N_f) sum_s (1 - 1/n_s).
  EVERY CONSTANT IN THAT ARITHMETIC COMES FROM
  `unified_face_head.head_layout(--approx-add)` and
  `common.masks.FACE_QUANT_DTYPES`, never from a literal
  (`gate_telemetry.face_head_geometry`): the slot count is the running
  width's, and the QUANT term is the FOUR-way dtype categorical
  (float32, bfloat16, float8_e5m2, float8_e4m3fn) with the operand's own
  dtype masked (ticket .40 D4), so a face whose mask says only "QUANT is
  legal" offers K - 1 = 3 casts, not 1. Under the 2026-09-13 head (103
  logits = 1 + 34*3) a 6-axis face with everything legal has
  n_s = 1 + 30 + 45 + 3 = 79 and N_f = 1 + 79^3 = 493,040, i.e. 13.11 nats
  per face; the live masks make it far smaller, which is why the floor is
  computed from them and not stated as a constant. `gate/g3/n_slots`,
  `gate/g3/head_width` and `gate/g3/n_quant_dtypes` put the derivation in the
  run so a floor can never quietly describe a head that stopped running.
- **Where the G3 masks come from.** `gate/g3/mask_source` is 2 on the LIVE
  path (`--live-faces`, every campaign arm): `LiveFaceStream.face_slot_legality`
  at the reset state, per face, per slot and per QUANT dtype. It is 1 on the
  oracle path and 0 when neither is available. Before 2026-09-13 only the
  oracle path existed and `ppo._NO_ORACLE` is true whenever `--live-faces` is
  on, so every `gate/g3/*` floor field read NaN in every run that mattered.
- **G5 drift floor.** The floor a front spread must beat is the spread of the
  PAIRED REFERENCE, not of candidates that happen to be rev-exact: ticket .9
  re-measures the same rev-exact plan once per candidate in the same actor,
  so `(max - min) / mean` of `ref_latency_ns` within an episode is the
  instrument's own noise on the same computation. The old definition needed a
  candidate that is rev-exact ON THE WIRE (strictly descending order, no rule,
  no face row), which under `--fixed-order markowitz` is never any of them, so
  it was structurally NaN on every campaign arm; it is kept under
  `*_revexact` for the reverse-order control of ticket .60.
- **G5 present.** `gate/g5/present` is 0 on an arm without
  `--preference-conditioned`: there is one implicit weighting and no simplex
  to spread over, which is a stated absence rather than a NaN with no reason.
- **G2** is per head, on `_value_target(estim_returns)` against the head's
  prediction (the pair the value loss compares), over every (env, step) of the
  episode. NaN when the target is constant.
- **G6** is a placeholder for ticket .42's number: `--gate-offline-contrast`.
- **G1** needs `--gate-winners-table` (CSV or JSON: `vertex`, `primitive`,
  `kind`), the output of the sweep .41. The sweep has not run; the gate accepts
  an absent table (`gate/g1/present = 0`).

Counters are drained in the process that increments them (ticket .7): the
toolchain and memory-parity rows below are logged by `ppo.py`'s plan-log block
off `env.consume_plan_records` (merged with the measure pool's) and are listed
here so the gate reads one table. `measure/drain/*` AUDITS that: the measure
actors count their own terminal plans, and `measure/drain/undrained` is what
those counters claim minus what actually reached this process. It reads 0 on a
healthy episode and the number of lost plans on the ticket .7 shape -- the
cancelled canary 65319 logged `pool_terminals=16` beside `pooled=0`, i.e. every
gate field that episode was computed over an empty record set.
`gate_telemetry.drain_provenance(..., strict=True)` raises on it.

`episode_fields` RAISES if the table below names a field it does not emit: a
gate input that is silently absent is the failure tickets .43 and .45 exist to
end.

| field | unit | source | gate |
|---|---|---|---|
| `paired/n` | count | plan records this episode, live (not sentinelled) | - |
| `paired/n_with_ref` | count | live records carrying ref_latency_ns (ticket .9) | - |
| `paired/lat_ratio_mean` | ratio | mean over live records of latency_ns / ref_latency_ns | G5 |
| `paired/lat_ratio_median` | ratio | median of the same | G5 |
| `paired/lat_ratio_best` | ratio | min of the same (fastest plan) | G5 |
| `paired/temp_ratio_mean` | ratio | mean of mem_temp_bytes / ref_mem_temp_bytes | G5 |
| `paired/temp_ratio_median` | ratio | median of the same | G5 |
| `paired/temp_ratio_best` | ratio | min of the same (leanest plan) | G5 |
| `paired/grad_cosine_mean` | cosine | mean quality (reward slot 'quality') over live records | G4 |
| `paired/grad_cosine_median` | cosine | median of the same | G4 |
| `paired/grad_cosine_best` | cosine | max of the same | G4 |
| `gate/g1/present` | 0/1 | 1 when a winners table was loaded (--gate-winners-table) | G1 |
| `gate/g1/n_winners` | count | rows in the winners table | G1 |
| `gate/g1/n_recovered` | count | winners whose (vertex, class) some plan this episode applied | G1 |
| `gate/g1/recovery` | fraction | n_recovered / n_winners | G1 |
| `gate/g1/recovery_skip` | fraction | recovery restricted to SKIP winners (NaN if none) | G1 |
| `gate/g1/recovery_diag` | fraction | recovery restricted to Diag winners | G1 |
| `gate/g1/recovery_compress` | fraction | recovery restricted to Reduce winners (code: compress) | G1 |
| `gate/g1/recovery_quant` | fraction | recovery restricted to Quant winners | G1 |
| `gate/g1/primitive_mismatch` | count | winners whose primitive label differs from the run's jaxpr at that vertex | G1 |
| `gate/g2/n` | count | (env, step) pairs with a finite target and prediction | G2 |
| `gate/g2/ev_{head}` | fraction | explained variance of value head {head}: 1 - Var[target - pred] / Var[target]; NaN when Var[target] = 0 | G2 |
| `gate/g3/face_entropy_nats` | nats | the trainer's arity-normalised face-head entropy (entropy/approx_head) | G3 |
| `gate/g3/uniform_floor_nats` | nats | the same quantity if the head were uniform over legal joint outcomes, under the run's legality masks | G3 |
| `gate/g3/uniform_floor_per_face_nats` | nats | mean over faces of log(number of legal joint outcomes) | G3 |
| `gate/g3/entropy_over_floor` | ratio | face_entropy_nats / uniform_floor_nats | G3 |
| `gate/g3/n_faces` | count | faces the floor was computed over | G3 |
| `gate/g3/n_outcomes_max` | count | largest legal joint-outcome count of any face | G3 |
| `gate/g4/n` | count | live records with a finite quality | G4 |
| `gate/g4/q_zero_frac` | fraction | share of those with quality exactly 0 (destroyed Jacobian) | G4 |
| `gate/g4/tau` | cosine | the quality floor in force (--quality-floor, ticket .9); NaN if none | G4 |
| `gate/g4/q_ge_tau_frac` | fraction | share of live records at or above tau (feasible); NaN if no floor | G4 |
| `gate/g5/{corner}/n` | count | live records whose env preference sat at the {corner} corner | G5 |
| `gate/g5/{corner}/best_lat_ratio` | ratio | min paired latency ratio among them | G5 |
| `gate/g5/{corner}/best_temp_ratio` | ratio | min paired temp ratio among them | G5 |
| `gate/g5/spread_lat` | ratio | max - min over corners of best_lat_ratio | G5 |
| `gate/g5/spread_temp` | ratio | max - min over corners of best_temp_ratio | G5 |
| `gate/g5/n_rev_exact` | count | live records whose plan IS rev-exact (reverse order, no approximation) | G5 |
| `gate/g5/drift_floor_lat` | ratio | max - min of the latency ratio over those rev-exact records (their ratio is pure drift) | G5 |
| `gate/g5/drift_floor_temp` | ratio | the same for the temp ratio (0 when the static temp is deterministic) | G5 |
| `gate/g5/n_unmatched` | count | records that matched no env row (no preference known) | G5 |
| `gate/g6/present` | 0/1 | 1 when --gate-offline-contrast was given | G6 |
| `gate/g6/offline_contrast` | fraction | ticket .42's offline contrast, copied from the flag; NaN when absent | G6 |
| `measure/compile_fallbacks_this_ep` | count | ppo.py plan-log block; env.consume_plan_records (ticket .21) | toolchain |
| `measure/compile_fallbacks_total` | count | same drain, process lifetime | toolchain |
| `measure/toolchain_ok` | 0/1 | same drain; 0 = a measure process failed the toolchain gate | toolchain |
| `measure/mem_parity/gap_mean_bytes` | bytes | ppo.py plan-log block; env.mem_parity_summary (ticket .49): watermark - temp | memory |
| `measure/mem_parity/gap_max_bytes` | bytes | same | memory |
| `measure/mem_parity/gap_min_bytes` | bytes | same | memory |
| `measure/mem_parity/temp_mean_bytes` | bytes | same: the channel | memory |
| `measure/mem_parity/watermark_mean_bytes` | bytes | same: the runtime watermark beside the channel | memory |
| `measure/mem_parity/static_fallbacks` | count | same: readings that were static substitutions | memory |
| `measure/mem_parity/n` | count | same | memory |
| `measure/mem_parity/n_paired` | count | same | memory |
| `measure/mem_parity/measured` | count | same | memory |
| `entropy/approx_head` | nats | ppo.py host_log; the loss's arity-normalised face-head entropy | G3 |
| `explained variance` | fraction | ppo.py host_log; EV of the SUM over heads (the pre-.45 panel) | G2 |

| config key | unit | source | role |
|---|---|---|---|
| `toolchain/jax` | version | jax.__version__ | fingerprint |
| `toolchain/jaxlib` | version | jaxlib.__version__ | fingerprint |
| `toolchain/xla_flags` | string | os.environ['XLA_FLAGS'] ('' if unset) | fingerprint |
| `toolchain/jax_platforms` | string | os.environ['JAX_PLATFORMS'] ('' if unset) | fingerprint |
| `toolchain/backend` | string | jax.default_backend() | fingerprint |
| `toolchain/sparse` | 0/1/-1 | env.config.sparse (jacve returns SparseTensor); -1 = unknown | fingerprint |
| `toolchain/hostname` | string | the trainer's host | fingerprint |
| `commit/alphagrad` | sha | ppo._repo_commits (already there) | fingerprint |
| `commit/graphax` | sha | ppo._repo_commits (already there) | fingerprint |
