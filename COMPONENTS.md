# Component reference — what works, where it lives, how to reuse it

Parts catalogue for building a **new, small trainer** out of the pieces that
already work. Every entry: what it does · exact location · the interface you
call · gotchas that cost us time. Line numbers are as of alphagrad `70f737a`
/ graphax `c4f76f8`; symbol names are the stable anchor if they drift.

Paths: `AG = alphagrad/src/alphagrad/approx/` · `GX = graphax/src/graphax/`

**Status key** — 🟢 use as-is · 🟡 works, needs adaptation · 🔴 broken/don't use

---

## 0. The 60-second mental model

```
jaxpr of the target fn  ──►  tokenized  ──►  palimpsa encoder
                                                   │
                    ┌──────────────────────────────┘
                    ▼
       pointer head picks a VERTEX to eliminate      (macro action, O(|V|))
                    │
                    ▼
       per-vertex SUB-EPISODE of micro-actions:      (approximation)
         DIAG(i,j,factor) | COMPRESS(axis,kind) | QUANT(dtype) | END
         each masked against the LIVE tensor structure
                    │
                    ▼
       env applies them via graphax, measures the result
         → 8 reward channels (flops, mem, latency, cosine, frob, …)
                    │
                    ▼
       PPO: 4 value heads, per-channel GAE, preference-weighted advantage
```

Two loops, one policy. The macro loop is ordinary AlphaGrad. **Everything hard
is in the micro loop**: legality depends on the live tensor's index structure,
which only exists *during* elimination — hence the host-side oracle (§2).

---

## 1. Environment & measurement — `AG/env.py` (~1900 lines)

### 🟢 `VertexEliminationEnv` — the env
| | |
|---|---|
| Construct | `VertexEliminationEnv.from_jaxpr(...)` — `env.py:1625` |
| Reset | `env.reset(num_envs=None) -> EnvState` — `env.py:1758` |
| Step | `env.step(state, StepAction(target_vertex, rule_specs)) -> EnvOut` — `env.py:1808` |

`step` is `@jit` with an `io_callback` to the host `_callback` (`env.py:1178`)
which does the real work: apply transforms → compile → measure.

**`EnvState`** (`env.py:331`) — the fields that matter:
- `order` **(N,) int32, 1-based** — elimination order so far, prefix `[:step_count]`
- `sparsity_specs` **(N, MAX_RULES_PER_VERTEX, 3) int32** — the rules applied at
  each step, index-aligned with `order`. *This is the replay history.*
- `tokens`, `eqn_ids` — the observation (padded to `MAX_TOKENS=4096`)
- `axis_state` **(V, MAX_AXES, 4)** — per-vertex axis features for the policy
- `step_count`, `max_steps`

**Wire format** for `sparsity_specs` rows — decode/encode with the ONE shared
function, never by hand:

```python
decode_vertex_rule_specs(jaxpr, vertex, spec_rows, is_last) -> tuple[Diag|Compress|Quant, ...]
# AG/env.py:1060
```
| `row[0]` | meaning |
|---|---|
| `-1` | end-of-sequence sentinel |
| `>= 0` | `DIAG(bi1=row[0], bi2=row[1], factor=row[2])`; `factor=-1` ⇒ joint gcd |
| `COMPRESS_SENTINEL (-2)` | `COMPRESS(axis=row[1], kind=COMPRESS_KINDS[row[2]])` |
| `QUANT_SENTINEL (-3)` | `QUANT(dtype=QUANT_DTYPES[row[1]])` |

> ⚠️ **COMPRESS is only legal on the LAST vertex of the partial order** — it
> reduces `val.ndim`, which trips graphax's shape assertion if a later
> elimination consumes that edge. `is_last` enforces it.

### 🟢 Reward channels — `env.py:210`
```
0 muls_adds_fmas   1 flops        2 latency_ns    3 max_io_sum
4 bytes_accessed   5 peak_memory  6 cosine_sim    7 frob_residual
```
**Convention: higher is better.** Costs are stored NEGATED; `cosine_sim ∈ [0,1]`
raw; `frob_residual` stored as `-residual`. Any new consumer must respect this
or the sign flips silently.

### 🟢 Quality metrics — `_quality_metrics` `env.py:958`
```python
cos, rel_frob = _quality_metrics(jac_exact, jac_approx)
```
- Calls `_align_jac` (`env.py:937`) first: transposes back 2-D leaves delivered
  in reversed layout. Without it a (256,784) vs (784,256) ravel is
  near-orthogonal and cosine is a **layout artifact**.
- `jnp.real(cos)` — some quant/compress combos produce complex Jacobians.
- **Degenerate (no leaves / shape mismatch / zero size) ⇒ `(0.0, 1.0)` = WORST.**
  It used to return `(1.0, 0.0)` = *perfect*, which was a reward backdoor: a
  shape-destroying plan out-scored every honest one.

### 🟢 Robust aggregation — `env.py:1017`, `env.py:1030`
```python
_winsorized_mean(stack, lo_q=0.25, hi_q=0.75)   # IQR-clip then mean
_aggregate_samples(values, want_top_quartile)    # winsorized for >=4 samples
```
Also `_LAT_FLOOR_NS = 100.0`: a positive latency below the floor is clamped UP,
so a zero-work plan can't report an unbeatable time.

### 🟡 Measurement loop — `_callback` `env.py:1178`, exec loop ~`env.py:1413`
Currently `n_samples = 10 if measure_latency else 1`, and **quality uses only
eval-sample 0**. The spec wants 5 samples × 4 reps = 20 winsorized. The loop
structure is right; only the counts and the quality-over-all-points are missing.
> ⚠️ `from_jaxpr(**_compat)` **silently swallows** `latency_samples`,
> `num_data_points`, `reps_per_point`, `percentile_keep`, `latency_winsor`
> (`env.py:1645`). Passing them does nothing. Wire them properly or drop them.

### 🟢 Host-side telemetry (escapes the jit's fixed reward width)
```python
consume_tokenization_truncation_stats()  # env.py:136  -> count/max_len/overflow
consume_xla_memory_stats()               # env.py:197  -> xla_peak_memory, compression_ratio
```
Pattern: `_callback` records into a module-level list; the driver polls once per
episode. Use this for any diagnostic that doesn't fit the reward vector.

---

## 2. Masking — `AG/common/masks.py` (~960 lines) — **the crown jewels**

Legality can't be precomputed: the operands are join intermediates whose index
structure only exists during elimination. So we replay it host-side.

### 🟢 `LiveVertexMaskOracle` — `masks.py:427`
```python
o = LiveVertexMaskOracle(jaxpr, consts, args, argnums, max_axes=8)
o.advance(vertex, rules=decoded_rules)   # masks.py:493  — keep in lockstep
pair, comp = o.masks()                   # masks.py:760  — (V+1,N,N), (V+1,N) 1-based
p, c       = o.vertex_mask(v)            # masks.py:573  — one vertex, AND over faces
fp, fc, n  = o.face_masks(v, max_faces)  # masks.py:666  — PER-FACE, not AND-ed
```
> ⚠️ **`advance` MUST get the rules the env actually applied.** A structural
> `rules=()` replay diverges from the real graph after the first landed
> approximation → masks computed against the wrong edges. Decode them from
> `state.sparsity_specs` with `decode_vertex_rule_specs`.

> ⚠️ It probes **both** elemental-dispatch modes and intersects
> (`_DISPATCH_MODES`, `masks.py:~480`) — an action must be legal on both paths
> or whichever runs first sentinels the measurement.

**`vertex_mask` ANDs over all faces** (a per-vertex rule list hits every face),
which is why per-vertex DIAG is nearly always illegal. **`face_masks` keeps them
separate** — that's the per-face payoff. Note: the win is *expressivity*
((L+1)^F configs vs L+1), not necessarily more legal actions; extra legality
only appears when faces have heterogeneous shapes.

### 🟢 Primitive legality predicates
```python
diag_valid_mask(st, max_dims)        # masks.py:184
compress_valid_mask(st, max_axes)    # masks.py:256
diag_pair_factor_space(st, i, j)     # masks.py:278 -> (base, span)
legal_diag_actions(st, max_dims)     # masks.py:350 -> list[Diag]
legal_compress_actions(st, ...)      # masks.py:365 -> list[Compress]
rule_is_legal(st, rule, ...)         # masks.py:836
```

### 🟢 Vertex availability
```python
vertex_valid_static = build_vertex_valid_static(env.valid_vertices, total_v)  # :144
mask = vertex_avail_at_step(state, vertex_valid_static, total_v, num_valid)   # :152
```
Enforce by `jnp.where(mask > 0.5, logits, -1e9)` **at both sample and loss time**
(store the mask in the trajectory — that's what keeps PPO ratio == 1).

### 🟢 Per-face chooser (for the per-face mode) — `masks.py:782`
```python
chooser = masked_micro_chooser(pick)     # pick(st, legal_actions) -> action | None
incr.eliminate(v, face_transforms={k: (chooser, chooser, chooser) for k in incr.faces(v)})
```
Enumerates legal actions *from the live tensor*, so illegal is unrepresentable.
Returning the chosen **action** (not a tensor) keeps it in the transform log.

---

## 3. Policy heads — `AG/heads.py` (~1700 lines)

### 🟢 `MicroActionPolicy` — the sub-episode
```python
actions, dists..., logps = policy.sample(v_context, features, factor_tables,
                                         quant_legality_mask, key,
                                         pair_valid=..., compress_valid=...)
logp, entropy, arity     = policy.evaluate(..., actions, ...)
```
`MicroAction` (`heads.py:852`) fields: `op_type, i, j, exponents, factor,
compress_kind, quant_dtype, quant_scale_sign` — **all 8 required**.

Ops (`heads.py:~79`): `DIAG=0, COMPRESS=1, QUANT=2, END=3`. END is always legal
(that's the 0-option guard). There is **no skip op** — END terminates the
sub-episode; the vertex is still eliminated.

### 🟢 Head zoo
| Head | Where | Emits |
|---|---|---|
| `OpTypeHead` | `heads.py:301` | masked categorical over the 4 ops |
| `AxisPointerHead` | `heads.py:333` | pointer over axis tokens (i, then j) |
| `PrimeExponentHead` | `heads.py:411` | factor as prime exponents (not a flat categorical) |
| `CompressKindHead` | `heads.py:890` | mean/min/max/median/abs_min/abs_max |
| `FactoredQuantHead` | `heads.py:931` | sign ± and 7 dtype factors |
| `QuantDtypeHead` | `heads.py:909` | 🔴 superseded by the factored head |

### 🟢 `FactoredQuantHead` — semantic quantization — `heads.py:931`
Instead of one flat softmax over ~20 dtypes, picks
`sign, kind(f/i/u), bits, exp, mantissa, bias, finite, unsigned_zero` and
resolves the tuple to a catalog dtype. Factor tables:
`AG/quant_factoring.py`. Masked per factor against what's consistent with
earlier picks *and* available in hardware.

### 🟢 The ≥2-options rule (head skipping)
A head contributes log-prob / entropy / arity **only when it has ≥2 legal
options**. 0 options ⇒ head dropped; 1 option ⇒ forced, contributes nothing.
Applied identically in `sample` and `evaluate` — this symmetry is load-bearing
for ratio == 1. See `log_prob_step` (`heads.py:1176`).

### 🟢 Axis masks — `_compute_axis_masks` `heads.py:792`, `_finalise_axis_masks` `:772`
ANDs the oracle masks into i/j/compress and **drops any `i` whose whole `j` row
is masked** — otherwise an all-`-1e9` softmax degrades to uniform-over-illegal.

---

## 4. graphax — the actual math

### 🟢 Entry points
```python
jacve(fn, order, argnums, transforms=..., face_transforms=...)   # GX/core.py:322
vertex_elimination_jaxpr(..., count_ops=True, transforms=...)    # GX/core.py:2479
extract_jaxpr(...).tokenized()                                   # GX/core.py:2699  (per-VERTEX only)
inline_call_primitives(jaxpr, consts)                            # GX/core.py:202
```
> ⚠️ Always `inline_call_primitives` before numbering vertices — jacve splices
> jit/pjit bodies, so raw-trace numbering addresses a different graph.

### 🟢 Micro-actions — `GX/sparse/micro_actions.py`
```python
Diag(i, j, factor)                    # :84    i,j must be opposite partitions
Compress(axes, kind)                  # :119
Quant(dtype, scale_sign=±1)           # :343
apply_diag / apply_compress / apply_quant   # :395 / :610 / :773
verify_hardware_compat()              # :229  -> (avail_mask, compat_matrix)
quant_hardware_masks()                # :263  memoized
```
Quantization is **sign-flip half-range**: symmetric about 0, `scale_sign`
picks the arm for unsigned targets, scale = `max|logical| / dtype_max(target)`,
sign folded into `scalar_mult`. (Not a zero-point shift — that was tried and
replaced.)

### 🟢 Per-face transforms — `test_face_transforms.py` (29 tests) is the spec
```python
incr = IncrementalJaxpr(jaxpr, argnums, consts, args)   # GX/incremental.py:47
keys = list(incr.faces(v))                              # out-edge major, in-edge minor
incr.eliminate(v, face_transforms={key: (lhs, rhs, res)})   # pre / post / new
```
Faces are **independent within a vertex** — untargeted faces stay numerically
exact. Slots take a `Diag|Compress|Quant`, a callable, or `None`.

### 🟢 Incremental tokenizer — `GX/jaxpr.py:969`
```python
tk = IncrementalPathTokenizer(jaxpr, argnums, consts, args, vocab_size=512)
stream = list(tk.base_tokens())
stream += list(tk.eliminate(v))      # append-only, one block per face
tk.max_token_id()                    # size embeddings from THIS
```
**Pass `vocab_size`** — unbounded mode lets ids exceed any fixed embedding
table. Bounded mode is verified id-safe for the incremental path
(`tests/misc/test_incremental_tokenizer.py`, 23 tests).

---

## 5. PPO machinery — `AG/common/`

| Module | Status | What you get |
|---|---|---|
| `gae.py` (189) | 🟢 | `make_get_advantages(use_symlog)` → per-channel GAE; `reward_normalization_fn` = symlog |
| `popart.py` (184) | 🟢 | `PopArtStats` (debiased EMA, MAD-winsorized, σ floors) + `popart_rescale_mlp_head` (output-preserving) |
| `pareto_archive.py` (235) | 🟢 | `ParetoArchive(obj_names, obj_idx)`, `.add(vec, seq, ep)`, `.add_many(iter, ep)`, `.hypervolume()`, `.pts`. Rejects sentinel + all-zero rows |
| `examples.py` (391) | 🟢 | `get_args/get_fn/infer_argnums`, MNIST `data_gen`, MLP/Encoder/ViT/LIF-SNN |
| `eval_samples.py` (47) | 🟢 | `generate_eval_samples` — fresh data + N(0,1) weights per episode |
| `instrumentation.py` (417) | 🟡 | per-vertex features; **arg-slot bug**: merges full tuples as if argnums-only, failure swallowed → features silently zero |
| `compile_cache.py` (350) | 🟡 | cluster cache via Ray; **no-op single-process** (falls straight through) |

### PopArt usage (from the ray worker — the only place it's wired correctly)
```python
popart.update(returns_per_channel)                    # EMA of mu/sigma
agent = popart_rescale_mlp_head(agent, ..., old, new) # output-preserving
targets = (G - mu) / sigma                            # normalized critic targets
adv     = sum_k (A_k / sigma_k) * w_k                 # scaled advantages
values  = values * sigma + mu                         # de-normalize BEFORE GAE
```

---

## 6. Trainer glue — `AG/ppo.py` (3949 lines after the trim)

Reusable patterns, not necessarily the file to keep:

### 🟢 Oracle bridge into a jitted rollout — `ppo.py:~2310`
```python
def _oracle_masks_host(order, spec_hist, step_count):
    o = LiveVertexMaskOracle(...)
    for k in range(int(step_count)):
        v = int(order[k])
        o.advance(v, rules=decode_vertex_rule_specs(jaxpr, v, spec_hist[k], is_last=(k==n-1)))
    return o.masks()

jax.pure_callback(_oracle_masks_host, out_shapes, order, specs, step_count,
                  vmap_method="sequential")
```
This is how host-side legality reaches a jit+vmap rollout. `vmap_method=
"sequential"` is required (the oracle is stateful Python).

### 🟢 Reward → advantage
```python
HEAD_REWARD_INDICES = (flops, peak_memory, cosine_sim, frob_residual)  # ppo.py:135
sl   = _symlog_rewards(traj.reward)         # symlog all but cosine   # ppo.py:205
_, returns, adv = get_advantages(sl[..., HEAD_IDX], done, value, next_value, disc, lam)
adv  = per_channel_zscore(adv)
adv  = jnp.sum(adv * preference, axis=-1)
```
> ⚠️ **cosine MUST be a trained head.** It wasn't (the head labeled "acc" was
> fed frob), so 2 of 3 trained channels rewarded zeroing the computation and
> the policy collapsed to flops=0/cos=0. Also: after the z-score, once the
> batch is uniformly degenerate the opposing channel's σ→0 and it *vanishes* —
> a one-way ratchet. Guard against it.

### 🟢 Multiplicative cosine gate — `_apply_mult_gate` `ppo.py:218`
```
g(cos)    = clip((cos - tau)/(1 - tau), 0, 1)
cheapness = max(0, W - sum_c w_c * symlog(cost_c))
reward    = g(cos) * cheapness      # + shaped anti-degeneracy penalty
```
Structurally kills the collapse: cos→0 ⇒ reward→0 no matter how cheap.
Verified: honest plan +6.0 vs degenerate −2.0.

### 🟢 Collapse guard (host side)
Exclude rows with `flops>=0 | mem>=0 | cos<=1e-6 | (measured latency>=0)` from
top-N heaps and `best_global`. Log `collapse/count_{this_ep,total}`.

### 🟢 Ratio-1 invariant — the thing that keeps PPO honest
Store in the trajectory and re-feed at loss time: the vertex mask, the oracle
`pair_valid`/`compress_valid`, and the raw factored-quant log-prob. Recompute
everything else identically. Test: `sample == evaluate ⇒ ratio == 1`.

---

## 7. Landmines (all verified, all cost us a day)

1. **`_quality_metrics` degenerate fallback** — fixed to `(0.0, 1.0)`. If you
   reimplement, don't "protect downstream normalisation" by awarding a perfect
   score.
2. **Oracle `advance(v, rules=())`** — desyncs after the first approximation.
3. **`from_jaxpr(**_compat)`** silently eats measurement flags.
4. **`op_marginal` labels** — index 2 is QUANT, 3 is END. Mislabeling hid the
   END-collapse mode.
5. **`MAX_TOKENS = 4096`** but nn256 needs ~4657 → the policy sees a clipped
   observation. Counters exist; poll them.
6. **`--exec-on-gpu` env-var ordering** (`ppo.py`): the backend initializes
   before `XLA_PYTHON_CLIENT_PREALLOCATE`/`CUDA_VISIBLE_DEVICES` are set —
   export them in the shell instead.
7. **`cost_analysis()` ≈9 MB/call, `ResourceMonitor` ≈18 MB/call** — documented
   leaks; per-step × per-env × 1000 episodes is the OOM we hit.
8. **Compile cache is a no-op without Ray** — every env-step recompiles.
9. **8 vs 10 reward channels** — `common/reward_scaling.py` still declares 10
   (`xla_peak_memory`, `bkstep_acc`); `env.py` has 8. Don't mix.
10. **`ppo_ray*.py` cannot run** — 5 independent hard failures. Read it for
    designs; don't try to execute it.

---

## 8. Minimal build order (a lean trainer, ~800–1200 lines)

1. **env + measurement** — reuse `VertexEliminationEnv` unchanged (§1).
2. **encoder + pointer head** — `Encoder` + `PointerVertexPolicy` (`ppo.py:454`),
   masked by `vertex_avail_at_step`.
3. **micro policy** — `MicroActionPolicy` unchanged (§3), fed by
   `_axis_features_from_state` (`ppo.py:605`) + the oracle bridge (§6).
4. **rollout** — `lax.scan` over `num_valid` steps; store the masks you sampled
   under (ratio-1).
5. **reward → advantage** — symlog, 4-head GAE, z-score, preference weights;
   or `--reward-mode mult` for the gated scalar.
6. **loss** — clipped PPO + per-channel value MSE + arity-normalized entropy.
7. **logging** — the P2 pack (§6, already written in `ppo.py`'s `host_log`).

Everything in steps 1–3 is reusable **as-is**; the new code is really steps
4–7, which is where the current file's remaining bulk is (argparse ~400 lines,
`Agent` ~500, rollout+loss ~900, logging ~400).

---

## 9. Local test loop (works on macOS, no cluster)

```bash
uv pip install --python graphax/.venv/bin/python \
    "jax==0.10.0" "jaxlib==0.10.0" distrax optax tqdm wandb
# jax_memory_monitor is a cluster-built C++ pkg — a no-op stub lives at
# graphax/.venv/lib/python3.12/site-packages/jax_memory_monitor.py

PYTHONPATH="graphax/src:alphagrad/src" ALPHAGRAD_DISABLE_RESOURCE_MONITOR=1 \
  graphax/.venv/bin/python -m pytest alphagrad/tests/<file> -q
```
Green today (44): `quality_metrics_test`, `rule_replay_test`,
`live_vertex_mask_test`, `head_skipping_test`, `factored_quant_head_test`,
`face_mask_oracle_test`, `face_slot_chooser_test`.
Known pre-existing failure: `micro_action_mask_test::test_coupled_pair_may_only_subdivide_never_coarsen`.
