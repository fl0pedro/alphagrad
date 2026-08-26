# Face read-point trace: does the approx head read palimpsa's recurrent state?

Scope: the live path only — `ALPHAGRAD_POLICY=palimpsa`, `--incremental-encode`,
`--live-faces`, `--unified-face-head`, `ALPHAGRAD_INCREMENTAL_TOKENS=1`,
`ALPHAGRAD_EXTEND_CHUNK=128`, `--seed-vertices --measure-grad`.
Read at `hostperf-caches` **fec993d** (the brief said `0e601ae`; that is not what
is checked out — every line number below is fec993d's).

## Headline

**The recurrence is NOT bypassed.** The face head's input is the mean of
**post-recurrence palimpsa hidden rows** — the residual-stream output `x` after all
encoder layers, conditioned on a side carry branched from the episode carry — not
`embedding(token_ids)`. The bypass hypothesis is refuted at
`ppo.py:2033-2059` → `ppo.py:2572-2573` → `ppo.py:2717` → `ppo.py:2745-2747` →
`unified_face_policy.py:242`.

**The real defect is the POOLING SPAN, not the read point's depth.** Face `f`'s
chunk is `[approx-echo(f-1) || header+contraction(f)]`, and the head pools **the
whole chunk**. So the vector fed to the 94-output head is a token-count-weighted
blend of the *previous* face's approximation with *this* face's contraction. The
carry underneath it is causally correct; the read window is not.

---

## Verdicts

| Claim | Verdict | Evidence |
|---|---|---|
| (A) base stream includes reserved Jacobian output vars | **PARTIAL / mostly REFUTED** | `graphax/jaxpr.py:1242-1261` |
| (B) append-only: per vertex, face accumulation then its approx echo | **CONFIRMED** | `graphax/jaxpr.py:1447-1497` |
| (C) palimpsa reads incrementally, state advances per appended chunk incl. per-face | **CONFIRMED** | `ppo.py:2717`, `ppo.py:2754`, `ppo.py:2565-2566` |
| (D) the approx head's input is palimpsa's own updated latent for that face | **CONFIRMED as to depth, PARTIAL as to span** | `ppo.py:2572-2573`, `unified_face_policy.py:227-242` |
| (E) three distinct chunk types, never conflated | **REFUTED** — types 2 and 3 are merged into one pooled unit, with a one-face offset | `live_faces.py:463-465`, `live_faces.py:682-696` |

---

## (A) The base stream — PARTIAL

`base_tokens()` emits `inputs <var> <shape> …` followed by the base equations
(primal forward + elemental edge partials) and **nothing else**. The reserved
Jacobian-output name list is explicitly *dropped*:

```
graphax/src/graphax/jaxpr.py:1242-1246
    def base_tokens(self):
        # inputs (+ shapes) then the base equations (primal forward + elemental
        # edge partials). No output/jac list: the Jacobian outputs ARE the
        # input->output edges produced by the path blocks; a reserved-name list
        # here was never referenced, so it is dropped.
```

The `outputs value … jac <ov>&<in>=<reserved block>` section (`_emit_outputs`,
`jaxpr.py:1682-1700`) exists **only** on the offline full-stream path
`capture_stream` (`jaxpr.py:1702-1706`). The live incremental path never calls it.
This matches the recorded finding that `capture_stream` states outputs and
`base_tokens` does not.

What *is* true: under `--seed-vertices --measure-grad` the tangent seed and the
`<ones/N, ·>` adjoint contraction are folded into the traced graph as **ordinary
eliminable vertices** before tokenization (`ppo.py:5017-5040`,
`grad_target_setup`), so those nodes appear in `ij.base_eqns()` and therefore in
the very first tokenization. The graph the policy acts on does carry the seed;
what it does **not** carry is a reserved-name declaration of the Jacobian output
blocks. So the head never sees "here is where the answer must land" stated up
front — it only ever sees it implied by the edges the path blocks create.

Base stream is its own separate first unit: `env.base_observation()` is encoded by
one dedicated `encode_extend` call in `carry_stream.init_carry`
(`common/carry_stream.py:91-165`, the call at `:134-137`), producing the base
carry + base slot memory, *before* any step delta. It is never merged into step 0's
delta. (Answers (E)(iii).)

## (B) Emission order — CONFIRMED

Per elimination step, `_emit_step_paths` walks the step's faces in order and, for
each, emits header + contraction, records the split, then the approximation echo:

```
graphax/src/graphax/jaxpr.py:1489-1497
    def _emit_step_paths(self, step_i, eqns):
        for fr in self.ij.step_faces(step_i):
            start = len(toks)
            split = self._emit_face(fr, eqns, toks)
            segs.append((start, split, len(toks)))
```

`_emit_face` (`jaxpr.py:1447-1487`): `_emit_face_header` (`path <central> & <pred>
& <succ>`, `:1438-1445`), then the contraction equations, `split = len(out)`
(`:1477`), then the `approx <pre> ^ <post> ^ <new>` head and three slot blocks. A
SKIPPED face is *emitted*, not silent: `path … {}` then a lone `approx SKIP {}`,
and it keeps its own segment index (`jaxpr.py:1521-1537`) — index `f` is face `f`,
consumers must not compensate for skips.

`last_face_segments()` returns `(start, split, end)` per face and its docstring
states the intended autoregressive contract verbatim:

```
graphax/src/graphax/jaxpr.py:1503-1507
    ``[start:split]`` is the face header plus its contraction/join
    equations; ``[split:end]`` is the face's APPROXIMATION part … it
    reads ``[start:split]`` of face f, decides, and then reads
    ``[split:end]`` of face f followed by ``[start:split]`` of face f+1.
```

So graphax exposes the type-2 / type-3 boundary cleanly. The conflation is
downstream.

## (C) Incremental reading and the per-face advance — CONFIRMED

`_extend_sequential` (`ppo.py:2010-2213`) is the recurrence. Per token it:
embeds (`:2033 x = self.embedding(tok)`), runs the palimpsa mixer per layer
updating the fast weights `M_l`, `I_l` (`:2049-2051`), applies `mu = M_l / I_l`
and the output/MLP residuals (`:2052-2056`), freezes the carry on invalid steps
(`:2057-2058`), and **emits the per-token hidden row**:

```
ppo.py:2059-2060
    row = jnp.where(ok, x, jnp.zeros_like(x))
    return (jnp.stack(new_M), jnp.stack(new_I), cumhist2, nvalid2), row
```

So there **is** a per-token output sequence (`rows`, `(window, E)`), in addition to
the carry `EncCarry(M, I, cumhist, nvalid, pos)`. `M`/`I` are `(L, H, d, d)` fast
weights — not directly consumable as an `E`-wide head input; `rows` are.

Two distinct advances:

* **Per elimination step (main stream).** `carry_stream.advance`
  (`common/carry_stream.py:231-291`) consumes the whole step delta once, scatters
  its rows into every *participating* vertex slot. Called once per step at
  `ppo.py:6755` / `:6763`. This delta is the **true** emission — contractions
  *and* approximations for every face.
* **Per face (side carry).** Inside `_face_loop`, the carry is threaded through
  every face:
  ```
  ppo.py:2717   carry, summ = self._face_encode(carry, tk_f, eq_f, ct_eff)
  ppo.py:2754   out = (f + 1, carry, logp + lp, ent + e, …)
  ```
  `_face_encode` (`ppo.py:2551-2578`) calls `encode_extend(..., start=0)` on the
  standalone chunk buffer (`ppo.py:2565-2566`) and skips the scan entirely on an
  empty chunk (`lax.cond` at `:2578`), so an empty chunk leaves the carry exactly
  where it was.

The side carry is **branched and discarded**: `_face_loop`'s `while_loop` result
unpacks it as `_c` and never returns it (`ppo.py:2761-2767`). That is correct — the
real emission is re-consumed once by `advance` — but it means the per-face state
lives only for the duration of the face loop.

So: yes, the latent is already face-updated at the moment each face decides, and
yes a genuine per-face recurrent state exists today (it is `carry` at `ppo.py:2717`
and the last row of `rows` inside `_face_encode`). The fast weights `M`/`I` **are**
in scope inside `_face_loop` — they are fields of the threaded `carry`.

## (D) What the head actually reads — CONFIRMED as to depth

`_face_encode`'s summary is a parameter-free single-segment mean over the
recurrence's own output rows:

```
ppo.py:2572-2573
    return c2, _vmem.scatter_mean(
        rows, jnp.zeros((rows.shape[0],), jnp.int32), valid, 1)[0]
```

That `summ` is passed straight through as the head input:

```
ppo.py:2745-2748
    sk, row, lp, e, _ar, _sp, _od = pol.sample_face(
        features, factor_tables, jrand.fold_in(key, f),
        f, f_pair[f], f_comp[f], f_valid[f], face_context=summ,
        op_legality_override=op_legality_override)
```

```
unified_face_policy.py:253-254 / 227-242
    ctx_f = self._repr(face_context)
    …
    def _repr(self, face_latent):
        …
        return face_latent          # identity; zero vector when absent
```

`UnifiedFacePolicy` reads `face_latent` and nothing else; `vertex_context` is
accepted and explicitly ignored (`unified_face_policy.py:42-45`, `:330`), and the
endpoint contexts were deleted (`ppo.py:2620-2624`). Head width is `E` (32),
widened to 3E/5E only under `--face-endpoint-read` / `--face-edge-mem`
(`unified_face_policy.py:101-104`; concatenation done by the caller at
`ppo.py:2718-2744`).

Contrast — the pointer/VE head (question 5). It reads the same kind of thing by the
same primitive, keyed differently: `init_carry` scatters the base stream's
**post-recurrence rows** into their owning vertex slots
(`carry_stream.init_carry:134-160`, owners from
`IncrementalPathTokenizer.last_owner_ids`), `advance` scatters each delta's rows
into participating slots, and `SetPointerVertexPolicy._score` runs its SetBlocks
over that pooled slot memory (`set_pointer.py:138-157`). So *both* heads read
pooled palimpsa hidden states. The difference is **accumulation and attribution**
(vertex slots persist across the episode and are dominated by the base stream)
versus **transience** (the face latent is a chunk-local mean of ~77 tokens that
exists for one decision).

One more thing the face head does not see: `sample_face` is called **without**
`face_sizes_f` (`ppo.py:2745-2748`), so `_face_feats_1` falls back to the shared
vertex-level `features` (`unified_face_policy.py:160-161`). The per-face axis
extents — the ablation's largest main effect — are still not wired.

## (E) The three chunk types — REFUTED (types 2 and 3 are merged, with a one-face offset)

### (i) The exact span

`LiveFaceStream.chunk_ex` builds face `f`'s chunk from
`tk.last_face_segments()` (`live_faces.py:549`) as:

```
live_faces.py:682-696
    chunk: list[int] = []
    if gi > 0:
        # face f-1's approximation: `approx <TYPE> <args>` + the equations
        # it produced. Empty when that face ran exact.
        _s, split, end = segs[gi - 1]
        chunk += toks[split:end]
        …
    head = len(chunk)
    start, split, _e = segs[gi]
    chunk += toks[start:split]
```

i.e. **chunk(f) = segs[f-1][split:end] ++ segs[f][start:split]**, stated in the
docstring too:

```
live_faces.py:463-465
    ``tokens`` is the ``=>`` handoff before face ``f``'s decision: face
    ``f-1``'s approximation equations followed by face ``f``'s contraction
    (face ``0`` gets the contraction alone).
```

Truncation keeps the **tail** and adjusts `head` accordingly
(`live_faces.py:702-708`). The chunks are checked to tile the emission exactly
(`live_faces.py:621-647`), and face `f`'s chunk is pinned to open on face `f`'s own
`path` header (`live_faces.py:667-680`).

### (ii) Where face `i`'s approximation lives

In face **i+1**'s chunk. The last face's approximation echo is in **no** chunk at
all — it is the documented slack (`live_faces.py:629-631`, `ppo.py:2777-2780`).

### (iii) Base as its own unit

Yes — see (A) above. Separate `encode_extend`, separate memory, never merged into
step 0's delta.

### (iv) Type tags

There is a **lexical** marker but no type channel. `inputs` opens the base,
`path` opens a type-2 unit (`jaxpr.py:1440`), `approx` opens a type-3 unit
(`jaxpr.py:1479` / `:1483`), and `SKIP` marks a skipped face
(`jaxpr.py:1527-1534`). These are ordinary vocabulary tokens; the recurrence sees
them only through `self.embedding(tok)`. The only side channel into the recurrence
is `eqn_ids`, and those feed **only** the causal relational forget-gate features
(`ppo.py:2020-2031`) — they are stream-global segment counters, explicitly not
type or vertex identifiers (`carry_stream.py:139-152`). The head itself
(`_repr`) receives **no** type tag whatsoever: it cannot tell how much of its
pooled input came from the previous face's approximation.

### (v) Write path vs read path — THEY DISAGREE. Explicitly.

The edge-mem write path attributes tokens by `last_face_segments`' **own** tiling
(face `f`'s contraction plus face `f`'s OWN echo), reconstructed by subtracting the
approx-echo prefix lengths:

```
ppo.py:1331-1340 (_edge_write_ids docstring)
    The write attribution is `last_face_segments`'s OWN tiling -- face f's
    span is its contraction plus its OWN approximation echo … the chunk the
    head read for face f is [approx(f-1) || contraction(f)], so with C = cumsum
    (counts) and a_f = the chunk's approx-echo prefix length,

        span_f = [C_{f-1} + a_f,  C_f + a_{f+1})            (f < last)
```
implemented at `ppo.py:1352-1362` (`a_next`, `ends = C + a_next`).

The head's read path pools `[C_{f-1}, C_f)` — the raw chunk boundaries, no `a`
correction — at sampling (`ppo.py:2572-2573`, one segment per chunk) and in the
loss (`_face_replay`, `ppo.py:2832-2841`: `_ends = jnp.cumsum(f_cnt)`,
`fid = jnp.searchsorted(_ends, pos, side="right")`).

**So writes and reads are attributed to different faces.** The edge memory writes
`approx(f)` into face `f`'s slot; the head reads `approx(f-1)` as part of face
`f`'s context. The two conventions are offset by exactly the approx-echo lengths.
Sampling and replay agree with *each other* (which is what preserves ratio-1), so
this is not a PPO-correctness bug — it is a semantics bug in what the head is told
a face is.

---

## Plain statement: what the face head sees today

At the moment face `f` is decided, palimpsa's side carry has causally consumed the
base stream, every prior step's full delta, every earlier face of this step
(contraction *and* approximation), and finally `[approx-echo(f-1) ||
header+contraction(f)]`. That carry is exactly the causally-correct state the owner
describes. But the head is not handed the carry. It is handed a **length-weighted
arithmetic mean of the post-recurrence hidden rows of that last chunk only** — a
single 32-wide vector mixing the previous face's approximation echo with this
face's contraction, with no marker separating them and no per-face axis extents
alongside. It then produces 94 logits from that vector alone. Everything before the
chunk reaches the head only through the recurrent state that *conditions* those
rows, never as content the pooling can weight.

---

## Fix sketch

### The causally correct read

Palimpsa's state **after** face `f`'s accumulation (type 2) and **before** its
approximation (type 3, which does not exist at decision time), with the type-3
chunk consumed immediately after so face `f+1` starts from a state that includes
it. **The current implementation already produces exactly that state** — no
restructuring of the stream, the tokenizer, or the loop is needed. `chunk(f)` is
precisely `[type-3(f-1) || type-2(f)]`, which is `last_face_segments`' documented
autoregressive order (`jaxpr.py:1503-1507`). Only the *readout* is wrong.

### Minimal change (recommended): pool face `f`'s own span

`head` — the approx-echo prefix length — is **already computed and already
returned** (`live_faces.py:693`, `:714`), and already plumbed to device under
`--face-edge-mem` (`ei_f[3]`, consumed by `_edge_write_ids` at `ppo.py:6751`,
`:7176`, `:7249`).

1. **`live_faces.chunk`** (`live_faces.py:435-441`) — return `head` in the
   back-compat tuple (widen the `[:5]` slice to `[:6]`, or return `chunk_ex`'s
   element 7 unconditionally). `face_driver.make_face_callbacks`
   (`common/face_driver.py:221`, `:260`, `:294`) already unpacks nine values from
   `chunk_ex`; make the non-edge-mem `pure_callback` return one more int32 scalar.
2. **`Agent._face_encode`** (`ppo.py:2551-2578`) — add `pool_from=0` and mask the
   scatter, leaving the carry advance untouched:
   ```python
   rows_ok = valid & (jnp.arange(rows.shape[0], dtype=jnp.int32) >= pool_from)
   return c2, _vmem.scatter_mean(rows, jnp.zeros(...), rows_ok, 1)[0]
   ```
   The carry must still consume the whole chunk — that is what keeps the
   recurrence causal.
3. **`Agent._face_loop`** (`ppo.py:2695-2717`) — unpack `head_f` from
   `face_chunk_fn`, pass it as `pool_from`, and store it in the per-face wire
   (a `(F,)` int32 alongside `cnts` at `ppo.py:2751`) so the loss can mirror it.
4. **`Agent._face_replay`** (`ppo.py:2832-2850`) — mirror gate for gate or the
   ratio is not 1 at epoch 0 (`ppo.py:2547-2549` is explicit about this). Keep
   `fid = searchsorted(cumsum(f_cnt), pos, "right")` and add a start gate:
   ```python
   starts = jnp.concatenate([jnp.zeros((1,), jnp.int32), _ends[:-1]]) + a
   live = live * (pos >= starts[fid])
   ```
   one extra gather and compare inside the existing fold; `_FOLD_DELTA`'s
   `(sums, counts)` accumulation is unaffected.
5. **`TrainState`** — store `face_heads` unconditionally next to `face_counts`
   (already stored under `--face-edge-mem`).

**Cost:** effectively zero. No extra encode, no extra scan, no new parameters, no
width change — the pooled rows are already materialized; only the mask changes.

**Shape/jit hazards:**
* `head` is a traced int32 from a `pure_callback`, so the mask must be
  `jnp.arange(W) >= head`, never a Python slice. No dynamic shapes.
* Empty pool: `scatter_mean` divides by `max(count, 1)`, giving a zero vector,
  which is exactly `_repr`'s documented absent-latent convention
  (`unified_face_policy.py:236-241`). A skipped face still emits its `path`
  header, so `[start:split)` is never truncated to zero in practice.
* Truncation already adjusts `head` (`live_faces.py:707`) — do not re-derive it
  device-side from counts.
* Steps 2 and 4 **must land in one commit**; a sampling-only change breaks ratio-1
  silently.

### Stronger variant, also nearly free: read the state, not the window

The literal form of claim (D) — feed palimpsa's *state* rather than a pooled window
— costs nothing extra either, because `rows[cnt-1]` **is** the recurrence's output
conditioned on everything consumed up to and including face `f`'s contraction. That
single row is a strictly better claim to "the face's own recurrent state" than any
mean, and it needs no new tokens and no extra scan step — just index instead of
pool inside `_face_encode._run`, with the same `_face_replay` mirror (gather at
`_ends[f] - 1` instead of a segment mean).

The fast weights `M`/`I` themselves are `(L, H, d, d)` and cannot be fed directly;
reading them out would need a synthetic query token appended after face `f`'s
contraction (one extra recurrence step per face). That is the only variant that
requires touching the token stream, and it is not necessary — the last real token's
row already carries the readout.

Suggested shape: one flag with three settings — `chunk-mean` (today),
`own-span-mean` (fix), `last-row` (state read) — so the three are an A/B/C on the
same run rather than a rewrite.

### Independently: the write/read disagreement

Once the read pools `[C_{f-1} + a_f, C_f)` and the write spans
`[C_{f-1} + a_f, C_f + a_{f+1})`, the two conventions share a start and differ only
by the trailing echo — which is correct, because the echo does not exist when the
face is read but does belong to the face when written. Today they share neither
endpoint. Fixing the read fixes the disagreement.

---

## Were the offline ablations a faithful surrogate?

**Yes for the mechanism and the read point; no for the trained system — and the
dossier already says so.** `docs/FACE_LATENT_INFO_LOSS.md` §1 corrects the
"raw embeddings" sketch in as many words:

> Two corrections to the working sketch: (a) the chunk rows ARE post-recurrence —
> full 3-layer palimpsa outputs, carry-dependent (`_extend_sequential`) — the face
> path does not read raw embeddings; (b) the vertex probe input is post-Set-pointer,
> not a raw slot row.

§2 H-C and §9 pooled **frozen random-init palimpsa rows** — the real recurrence with
untrained weights — not `embedding(token_ids)`. So the read point, the pooling
primitive, the chunk span and the dimensionality all match production exactly. The
decodability numbers (chunk-mean ndim 0.82-0.89, size R² 0.45-0.55) therefore
describe the true read point **at initialization**.

Three caveats on how far they carry:

1. **Untrained ≠ trained.** §2 H-D(2) measures the online decay separately (lhs
   ndim peaks 0.64 at ~ep17 then decays to the 0.47 majority baseline by ~ep60,
   probe loss frozen at 3.228). That is the number that describes the live system;
   the offline table describes its starting point.
2. **The offline note "raw-embedding mean is comparable too"** (§2 H-C) says the
   information at this read point is token-content-level — i.e. at random init the
   recurrence contributes little *above* the embeddings. That is a statement about
   an untrained recurrence, not evidence that the head reads embeddings.
3. **The span conflation was never ablated.** Every offline arm pooled the
   production chunk, echo prefix included. There is no measurement of pooling face
   `f`'s **own** span, nor of the last-row state read. Both are missing arms, and
   both are cheap — which is the strongest argument for running them before
   concluding anything further about face-latent capacity.
