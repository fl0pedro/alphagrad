# Unbiased Pareto re-measurement and the measurement protocol it required

**Written 2026-08-27** from the artifacts under `/Users/assmuth/dsnn/run_analysis/landscape/`
produced by jobs **62391 / 62394** (probes), **62414 `face-attribution`** (pgi15-gpu18,
phases QA/QB/QC) and the earlier landscape phases PA–PH, plus the live logs of
**62411 `r1-trim`** (pgi15-gpu15), **62412 `r2-trimplus`** (pgi15-gpu16) and
**62413 `r3-credit`** (pgi15-gpu17). Companion to
[`QUALITY_COLLAPSE_INVESTIGATION.md`](QUALITY_COLLAPSE_INVESTIGATION.md) §14, which
this file continues.

Every claim below carries one of four labels:

| label | meaning |
|---|---|
| **CONFIRMED** | measured in a named artifact in this repo/tree, number quoted from that artifact |
| **PENDING** | the experiment that decides it is registered and either running or queued; no verdict exists yet |
| **WORKED-AROUND** | the failure is understood and neutralised by configuration; the root cause is still in the tree |
| **NOT REPRODUCED** | asserted elsewhere; the artifacts read for this document do not contain it, and the number they do contain is given |

---

## 1. PROTOCOL — what a latency number has to satisfy before it means anything

Every measurement in this document was produced by
`.ag_pin_landscape/src/alphagrad/approx/tools/landscape_map.py` under this
protocol. Each element is here because something went wrong without it.

### 1.1 Paired candidate-vs-its-own-exact-reference, reported as a ratio

Each candidate plan is measured **back to back with an exact-reverse reference
built from the same elimination order in the same process on the same GPU**, and
only the **ratio** is reported. Absolute nanoseconds are recorded but are never
the claim.

*Why.* Project memory, `feedback_paired_measurement_drift`: an earlier "17.5 %
beats reverse" result was unpaired-comparison drift — the same plan measured
back-to-back against its own reference came out at ratio 1.00. GPU state on this
cluster moves ~18–20 % over a session. §5.2 below shows a residual ~1.3 %
*systematic* per-actor offset that survives even within one job; pairing removes
it exactly, because both halves of the ratio are taken on the same device in the
same second.

**CONFIRMED.** The instrument's own null: the identity plan measured against
itself gives **1.0007 ± 0.0008 (n=6), range 1.0001–1.0020**
(`summary_c_winners_warm.md`). Nothing inside that band is evidence of anything.

### 1.2 Warm — N discarded rounds before any recorded round

`--warmup-trials 2` (phases PD/PE/PF/PG/PH) or `3` (phase PC), plus the tool's
whole-plan warmup pass that pays the XLA compile, plus the in-loop
`ALPHAGRAD_MEASURE_WARMUP=1`. The `warmup_src` column of every `rows_*.csv`
records which of these fired (`script:2+env:1/cfg:1`).

*Why.* A cold process mis-times its first measurement. Measured directly
(§5.1): the identity plan's **first** cold reading is **−7.2 %** below its
settled value and is back inside 2 % by reading #2.

### 1.3 `--latency-inner-reps 50`

*Why.* Two reasons, both measured. (a) Project memory,
`feedback_drop_count_pass`: 50 is the standing project rule. (b) At
`--latency-inner-reps 5` the *whole scale* moves: phase PH (`in5`) reads the
identity plan at **169 038 ns** where phase PE (`in50`) reads **140 090 ns**,
and the archived winner's ratio moves **0.5280 → 0.6012** — a 7-point shift in
the headline number purely from the inner-rep count. §7(a) shows the same
protocol mismatch, between the quality gate's floor and the plan it floors,
paid destruction a −13 % bonus in all twelve campaign runs.

### 1.4 Pinned code: alphagrad `c2b8104`, graphax `4ea0bf8`

Both libraries are imported from **archived snapshots** (`.ag_pin_landscape/src`,
`.gx_pin_landscape/src`), not from the live working trees, and the launcher
**aborts (exit 70)** if `graphax.__file__` / `alphagrad.__file__` do not resolve
inside the pins.

*Why.* At the time of the run the live `~/dsnn/graphax` tree had four modified
uncommitted files (`core.py`, `incremental.py`, `jaxpr.py`,
`sparse/dtype_compute.py`) touching the quantization dtype policy, and a DIAG
change (`39d8bd1`) was landing in alphagrad. PYTHONPATH does **not** reliably beat
a PEP 660 editable install — an editable install registers a meta-path finder
that runs before `sys.path` is consulted — so the pin is *verified*, not assumed.
The instrument itself (`landscape_map.py`) is deliberately the working version
laid over the pinned library, and its `sha256` prefix is stamped into the
`config_note` of every row (`tool=cb7c7667`, `tool=3cb4ceb2`).

### 1.5 `ALPHAGRAD_NEW_SLOT_JOIN=0`

Set explicitly in all five launchers. See §7(c): the default (`1`) emits a face
form no graphax on disk accepts. **WORKED-AROUND.**

### 1.6 `GRAPHAX_QUANT_PULLDOWN=1` for the archive phases

The archived Pareto points were *produced* under pulldown, so they are
*re-measured* under pulldown. Comparing them under pullup would be comparing two
compute stacks and calling the difference a result. Phases PE (pd1) and PG (pd0)
run the identical hand-specified ladder under both settings so the effect of the
flag itself is measurable — see §6.3.

### 1.7 One job holds the whole node

`--gres=gpu:4` for a job that only ever uses `CUDA_VISIBLE_DEVICES=0`.
*Why:* `peak_memory` is a **device-wide** counter (`peak_bytes_in_use` delta); a
co-resident process inflates it, and CV was observed going 0.0000 % → 49.7 %
under a noisy neighbour. Holding the node is what makes the memory column mean
anything.

---

## 2. THE WIN IS REAL — **CONFIRMED**

### 2.1 The archived winners reproduce

Phase **PC** (`summary_c_winners_warm.md`), warm, paired, `--reps 6`,
`--warmup-trials 3`, `--latency-inner-reps 50`, pd1, gpu0, ag=c2b8104 gx=4ea0bf8
newslotjoin=0 tool=cb7c7667:

| plan | approx faces | rules applied / skipped | latency ratio (mean ± sd, n=6) | mem ratio | quality | latency ns |
|---|---:|---:|---|---|---|---:|
| `archive:v63` | 1 | 0 / 0 | **0.5222 ± 0.0122** | 1.0000 ± 0.0000 | 0.8981 | 73 318 |
| `archive:v64b` | 1 | 0 / 0 | **0.5280 ± 0.0137** | 1.0000 ± 0.0000 | 0.8981 | 74 120 |
| `archive:v60` | 13 | 16 / 6 | **0.5291 ± 0.0040** | 1.0000 ± 0.0000 | 0.8981 | 74 284 |
| `archive:v57` | 16 | 13 / 3 | **0.6461 ± 0.0007** | 1.0000 ± 0.0000 | 0.9198 | 90 691 |
| `identity` (null) | 0 | 0 / 0 | 1.0007 ± 0.0008 | 1.0000 ± 0.0000 | 0.9260 | 140 470 |

Range **0.522–0.646** against a drift floor of **1.0007 ± 0.0008**. The winners
sit **35–60 sd** outside the null. Recovered from the runs' own
`pareto_front.json` dumps (95 vertices, 0 per-vertex rules) with the recorded
objectives matching: v64b `latency −69 801.8, peak_memory −55 380 992,
cosine_sim 0.86058`; v63 `−72 215.3 / −55 380 992 / 0.86058`; v60
`−76 778.8 / −55 605 376 / 0.86055`; v57 `−95 766.8 / −55 057 280 / 0.88242`.

Cross-checks, all in `summary_COMBINED.md`:

* four independent GPUs, `--reps 5` each: v64b at **0.5174 / 0.5286 / 0.5307 /
  0.5227** (phases PD gpu0–3);
* a second, independent replication path — phase PF re-measured 8 points from
  each of eight archives at `--reps 3` and found the same clusters (§3.1);
* at `--latency-inner-reps 5` the win is smaller but still far outside the null
  (0.5880–0.6978, phase PH).

### 2.2 The cold-measurement hypothesis is **REFUTED**

An earlier session hypothesised that the archived winners were artifacts of a
cold first measurement. Phases **PA** and **PB** were run to decide it, and they
decide it against.

`summary_a_cold_ident.md` — identity, fresh process, 12 consecutive readings:

```
ns  130220 139550 140289 140474 140246 140456 140565 140510 140454 140364 140381 140374
settled (median of last half) = 140417 ns
FIRST measurement error = -7.3 % ; rounds until within 2 % = 1
```

`summary_COMBINED.md` extends the same plan to 24 readings: first 130 220 ns
against a settled 140 330 ns, **−7.2 %**, back inside 2 % by reading #1.

`summary_b_cold_v64b.md` — the winner itself, 12 cold readings, settled
71 269 ns, **first-measurement error −0.3 %, 0 rounds to settle**.
`quant@5`: **+0.1 %, 0 rounds.**

Three things follow, and together they close the hypothesis:

1. The cold effect is **−7.2 %, on measurement #1 only**, and it is a property of
   the *identity* plan, not of every plan.
2. Its **sign is wrong for the hypothesis.** A cold reading makes the *reference*
   look 7 % **faster**, which inflates a candidate's ratio. Cold measurement can
   only ever *hide* a winner, never invent one.
3. It is **7 %, not 48 %.** Nothing of that size can manufacture a 0.52 ratio.

**Verdict: the earlier "cold artifact" hypothesis is REFUTED. The 1.9× latency
win is real, and the true size of the cold effect is recorded above.**

---

## 3. THE WIN IS ONE FACE — **CONFIRMED**

### 3.1 Five independent runs converge on the identical one-face plan

From `summary_f_arch_pd1_in50.md` (phase PF, paired, warm, `--reps 3`, pd1,
in50), reading only the plans that carry **one** approximated face and **zero**
applied rules — i.e. plans whose entire content is a single SKIP:

| plan | ratio | quality | source run |
|---|---|---|---|
| `arch:v64b:1` | 0.5784 ± 0.0003 | 0.9258 | v64b |
| `arch:v66a:1` | 0.5784 ± 0.0006 | 0.9258 | v66a |
| `arch:v65:6`  | 0.5795 ± 0.0026 | 0.9258 | v65 |
| `arch:v63:1`  | 0.5796 ± 0.0027 | 0.9258 | v63 |
| `arch:v66c:1` | 0.5796 ± 0.0010 | 0.9258 | v66c |
| `arch:v66c:16`| 0.5806 ± 0.0009 | 0.9258 | v66c |

Five separate training runs, on separate GPUs, weeks apart, each independently
archived a Pareto point whose measured ratio agrees to **±0.2 %** and whose
quality agrees to **four decimals** — against an identity quality of 0.9260,
i.e. the quality cost is **0.0002**. A second cluster does the same at a
different face: `arch:v63:4` 0.5409, `arch:v64b:3` 0.5409, `arch:v65:1` 0.5449,
`arch:v66a:4` 0.5512, all at quality **0.8981** (four runs).

### 3.2 Naming the faces

Phase **QB** (`--singleton-skip-sweep --sweep-stride 1`, job 62414, still
running at the time of writing) skips **each live face alone** and measures it
paired. The face inventory (`face_inventory_qa_attrib.json`, 118 live faces on
the exact reverse prefix) resolves each to its elimination step `k`, vertex,
primitive and operand shapes. Warm paired pairs from `rows_qb_sweep.csv`:

| k / f | vertex | primitive | operand shapes | candidate ns | its paired ref ns | ratio | quality |
|---|---|---|---|---:|---:|---|---|
| 13 / 1 | v82 | `dot_general` | (32,128) × (128,1024) → (32,1024) | 77 136 | 140 194 | **0.550** | 0.8981 |
| 19 / 0 | v76 | `add` | (32,128) + **(1,128)** → (32,128) | 74 325 | 140 240 | **0.530** | 0.9071 |
| 21 / 1 | v74 | `dot_general` | (32,128) × (128,128) → (32,128) | 74 412 | 139 984 | **0.532** | 0.9257 |
| 22 / 0 | v73 | `add` | (32,128) + **(1,128)** → (32,128) | 78 943 | 140 112 | **0.563** | 0.9257 |
| 24 / 0 | v71 | `mul` | (32,128) × **(1,128)** → (32,128) | 81 084 | 140 233 | **0.578** | 0.9258 |
| **25 / 0** | **v70** | **`broadcast_in_dim`** | **(128) → (1,128)** | **138 590** | **140 202** | **0.988** | **0.9260** |

The last row is **the dud**, and it is the point of the table. `k25/f0` is the
adjacent face — the broadcast that *feeds* `k24`'s `(1,128)` operand — and
skipping it buys **nothing** (0.988, inside/adjacent to the null band) and costs
nothing. The archived one-face point `arch:v57:2` sits at **0.9952 ± 0.0010**
with quality 0.9260, i.e. v57 archived a one-face plan that is a pure no-op.
*Which* face is skipped decides everything.

> Attribution caveat: the `k`↔plan mapping above is established by matching the
> QB singleton sweep's (ratio, quality) signatures to the archived plans'
> signatures, which is tight — quality is bit-deterministic (§5.3) and the
> clusters are separated by ≫ the noise. Byte-level wire-to-face resolution for
> each archived point is what phase QA's `plans_qa_attrib.json` holds; job 62414
> was still writing when this was drafted. The claim "`arch:v57:2` is `k25/f0`"
> specifically is **PENDING** on that file; the claim "`arch:v57:2` is a one-face
> plan that buys nothing" is CONFIRMED from the ratio.

### 3.3 The whole win lives in elimination steps 13–24

The QB sweep over all 118 live faces finds **14** single faces that beat the
drift floor at quality ≥ 0.92, and their ratios fall into two regimes:

* **steps 13–24** (`k13/f1, k19/f0, k21/f1, k22/f0, k24/f0`): ratio **0.53–0.58**;
* **steps 38–78** (`k38/f1, k39/f1, k43/f0, k44/f0, k59/f0, k61/f1, k62/f0,
  k64/f0, k78/f0`): ratio **0.80–0.90** — real, but half the size;
* everything else: inside the null, or destructive (`k0–k3, k7, k8` drop quality
  to 0.0000 at ~62 000 ns, which is the SKIP cliff, §6.4).

**The single best face in the entire fixed-reverse SKIP space is `k19/f0`
(v76 `add`) at ratio 0.530, quality 0.9071**, with `k21/f1` at 0.532 / 0.9257
essentially tied and *free*. Nothing in the archives beats it.

### 3.4 More applied rules do not buy more speed, and do cost quality

Over the 64 distinct archived points measured in phase PF:

```
pearson(applied rules, latency ratio)  = -0.12      (no relationship)
pearson(applied rules, quality)        = -0.44
pearson(approx faces,  quality)        = -0.39
mean latency ratio, 12 one-face plans          0.598   mean quality 0.915
mean latency ratio, 18 plans with >=20 faces   0.607   mean quality 0.662
```

Adding 20–43 approximations on top of the one skip buys **zero** additional
latency (0.598 → 0.607, i.e. slightly *worse*) and costs **0.25 of quality**.
Seven of those dense plans are at quality < 0.6, including `arch:v60:34` at
**q = −0.17** and `arch:v63:36` at **q = 0.096**. This is §14.3.2's
"loses on both axes" restated on paired data.

### 3.5 Memory is not on the table

**The memory ratio is 1.0000 ± 0.0000 for every archived winner and for every
rung of the hand-specified ladder except two** (`arch:v64b:41` 0.9893,
`arch:v64b:39` 1.0069) — and for the quant rungs, which reach 0.9816. The one
lever that moves memory hard is SKIP-to-destruction: `skip@all` reads **0.0406**,
a 25× reduction, at quality 0.0000. There is no memory Pareto front here.

---

## 4. THE OPEN CAVEAT — is the win "stop differentiating a parameter"? **PENDING**

### 4.1 The worry, stated precisely

Three of the five winning faces — `k19/f0`, `k22/f0`, `k24/f0` — have the same
shape signature: a `(32,128)` activation-shaped operand combined with a
**`(1,128)`** operand, which is a broadcast **parameter** (bias / scale) reshaped
for the batch. `k25/f0`, the `broadcast_in_dim (128) → (1,128)` that produces
exactly such an operand, is one step later in the elimination.

If skipping those faces removes the path along which the parameter's gradient is
accumulated, then the "win" is *not* a cheaper Jacobian — it is **not computing a
gradient at all** for that parameter. A 200-step Adam walk on a fixed probe batch
would barely notice: 200 steps is short enough that a handful of frozen bias
vectors changes the loss trajectory by ~0.0002, which is exactly the quality gap
we measure (0.9260 → 0.9258). At 3 200 steps it should not be invisible.

This matters beyond the archive: **three training runs (62411/62412/62413) are
currently optimising against that same 200-step probe.** If the probe cannot see
the cost, the campaign target is a reward hack.

### 4.2 The registered tests

Job **62415 `probe-forensics`** (`/Users/assmuth/dsnn/fq_forensics.sbatch`,
pgi15-gpu18, `-t 2:00:00`, pinned ag=c2b8104 gx=4ea0bf8, `NEW_SLOT_JOIN=0`,
`GRAPHAX_QUANT_PULLDOWN=1`) runs two phases:

* **T1 — jaxpr diff + per-leaf gradient coverage.** `ls_face_forensics.py`
  compares the exact and skipped jaxprs and reports, **per parameter leaf**, the
  gradient norm under each. Measured, not inferred.
* **T2 — horizon sweep.** The same paired identity-vs-skip comparison at
  `--walk-steps` **200 / 800 / 3200**, for `--skip-face 24:0` and `--skip-face
  19:0`, one process per horizon (`ALPHAGRAD_WALK_STEPS` is read before the env
  import). `--reps 1` is sufficient because the walk is bit-deterministic (§5.3).

### 4.3 Status and pre-registered prediction

> **PENDING — UNRESOLVED AT TIME OF WRITING.** At 2026-08-27 job **62415 is
> `PENDING (Resources)`**, waiting on 62414 to release pgi15-gpu18. No `t1_*` /
> `t2_*` outputs exist under `run_analysis/landscape/`. The verdict below is a
> **prediction recorded before the data**, and must not be edited afterwards.

| outcome | reading |
|---|---|
| T1 shows one or more parameter leaves with **‖grad‖ = 0** under the skip and non-zero under exact | The win is "stop differentiating a parameter". The 200-step probe is not a valid quality metric for this action class, and the archived front is a reward hack. |
| T1 shows all leaves non-zero, T2 gap **flat** across 200 / 800 / 3200 | The win is a genuinely cheaper Jacobian path. The front stands. |
| T1 shows all leaves non-zero, T2 gap **grows monotonically** with horizon | The approximation is real but its cost is under-measured by the short probe; the front stands qualitatively, the quality numbers do not. |

**Predicted:** `k24/f0` and `k19/f0` differ. `k24/f0` (a `mul` by a `(1,128)`
scale, quality cost 0.0002) is the one most likely to show a zeroed leaf;
`k19/f0` costs 0.019 of quality already at 200 steps, which is 95× the `k24`
cost, so something there is *already* visible to the short probe. If that
asymmetry appears in T1/T2 it is itself the answer.

---

## 5. MEASUREMENT FACTS now established

### 5.1 The cold effect — **CONFIRMED**, and it is small

−7.2 % / −7.3 % on measurement **#1 only**, settling by #2, and only on the
identity plan (v64b −0.3 %, `quant@5` +0.1 %; both settle in 0 rounds). See §2.2
for the sign argument.

### 5.2 Per-actor GPU offsets are systematic and pairing removes them — **CONFIRMED**

Phase PD ran the identical two-plan comparison on each of the four GPUs of one
node, `--reps 5` each (`summary_d_actor_gpu{0,1,2,3}.md`):

| actor | identity absolute ns | identity self-ratio (drift floor) | v64b ratio |
|---|---:|---|---|
| gpu0 | 140 154 | 1.0006 ± 0.0007 | 0.5174 ± 0.0061 |
| gpu1 | 139 040 | 1.0015 ± 0.0018 | 0.5286 ± 0.0044 |
| gpu2 | 140 139 | 1.0014 ± 0.0012 | 0.5307 ± 0.0083 |
| gpu3 | 140 855 | 0.9993 ± 0.0015 | 0.5227 ± 0.0047 |

The absolute identity latency spans **1.30 %** across the four actors
(sd 0.46 %), while each actor's own paired within-actor noise is
**0.07–0.18 %** — i.e. the between-actor offset is **7–19× the within-actor
noise** and it is *systematic*, not random: gpu1 is consistently fast and gpu3
consistently slow across every plan.

**Consequence.** Any cross-plan comparison of *absolute* latencies collected from
different actors carries a ~1.3 % bias floor and cannot resolve anything smaller.
The whole quant ladder (§6.3, −0.2 % to −5.0 %) lives partly inside that band.
Pairing removes it exactly: every actor's own drift floor is within 0.15 % of 1.

> **RECOMMENDATION (not implemented).** The training reward should become a
> **paired per-actor ratio** — measure candidate and its own exact-order
> reference back to back in the same actor and reward the ratio — rather than a
> raw nanosecond count compared against a reference measured elsewhere. This is
> the same defect as §7(a) at a different scale.

### 5.3 The 200-step quality walk is BIT-DETERMINISTIC — **CONFIRMED**

`summary_COMBINED.md` "Quality noise floor", three different plans, three
independent runs:

| plan | n | quality mean | quality sd | quality range | latency CV |
|---|---:|---:|---:|---|---:|
| `identity` | **24** | 0.925983 | **0** | 0.925983 – 0.925983 | 1.47 % |
| `archive:v64b` | **12** | 0.898126 | **0** | 0.898126 – 0.898126 | 1.47 % |
| `quant@5` | **12** | 0.925856 | **0** | 0.925856 – 0.925856 | 0.11 % |

The walk reads a fixed probe batch and the env's fixed initial weights
(`env.args`, never re-drawn), so this is not luck: **48 readings, three plans,
zero variance.**

**Consequence.** Episode-to-episode quality variation inside a training run is
**real signal about different plans**, not measurement noise. Any explanation of
the collapse that appeals to "noisy quality readings" is dead. (§14 already
recorded H4 as REFUTED; this is the direct measurement.) It also licenses
`--reps 1` per horizon for T2 (§4.2).

---

## 6. LEVER INVENTORY on the TLM target

Target: `TransformerLM` / `wikitext2`, `ALPHAGRAD_TLM_SEQ=32 DMODEL=128
VOCAB=1024`, fixed reverse order, 95 vertices, **118 live faces**,
`--measure-grad --seed-vertices`. Numbers from `summary_e_lad_pd1_in50.md`
(pd1) and `summary_g_lad_pd0_in50.md` (pd0), paired, warm, in50, n=5.
Drift floor for this phase: **0.9999 ± 0.0015**.

### 6.1 DIAG applies **0 of 103** requested rules — **CONFIRMED**

| rung | requested faces | applied / skipped | latency ratio | quality |
|---|---:|---:|---|---|
| `diag@1` | 1 | **0 / 1** | 0.9985 ± 0.0017 | 0.9260 |
| `diag@5` | 5 | **0 / 5** | 0.9994 ± 0.0012 | 0.9260 |
| `diag@15` | 15 | **0 / 14** | 0.9983 ± 0.0017 | 0.9260 |
| `diag@50` | 50 | **0 / 44** | 1.0005 ± 0.0004 | 0.9260 |
| `diag@all` | 118 | **0 / 103** | 0.9999 ± 0.0010 | 0.9260 |

Zero rules land, at every budget, under both pd1 and pd0. Latency is the drift
floor to three decimals and quality is bit-identical to identity at every rung.

> **Every "diag" result in the campaign is an identity plan wearing a diag
> label.** Any Pareto point, ablation or op-mix statistic that credits DIAG on
> this target is crediting a no-op. This does *not* by itself mean DIAG is
> broken — see §7(d): most of those skips are correct no-ops on operands that
> are already diagonal — but it does mean the *action* was inert.

### 6.2 COMPRESS lands but costs

| rung | applied / skipped | latency ratio | quality |
|---|---:|---|---|
| `compress@1` | 1 / 0 | 0.9992 ± 0.0009 | 0.9260 |
| `compress@5` | 4 / 1 | **1.0327 ± 0.0005** | **0.1174** |
| `compress@15` | 13 / 1 | 1.0015 ± 0.0004 | 0.5994 |
| `compress@50` | 47 / 2 | 1.0125 ± 0.0020 | 0.6017 |
| `compress@all` | 111 / 4 | 1.0100 ± 0.0013 | 0.5928 |

COMPRESS **does** apply — 111 of 115 attempted at the `all` rung, ~96 % — and it
is **net negative on latency at every budget above 1** (+1.0 % to +3.3 %, all
outside the drift floor) while costing 0.33–0.81 of quality. `compress@5` is
worse than `compress@all` on both axes, so the damage is not monotone in budget:
*which* faces get compressed dominates *how many*.

> **NOT REPRODUCED:** a figure of "COMPRESS ~42 % applied" is in circulation.
> The artifacts read here give **96 % applied** (111/115) at the `all` rung and
> 80–93 % at the intermediate rungs. The campaign telemetry uses a different
> denominator (`frac_compress` in the R1 log reads 0.11–0.54 over the first eight
> episodes, mean 0.25); the two counters are not the same quantity and should not
> be quoted interchangeably.

### 6.3 QUANT applies 100 %, and pullup vs pulldown changes nothing measurable

| rung | applied / skipped | ratio (pd1) | ratio (pd0) | mem (pd1) | quality |
|---|---:|---|---|---|---|
| `quant@1` | 3 / 0 | 1.0002 ± 0.0010 | 0.9994 ± 0.0005 | 1.0000 | 0.9260 |
| `quant@5` | 15 / 0 | 1.0120 ± 0.0023 | 1.0108 ± 0.0015 | 1.0000 | 0.9259 |
| `quant@15` | 42 / 0 | 0.9781 ± 0.0013 | 0.9772 ± 0.0014 | 0.9899 | 0.9258 |
| `quant@50` | 147 / 0 | 0.9848 ± 0.0023 | 0.9828 ± 0.0016 | 0.9892 | 0.9258 |
| `quant@all` | **345 / 0** | **0.9497 ± 0.0017** | **0.9483 ± 0.0010** | 0.9816 | 0.9258 |

**CONFIRMED:** QUANT applies every rule it is asked for (0 skipped at every
budget, 345 applications for 118 faces), and `quant@all` is a **real but small
win — −5.0 % latency and −1.8 % memory at a quality cost of 0.0002**, comfortably
outside the 0.15 % drift floor. It is the only one of the three rule ops that
produces any measured win at all.

**CONFIRMED and unexplained:** the pulldown flag does essentially **nothing**.
`quant@all` reads 0.9497 (pd1) vs 0.9483 (pd0) — a 0.15 % difference, i.e. the
drift floor — with byte-identical applied counts and memory ratios. Whatever
pulldown is meant to buy, it is not buying it here.

**The mechanism, cited from source.** In
`graphax/src/graphax/sparse/dtype_compute.py`:

* `_quant_pulldown()` (line 124) reads `GRAPHAX_QUANT_PULLDOWN`, **default `"0"`**.
* `_compute_dtype()` (line 133) is the arithmetic-dtype policy. Under
  **pulldown=1**, an all-float operand set containing any half type computes at
  that half type ("the quantization wins"). Under **pulldown=0 (pullup, the
  default)** it returns `jnp.result_type(*dts)` — i.e. a `{f32, bf16}` pair is
  **deliberately promoted back to f32**.
* The deferral of the dequantizing multiply already exists: `_scaled_mul(...,
  keep_narrow=True)` (line 170) broadcasts `scalar_mult` *down* into the narrow
  value dtype so a Quant survives densification. Its own comment names the
  failure it was written for:

  > *"the densify/materialize read (`val * f32 scalar_mult`) was re-promoting the
  > Quant'd array to f32 right BEFORE the contraction consumed it, so the
  > heavyweight dots never ran bf16"*

  So the blocker is **not** a missing deferral. It is that the *other* operand of
  every heavyweight contraction — the accumulated left-hand side — stays f32, so
  the pair is mixed `{f32, bf16}` and `_compute_dtype` upcasts it by design.

> **NOT INDEPENDENTLY VERIFIED here:** the further claims that this deferral is
> present in *all three* contraction paths and that chain propagation breaks
> specifically at the `old += new` accumulation join. Those are consistent with
> the code read above and with the pd1-vs-pd0 null result, but this document did
> not trace the three paths. Note also that `dtype_compute.py` is one of the four
> **uncommitted** files in the live graphax tree; the pinned `4ea0bf8` snapshot
> that produced the measurements above may differ from the lines quoted.

### 6.4 SKIP is the only lever with a large win — and it has two regimes

| plan | ratio (pd1) | ratio (pd0) | mem ratio | quality |
|---|---|---|---|---|
| `skip@all` | 0.4738 ± 0.0329 | 0.4620 ± 0.0094 | **0.0406** | **0.0000** |
| best single skip (`k19/f0`) | ~0.530 | — | 1.0000 | 0.9071 |
| best *free* single skip (`k21/f1`) | ~0.532 | — | 1.0000 | 0.9257 |

`skip@all` is the SKIP cliff — 2.1× faster, **25× less memory**, and the gradient
is gone. A *single, well-chosen* skip gets **most of the latency win (0.53 vs
0.47) at full quality and full memory**. That gap — 0.53 at q 0.926 versus 0.47
at q 0 — is the entire prize the campaign is trying to learn to find, and §7(a)
explains why the reward has been pointing at the wrong end of it.

### 6.5 Summary of the levers

| lever | applies? | best measured latency | memory | quality cost | verdict |
|---|---|---|---|---|---|
| **DIAG** | **0 of 103** | none (0.998–1.001) | none | none | inert on this target |
| **COMPRESS** | 96 % | **negative** (+1.0…+3.3 %) | none | 0.33–0.81 | actively harmful |
| **QUANT** | 100 % | −5.0 % | −1.8 % | 0.0002 | small, real |
| **SKIP (one face)** | n/a | **−47 %** | none | 0.0002–0.019 | the win |
| **SKIP (all)** | n/a | −53 % | −96 % | total | the absorber |

---

## 7. BUGS FOUND, AND THEIR STATUS

### (a) The quality gate's floor and the plan it floors were timed by different instruments — **FIXED (`1c1e480`)**

`ALPHAGRAD_QUALITY_GATE_MIN=0.05` is set in all twelve campaign launchers, to
make destruction cost-neutral: a plan whose quality falls below 0.05 has its
latency floored at what its own order costs done exactly. It did not do that.

`_measure_exec_cost` (`env.py:1912-1952`) timed the **floor** as the median of
3 laps of 20 back-to-back executions — a throughput timing that amortises
per-call dispatch — while the **plan** was timed by the campaign path
(`--num-data-points 5 × --reps-per-point 4` at `--latency-inner-reps 5`, reduced
by a median). Measured from the gate's own clamp prints:

```
v63  (61866)  CLAMP #1   : q=0.0000 < 0.05; lat  65.8 -> 138.5us
v64b (61983)  CLAMP #1   : q=0.0000 < 0.05; lat  67.9 -> 137.9us
              CLAMP #2550: q=0.0000 < 0.05; lat  82.9 -> 137.5us
```

against identity/exact plans charged **155–160 µs** in the same runs (ep3–10
means: v63 159.0, v64b 158.3, v66a 155.1, v65 157.6 µs). **The floor is 13–15 %
below what an honest exact plan is charged**, so a plan that destroys the
gradient collected a guaranteed **−13 % latency and −1.4 % memory bonus, in all
twelve campaign runs.**

**FIXED** — commit `1c1e480` *"env(measure): gate floor uses the campaign
instrument, not a 3x20 throughput timing (+per-order floor cache, ratio
telemetry, default-on warmup)"*. Historical consequence stands: treat every
clamped-plan cost number in v57–v66 as carrying a −13 % artefact, and treat the
group-B absorber's "136 µs" as the floor's own value, not a measurement.

### (b) All 264 archived Pareto points were unreplayable — **FIXED (`907c231`)**

The Pareto dumps record the elimination as **1-based vertex ids**; the replay
path read them as **0-based action indices**. Every archived point therefore
rebuilt a different plan, and recovery failed silently.

**FIXED** — commit `907c231` *"tools/landscape_map: fix archive replay (recorded
column is 1-based vertex ids, not action indices) + per-config stamping,
cold-sequence and per-actor reporting"*. Recovery went **0/42 → 42/42** on v64b.
Post-fix census (`archive_census_qa_attrib.json`), **held = recovered for all
eight archives, 264 points total**:

```
v57 11/11   v60 42/42   v63 37/37   v64b 42/42
v65 39/39   v66a 31/31  v66b 27/27  v66c 35/35
```

Every number in §2 and §3 depends on this fix.

### (c) `ALPHAGRAD_NEW_SLOT_JOIN` defaults to 1 and emits a face form graphax rejects — **WORKED-AROUND, ROOT CAUSE UNFIXED**

With the default `=1`, `_face_dict_for_vertex` emits
`((lhs, rhs, res), (None, res_hook, None))`. Measured 2026-08-27, jobs **62391**
and probe **62394**: graphax's `_unpack_face_slots` raises
*"face_transforms entry at vertex 95 must be a 3-tuple"* — in **both** the pinned
`4ea0bf8` tree and the live working tree. So **every plan that puts a rule in the
`res`/`new` slot dies at elimination.**

Worse, `env.py` routes graphax trace failures through
`_is_graphax_trace_failure → _trace_truncate`, so such a plan is **silently
excluded from the gradient** rather than scored. If the campaign ran with the
default, the res slot was unreachable all along and every plan that tried it was
dropped without trace.

**WORKED-AROUND:** `export ALPHAGRAD_NEW_SLOT_JOIN=0` is set explicitly in all
five launchers (`fq_r1_trim.sbatch`, `fq_r2_trimplus.sbatch`,
`fq_r3_credit.sbatch`, `fq_face_attrib.sbatch`, `fq_forensics.sbatch`), which
emits the plain 3-tuple graphax accepts. That form still carries the res-slot
approximation; it just does not also apply it to the existing edge at the join —
and `env.py` states the two forms are **not comparable**. **The root cause is
unfixed and the v57–v66 logs have not been audited for how many plans it ate.**

### (d) DIAG per-face masking: implemented, measured, changes nothing — **commit `39d8bd1`, flag default OFF**

Commit `39d8bd1` *"masks: `--diag-per-face` masks DIAG by each face own legal
factor set (default off); measured: on NN it repairs nothing, 83 % of skips are
idempotent no-ops"*. Zero projections fire. The 0-of-103 applied fraction in
§6.1 is therefore **mostly correct behaviour** — the operands are already
diagonalised, so the rule is a legitimate no-op — not a masking bug. The flag
stays default OFF.

The open question this leaves is not "why does DIAG skip?" but **"why is DIAG in
the action space of this target at all?"**, since on TLM it can never do
anything, while occupying ~0.2 of the head's probability mass in every campaign
run.

---

## 8. THE FOUR RUNS (62411 / 62412 / 62413, plus 62414/62415)

All three arms share: `ALPHAGRAD_POLICY=palimpsa`, `ALPHAGRAD_FORCE_REV_ORDER=1`
(so **approximation is the only lever**), `GRAPHAX_QUANT_PULLDOWN=0` (**pullup**
— every previous TLM arm ran pulldown=1), `ALPHAGRAD_NEW_SLOT_JOIN=0`,
`GRAPHAX_PLANNER_EXACT=1`, `GRAPHAX_DEMAND_EMIT=1`, target
`TLM_SEQ=32 / DMODEL=128 / VOCAB=1024`, `MAX_FACES=2538`,
`MAX_DELTA_TOKENS=32768`, `SKIP_COUNT_OPS=1`, `DIRECT_MEASURE=1`,
`CLEAR_JIT_CACHES_EVERY=0`, 4 GPUs, 250 episodes, `--wandb online` to
`dll-streetview/dsnn-vertex`.

Common CLI: `--measure-latency --latency-inner-reps 50 --num-data-points 5
--reps-per-point 4 --incremental-encode --cmp-type latency --mem-type
peak_memory --terminal-rewards-only --rewards cmp mem acc --reward-mode additive
--symlog-channels cost --lambda-cmp 1 --lambda-mem 1 --lambda-acc 16
--advantage-norm none --entropy-weight 0 --face-entropy-weight 0
--face-entropy-floor 0 --face-logit-clamp 0 --quality-metric loss_drop
--init-scheme classic --face-read last-row --num-envs 16 --minibatches 4
--grad-window 0 --measure-grad --seed-vertices --set-pointer --face-actions
--unified-face-head --live-faces --hidden-dim 256 --vocab-size 512 --num-layers 3
--ray-measure 3 --ray-measure-timeout 600 --pareto-dump-every 10`.

Each launcher pre-flights `--symlog-channels`, `--init-scheme` and `--face-read`
against `ppo.py` and aborts with exit 64 naming the missing flag.

### The target all three are judged against

> **A one-face plan at latency ratio 0.578 and quality 0.926** (§3.1) — five
> independent runs already archived it, and phase QB says a slightly better one
> exists at 0.530–0.532. Any arm that ends without a plan in that region has not
> found what is there to find.

### R1 `r1-trim` — job 62411, pgi15-gpu15, `fq_r1_trim.sbatch`

**Purpose.** The owner's minimal design, run exactly as specified, as the control
the other arms are read against. Nothing tuned, nothing rescued.
**Differentiating flags:** `ALPHAGRAD_FACE_NONE_BIAS=0`,
`ALPHAGRAD_QUALITY_GATE_MIN=0`, `--discount 1.0 --gae-lambda 1.0`,
`--ppo-epochs 2`.

**REGISTERED PREDICTION** (from the launcher, recorded before launch): the policy
starts at ~372 approximations per plan; with `FACE_NONE_BIAS=0` the NONE arm
carries no prior, so episode 0 sits far outside the 1–15-approx region where every
good plan the campaign has found actually lives. The thing to watch is whether
**any batch contrast exists at ep 0–30** — with `--advantage-norm none`,
γ = λ = 1 and every entropy term at 0, if all 16 envs return the same scalarized
reward the advantage is ~0 everywhere and there is no learning signal at all.

**OBSERVED, ep 0–4** (`r1_trim_62411.log`) — the predicted no-contrast absorber:

```
[plan ep=0] n=16 live=16 | q med=+0 [+0,+0] sd=0      | lat_us med=61.96 [58.63,70.75] | skips=23 | WIN none
[plan ep=1] n=16 live=16 | q med=+0 [+0,+0] sd=0      | lat_us med=60.89 [43.08,72.10] | skips=18 | WIN none
[plan ep=2] n=16 live=16 | q med=+0 [+0,+0.0014] sd=3.4e-4 | lat_us med=62.23 | skips=29 | WIN none
[plan ep=3] n=16 live=16 | q med=+0 [+0,+0.8606] sd=0.2083 | lat_us med=61.00 | skips=19 | WIN none
[plan ep=4] n=16 live=16 | q med=+0 [+0,+0.5616] sd=0.1359 | lat_us med=58.34 | skips=21 | WIN none
```

At **ep0 and ep1 the quality of all 16 plans is 0 with sd EXACTLY 0**, at a median
latency of **62 µs**. Cross-reference §6.4: `skip@all` measures 0.4738 × identity
= **66 µs at quality 0.0000, memory ratio 0.0406**. R1's episode 0 *starts inside
the SKIP-cliff absorber*, with 16–29 skips per batch and zero batch contrast —
and with `QUALITY_GATE_MIN=0` there is nothing to floor it. The registered
prediction is confirmed at the first data point. `WIN none` at every episode so
far.

### R2 `r2-trimplus` — job 62412, pgi15-gpu16, `fq_r2_trimplus.sbatch`

**Purpose.** The best-shot arm. Three changes from R1, each with a named job:
`ALPHAGRAD_FACE_NONE_BIAS=6` starts the policy *inside* the 1–15-approx region;
`--ppo-epochs 1` makes the importance ratio identically 1 for the whole update,
removing the negative-advantage epoch asymmetry present in every previous arm;
`ALPHAGRAD_QUALITY_GATE_MIN=0.05` keeps the SKIP cliff from earning anything.

**REGISTERED PREDICTION.** (i) `none` stays ≥ 0.9 for 100+ consecutive episodes.
(ii) The per-plan joint shows plans at **≤ 20 approximations with quality ≥ 0.8
AND latency ≤ 0.9× the exact-rev reference — measured paired, as a ratio, never
as an absolute against a reference from another window.**

**OBSERVED, ep 0–2** (`r2_trimplus_62412.log`):

```
[plan ep=0] q med=+0.8853 [+0.8849,+0.8853] sd=1.2e-4 | lat_us med=131.5 [77.55,133.6] | WIN env=9 lat=77.55us q=+0.885 ops=0
[plan ep=0] q med=+0.8853 [-1,+0.8853]      sd=0.4889 | lat_us med=131.7 [116.1,135.0] | WIN env=8 lat=116.1us q=+0.885 ops=0
[plan ep=1] q med=+0.8853                   sd=1.5e-5 | lat_us med=132.0 [130.1,133.6] | WIN none
[plan ep=2] q med=+0.8853 [+0.727,+0.8853]  sd=0.0457 | lat_us med=132.2 [122.1,133.1] | WIN none
```

The bias does what it was asked to do: **ep0 sits at identity quality
(0.8853 vs R1's 0), at 131–132 µs, with 1–2 requested ops per plan** instead of
R1's 372. And at ep0 one env already returned **77.55 µs at q +0.885** — ratio
0.59 against the batch median, which is the §3.1 target region, hit by random
exploration at episode zero. **Caveat:** that is an *unpaired single* measurement
from the trainer's own reward path, not the landscape protocol, and R2's quality
scale (0.885 identity) is not the landscape's (0.926 identity) — different probe
setup — so it corroborates, it does not confirm.

### R3 `r3-credit` — job 62413, pgi15-gpu17, `fq_r3_credit.sbatch`

**Purpose.** Identical to R2 **except** `--discount 0.99 --gae-lambda 0.95`.
This isolates the one knob the campaign has never varied: the credit horizon. At
γ = 0.99, λ = 0.95 over a ~95-step episode, the GAE weight on the terminal reward
at the *first* elimination is (γλ)^95 = **0.0079** — the first two thirds of the
plan receive essentially no credit for the plan's measured cost. Every previous
arm ran 1.0 / 1.0.

**REGISTERED PREDICTION.** If R2 holds (`none` ≥ 0.9 for 100+ eps) and R3 drifts,
the credit horizon is the cause of the drift and R3 is the falsifier that says
so. **If both hold, or both drift, the credit horizon is NOT the lever and this
arm is a clean negative result — say so, do not reinterpret it.**

**OBSERVED, ep 0–2:** indistinguishable from R2 so far (ep0 q med +0.8853, lat med
131.3 µs, the same env=9 win at 77.46 µs / q +0.885 / ops=0). Too early to read.

### 62414 `face-attribution` (pgi15-gpu18) and 62415 `probe-forensics` (queued)

62414 runs QA (name the skipped face of every archived plan, measures nothing),
QB (singleton skip sweep, stride 1, all 118 faces, paired, `--reps 3
--warmup-trials 2`) and QC (combined report). Its QB output is the basis of §3.2
and §3.3 and was **still being written** when this document was drafted — the QB
table is a valid partial snapshot, not the final n=3 aggregate. 62415 is §4.2 and
was `PENDING (Resources)`.

---

## 9. What would close the remaining questions

1. **T1/T2 (job 62415).** Decides §4. Until it lands, every quality number in
   this document is "quality as seen by a 200-step Adam walk", and that qualifier
   is load-bearing.
2. **Finish QB and read `plans_qa_attrib.json`.** Gives byte-level face
   attribution for all 264 archived points instead of signature matching (§3.2
   caveat), and the final n=3 paired aggregate for all 118 faces.
3. **Trace the three contraction paths for the pullup blocker** (§6.3) and
   explain the pd1-vs-pd0 null result. Either pulldown never fires on this graph,
   or bf16 dots are not where the −5 % comes from.
4. **Audit v57–v66 for `NEW_SLOT_JOIN=1` casualties** (§7(c)) — how many plans
   were silently dropped from the gradient, and whether that biases the archives
   this document just validated.
5. **Move the training reward to a paired per-actor ratio** (§5.2). It is the
   same class of defect as §7(a), it is measured, and it is un-fixed in the
   learner.

---

## Provenance index

| artifact | what it establishes |
|---|---|
| `run_analysis/landscape/summary_COMBINED.md` | the full 128-row ratio table, cold sequences, quality noise floor, archive census |
| `run_analysis/landscape/summary_c_winners_warm.md` | §2.1 winners, drift floor 1.0007 ± 0.0008 |
| `run_analysis/landscape/summary_a_cold_ident.md`, `summary_b_cold_v64b.md` | §2.2 cold effect −7.3 % / −0.3 % |
| `run_analysis/landscape/summary_d_actor_gpu{0..3}.md` | §5.2 per-actor offsets |
| `run_analysis/landscape/summary_e_lad_pd1_in50.md`, `summary_g_lad_pd0_in50.md` | §6 lever inventory, pd1 vs pd0 |
| `run_analysis/landscape/summary_f_arch_pd1_in50.md` | §3.1 five-run cluster, §3.4 correlations |
| `run_analysis/landscape/summary_h_corr_pd1_in5.md` | §1.3 inner-reps 5 vs 50 |
| `run_analysis/landscape/rows_qb_sweep.csv`, `face_inventory_qa_attrib.json` | §3.2/§3.3 named faces |
| `run_analysis/landscape/archive_census_qa_attrib.json` | §7(b) 264/264 recovered |
| `fq_r1_trim.sbatch`, `fq_r2_trimplus.sbatch`, `fq_r3_credit.sbatch`, `fq_face_attrib.sbatch`, `fq_forensics.sbatch` | §8 flag sets and registered predictions |
| `r1_trim_62411.log`, `r2_trimplus_62412.log`, `r3_credit_62413.log` | §8 observations |
| `graphax/src/graphax/sparse/dtype_compute.py:124-212` | §6.3 pullup mechanism |
| commits `1c1e480`, `907c231`, `39d8bd1`, `c2b8104` | §7 bug fixes |
