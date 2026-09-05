# How every run in the approximation campaign actually ran (v57 → v66c)

> **Contamination note (2026-09-03).** All twelve runs described here (v57 through v66c) ran with discount 0.99 and GAE-lambda 0.95 inherited from `ppo.py` defaults, so the first eliminations of each episode received about 0.31 % of the terminal reward, and all of them also ran under the -13 % timer bug fixed in alphagrad `1c1e480f`. The trajectories are recorded only; any conclusion drawn from them about the order axis, PopArt, or PPO's ability to learn is superseded by the campaign in the dsnn-3qm map. See `.scratch/trustworthy-approx-search/findings/55-gamma-lambda-rot.md` (dsnn superproject).

*Generated 2026-08-26. Purely **descriptive**: trajectories, transitions and
landmarks. No causal claims — the "why" is a separate piece of work.*

Source: full `run.scan_history()` for all 12 runs of `dll-streetview/dsnn-vertex`
(pulled to `run_analysis/data/*.csv`, key inventory in `data/meta.json`),
cross-checked against the STDOUT logs in `/Users/assmuth/dsnn/*.log`
(`[approx]`, `[entropy]`, `[lagrangian]`, `[vprobe]`, `[ppdec]/prof` lines,
mined to `data/*_log.csv` + `data/logmeta.json`) and against `sacct`.

Figures: `/Users/assmuth/dsnn/run_analysis/figs/` (18 PNGs, 150 dpi).
Manifest: `/Users/assmuth/dsnn/run_analysis/manifest.txt`.

---

## 0. Conventions, definitions and caveats

**Run identities.**

| run | wandb name | wandb id | slurm job |
|---|---|---|---|
| v57  | v57-tlm-revapprox-fullT   | `it05ku34` | 61485 |
| v58b | v58b-tlm-fullT-env16      | `ud3cla83` | 61498 |
| v59  | v59-tlm-fullT-warmup      | `3faj3e36` | 61507 |
| v60  | v60-tlm-fullT-multwarmup  | `ygm8n2jy` | 61515 |
| v61  | v61-tlm-lagrangian        | `auw0vzlm` | 61610 |
| v62  | v62-tlm-lagrangian        | `w6sh91ya` | 61844 |
| v63  | v63-tlm-endpoint          | `as9s5yrl` | 61866 |
| v64b | v64b-tlm-creditfix        | `38oyqf4g` | 61983 |
| v65  | v65-tlm-rawviol           | `8sht6x1m` | 62075 (LIVE) |
| v66a | v66a-static-lam10         | `318ktrgq` | 62072 (LIVE) |
| v66b | v66b-static-lam16         | `0olsxsjl` | 62073 (LIVE) |
| v66c | v66c-static-lam13-nomask  | `s1537jdd` | 62074 (LIVE) |

**Sign convention.** `mean_latency_ns` and `mean_peak_memory` are logged as
*negated reward channels* (they are ≤ 0 throughout every run). Every latency
and memory number in this document is the **absolute value**, i.e. the physical
cost, and "best" means the smallest physical cost.

**De-diluted entropy.** Per `docs/QUALITY_COLLAPSE_INVESTIGATION.md` §13.3–13.4,
the logged `entropy/approx_head` is `jnp.mean(face_ents)` over the flat
(env × step) batch, so steps whose vertex carries **no live face** enter the
mean as a structural `0`. The de-diluted quantity plotted everywhere alongside
it is

```
H_dediluted(ep) = entropy/approx_head(ep)  ÷  faces/mean_valid(ep)
```

which divides the diluted metric by its own population (a lower bound on
per-face entropy, since the unlogged per-face-bearing-step count `k ≥ 1`).
Both panels are always shown: a falling **raw** panel can mean the decisions
being measured have disappeared rather than that the head became deterministic.

**Phase criteria (explicit, applied per episode).** With
`base_faces := median(faces/mean_valid)` over the run's first 10 logged episodes:

* **ABSORBED** — `faces/mean_valid < 0.25 · base_faces`, sustained ≥ 3
  consecutive episodes (the live-face population the head acts on has
  collapsed); the phase is latched from the first such episode onward.
* **DRIFT** — not absorbed **and** `approx_prob/none < 0.5` (the head no longer
  mostly emits the exact / `OP_NONE` op).
* **HEALTHY** — everything else, i.e. `approx_prob/none ≥ 0.5`.

These are thresholds on logged telemetry, nothing more; they are not a claim
about what is good or bad.

**Caveats.** (1) `mean_quality`, `mean_latency_ns` and `mean_peak_memory` are
**episode means**, so "best-ever" below is the best episode mean, not the best
individual plan (only v58b logged `measure/*/best_alltime`). (2) `mean_quality`
is strongly **bimodal** in several runs — see §3 v57 — so a low tail *mean* is
not by itself evidence of a low tail *level*. (3) The `[lagrangian]` STDOUT line
count is short of the episode count in v65 (49 lines vs 203 episodes) and v66b
(71 vs 193) because the tqdm progress bar overwrites line starts; the wandb
`lagrangian/*` series are complete and are what is used here. (4) No run's log
tail contains a Python `Traceback`; wandb's `crashed` state on v57/v58b/v59/
v61/v62 reflects the process being killed without finalising, not an in-process
exception.

---

## 1. Landmark table

### T1 — census / entropy landmarks

| run | state | eps | h | none<0.9 | none<0.5 | none<0.1 | Hraw max (ep) | Hded max (ep) | fracviol>0.5 |
|---|---|---|---|---|---|---|---|---|---|
| v57 | crashed | 164 | 0.68 | 48 | — | — | 1.426 (ep163) | 1.190 (ep160) | — |
| v58b | crashed | 191 | 3.90 | 18 | 38 | — | 1.222 (ep26) | 1.706 (ep165) | — |
| v59 | crashed | 49 | 1.36 | 46 | — | — | 1.127 (ep49) | 0.927 (ep49) | — |
| v60 | finished | 507 | 8.64 | 46 | 71 | 260 | 1.366 (ep75) | 1.844 (ep343) | — |
| v61 | crashed | 479 | 35.96 | 16 | 64 | 73 | 1.329 (ep20) | 1.614 (ep66) | 17 |
| v62 | crashed | 121 | 5.72 | 50 | 64 | 92 | 1.485 (ep96) | 1.261 (ep99) | 55 |
| v63 | finished | 507 | 13.55 | 61 | 64 | 68 | 1.433 (ep72) | 1.216 (ep73) | 64 |
| v64b | finished | 507 | 12.73 | 62 | 71 | 79 | 1.025 (ep77) | 0.841 (ep78) | 74 |
| v65 | **running** | 202 | 9.35 | 103 | 112 | — | 1.089 (ep139) | 0.880 (ep139) | 118 |
| v66a | **running** | 199 | 9.37 | 80 | 83 | 93 | 1.596 (ep111) | 1.285 (ep111) | 83 |
| v66b | **running** | 192 | 9.37 | 79 | 82 | 98 | 1.215 (ep155) | 0.981 (ep156) | 84 |
| v66c | **running** | 198 | 9.35 | 78 | 81 | 96 | 1.422 (ep157) | 1.155 (ep156) | 81 |

`eps` = last logged step; `h` = wall-clock hours; `—` = threshold never reached.

### T2 — bests and finals

| run | best lat µs (ep) | best mem MB (ep) | best quality (ep) | final none | final skip | final Hraw | final Hded | final faces | final quality | final lat µs |
|---|---|---|---|---|---|---|---|---|---|---|
| v57 | 95.8 (ep117) | 54.66 (ep48) | 0.8854 (ep32) | 0.766 | 0.000 | 1.379 | 1.110 | 1.242 | 0.884 | 195.9 |
| v58b | 126.5 (ep53) | 54.64 (ep154) | 0.8853 (ep8) | 0.226 | 0.215 | 0.139 | 1.622 | 0.086 | 0.028 | 132.3 |
| v59 | 147.5 (ep46) | 55.35 (ep14) | 0.8853 (ep8) | 0.820 | 0.008 | 1.127 | 0.927 | 1.216 | 0.702 | 153.4 |
| v60 | 126.3 (ep117) | 54.57 (ep302) | 0.8853 (ep8) | 0.103 | 0.462 | 0.038 | 1.481 | 0.026 | 0.000 | 135.4 |
| v61 | 117.4 (ep43) | 54.59 (ep67) | 0.8853 (ep4) | 0.000 | **1.000** | 1e-05 | 0.0006 | 0.011 | 0.000 | 136.7 |
| v62 | 133.9 (ep118) | 54.83 (ep120) | 0.8853 (ep4) | 0.019 | 0.196 | 0.156 | 1.106 | 0.141 | −0.085 | 135.1 |
| v63 | 126.1 (ep95) | 54.62 (ep142) | 0.8853 (ep4) | 0.000 | 0.197 | 0.121 | 0.824 | 0.147 | 0.049 | 135.7 |
| v64b | 132.4 (ep281) | 54.59 (ep442) | 0.8853 (ep4) | 0.000 | 0.315 | 0.033 | 0.696 | 0.048 | 0.000 | 136.8 |
| v65 | 149.7 (ep86) | 55.27 (ep71) | 0.8853 (ep4) | 0.434 | 0.002 | 0.897 | 0.724 | **1.238** | 0.580 | 174.1 |
| v66a | 149.8 (ep57) | 55.23 (ep132) | 0.8853 (ep75) | 0.000 | 0.000 | 0.872 | 0.702 | **1.242** | −0.120 | 176.5 |
| v66b | 152.7 (ep12) | 55.34 (ep44) | 0.8853 (ep4) | 0.000 | 0.000 | 1.033 | 0.832 | **1.242** | 0.087 | 185.8 |
| v66c | 148.2 (ep12) | 55.35 (ep71) | 0.8853 (ep4) | 0.000 | 0.001 | 1.060 | 0.853 | **1.229** | −0.176 | **304.3** |

### T3 — phases (episode ranges)

| run | base_faces | healthy | drift | absorbed | phase @ last logged ep |
|---|---|---|---|---|---|
| v57 | 1.242 | 3–164 | — | — | healthy |
| v58b | 1.194 | 3–37 | 38–66 | **67–191** | absorbed |
| v59 | 1.229 | 3–49 | — | — | healthy |
| v60 | 1.229 | 3–70, 73–74 | 71–72, 75–97 | **98–502** | absorbed |
| v61 | 1.229 | 3–34 | — | **35–479** | absorbed |
| v62 | 1.228 | 3–63 | 64–117 | **118–121** | absorbed |
| v63 | 1.188 | 3–63 | 64–82 | **83–502** | absorbed |
| v64b | 1.188 | 3–70 | 71–92 | **93–502** | absorbed |
| v65 | 1.188 | 3–111 | **112–202** | — | drift |
| v66a | 1.189 | 3–82 | **83–199** | — | drift |
| v66b | 1.189 | 3–81 | **82–192** | — | drift |
| v66c | 1.189 | 3–80 | **81–198** | — | drift |

### T4 — critic

| run | value loss median | value loss final | rolling(10) std, median | value loss ÷ &#124;weighted_mean_return&#124;, median |
|---|---|---|---|---|
| v57 | 0.2174 | 0.4114 | 0.2954 | 22.06 |
| v58b | 0.0364 | 0.0368 | 0.0188 | 0.130 |
| v59 | 0.1696 | 0.1856 | 0.0541 | 9.212 |
| v60 | 0.0339 | 0.0672 | 0.0170 | 0.0499 |
| v61 | 1.009 | 1.192 | 0.0122 | 4.659 |
| v62 | 0.3346 | 0.7048 | 0.1581 | 0.744 |
| v63 | 0.9336 | 1.102 | 0.0410 | 1.732 |
| v64b | 0.9301 | 1.211 | 0.0206 | 1.729 |
| v65 | 0.3623 | 0.5074 | 0.2082 | 0.433 |
| v66a | **0.00587** | 0.00637 | **0.00198** | **0.0107** |
| v66b | **0.00630** | 0.01496 | **0.00269** | **0.00979** |
| v66c | **0.00675** | 0.00533 | **0.00281** | **0.00937** |

Per-channel `value_loss/{quality,latency,mem}` exists only for v65 and
v66a/b/c and is plotted in figure (iv), lower-left.

> **Do not read this table as critic accuracy.** `value loss` is in each arm's
> own target units, which differ by orders of magnitude between the PopArt and
> the `advantage_norm=none` arms. See §6.4 for the scale-free version, which
> reverses the ordering on the cost channels.

### T5 — trailing 20 episodes (basis for the LIVE verdicts)

| run | none | skip | faces | Hraw | Hded | quality | raw_q | fracviol | lat µs |
|---|---|---|---|---|---|---|---|---|---|
| v57 | 0.697 | 0.013 | 1.148 | 1.268 | 1.097 | 0.037¹ | — | — | 163.9 |
| v58b | 0.194 | 0.214 | 0.098 | 0.152 | 1.557 | −0.005 | — | — | 133.8 |
| v59 | 0.927 | 0.006 | 1.181 | 0.639 | 0.544 | 0.672 | — | — | 154.1 |
| v60 | 0.116 | 0.491 | 0.025 | 0.038 | 1.482 | 0.000 | — | — | 135.0 |
| v61 | 0.000 | 1.000 | 0.011 | 0.00001 | 0.0006 | 0.000 | — | 1.000 | 136.3 |
| v62 | 0.022 | 0.108 | 0.523 | 0.614 | 1.160 | −0.209 | — | 0.984 | 148.1 |
| v63 | 0.000 | 0.264 | 0.078 | 0.063 | 0.804 | 0.004 | 0.004 | 0.988 | 136.0 |
| v64b | 0.000 | 0.366 | 0.036 | 0.024 | 0.654 | −0.001 | −0.001 | 1.000 | 136.3 |
| **v65** | 0.452 | 0.004 | 1.188 | 0.859 | 0.723 | **0.594** | 0.594 | 0.378 | 171.3 |
| **v66a** | 0.000 | 0.0003 | 1.242 | 0.873 | 0.703 | −0.021 | −0.021 | 0.969 | 179.1 |
| **v66b** | 0.000 | 0.0003 | 1.242 | 1.015 | 0.818 | 0.233 | 0.233 | 0.909 | 179.6 |
| **v66c** | 0.000 | 0.0012 | 1.229 | 1.066 | 0.867 | −0.105 | −0.105 | 1.000 | 300.5 |

¹ v57's trailing quality is a mean over a **bimodal** series; the last 16 values
are `0.716, −0.039, −1.00, −0.003, 0.813, −0.278, 0.885, −0.288, 0.645, 0.884,
0.649, −0.220, −1.00, 0.885, −0.210, 0.884` — it alternates between ≈0.88 and
≈−1, it does not sit near 0.04.

### T6 — how each run ended

| run | slurm state | elapsed | walltime limit | ended by |
|---|---|---|---|---|
| v57 | CANCELLED | 00:41:28 | 1-12:00:00 | operator cancel (early) |
| v58b | CANCELLED | 03:55:31 | 1-12:00:00 | operator cancel |
| v59 | CANCELLED | 01:22:42 | 1-12:00:00 | operator cancel (early) |
| v60 | COMPLETED | 08:38:56 | 1-12:00:00 | ran to completion (500 ep) |
| v61 | TIMEOUT | 1-12:00:16 | 1-12:00:00 | **walltime** (stopped at ep479) |
| v62 | CANCELLED | 05:43:55 | 1-12:00:00 | operator cancel |
| v63 | COMPLETED | 13:33:38 | 1-12:00:00 | ran to completion (500 ep) |
| v64b | COMPLETED | 12:44:18 | 1-12:00:00 | ran to completion (500 ep) |
| v65, v66a/b/c | RUNNING | 09:29:5x | 1-12:00:00 | **still running** |

---

## 2. Phase-timing summary

Under the §0 criteria, the campaign splits into three observed shapes.

**(A) Ended while still healthy — v57, v59.** Both were cancelled inside the
first 1.4 h, before `approx_prob/none` ever reached 0.5 (v57 bottomed at 0.657,
v59 at 0.820). `faces/mean_valid` never left 1.15–1.24; both raw and de-diluted
entropy were still *rising* at the last episode. These two runs contain no
transition to describe — they stop before one.

**(B) healthy → drift → absorbed, with `skip` taking over — v58b, v60, v61,
v62, v63, v64b.** Every one of these crosses `none < 0.5` between ep38 and
ep71, and every one subsequently loses its live-face population: final
`faces/mean_valid` is 0.011–0.147 against a `base_faces` of ≈1.19–1.23, i.e.
an 8×–115× reduction, while `approx_prob/skip` rises to 0.20–1.00. The absorbed
phase begins at ep67 (v58b), ep98 (v60), ep35 (v61), ep118 (v62), ep83 (v63),
ep93 (v64b) and is never exited in any of them: once latched it runs to the end
of the run, for as many as 420 further episodes (v60, v63, v64b). v61 is the
extreme point — `skip` = 1.000, `faces/mean_valid` = 0.011, raw entropy 1e-05
and de-diluted entropy 0.0006, held flat from ep35 to ep479 (36 h of wall
clock). The drift window is short in every case: 29 episodes (v58b), 25 (v60),
0 (v61 — it went straight from healthy to absorbed within one episode of
crossing), 54 (v62), 19 (v63), 22 (v64b).

**(C) healthy → drift, with `skip` staying at zero and the graph intact — v65,
v66a, v66b, v66c (all four LIVE).** All four cross `none < 0.5` at ep81–ep112
and are still in drift at ep192–202. What separates them from group (B) is that
`approx_prob/skip` stays at 0.0002–0.004 (versus 0.20–1.00) and
`faces/mean_valid` stays at 1.23–1.24 — **at or above** their own
`base_faces` of 1.189. No absorbed phase has begun in any of them, by the
face-population criterion or by the entropy one: raw entropy is 0.86–1.07 and
de-diluted entropy 0.70–0.87, both near their run maxima, not near zero.

The crossing points cluster: the ep of first `none < 0.5` is 38 (v58b), 64
(v61, v62, v63), 71 (v60, v64b), 81–83 (v66c, v66b, v66a), 112 (v65) — later
in every v66 arm than in every lagrangian-era run, and latest of all in v65.

---

## 3. Per-run description

**v57** (`run_v57.png`) — 165 episodes in 41 min, cancelled. `none` decays
1.00 → ≈0.70 by ep50 and then plateaus there for 110 episodes; `skip` stays
≈0.01. Raw entropy climbs monotonically to 1.426 (ep163) and de-diluted to
1.19; `faces/mean_valid` holds at 1.24 with occasional single-episode dips to
0.3–1.0. Quality is violently bimodal, alternating between ≈0.885 and ≈−1.0
episode to episode. Value loss is the noisiest of the whole campaign relative
to its own reward scale (ratio median 22.1, rolling-std median 0.295).

**v58b** (`run_v58b.png`) — 192 episodes, 3.9 h, cancelled. First run to cross
`none < 0.5` (ep38) and the first to absorb (ep67). `skip` rises to 0.215 and
`faces/mean_valid` falls 1.19 → 0.086. Raw entropy peaks early (1.222 @ ep26)
then falls to 0.139, while the de-diluted quotient goes the other way and
reaches its maximum at ep165 (1.706) — the two panels move in opposite
directions across the absorbed window. Quality drops from 0.885 to ≈0.03.
Value loss is small and steady (median 0.036).

**v59** (`run_v59.png`) — 50 episodes, 1.4 h, cancelled. Shortest run; `none`
only reaches 0.820, `faces/mean_valid` 1.22, quality 0.70 at the last episode,
both entropies still rising at cutoff.

**v60** (`run_v60.png`) — 508 rows, 8.6 h, ran to completion. Crosses
`none < 0.5` at ep71 and `< 0.1` at ep260; absorbs at ep98 and stays absorbed
for 405 episodes. `skip` ends at 0.462, `faces/mean_valid` at 0.026, quality at
exactly 0.0. Raw entropy ends at 0.038 while de-diluted ends at 1.48 and peaks
at 1.844 (ep343) — the widest raw-vs-de-diluted divergence in the campaign.

**v61** (`run_v61.png`) — 480 episodes, 36 h, ended on **walltime**. The
fastest and most complete absorption: `frac_violating` exceeds 0.5 at ep17,
`none < 0.5` at ep64 but the face population is already gone from ep35, and
from there `skip` = 1.000, `faces/mean_valid` = 0.011, both entropies ≈0 and
quality 0.0, unchanged for 445 episodes. λ pins at 10.0 and `frac_violating`
at 1.0. Value loss is large (median 1.01) but extremely smooth
(rolling-std median 0.012).

**v62** (`run_v62.png`) — 122 episodes, 5.7 h, cancelled. `frac_violating`
crosses 0.5 at ep55, `none < 0.5` at ep64, absorption only at ep118 — the
longest drift window (54 episodes) of group (B). λ rises off its floor to
12.53. Raw entropy actually peaks late (1.485 @ ep96, inside drift). Ends with
quality −0.085, `faces/mean_valid` 0.141. Its logged cost channel is the worst
of the pre-v66 runs (best 133.9 µs / 54.8 MB).

**v63** (`run_v63.png`) — 508 rows, 13.6 h, completed. The tightest sequence in
the campaign: `none` 0.9→0.5→0.1 in seven episodes (ep61, 64, 68),
`frac_violating > 0.5` at ep64, absorbed at ep83, then 420 further episodes in
the absorbed state with `skip` ≈0.20–0.26, `faces/mean_valid` ≈0.08–0.15, raw
entropy 0.02–0.14 and de-diluted entropy flat at 0.80–0.82.

**v64b** (`run_v64b.png`) — 508 rows, 12.7 h, completed. Same shape as v63,
shifted ≈10 episodes later (`none < 0.9/0.5/0.1` at ep62/71/79, absorbed ep93).
Lowest entropy maxima of any run (raw 1.025, de-diluted 0.841). Ends at
`skip` 0.315, `faces/mean_valid` 0.048, quality 0.000, `frac_violating` 1.000.

**v65** (`run_v65.png`) — LIVE, 203 episodes so far of a 500-episode budget.
Latest crossing of the campaign (`none < 0.9` at ep103, `< 0.5` at ep112) and
`none` has since sat at 0.42–0.48 rather than continuing to 0. `skip` ≈0.002–
0.006, `faces/mean_valid` 1.11–1.24, entropy raw 0.78–0.93 and de-diluted
0.72–0.88. Quality over the last 16 episodes ranges 0.34–0.80 (mean 0.59) and
`frac_violating` ranges 0.19–0.56 (mean 0.38) — the only run whose violating
fraction is not pinned near 1. Value loss median 0.362 with the second-highest
rolling-std (0.208).

**v66a** (`run_v66a.png`) — LIVE, 200 episodes. `none` reaches exactly 0.000 by
ep93 and stays there, but `skip` is 0.000–0.0016 and `faces/mean_valid` is a
flat 1.242 for the entire trailing window: the head emits an approximation on
essentially every slot while destroying no faces. Raw entropy is pinned at
0.86–0.88 and de-diluted at 0.70. Quality over the last 16 episodes swings
−0.28…+0.25 (mean −0.02); `frac_violating` 0.875–1.0. Value loss is ~150×
smaller than v63/v64b (median 0.0059).

**v66b** (`run_v66b.png`) — LIVE, 193 episodes. Same shape as v66a with a
higher entropy plateau (raw 1.03, de-diluted 0.83) and the least-degraded
quality of the three v66 arms (trailing mean 0.233). `frac_violating` 0.909.

**v66c** (`run_v66c.png`) — LIVE, 199 episodes. Same census and face behaviour
as v66a/b (none 0.000, skip 0.001, faces 1.229), highest entropy plateau of the
four live arms (raw 1.066, de-diluted 0.867), lowest quality (trailing mean
−0.105) and `frac_violating` pinned at exactly 1.000 for the whole trailing
window. Its latency channel is distinct: ≈300 µs against ≈179 µs for v66a/b,
a 1.7× difference held steadily over the last 16 episodes.

---

## 4. Status of the four LIVE arms, by the numbers

All four are at ep192–203 of 500 with ≈9.5 h elapsed of a 36 h limit. Reading
`approx_prob/skip`, `faces/mean_valid` and the **de-diluted** entropy together
(a falling raw-entropy panel would otherwise be ambiguous between commitment
and graph destruction):

| arm | verdict | evidence |
|---|---|---|
| **v65** | **DRIFTING, least advanced** | `none` 0.452 (crossed 0.5 only at ep112 — the latest in the campaign) and not falling further; `skip` 0.004; `faces/mean_valid` **1.188 = base**; H_ded **0.723** (near its own max 0.880); quality **0.594**; `frac_violating` **0.378**, the only arm not pinned near 1 |
| **v66a** | **DRIFTING, quality collapsed; graph NOT destroyed** | `none` **0.000** since ep93; `skip` **0.0003**; `faces/mean_valid` **1.242 ≥ base 1.189**; H_raw 0.873 / H_ded 0.703, flat, not collapsing; quality **−0.021**; `frac_violating` 0.969 |
| **v66b** | **DRIFTING, quality partly retained; graph NOT destroyed** | `none` **0.000**; `skip` **0.0003**; `faces/mean_valid` **1.242**; H_raw 1.015 / H_ded 0.818; quality **0.233**; `frac_violating` 0.909 |
| **v66c** | **DRIFTING, quality collapsed; graph NOT destroyed; cost channel 1.7× worse** | `none` **0.000**; `skip` **0.0012**; `faces/mean_valid` **1.229**; H_raw 1.066 / H_ded 0.867 (highest of the four); quality **−0.105**; `frac_violating` **1.000**; latency **300 µs** vs 179 µs for v66a/b |

**None of the four is absorbed** by the stated criterion, and none is close:
the absorbed threshold is `faces/mean_valid < 0.30` and all four sit at
1.19–1.24. This is the qualitative difference from v58b/v60/v61/v62/v63/v64b,
each of which had lost 88–99% of its live faces by this point in its own
trajectory. Equally, none of the four shows the raw-entropy collapse of group
(B): where v61/v63/v64b reached raw entropy 1e-05–0.12, the live arms are at
0.86–1.07 with de-diluted values of 0.70–0.87.

**v65 is the only arm still holding quality** (0.59 trailing mean, versus
−0.02 / 0.23 / −0.11), and it is also the only one whose `none` has stalled
part-way (0.45) rather than going to exactly 0.

**What the probability mass moved *to*.** The last `[approx]` census line of
each run shows where the vacated `none` mass ended up. The live arms put it
entirely into the three approximation ops; the completed group (B) runs put a
large share into `skip`:

| run | none | quant | diag | compress | skip |
|---|---|---|---|---|---|
| v65 (LIVE) | 0.421 | 0.102 | 0.301 | 0.172 | 0.004 |
| v66a (LIVE) | 0.000 | 0.380 | 0.404 | 0.216 | **0.000** |
| v66b (LIVE) | 0.000 | 0.398 | 0.388 | 0.215 | **0.000** |
| v66c (LIVE) | 0.000 | 0.337 | 0.344 | 0.319 | **0.000** |
| v63 | 0.000 | 0.336 | 0.295 | 0.172 | **0.197** |
| v64b | 0.000 | 0.306 | 0.224 | 0.155 | **0.315** |

v66c is the one live arm whose mix is compress-heavy (0.319 versus 0.215–0.216
for v66a/b), and it is also the arm carrying the ≈300 µs latency.

---

## 5. Figures

All at `/Users/assmuth/dsnn/run_analysis/figs/`, 150 dpi.

**Per-run flight recorders** — `run_v57.png` … `run_v66c.png` (12 files). Each
is an eight-panel grid over the run's episode axis: **(a)** stacked
`approx_prob/{none,quant,diag,compress,skip}` census with a dashed marker at
the first episode where `none < 0.5`; **(b)** raw `entropy/approx_head`
together with the de-diluted quotient `H ÷ faces/mean_valid`, plus
`entropy_floor/H_face` and, on a right axis, `entropy_floor/penalty`;
**(c)** `mean_quality`, `weighted_mean_quality`, `lagrangian/mean_raw_q` and
the best-so-far envelope; **(d)** mean and best-so-far latency (µs, left) and
peak memory (MB, right); **(e)** λ against `frac_violating`,
`mean_violation`, `mask_fraction`, `popart_frozen`; **(f)** PopArt μ and σ for
the quality / latency / mem channels on a symlog axis; **(g)** PPO, value,
total and per-channel value losses plus `kl/approx` on a log axis;
**(h)** `faces/mean_valid` with the `0.25·base` absorbed threshold and the
collapse / truncation counters. Every panel carries the same green / orange /
red phase ribbon (healthy / drift / absorbed), so a transition can be read
across all eight channels at one glance. The four live arms are titled
`[PARTIAL, LIVE as of ep N]`.

**(i) `cross_i_census_none.png`** — `approx_prob/none` for all 12 runs on one
axis, linear-x and log-x, with a ringed marker on each run's crossing point
(first episode with `none < 0.5`). The markers fall in three groups: v58b at
ep38, the lagrangian-era runs at ep64–71, and the four live arms at ep81–112;
v57 and v59 have no marker because they were stopped first.

**(ii) `cross_ii_quality.png`** — `mean_quality` (left) and
`lagrangian/mean_raw_q` (right) for every run that logs them. The completed
group (B) runs converge on 0.0 or slightly below and stay there; v65 is the
only trace holding above 0.5 at its current episode.

**(iii) `cross_iii_entropy.png`** — raw `entropy/approx_head` (left) against
the de-diluted quotient (right), same colours both panels. The two panels tell
different stories for the same runs: the raw traces of v60/v61/v63/v64b fall
to ≈0 while their de-diluted traces stay between 0.65 and 1.85, and only v61's
de-diluted trace also goes to zero.

**(iv) `cross_iv_value_loss.png`** — four panels: value loss (log y), its
rolling-10 standard deviation, the per-channel `value_loss/*` traces for the
runs that log them (v65, v66a/b/c), and value loss divided by
`|weighted_mean_return|` for that same run. The vertical spread is ≈3.5 orders
of magnitude: the v66 arms sit at 0.006 with rolling-std 0.002–0.003, v63/v64b
at 0.93 with rolling-std 0.02–0.04, and v57/v59 have the largest loss relative
to their own reward scale (ratios 22.1 and 9.2).

**(v) `cross_v_phase_timing.png`** — one horizontal bar per run, coloured
green / orange / red for healthy / drift / absorbed under the §0 criteria, with
the run's final episode count and end state annotated at the right. It makes
the group (B) versus group (C) split immediate: six bars go red between ep35
and ep118 and stay red; the four live bars are green-then-orange with no red at
all.

**(vi) `cross_vi_live_arms.png`** — the four live arms only, four panels:
`approx_prob/none`, `faces/mean_valid`, raw entropy and de-diluted entropy.
The `faces/mean_valid` panel is flat at 1.19–1.24 for all four across the whole
range, and the two entropy panels track each other rather than diverging.

---

## 6. Follow-up interrogation (normalized age, op mix, cost/quality, critic, quality signal)

Same descriptive rule as above. Raw output of this pass:
`run_analysis/deep_findings.md`; mined archives: `data/*_archive.csv`.

### 6.1 Group B vs group C at a **normalized age** (Q2)

Every run aligned on **its own** `none < 0.5` crossing, so the comparison is
like-for-like rather than at different run ages. `faces/base` is
`faces/mean_valid ÷ base_faces`.

| run | cross ep | +0: skip / faces-base | +20: skip / faces-base | +50: skip / faces-base |
|---|---|---|---|---|
| v58b | 38 | 0.060 / 0.589 | 0.102 / 0.270 | 0.169 / **0.107** |
| v60 | 71 | 0.021 / 0.681 | 0.051 / 0.557 | 0.173 / **0.180** |
| v61 | 64 | 0.396 / 0.026 | 1.000 / 0.009 | 1.000 / **0.009** |
| v62 | 64 | 0.006 / 0.925 | 0.004 / 0.983 | 0.100 / **0.433** |
| v63 | 64 | 0.002 / 0.963 | 0.154 / 0.198 | 0.221 / **0.070** |
| v64b | 71 | 0.004 / 0.869 | 0.113 / 0.422 | 0.667 / **0.013** |
| **v65** | 112 | 0.005 / 1.030 | 0.003 / 1.035 | **0.004 / 0.899** |
| **v66a** | 83 | 0.003 / 1.034 | 0.002 / 1.023 | **0.002 / 0.995** |
| **v66b** | 82 | 0.004 / 0.957 | 0.002 / 0.994 | **0.001 / 1.045** |
| **v66c** | 81 | 0.001 / 1.044 | 0.002 / 1.044 | **0.003 / 1.032** |

**At +20 the split is not clean** — v62 sits at `faces/base` 0.983 and
`skip` 0.004, i.e. indistinguishable from the v66 arms, and only loses its
faces later. **At +50 it is close to binary**: every group-B run is at
`faces/base` ≤ 0.433 (five of six at ≤ 0.180) and every group-C run at ≥ 0.899,
with an empty band between 0.433 and 0.899. On `skip` there is no continuum at
all at +50: group B spans 0.100–1.000, group C spans 0.001–0.004, a 25×–1000×
separation with nothing in between.

**Does v65 look like v66a/b/c or like v64b?** At the same normalized age (+50)
v65 is `skip` 0.0037 / `faces/base` 0.899; v66a/b/c are 0.001–0.003 / 0.995–
1.045; v64b is 0.667 / 0.013. On this normalized basis **v65 is with the v66
arms**, separated from v64b by 180× in `skip` and 69× in surviving faces —
even though v65 shares v64b's `adv_winsorize=3` / `advantage_norm=popart`
configuration.

### 6.2 What replaces `none` (Q3)

Mix as a fraction of the **non-none mass** `1 − p(none)`:

| run | +0 quant/diag/compress/skip | +20 | +50 | winner @+50 |
|---|---|---|---|---|
| v58b | .462/.308/.111/.119 | .499/.249/.094/.157 | .367/.231/.161/.241 | quant |
| v60 | .572/.287/.100/.041 | .509/.261/.154/.076 | .378/.234/.133/.255 | quant |
| v61 | .143/.078/.039/.740 | .000/.000/.000/1.00 | .000/.000/.000/1.00 | **skip** |
| v62 | .338/.223/.428/.011 | .274/.258/.463/.005 | .256/.304/.338/.102 | **compress** |
| v63 | .389/.286/.322/.004 | .345/.297/.204/.154 | .325/.297/.157/.220 | quant |
| v64b | .327/.329/.337/.007 | .322/.371/.194/.113 | .194/.111/.028/.667 | **skip** |
| v65 | .157/.503/.330/.009 | .128/.550/.317/.004 | .153/.537/.304/.006 | **diag** |
| v66a | .481/.464/.049/.006 | .435/.272/.291/.002 | .393/.385/.220/.002 | quant |
| v66b | .474/.478/.041/.007 | .457/.469/.072/.002 | .498/.375/.126/.001 | quant |
| v66c | .478/.474/.047/.002 | .394/.510/.094/.002 | .393/.524/.079/.003 | **diag** |

**The same op does not win everywhere.** All four ops win somewhere: quant in
v58b/v60/v63/v66a/v66b, diag in v65/v66c, compress in v62, skip in v61/v64b.

Observed co-variation with config (association only, no mechanism claimed):

* The two arms whose winner is **diag** by a wide margin are v65 (diag 0.537,
  quant 0.153) and v66c (diag 0.524). v65 is the only run with
  `advantage_norm=popart` **and** the raw-violation change; v66c is the only
  v66 arm with `--no-mask`.
* The three arms sharing `advantage_norm=none, adv_winsorize=0` (v66a/b/c,
  differing only in λ = 10 / 16 / 13 and the mask) start from near-identical
  mixes (quant ≈ 0.47–0.48, diag ≈ 0.46–0.48, compress ≈ 0.04–0.05 at +0) and
  diverge afterwards, so λ alone does not fix the winner.
* `skip` only ever wins in the two runs where the face population also
  collapsed (v61, v64b), and never in a run whose faces survived.
* Note the mix keeps moving after +50: v66c's compress share is 0.079 at +50
  but 0.319 in its **last** logged census, so the "+50 winner" is a snapshot,
  not a terminal state.

### 6.3 Joint cost/quality — is anything actually trading? (Q4)

Identity baseline per run = median over that run's own episodes with
`approx_prob/none > 0.95`. Across the 12 runs these are tightly clustered:
latency **154.8–160.2 µs** and quality **0.778–0.885**. Campaign-median
identity point: **157.5 µs, q 0.8125**.

**Yes — genuinely better cost/quality points exist, and they were found early.**
The Pareto archive dumped to STDOUT (39 entries each, recoverable only for
v60, v63, v64b) contains **26 plans that beat the identity point on both axes**
(latency lower, q ≥ 0.8). The strongest:

| run | ep | latency µs | quality | mem MB | vs identity |
|---|---|---|---|---|---|
| **v64b** | **3** | **69.8** | **0.8606** | 55.40 | **2.26× cheaper, q +0.048** |
| **v63** | **3** | **72.2** | **0.8606** | 55.40 | **2.18× cheaper, q +0.048** |
| **v60** | **239** | **76.8** | **0.8606** | 55.60 | **2.05× cheaper, q +0.048** |
| v63 | 113 | 78.2 | 0.8605 | 55.40 | 2.01× cheaper |
| v60 | 196 | 79.1 | 0.8188 | 55.60 | 1.99× cheaper |
| v60 | 142 | 83.5 | 0.8651 | 55.60 | 1.89× cheaper |
| v64b | 39 | 84.7 | 0.8660 | 55.40 | 1.86× cheaper |
| v64b | 71 | 87.2 | 0.8605 | 55.60 | 1.81× cheaper |

At the **episode-mean** level the same conclusion holds more weakly: every run
has at least 2 episodes whose mean latency is below its own identity latency
with mean quality ≥ 0.8. The single best episode mean is **v57 ep117 at
95.8 µs with mean quality 0.8824** — 1.62× cheaper than v57's own identity
(155.5 µs) at essentially unchanged quality.

Quadrant census of episode-mean points (cheaper = lat < own identity;
quality kept = q ≥ identity q − 0.02):

| run | down-left (real trade) | down-right | up-left | up-right |
|---|---|---|---|---|
| v57 | **46** | 36 | 33 | 47 |
| v58b | 2 | 178 | 4 | 5 |
| v59 | 9 | 18 | 12 | 8 |
| v60 | 9 | 442 | 12 | 37 |
| v61 | 4 | 457 | 3 | 13 |
| v62 | 12 | 26 | 14 | 67 |
| v63 | 15 | 433 | 17 | 35 |
| v64b | 13 | 426 | 18 | 43 |
| v65 | 25 | 24 | 34 | **117** |
| v66a | 22 | 16 | 24 | **135** |
| v66b | 22 | 16 | 23 | **129** |
| v66c | 24 | 13 | 21 | **138** |

Two distinct shapes: the completed group-B runs pile up **down-right** (426–457
episodes at lower latency with quality lost — the cheap-because-destroyed
corner), while the four live arms pile up **up-right** (129–138 episodes that
are *both* dearer and worse than their own identity). No run spends the
majority of its episodes down-left; v57 comes closest at 46/162 (28%).

The **latency** side of these points is the robust part (1.8×–2.3×). The
quality side should be read as "inside the identity band", not as an
improvement: q 0.8606 sits within the 0.778–0.885 spread that the identity
baseline itself takes across the 12 runs, so `q +0.048` against the campaign
median is not a quality gain, only evidence that quality was **not** given up.

Caveats on this subsection: archive latencies are single-plan measurements
taken when the plan was archived, not re-measured back-to-back against
identity, so the 1.8×–2.3× figures carry the usual unpaired-measurement risk;
the archive was only recoverable for 3 of 12 runs (the Top-N dump is printed
at run end, which the cancelled runs never reached), so the absence of archive
points for the other 9 is a **logging** absence, not evidence they found none.

### 6.4 Value loss: units, not accuracy (Q5)

The earlier observation that the v66 arms' `value loss` is ≈150× smaller than
v63/v64b **does not survive** a scale-free check. It is a change of units.

The target scales differ by orders of magnitude. `diag/estim_return_raw_latency`
has mean −8.2e4 and std 1.4e4 in v64b (raw ns) but mean −7.7 and std 0.72 in
v66a (the `advantage_norm=none` arms never leave the scaled space):

```
v63  V: -108074.7 -101735.0 -98020.8  ...  -76722.9 -78411.9 -79310.6
v63  R: -102066.0  -98853.6 -96332.6  ...  -76861.9 -77940.8 -78701.9
v66a V:      -0.142     -0.384    -0.809 ...    -2.104   -2.142   -2.170
v66a R:      -2.218     -2.450    -3.025 ...    -7.550   -7.820   -8.031
```

R² of `diag/value_raw_X` against `diag/estim_return_raw_X`, in raw units:

| run | quality R² | latency R² | mem R² | latency bias (v−r) | target std | bias in σ | latency R² after removing bias |
|---|---|---|---|---|---|---|---|
| v57 | 0.629 | 0.878 | 0.869 | −0.0007 | 0.011 | −0.06 | 0.883 |
| v60 | 0.997 | 0.875 | 0.851 | −0.0005 | 0.011 | −0.05 | 0.878 |
| v61 | 0.992 | 0.990 | 0.967 | 26.5 | 5599 | 0.005 | 0.990 |
| v63 | 0.994 | 0.996 | 0.964 | 28.7 | 12660 | 0.002 | 0.996 |
| v64b | 0.996 | 0.996 | 0.980 | 75.6 | 13950 | 0.005 | 0.996 |
| v65 | 0.915 | 0.982 | 0.904 | 33.5 | 6371 | 0.005 | 0.982 |
| **v66a** | 0.904 | **−58.3** | **−69.0** | **+5.557** | 0.725 | **+7.7σ** | **0.499** |
| **v66b** | 0.907 | **−56.9** | **−63.9** | **+5.490** | 0.725 | **+7.6σ** | **0.506** |
| **v66c** | 0.881 | **−54.7** | **−72.8** | **+5.598** | 0.753 | **+7.4σ** | **0.484** |

In scale-free terms the static arms' critic is **not** more accurate. On the
quality channel it is comparable (0.88–0.91 vs 0.92–0.996). On latency and
memory it is negative, driven by a systematic offset of 5.5–8.9 raw units
against a target standard deviation of 0.72–1.10 — a **7.4–8.2 σ** bias — and
even after that offset is removed it explains only 0.42–0.51 of the variance,
against 0.90–0.996 for every PopArt arm.

The independently logged `explained variance` (the in-update, per-sample
measure) points the same way on the ordering of the live arms:

| run | median | last-20 median |
|---|---|---|
| v57 | 0.498 | −1.062 |
| v58b | 0.061 | 0.059 |
| v59 | 0.336 | 0.546 |
| v60 | 0.0004 | −0.0004 |
| v61 | 0.005 | 0.005 |
| v62 | 0.884 | 0.636 |
| v63 | 0.028 | 0.030 |
| v64b | 0.008 | 0.014 |
| **v65** | **0.874** | **0.881** |
| v66a | 0.186 | 0.197 |
| v66b | 0.184 | 0.193 |
| v66c | 0.184 | 0.201 |

Note the two scale-free measures **disagree in level** for v63/v64b (logged EV
0.01–0.03 vs across-episode R² 0.996) and must not be conflated: the
across-episode R² asks whether the critic tracks the slow episode-to-episode
movement of returns (easy, because both share a trend), while the logged
`explained variance` asks whether it separates samples *within* an update. Both
nevertheless agree that the v66 arms' small `value loss` is not a smaller
critic error, and that v65 (0.874) sits well above v66a/b/c (0.184–0.186).

### 6.5 The quality signal: how noisy, and is it binary? (Q6)

`mean_quality` is an **episode mean over the envs**; no per-env quality key is
logged in any run, so within-episode spread across the 16 envs cannot be
recovered from wandb. The two proxies available are the distribution of
episode means and the mined per-plan archive qualities.

| run | n eps | frac q>0.8 | frac q<0.1 | frac MIDDLE (0.1<q<0.8) | lag-1 autocorr | lag-1, first 20 eps | min | max |
|---|---|---|---|---|---|---|---|---|
| v57 | 162 | 0.568 | 0.290 | **0.142** | 0.150 | **−0.021** | −1.000 | 0.885 |
| v58b | 189 | 0.032 | 0.720 | **0.249** | 0.838 | 0.355 | −0.253 | 0.885 |
| v59 | 47 | 0.447 | 0.000 | 0.553 | 0.435 | **0.001** | 0.318 | 0.885 |
| v60 | 500 | 0.042 | 0.826 | **0.132** | 0.925 | **0.001** | −0.392 | 0.885 |
| v61 | 477 | 0.019 | 0.916 | **0.065** | 0.917 | 0.854 | −0.209 | 0.885 |
| v62 | 119 | 0.193 | 0.462 | 0.345 | 0.949 | 0.073 | −0.539 | 0.885 |
| v63 | 500 | 0.062 | 0.852 | **0.086** | 0.959 | 0.117 | −0.263 | 0.885 |
| v64b | 500 | 0.062 | 0.850 | **0.088** | 0.965 | 0.077 | −0.332 | 0.885 |
| v65 | 200 | 0.250 | 0.015 | 0.735 | 0.488 | 0.078 | −0.002 | 0.885 |
| v66a | 197 | 0.223 | 0.492 | 0.284 | 0.907 | 0.160 | −0.770 | 0.885 |
| v66b | 190 | 0.221 | 0.226 | 0.553 | 0.827 | 0.160 | −0.358 | 0.885 |
| v66c | 196 | 0.219 | 0.291 | 0.490 | 0.894 | 0.161 | −0.279 | 0.885 |

Per-plan archive qualities (v60/v63/v64b, the only recoverable archives) show
the **same** bimodality without any averaging: middle-mass 0.128 / 0.205 /
0.103, with 0.487–0.564 of archived plans at q < 0.1 and 0.308–0.385 at
q > 0.8.

Two things are visible here. **(a) The signal is close to binary in the
completed runs**: middle-mass 0.086–0.142 for v57/v60/v63/v64b, i.e. 86–91% of
episode means sit either above 0.8 or below 0.1, and the per-plan archive
agrees. It is *not* binary everywhere — v65 (0.735), v66b (0.553) and v59
(0.553) put most of their mass in the middle band. **(b) For a policy that is
barely moving, the episode-to-episode signal is close to white**: over each
run's first 20 episodes the lag-1 autocorrelation of `mean_quality` is
−0.021 to +0.161 in 11 of 12 runs (v61's 0.854 is the sole exception). The high
whole-run autocorrelations (0.83–0.97) come from the long absorbed stretches
where quality is pinned at a constant, not from a smooth signal.

The value 0.885 is the maximum in **every one of the 12 runs**, and −1.000 is
hit exactly in v57 — the channel is bounded at both ends.

### 6.6 Anomalies noted while building the figures

Recorded as observations, without explanation:

* The best cost/quality plans in the recoverable archives were found at
  **ep3, ep9, ep19, ep22, ep39, ep42** — i.e. at the very start of the runs.
  v63 and v64b both list their best-quality entry as *ep42, q 0.8854,
  165.0 µs*, and both archives hold exactly 39 entries.
* The identity baseline is reproduced to within 3.5% across 12 runs
  (154.8–160.2 µs) and the peak-memory channel to within 1.5% (54.57–55.35 MB).
* v58b and v60 have their **de-diluted** entropy maximum deep inside the
  absorbed phase (ep165 → 1.706; ep343 → 1.844), above anything reached while
  healthy.
* v57 — the run with the most down-left episodes (46) and the single best
  episode-mean cost/quality point (ep117, 95.8 µs, q 0.882) — was cancelled
  after 41 minutes, the shortest run in the campaign.
* v61's first-20-episode quality autocorrelation (0.854) is an order of
  magnitude above every other run's (−0.02 to 0.16).
* `lagrangian/raw_adv_mean` and `raw_adv_min` exist **only** in v65; the
  `[lagrangian]` STDOUT line logs `raw_adv=nan` in v66a from ep0 onward.

## 7. Manifest

```
/Users/assmuth/dsnn/run_analysis/figs/run_v57.png
/Users/assmuth/dsnn/run_analysis/figs/run_v58b.png
/Users/assmuth/dsnn/run_analysis/figs/run_v59.png
/Users/assmuth/dsnn/run_analysis/figs/run_v60.png
/Users/assmuth/dsnn/run_analysis/figs/run_v61.png
/Users/assmuth/dsnn/run_analysis/figs/run_v62.png
/Users/assmuth/dsnn/run_analysis/figs/run_v63.png
/Users/assmuth/dsnn/run_analysis/figs/run_v64b.png
/Users/assmuth/dsnn/run_analysis/figs/run_v65.png
/Users/assmuth/dsnn/run_analysis/figs/run_v66a.png
/Users/assmuth/dsnn/run_analysis/figs/run_v66b.png
/Users/assmuth/dsnn/run_analysis/figs/run_v66c.png
/Users/assmuth/dsnn/run_analysis/figs/cross_i_census_none.png
/Users/assmuth/dsnn/run_analysis/figs/cross_ii_quality.png
/Users/assmuth/dsnn/run_analysis/figs/cross_iii_entropy.png
/Users/assmuth/dsnn/run_analysis/figs/cross_iv_value_loss.png
/Users/assmuth/dsnn/run_analysis/figs/cross_v_phase_timing.png
/Users/assmuth/dsnn/run_analysis/figs/cross_vi_live_arms.png
/Users/assmuth/dsnn/run_analysis/figs/cross_vii_normalized_age.png
/Users/assmuth/dsnn/run_analysis/figs/cross_viii_cost_quality.png
/Users/assmuth/dsnn/run_analysis/figs/cross_ix_critic_scalefree.png
/Users/assmuth/dsnn/run_analysis/figs/cross_x_quality_distribution.png
```

**(vii) `cross_vii_normalized_age.png`** — `approx_prob/skip`,
`faces/mean_valid ÷ base_faces` and `approx_prob/none`, with every run's x-axis
shifted so 0 is its **own** `none < 0.5` crossing. The middle panel shows the
group-B traces diving through the 0.25 line within 20–50 episodes of crossing
while the four live traces stay flat at ≈1.0 out to +90.

**(viii) `cross_viii_cost_quality.png`** — joint (latency, quality) points: on
the left every run's episode means, joined in time order; on the right the
per-plan Pareto-archive points mined from STDOUT. A star marks the campaign
identity point (157.5 µs, q 0.8125). The right panel shows three separated
clusters — ≈70–95 µs at q 0.78–0.87, ≈130–140 µs at q ≈ 0, and ≈142–185 µs at
q 0.885 — the first of which lies down-left of the star.

**(ix) `cross_ix_critic_scalefree.png`** — R² of V(s) against realized return
per channel, in raw units, one bar per run. The quality panel is uniformly
0.63–0.997; the latency and mem panels show the three v66 bars going sharply
negative while every other bar is 0.85–0.996.

**(x) `cross_x_quality_distribution.png`** — histogram of episode-mean quality
per run with the 0.1–0.8 "middle" band shaded and its mass printed in each
title. v57/v60/v63/v64b are visibly two-spiked at the extremes; v59/v65/v66b
fill the band.

Supporting artefacts (not figures): `run_analysis/data/*.csv` (per-run wandb
history), `data/*_log.csv` (STDOUT-mined telemetry), `data/*_archive.csv`
(mined Pareto archives), `data/meta.json`, `data/logmeta.json`,
`landmarks.json`, `landmarks.md`, `deep_findings.md`, `manifest.txt`, and the
scripts `pull_all.py`, `mine_logs.py`, `analyze.py`, `verify.py`, `deep.py`,
`q5b.py`.
