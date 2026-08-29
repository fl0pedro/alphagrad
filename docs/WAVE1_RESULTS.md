# WAVE 1 RESULTS — CONTRAST × PRICE

Four arms, 250/250 episodes each, all COMPLETED. Target `TransformerLM` /
`wikitext2`, 96 vertices (95 valid), order pinned (`ALPHAGRAD_FORCE_REV_ORDER=1`)
so only approximations are learned. 16 envs, terminal rewards only,
`--discount 1.0 --gae-lambda 1.0 --advantage-norm none --walk-rotate`,
`--reject-frozen-grads`, `--quality-metric grad_cosine`, `ALPHAGRAD_GRAD_COSINE_K=1`.

| arm | job | node | B | λ_acc | wandb | elapsed | sec/ep (last 50) |
|---|---|---|---|---|---|---|---|
| w1a | 62762 | pgi15-gpu15 | 6 | 170 | `phjgw3fv` | 05:06:14 | 67.4 |
| w1b | 62763 | pgi15-gpu16 | 5 | 170 | `fns79wak` | 04:55:59 | 65.9 |
| w1c | 62764 | pgi15-gpu17 | 4 | 170 | `jzw2jewg` | 06:36:13 | 89.7 |
| w1d | 62765 | pgi15-gpu18 | 5 | 16  | `57fn5ecb` | 04:50:35 | 63.1 |

Evidence base: the four A6 plan logs (`--plan-log auto`), **4048 terminal plans
per arm, 16192 total**, every one carrying all 11 reward slots, per-kind
applied/requested/skipped/idempotent counts and the coverage census; plus the
full wandb histories (446 keys, 253 rows/run; `_step = episode + 3`).

```
plan_log_w1a-bias6-lam170.jsonl   wandb/run-20260829_002317-phjgw3fv/files/
plan_log_w1b-bias5-lam170.jsonl   wandb/run-20260829_002316-fns79wak/files/
plan_log_w1c-bias4-lam170.jsonl   wandb/run-20260829_002316-jzw2jewg/files/
plan_log_w1d-bias5-lam16.jsonl    wandb/run-20260829_002317-57fn5ecb/files/
```
Derived CSVs and scripts: `/Users/assmuth/dsnn/run_analysis/w1{a,b,c,d}_hist.csv`,
`w1_analyze.py`, `w1_deep.py`, `w1_null.py`, `w1_skip.py`, `w1_rep.py`, `w1_reward.py`.

---

## VERDICT

**Selected: decision-table row 14** — *"B=5 shows spread > 1e-3 and ≥1 plan
outside the drift floor"* → **B=5 is the operating point. Export `W1_BIAS=5`;
run wave 2, then wave 3.**

Both conjuncts hold for B=5. Quality spread > 1e-3 in 48 of 250 episodes
(23/25 of ep0–24, 10/50 of ep50–99), and 85 of 1703 plans at grad-cosine ≥ 0.999
sit below w1b's own measurement null — including 6 in ep225–249, minimum 0.977.
The second conjunct also holds on the *other* B=5 arm, w1d, which took **zero**
compile fallbacks: 97 of 1796, minimum 0.899, present in every window to ep249.
So the row does not rest on the contaminated arm.

Row 14 is the only row whose antecedent the data satisfies, but it fires for a
reason the plan did not anticipate, and **two of its neighbours must be
overridden**: row 18 is *refuted* (λ is a live knob, so wave 3's price arms are
kept, not cancelled), and the B=5 selection is a **default, not a preference** —
the bias sweep did not separate the arms on anything that mattered. See
*Amendments* at the end.

### Why the other rows are excluded

* **Row 15** (*B=4 also shows contrast but worse latency at worse quality*) —
  half true, so it does not fire. B=4 **does** show contrast at worse quality
  (ep0–24 mean quality 0.822 vs w1a 0.961) — but its latency is the **best** in
  the battery, not the worst: w1c has 150 quality≥0.999 plans below its own null
  (8.7% of its high-quality plans, against 1% by construction), the most of any
  arm, and its late-window minimum ratios (0.966–0.974) are the lowest. The
  "more rules, no more speed" effect is *not* what happened.
* **Row 16** (*no B shows contrast*) — false. All four arms show quality spread
  > 1e-3 in 23–25 of the first 25 episodes, and all four contain plans outside
  their own measurement null at grad-cosine ≥ 0.999 **in every 25-episode window
  through ep249** (Panel 4). The registered falsifier — *"if none of a/b/c
  reaches spread > 1e-3 by ep50"* — is not met: all three reach it, repeatedly,
  well before ep50. **The NONE-bias hypothesis is not dead, and wave 3 is not
  cancelled.**
* **Row 17** (*`grad_cov/rejected_this_ep` ≈ `plan/n_live`*) — **excluded, on
  two independent counts.** Detail in Panel 2. The policy was not being refused.
* **Row 18** (*λ=16 ≈ λ=170*) — **refuted**, and in the informative direction.
  Detail in Panel 6.

---

## PANEL 1 — quality spread across the 16 envs (threshold 1e-3)

Within-episode `max − min` of reward slot 6 over the 16 plans of that episode,
per the `--walk-rotate` discipline (never across episodes).

Episodes with spread > 1e-3:

| window | w1a (B=6) | w1b (B=5) | w1c (B=4) | w1d (B=5, λ16) |
|---|---|---|---|---|
| ep 0– 24 | 23/25 | 23/25 | 24/25 | **25/25** |
| ep 25– 49 |  9/25 | 10/25 | 18/25 | **22/25** |
| ep 50– 99 |  9/50 | 10/50 | 15/50 | **32/50** |
| ep100–149 |  1/50 |  1/50 |  4/50 |  6/50 |
| ep150–199 |  3/50 |  4/50 |  8/50 |  9/50 |
| ep200–249 |  **0/50** | **0/50** | **0/50** | 1/50 |

Median spread is **exactly 0.000e+00 in every window from ep50 onward, in every
arm.** Spread at ep50 itself: 0.000 / 0.000 / 0.000 / 1.810e-02.

By ep200 quality has collapsed to a point mass: over ep200–249, quality is
**exactly 1.0 for all 800 plans** in w1a, w1b and w1c, and for 799/800 in w1d
(the exception is 0.97162 at ep247, an applied COMPRESS). Compare ep0–24, where
83 / 156 / 259 / 209 of 448 plans scored below 0.999.

Monotone in B at fixed λ, as predicted (9 < 10 < 15 over ep50–99) — but the
**λ arm dominates the bias sweep**: w1d holds contrast in 32/50 episodes where
w1c (a full bias step further out) holds it in 15/50.

**Reading.** Contrast is real early and decays to nothing. It does not decay to
the *identity plan* — see Panel 5, which is the substantive finding.

---

## PANEL 2 — `grad_cov/rejected_this_ep` against `plan/n_live` (row 17)

**`grad_cov/rejected_this_ep` is identically 0 in all 250 episodes of all four
arms; `plan/n_live` is identically 16. They are equal in 0 of 1000 episodes.**
Row 17 is excluded. Corroborated by the plan logs independently:
**0 of 16192 terminal plans carry `sentinelled: true`**, and
`plan_log/sentinelled_this_ep = 0` in every episode.

Verbatim from `w1a_hist.csv` (identical shape in b/c/d):

```
{'_step': '252', 'grad_cov/rejected_this_ep': '0', 'grad_cov/measured_this_ep': '0',
 'grad_cov/undefined_this_ep': '0', 'plan/n_live': '16',
 'plan_log/records_this_ep': '16', 'measure/grad_coverage/mean_ep': '1'}
```

The guard did not merely fail to fire — **it was never armed.** In all 16192
plan-log records, in all four arms, the coverage census reads:

```
"coverage":{"measured":true,"defined":false,"min_leaf_ratio":1.0,"frac_leaves_zeroed":0.0,
 "channel":1.0,"n_leaves":15,"n_counted":0,"n_uncounted":15,"n_uncountable":15,
 "n_zeroed":0,"zeroed":[],"leaf_ratios":["nan", ...×15],"exact_norms":["nan", ...×15],
 "leaf_mismatch":false,"leaves_recorded":15,"leaves_elided":0}
```

`n_leaves = 15` and `n_uncountable = 15` in **4048/4048 records per arm**. Every
leaf's *exact* reference norm is non-finite, so `_grad_coverage`
(`env.py:1037`) takes its `if not counted:` branch, returns `defined: False`
and pins slot 7 at the fallback `channel: 1.0`. `--reject-frozen-grads` is ON
and reachable (`env.py:6295` requires `_cov["defined"]`), so it can never
trigger. This is target-specific: the W0-C census on the NeuralNetwork stand-in
counted 15/15 leaves. On `TransformerLM` it counts 0/15.

Two candidate causes, both consistent with `_leaf_norms` (`env.py:974`) —
(a) every differentiated output leaf is 0-d, which the A4 leaf-set convention
marks `nan` by design; (b) the exact reference gradient itself is non-finite.
**Not resolved here** — it needs a targeted one-process probe, not a battery.
(b) would be far more serious, and slot 6's finite grad-cosines argue against
it, but they may not travel the same code path.

**Second, independent defect:** `grad_cov/measured_this_ep` also reads 0, while
the plan log shows `"measured": true` on all 16192 records. The counters are
incremented inside the Ray measure actors (`_record_grad_coverage`) but drained
in the trainer process (`ppo.py:11300`, `consume_grad_coverage_stats()`), which
never increments them. **The whole `grad_cov/*` wandb family is
process-mismatched and would read 0 even on a target where the guard works.**
The plan-log census is the only working readout; it worked, and this is its
first real use.

The wave-1 design leaned on this guard to make exploration safe
(*"made safe by the gradient-coverage guard so that exploring toward skips
cannot be rewarded for freezing gradients"*). **It was inert for the entire
battery.** Nothing in wave 1 was protected by it. Nothing was harmed by that
either — see Panel 5's v1/v2/v4/v8/v9 rows, which the *price*, not the guard,
kept out.

---

## PANEL 3 — `entropy/approx_head ÷ faces/mean_valid`

The quotient, never the raw numerator:

| ep | 0 | 5 | 25 | 50 | 100 | 150 | 200 | 249 | fall |
|---|---|---|---|---|---|---|---|---|---|
| w1a (B=6) | 0.0940 | 0.0784 | 0.0427 | 0.0207 | 0.0167 | 0.0127 | 0.0122 | 0.0117 | 8.0× |
| w1b (B=5) | 0.2076 | 0.1775 | 0.0825 | 0.0503 | 0.0280 | 0.0268 | 0.0274 | 0.0238 | 8.7× |
| w1c (B=4) | 0.4533 | 0.3450 | 0.1365 | 0.1902 | 0.2229 | 0.0835 | 0.1234 | 0.0978 | 4.6× |
| w1d (B=5,λ16) | 0.2076 | 0.1936 | 0.1656 | 0.0754 | 0.0439 | 0.0387 | 0.0427 | 0.0419 | 5.0× |

`faces/mean_valid` is flat: 1.13–1.20 at ep0, pinned at **1.23158** from ~ep25
to ep249 in every arm (full-run range 0.872–1.232, entirely inside the first
~25 episodes). **The fall is entirely in the numerator — a genuine entropy
collapse, not the graph-destruction signature v64b showed.**

Ordering is monotone in the bias as designed (w1c ≫ w1d > w1b > w1a) and λ
moves it at fixed bias (w1d 1.8× w1b at ep249). `entropy/ve_head`,
`entropy/macro_vertex` and `entropy/macro_vertex_norm` are 0 in every episode
of every arm, confirming the order pin.

---

## PANEL 4 — `approx_prob/none`, and per-plan ratios against the null

### `approx_prob/none`

| arm | ep0 | ep25 | ep50 | ep100 | ep200 | ep249 | first ep > 0.99 |
|---|---|---|---|---|---|---|---|
| w1a | 0.99196 | 0.99822 | 0.99947 | 0.99911 | 0.99929 | 0.99929 | **ep0** (starts above) |
| w1b | 0.98061 | 0.99537 | 0.99767 | 0.99893 | 0.99840 | 0.99785 | ep11 |
| w1c | 0.95562 | 0.99141 | 0.98629 | 0.98362 | 0.99110 | 0.99306 | ep23 |
| w1d | 0.98061 | 0.98819 | 0.99572 | 0.99840 | 0.99715 | 0.99769 | ep28 |

The registered prediction that B=6 parks with `none` > 0.99 holds — but so do
B=5 and B=4, just later. **Lowering the bias from 6 to 4 buys ~23 episodes of
exploration and a ~0.6 % residual approximation probability. It does not buy a
different endpoint.** `approx_prob/skip` reaches exactly 0 by ep25–50 in every
arm and stays there apart from 5.3e-4 blips.

### The measurement null actually available

The plan's protocol null is **1.0007 ± 0.0008**, back-to-back. **The plan log
cannot supply back-to-back pairs.** The tightest honest pairing it supports is
*candidate against identity plans measured in the same episode by the same
measure actor*. Its null, derived from this battery's own identity-vs-identity
pairs at that granularity:

| arm | n | p1 | p50 | p99 | min | sd |
|---|---|---|---|---|---|---|
| w1a | 3001 | 0.9858 | 1.0000 | 1.0147 | 0.9789 | 0.0034 |
| w1b | 1964 | 0.9818 | 1.0000 | 1.0168 | 0.9790 | 0.0049 |
| w1c |  149 | 0.9820 | 1.0000 | 1.0106 | 0.9812 | 0.0046 |
| w1d | 1626 | 0.9868 | 1.0000 | 1.0090 | 0.9795 | 0.0031 |

**Every ratio in this document uses that self-derived null, not 1.0007 ±
0.0008.** It is 4–6× wider, and using the narrow one would have been the
unpaired-comparison error the plan forbids.

### Plans at grad-cosine ≥ 0.999 below their arm's own p1 null edge

`n_below / n_at_quality≥0.999`, and the minimum ratio in the window:

| window | w1a | w1b | w1c | w1d |
|---|---|---|---|---|
| ep 0– 24 | 21/215 min 0.914 | 25/189 min 0.902 | 12/56 min 0.906 | 30/115 min 0.902 |
| ep 25– 49 |  4/115 min 0.927 | 18/247 min 0.918 | 28/152 min 0.920 | 29/260 min 0.899 |
| ep 50– 74 |  1/93 min 0.982 |  4/215 min 0.980 |  0/15 min 0.993 |  9/218 min 0.920 |
| ep 75– 99 |  0/73 min 0.990 |  3/185 min 0.973 |  1/29 min 0.981 |  4/205 min 0.968 |
| ep100–124 |  2/75 min 0.969 |  4/151 min 0.972 | 14/173 min 0.967 |  6/168 min 0.967 |
| ep125–149 |  1/68 min 0.984 | 11/147 min 0.975 | 21/310 min 0.966 |  5/154 min 0.918 |
| ep150–174 |  0/56 min 0.996 |  5/150 min 0.978 | 23/305 min 0.966 |  3/182 min 0.978 |
| ep175–199 |  1/72 min 0.983 |  5/164 min 0.976 | 14/173 min 0.970 |  5/186 min 0.983 |
| ep200–224 |  1/72 min 0.983 |  4/154 min 0.976 | 17/251 min 0.974 |  2/191 min 0.978 |
| ep225–249 |  3/67 min 0.981 |  6/142 min 0.977 | 20/289 min 0.967 |  4/188 min 0.981 |

Run totals: 34/873 (3.9 %), 85/1703 (5.0 %), 150/1728 (8.7 %), 97/1796 (5.4 %)
— against **1 % by construction** (the threshold is the null's own p1). Every
arm carries a real excess, in every window, to the last episode.

**The `best:` field of the progress bar is a per-channel best-so-far and is
not a coherent plan.** Its quality 1.00 / sparsity 0.00 / latency ≈1.27e5 final
line is the identity plan's *cost slots* pasted next to another plan's quality
slot. Verified against the plan log rather than inherited.

---

## PANEL 5 — WHAT THE POLICY ACTUALLY CONVERGED TO (not in the plan)

### The arms do not park at the identity plan

| | w1a | w1b | w1c | w1d |
|---|---|---|---|---|
| identity plans (nothing requested) / 4048 | 3035 | 2049 | **302** | 1733 |
| requested approximations per plan | 0.29 | 0.76 | **3.46** | 0.99 |
| requested: diag / compress / quant / skip | 267/290/613/141 | 587/621/1865/317 | 7857/1588/4554/660 | 1064/987/1950/466 |
| **applied**: diag / compress / quant | **3**/57/548 | **6**/108/1707 | **16**/181/4284 | **20**/188/1773 |
| applied quant in ep200–249 | 102 | 279 | 659 | 369 |

w1c requests 34–51 approximations *per episode* at ep249 and applies ~20 of
them. **The provisional read that "all four arms end at the identity plan" is
wrong.** They end at a **quality- and cost-neutral QUANT plan**: quality
exactly 1.0, ratio at the null. Under `GRAPHAX_QUANT_PULLDOWN=0` (pullup) the
surviving action is an exact no-op in both channels.

Two structural facts fall out:

* **DIAG applies essentially never.** Apply rate against requests:
  1.1 % / 1.0 % / **0.2 %** / 1.9 %. In w1c's ep200–249 alone, 840 DIAG
  requests → 594 refused by the applier → **0 applied**. These arms ran with
  `ALPHAGRAD_PER_FACE_MASKS=0`, which is the flag W0-B P2 exists to test, so
  this is consistent with — not yet a substitute for — decision rows 5/6.
* **The memory channel is dead.** `plan/peak_memory/median_ep` is −5.139e7 at
  every episode of every arm; full-run range across all four arms is
  −5.145e7 to −5.135e7, a **0.2 %** spread. Peak memory is byte-identical
  between the identity plan and every skip plan in Panel 5's table. As the plan
  already states, this channel is only alive across *orders* — i.e. in wave 2.

### The per-face map: one SKIP, priced

Every plan whose wire is exactly one live face carrying the skip flag, pooled
over all four arms, keyed on the **exact recorded plan wire** (face rows + rule
specs), ≥3 measurements. `Δsymlog(lat)` is the reward gained on the latency
head; `NET` is `Δsymlog(lat) − λ·(1−q)`, the whole change in the terminal
reward that skipping that one face buys.

| vertex | n | ratio (median) | grad-cosine | Δsymlog(lat) | NET @λ=170 | NET @λ=16 |
|---|---|---|---|---|---|---|
| **v84** | 7 | **0.9184** | **0.999737** | +0.0851 | **+0.040** | **+0.081** |
| **v91** | 5 | 0.9698 | 0.999944 | +0.0306 | +0.021 | +0.030 |
| **v93** | 7 | 0.9674 | 0.999900 | +0.0332 | +0.016 | +0.032 |
| **v90** | 6 | 0.9878 | 0.999969 | +0.0123 | +0.007 | +0.012 |
| v85 | 3 | 0.9988 | 1.000000 | +0.0012 | +0.001 | +0.001 |
| v10 | **39** | 1.0000 | 1.000000 | −0.0000 | −0.000 | −0.000 |
| v86 | **18** | 1.0007 | 1.000000 | −0.0007 | −0.001 | −0.001 |
| v46 | **35** | 1.0014 | 1.000000 | −0.0014 | −0.001 | −0.001 |
| v80 | 7 | 0.9496 | 0.969584 | +0.0517 | −5.12 | −0.44 |
| v62 | 7 | 0.8402 | 0.960651 | +0.1741 | −6.52 | −0.46 |
| v79 | 4 | 0.8825 | 0.964807 | +0.1250 | −5.86 | −0.44 |
| v60 | 3 | 0.8213 | 0.778308 | +0.1969 | −37.5 | −3.35 |
| v25 | 6 | 0.5855 | 0.653315 | +0.5352 | −58.4 | −5.01 |
| v23 | 7 | 0.5615 | 0.539375 | +0.5772 | −77.7 | −6.79 |
| **v20** | 4 | **0.5049** | **0.406619** | +0.6834 | −100.2 | −8.81 |
| v1,v2,v4,v8,v9 | 3–8 ea. | 0.994–1.000 | **0.000000** | ≈0 | −170.0 | −16.0 |

Four separable classes, none of them anticipated as a *map*:

1. **Profitable skips — v84, v91, v93, v90.** v84 costs **8.2 % of latency for
   2.6e-4 of gradient cosine.** Net reward **+0.040 at λ=170**, the largest
   single action available in the whole battery.
2. **The known SKIP cliff — v20, v23, v25.** 40–50 % of latency for 35–59 % of
   quality: the graph is DCE'd. Correctly priced out at both λ.
3. **Free destruction — v1, v2, v4, v8, v9.** Quality **exactly 0.0** at ratio
   0.994–1.000: they annihilate the gradient and save nothing. These are the
   plans `--reject-frozen-grads` was built to refuse, and the guard was inert
   (Panel 2) — the λ price alone kept them out, at −170.
4. **Exact no-ops — v10, v46, v86, v85.** Ratio 1.0000, quality 1.000000, net
   reward ±0.001.

**Cross-arm replication on the exact wire.** 318 (episode, plan-wire) keys were
measured on ≥3 of the four nodes. Examples, one plan measured on three or four
different GPUs of three or four different nodes:

```
ep 32  q=0.999944  skip v91 :  w1a(gpu15)=0.9698  w1b(gpu16)=0.9708  w1c(gpu17)=0.9688
ep121  q=0.999900  skip v93 :  w1a(gpu15)=0.9692  w1b(gpu16)=0.9718  w1d(gpu18)=0.9668
ep153  q=0.956207  skip v52 :  w1a=0.9668  w1b=0.9669  w1c=0.9872  w1d=0.9687
ep  7  q=0.539159  skip v23 :  w1a(gpu15)=0.5827 w1b(gpu16)=0.5639  w1d(gpu18)=0.5538
```

The v91 triple agrees to **0.2 %** across three nodes. The v84 class is measured
five times on the two arms with **zero** compile fallbacks (w1a: 0.9143, 0.9194;
w1d: 0.9111, 0.9184, 0.9200), so it does not rest on the contaminated arms.
Against a null of sd 0.0031–0.0034 on those arms, 0.918 is ~24 σ.

### The policy chose the no-ops over the wins

Single-face-skip occurrences, all arms pooled:

| window | profitable (v84,90,91,93) | exact no-op (v10,46,85,86) | other |
|---|---|---|---|
| ep 0– 49 | **18** | 9 | 111 |
| ep 50– 99 | 2 | 17 | 20 |
| ep100–149 | 4 | 11 | 5 |
| ep150–199 | **0** | 36 | 9 |
| ep200–249 | **1** | **22** | 0 |

v10 appears 39 times, ep95–243; v46 35 times, ep12–232; v86 18 times, ep0–212.
**The policy found the +8 % action repeatedly (v84 at ep9 … ep146, across three
arms) and converged away from it, onto the faces whose reward contribution is
exactly zero.**

### Why — the scale arithmetic

The composition, quoted verbatim from every run log:

```
[cfg] --symlog-channels cost: latency/memory symlog'd, QUALITY channel RAW.
Additive composition is 1*symlog(lat) + 1*symlog(mem) + 170*q_raw.
```

The cost slots are stored negated (slot 2 = −latency_ns, slot 5 = −peak_memory),
so their symlog terms enter negative. For the identity plan:
`−11.77 − 17.76 + 170·1.0 = 140.5`. The best single
action in the action space is worth **+0.0403**, i.e. **0.029 % of |R|**.
At λ=16 the same action is worth +0.081 on |R| = 13.5 → **0.298 %, 10× the
signal-to-offset ratio.**

The direct corroboration is in wandb: **`explained variance` is negative in
every episode of all four arms** (run medians −1.22 / −3.05 / −2.94 / −3.60;
worst −20627). The critic never beat a constant predictor. With
`--advantage-norm none`, the advantage that must carry a 0.029 % perturbation
is drawn from a value function that cannot resolve the mean. `kl/approx` ends
at 1e-8–3e-8 and `ratio/max_log` at 8e-4–2e-3: the policy stopped moving.

**Mechanism (derived, not measured):** λ·q_raw contributes λ·1.0 of *constant*
to every terminal reward and λ·δ of *signal*, with δ ≈ 2.6e-4 for the best
action. Raising λ raises the offset far more than the contrast. The
prescription "match the pricing PopArt actually realized, λ ≈ 134–199" priced
the *gradient* of quality against cost correctly and did not account for q
sitting at ≈1.0, where the term is almost entirely a constant the critic has to
learn before any of it is usable.

---

## PANEL 6 — w1b vs w1d: the price, at fixed bias (row 18)

Same B=5. **They differ, clearly, and only w1d has zero compile fallbacks**, so
the comparison is made on the fallback-independent channels (quality, entropy,
action counts) and *not* on cross-arm latency.

| | w1b (λ=170) | w1d (λ=16) |
|---|---|---|
| episodes ≥50 with quality spread > 1e-3 | 15 | **48** |
| ep50–99 episodes with spread > 1e-3 | 10/50 | **32/50** |
| entropy quotient at ep249 | 0.0238 | **0.0419** (1.8×) |
| entropy quotient, run median | 0.0298 | **0.0433** |
| identity plans, ep0–49 | 172/848 | **83/848** |
| requested skips, ep0–49 | 248 | **361** |
| requested approximations per plan (run) | 0.76 | **0.99** |
| plans with quality < 0.999, ep50–99 | 10 | **40** |
| plans with quality < 0.999, ep150–199 | 4 | **11** |
| `approx_prob/none` first > 0.99 | ep11 | ep28 |
| Pareto archive size at ep249 | 18 | **22** |

**Row 18's antecedent is false: λ=16 is not λ=170.** The plan's falsifier —
*"if d is indistinguishable from b, λ is not a live knob and the {130,170,210}
relaunch is closed as answered"* — does not fire. λ is the strongest knob in
this battery: at fixed bias it more than triples the number of post-ep50
episodes that retain contrast, more than a whole step of the bias sweep buys
(w1c, B=4 at λ=170, manages 27).

**Direction.** The A5 arithmetic argued λ=16 under-prices quality ~10×. On the
evidence here, λ=16 is the *better* of the two settings on every contrast
measure, and the scale argument in Panel 5 says why. **This inverts the sign of
the plan's λ recommendation.** No cross-arm latency conclusion is drawn: w1b is
51 % contaminated by degraded-fusion compiles and w1d is 0 %, and the
contamination biases in the direction opposite to the raw reading.

---

## MEASUREMENT CAVEATS

### The nvlink compile fallback is an arm-asymmetric confound

Root cause: CUDA driver newer than toolkit on **pgi15-gpu16 and pgi15-gpu17**
only. Verbatim (w1b):

```
E0829 00:29:45.562679 3417192 gpu_compiler.cc:2450] The CUDA linking API did not work.
... Original error: INTERNAL: nvlink exited with non-zero error code 256, output:
nvlink fatal : Input file '/tmp/tempfile-pgi15-gpu16...cubin' newer than toolkit (129 vs 128)
```
```
w1b_bias5_lam170_62763.log:10553  (CpuApproximationActor pid=3417192)
[measure] compile FALLBACK #649 (degraded fusion) after: INTERNAL: nvlink exited with
non-zero error code 256, ... 'newer than toolkit' [repeated 3x across cluster]
```

`#649` is the running fallback **counter**, not an error code. Printed line
counts undercount because Ray dedups; the reliable figure is the per-actor
counter maximum:

| arm | `nvlink` lines | fallback events (Σ per-actor max) | share of the 4000 measurements |
|---|---|---|---|
| w1a | **0** | **0** | 0 % |
| w1b | 1228 | 630 + 769 + 649 = **2048** | ~51 % |
| w1c | 1298 | 1473 + 1201 + 1221 = **3895** | ~97 % |
| w1d | **0** | **0** | 0 % |

Four independent greps (`nvlink`, `129 vs 128`, `gpu_compiler.cc:2450`,
`compile FALLBACK`) all return 0 for w1a and w1d.

The fallback is `env.py:_compile_measure()`'s own retry at
`xla_gpu_autotune_level: 0`, Triton GEMM off, runtime fusion off. Its docstring:
*"The fallback executable is less fused, so its latency reads conservatively —
a real measurement, not a sentinel."* So affected measurements are real and
systematically **pessimistic**.

**Wall time: no detectable cost.** w1b took 2048 fallbacks and is the second
*fastest* arm (65.9 s/ep last-50) — faster than w1a, which took none. w1c's
extra 1.5 h tracks its approximation workload (run-median
`approx_requested/total` 45.5 vs 9–12), not its fallbacks.

**Measurements: a real confound, contained.** All ratios here are within-arm
(candidate against identity in the same episode and actor), so the bias
partly cancels and what leaks shows up as the wider null on the contaminated
arms (w1b sd 0.0049, w1c 0.0046 vs w1a 0.0034, w1d 0.0031) — which the p1-edge
criterion already absorbs. The load-bearing v84/v90/v93 results replicate on
w1a and w1d, both at 0 %. **No cross-arm absolute latency comparison is made
anywhere in this document.**

*(Note: the logs contain only three `[health epN] ... sec/ep=` lines each —
eps 0, 1, 2 — so wall time came from wandb `time/sec_per_episode` and `sacct`,
not from the logs.)*

### Nothing else went wrong

`faces_truncated > 0`: 0/16192. `replayable: false`: 0/16192.
`collapse/{count,truncated,untraceable,zero_work}_this_ep`,
`tokenization/{truncated_count,overflow_sum_this_ep}`, `oracle/probe_failures`,
`plan_log/dropped_this_ep`, `per_face/skipped_raised`,
`measure/peak_memory_static_fallback`: all 0 in all 1000 episodes.
`plan_log/actors_polled = 3`, `records_this_ep = 16` throughout — the measure
pool stayed live for all four runs.

---

## WHAT THE PLAN DID NOT ANTICIPATE

1. **The action space has a clean, measurable per-face price map** (Panel 5) —
   four profitable skips, a three-face cliff, five free-destruction faces, four
   exact no-ops — and it replicates across nodes on the exact plan wire. This is
   the plan log's first real use and it is the most useful artefact wave 1
   produced.
2. **The convergence point is a QUANT no-op, not the identity plan.** w1c
   applies 659 quant rules in its last 50 episodes at quality exactly 1.0 and
   ratio 1.000.
3. **The coverage guard was inert for the whole battery**, on this target
   (`defined: false`, 16192/16192), *and* its wandb telemetry is drained in the
   wrong process. The wave's safety argument rested on it.
4. **The reward's constant offset, not the bias, is what killed learning.**
   0.029 % of |R| for the best action at λ=170, with `explained variance` < 0 in
   every episode of every arm.
5. **λ dominates B.** The plan treated λ as the secondary knob (one arm, "the
   price") and B as the contrast knob. The data says the reverse, and says the
   λ recommendation points the wrong way.
6. **A5's approximation-rate arithmetic is ~6–9× high.** Predicted 2.6 / 7.2 /
   19.5 approximations per plan at B = 6/5/4; observed **0.29 / 0.76 / 3.46**.
   The instruction *"do not sweep B wider — B=4 is already 19.5 approx/plan,
   past the band every win lives in"* rests on that arithmetic. B=4 is at 3.46,
   inside the 1–15 band, not past it.
7. **`plan/n_live` is not a live-face count** — it is pinned at 16, the env
   count, and equals `plan_log/records_this_ep`. Panel 2's comparison is
   "rejections against plans logged", which is still the right test for row 17.
8. **`plan/sparsity*` does not exist as a wandb key** (only `mean_sparsity` and
   `measure/sparsity/*`), and the PPO policy loss is logged as `ppo loss`, with
   a space.

---

## NEXT ACTIONS

Row 14's prescription, plus the amendments the evidence forces.

1. **Export `W1_BIAS=5` and run wave 2 as scheduled.** Wave 2 runs regardless
   of wave 1 and is now clearly the main line: the order axis is the only place
   the memory channel is alive (0.2 % across every plan in this battery), and
   nothing here contradicts that. **B=5 is a default, not a preference** — the
   bias sweep separated the arms only on how many approximations they requested.
2. **DO NOT cancel wave 3's price arms.** Row 18 is refuted. Re-point the
   sweep: the informative direction is **below** 16, not the plan's
   {16, 130, 400}. λ=400 will park harder, for the reason in Panel 5, and costs
   a node to confirm something already implied.
3. **Re-base the reward before wave 3, and preferably before wave 2's analysis.**
   Reward the *paired deltas* — `−symlog(lat/lat_exact)`, `−symlog(mem/mem_exact)`,
   `λ·(q−1)` — so the identity plan scores ≈0 and the +0.040 of v84 is the whole
   signal rather than 0.029 % of it. This is the same paired-ratio discipline
   the plan already mandates for *reporting*, applied to the *reward*, and it is
   the same failure recorded in `project_ppo_uniform_policy_fixed_point`.
4. **Confirm the v84 / v91 / v93 single-skip plans properly**, back-to-back,
   warm, `--latency-inner-reps 50`, 5 seeds, against exact reverse — one GPU,
   a few hours. Wave 1's pairing granularity (same episode, same actor) is not
   the protocol's. If v84 holds at ~0.92 with grad-cosine 0.99974, it is the
   first genuine approximation win on this target and it feeds decision rows 4
   and 8 directly.
5. **Fix the `grad_cov/*` process mismatch** (counters incremented in the Ray
   measure actors, drained in the trainer) and **diagnose why the coverage
   census counts 0/15 leaves on `TransformerLM`** when it counts 15/15 on the
   NeuralNetwork stand-in. Until then `--reject-frozen-grads` is a no-op on this
   target and must not be cited as a safety property. One-process probe, no
   battery.
6. **Fix or disable the compile fallback on pgi15-gpu16 / pgi15-gpu17** (driver
   129 vs toolkit 128) before any wave that compares arms on latency. Wave 2
   assigns gpu16 and gpu17 to `w2b` and to `w2c`, **the decisive arm**.

## WHAT THIS BATTERY DOES NOT SHOW

Four single-seed screens. No distribution is claimed and none is available; the
plan forbids it and this document does not do it. The quality-spread collapse is
consistent across all four arms and is the strongest negative here. The v84
result is a *replicated observation across four measurement contexts*, not a
5-seed confirmation, and item 4 above is what would make it one.
