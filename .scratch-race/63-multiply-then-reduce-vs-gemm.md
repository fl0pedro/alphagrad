# 63 - Multiply-then-reduce against a GEMM (ticket dsnn-3qm.28.4)

Plain English. Short sentences. Every cost number is a ratio. Every ratio was
measured in one process, back to back, with a drift floor beside it.

## The question

A single implicit sparse axis is the meta axis of a diagonal pair. Exactly one
operand stores it. The engine has three ways to emit its contraction.

1. **Broadcast-in-dot.** Broadcast the side that does not store the axis. Then
   call `dot_general` with the axis in the batch list. The incumbent form.
2. **Kept on the storing side.** The axis rides as a free axis of the operand
   that stores it. The contraction is a clean 2-D dot. This is the planner's
   form and `GRAPHAX_TILED_LAZY=full`.
3. **Multiply-then-reduce.** Never emit a dot. `sum(lhs * rhs, axis=k)` over
   the broadcast shapes. On GPU this is the form XLA already rewrites (1) into.

The owner asked what (3) costs against a GEMM. The channels are latency,
static temp, the runtime watermark, the kernel census and the values.

## Verdict

1. **Multiply-then-reduce is not free, and its price is memory at a large
   multiply grid.** In the isolated sweep it allocates nothing: static temp is
   zero in 80 of 80 CPU cells and in 56 of 80 GPU cells. Inside the real
   engine that stops being true. On TLM at the campaign shape it needs 18.00
   MB of temp on the reverse order, against 394 kB for the 2-D dot. On the
   Markowitz order it asks for 3.3 TB on CPU and runs out of memory. Above
   some grid size XLA stops fusing the product into the reduce, and then it
   writes the whole grid.
2. **On GPU it is the same program as the incumbent.** On TLM reverse it
   matches emission (1) on static temp to the byte, on Triton temp to the
   byte, on fusion count and on cuBLAS count. Its gradients are bit-identical
   to emission (1) on the exact and Reduce plans. This is grill2/F2 section 1
   confirmed inside the engine.
3. **The 2-D dot costs about 10 percent on GPU and buys everything on CPU.**
   TLM reverse: emission (2) reads 1.098 on exact and 1.078 on Quant against
   emission (1), reproducing T28B-RESULT.md section 4. On CPU the same
   emission reads 0.058 on exact: 17 times faster than the incumbent, 3.8
   times faster than multiply-then-reduce.
4. **At a small multiply grid both devices prefer multiply-then-reduce.** On
   NeuralNetwork/CPU it is the fastest of four arms in 5 of 6 cells. In the
   sweep it is the best emission in 52 of 80 GPU cells and its regret is
   1.000 median. On NeuralNetwork/GPU every arm is inside the drift floor.
5. **A GEMM costs 32 MiB of workspace per executable on GPU.** In the sweep
   both dot forms carry 33 554 688 B in almost every cell. Multiply-then-reduce
   makes no cuBLAS call, so it reserves nothing: 0.5 MB across all 80 cells
   against 2.68 GB. It is also the only arm that never raised the runtime
   watermark.
6. **A GEMM also costs copies.** On TLM reverse exact, emission (2) has 28
   cuBLAS calls and 54 `wrapped_slice` copy kernels. Emissions (1) and (3)
   have 9 and 0.
7. **Every emission computes the same gradient.** All three are within 1.1e-06
   of `jax.grad` on TLM/CPU exact and within 5.7e-04 on TLM/GPU exact, the
   same float32 reduction-order size finding 61 measured. All three return
   parameter layout on every non-Reduce plan.
8. **Emission (1) can be deleted.** It is byte-for-byte emission (3) on GPU,
   and on CPU it is dominated: sweep regret 2.031 median with a worst cell of
   83.30, and it is the only emission whose CPU static temp is never zero.

## Apparatus

Code: graphax `wip/t284-20260906` at `d4b9c49`, alphagrad `wip/t284-20260906`
at `6888e9c7`. The graphax change adds one race-only knob,
`GRAPHAX_TILED_MULREDUCE`, default OFF. It selects emission (3) at the single
frame-contraction site of `_execute_block_sparse_contraction`
(`src/graphax/sparse/ops/matmul.py`). No default changes. `mul_reduce_test.py`
pins the emission against `lax.dot_general` and pins the flag-off jaxpr.

Probes: `alphagrad/.scratch-race/probes/t284/`.

| job | node | what |
|---|---|---|
| 63828 | pgi15-gpu15 | the synthetic sweep, GPU |
| 63829 | pgi15-cpu1 | the synthetic sweep, CPU |
| 63830 | pgi15-gpu15 | the engine probe, GPU |
| 63831 | pgi15-cpu1 | the engine probe, CPU |

Jobs 63828 and 63829 also ran the engine probe and it died at import. The
campaign SHARED_ENV still exports `ALPHAGRAD_FORCE_REV_ORDER`, which ticket
dsnn-3qm.64 deleted from the code, so `masks.py` raises. The probe now drops
that variable. Their sweep halves are the data of section (a).

## Deliverable (a): the synthetic sweep

One contraction. Meta extent M, block extent P, contracted extent K. Output
extent Q is fixed at 32. The grid is finding 62's, 80 cells, with emission (3)
added. M runs over 4, 16, 64, 256 and 1024. P runs over 1, 4, 16 and 64. K
runs over 4, 16, 64 and 256.

Every cell compiles all three emissions and a second executable of (2). That
second executable is the drift floor. All four are timed in the same rounds,
back to back.

The **multiply grid** below is M times P times K times Q. It is the number of
scalar products the contraction performs. Emission (3) writes that many
multiplies. A GEMM performs the same number, inside the library.

### The sweep on CPU  (80 cells, Q=32)
drift floor 0.997 (0.740..1.547)

| multiply grid | cells | lat a/b | lat c/b | lat c/a | temp a | temp b | temp c |
|---|---|---|---|---|---|---|---|
| 2^9 | 1 | 0.97 | 0.92 | 0.95 | 3 kB | 1 kB | 0 kB |
| 2^11 | 3 | 1.04 | 1.04 | 1.00 | 9 kB | 1 kB | 0 kB |
| 2^13 | 6 | 0.99 | 1.03 | 1.02 | 21 kB | 0 kB | 0 kB |
| 2^15 | 10 | 1.14 | 1.19 | 0.97 | 33 kB | 0 kB | 0 kB |
| 2^17 | 13 | 1.78 | 1.21 | 0.72 | 131 kB | 0 kB | 0 kB |
| 2^19 | 14 | 2.65 | 1.38 | 0.63 | 328 kB | 0 kB | 0 kB |
| 2^21 | 13 | 2.19 | 1.31 | 0.62 | 524 kB | 0 kB | 0 kB |
| 2^23 | 10 | 3.43 | 2.87 | 0.82 | 2097 kB | 0 kB | 0 kB |
| 2^25 | 6 | 5.68 | 2.87 | 0.50 | 5243 kB | 0 kB | 0 kB |
| 2^27 | 3 | 8.57 | 4.19 | 0.49 | 8389 kB | 0 kB | 0 kB |
| 2^29 | 1 | 22.77 | 18.25 | 0.80 | 33554 kB | 0 kB | 0 kB |

| fixed emission | latency regret median | p90 | worst | best in | static temp over 80 cells |
|---|---|---|---|---|---|
| (1) broadcast-in-dot | 2.031 | 6.667 | 83.30 | 6/80 | 238.1 MB |
| (2) kept on the storing side | 1.000 | 1.014 | 1.44 | 70/80 | 0.7 MB |
| (3) multiply-then-reduce | 1.308 | 3.327 | 18.25 | 4/80 | 0.0 MB |

### The sweep on GPU  (80 cells, Q=32)
drift floor 0.995 (0.969..1.050)

| multiply grid | cells | lat a/b | lat c/b | lat c/a | temp a | temp b | temp c |
|---|---|---|---|---|---|---|---|
| 2^9 | 1 | 0.92 | 0.90 | 0.98 | 0 kB | 33555 kB | 0 kB |
| 2^11 | 3 | 0.92 | 0.91 | 1.00 | 0 kB | 33555 kB | 0 kB |
| 2^13 | 6 | 0.94 | 0.88 | 0.96 | 16777 kB | 33555 kB | 0 kB |
| 2^15 | 10 | 0.95 | 0.88 | 0.95 | 33555 kB | 33555 kB | 0 kB |
| 2^17 | 13 | 0.97 | 0.88 | 0.93 | 33555 kB | 33555 kB | 0 kB |
| 2^19 | 14 | 0.97 | 0.89 | 0.92 | 33555 kB | 33555 kB | 0 kB |
| 2^21 | 13 | 0.99 | 0.96 | 0.99 | 33555 kB | 33555 kB | 8 kB |
| 2^23 | 10 | 1.00 | 1.04 | 1.02 | 33555 kB | 33555 kB | 8 kB |
| 2^25 | 6 | 1.04 | 1.25 | 1.21 | 33555 kB | 33555 kB | 20 kB |
| 2^27 | 3 | 1.05 | 3.11 | 3.69 | 33555 kB | 33555 kB | 33 kB |
| 2^29 | 1 | 0.77 | 5.05 | 6.56 | 33555 kB | 33555 kB | 33 kB |

| fixed emission | latency regret median | p90 | worst | best in | static temp over 80 cells |
|---|---|---|---|---|---|
| (1) broadcast-in-dot | 1.054 | 1.136 | 1.89 | 15/80 | 2013.4 MB |
| (2) kept on the storing side | 1.110 | 1.168 | 1.40 | 13/80 | 2684.4 MB |
| (3) multiply-then-reduce | 1.000 | 1.218 | 9.47 | 52/80 | 0.5 MB |

### The threshold policy: emission (3) while the multiply grid <= T, else emission (2)

| T (products) | cpu regret median | cpu p90 | gpu regret median | gpu p90 | gpu static temp over 80 cells | cells taking (3) |
|---|---|---|---|---|---|---|
| 2^14 = 16384 | 1.000 | 1.044 | 1.099 | 1.163 | 2349 MB | 10/80 |
| 2^15 = 32768 | 1.000 | 1.135 | 1.038 | 1.158 | 2013 MB | 20/80 |
| 2^16 = 65536 | 1.000 | 1.135 | 1.038 | 1.158 | 2013 MB | 20/80 |
| 2^17 = 131072 | 1.000 | 1.307 | 1.000 | 1.134 | 1577 MB | 33/80 |
| 2^18 = 262144 | 1.000 | 1.307 | 1.000 | 1.134 | 1577 MB | 33/80 |
| 2^19 = 524288 | 1.055 | 1.801 | 1.000 | 1.117 | 1107 MB | 47/80 |
| 2^21 = 2097152 | 1.130 | 1.971 | 1.000 | 1.084 | 671 MB | 60/80 |
| 2^23 = 8388608 | 1.202 | 2.786 | 1.000 | 1.082 | 336 MB | 70/80 |
| 2^29 = 536870912 | 1.308 | 3.327 | 1.000 | 1.218 | 1 MB | 80/80 |

### Values
cpu: (1) against (2): bit-identical in 32/80 cells, largest relative L2 3.27e-07
cpu: (3) against (2): bit-identical in 68/80 cells, largest relative L2 3.27e-07
gpu: (1) against (2): bit-identical in 26/80 cells, largest relative L2 4.66e-04
gpu: (3) against (2): bit-identical in 0/80 cells, largest relative L2 4.66e-04

## Deliverable (b): the three emissions inside the engine

### CPU (job 63831, pgi15-cpu1)

Ratios against the reverse-order exact plan of emission (1), measured
immediately before every arm, five pairs.

**TransformerLM at the campaign shape, reverse order.** Drift floor 0.998
(0.986 to 1.012).

| plan | (1) bcast-in-dot | (2) kept on storing side | (3) mul-then-reduce |
|---|---|---|---|
| exact | 1.001 | **0.058** | 0.221 |
| quant_all | 1.026 | **0.065** | 0.161 |
| compress_all (Reduce) | 0.047 | **0.036** | 0.061 |

Static temp, same runs:

| plan | (1) | (2) | (3) |
|---|---|---|---|
| exact | 56.85 MB | **394 kB** | 18.00 MB |
| quant_all | 56.89 MB | **996 kB** | 17.74 MB |
| compress_all | **718 kB** | 800 kB | 1.34 MB |

Emission (3) is 4.5 times faster than the incumbent on the exact plan. It is
3.8 times slower than keeping the axis on the storing operand. Its temp sits
between the two: a third of the incumbent's, 46 times the 2-D dot's. There is
no cuBLAS call on CPU at all, so the kernel census only differs in fusion
count (127 for (1), 113 for (2), 128 for (3) on exact).

All three return the same gradient. Against `jax.grad`: 1.08e-06, 1.02e-06 and
0.95e-06 relative L2 on exact. All three are in parameter layout.

**TransformerLM, Markowitz order. The run died, and that is the result.**

| arm | static temp on markowitz:exact |
|---|---|
| (1) bcast-in-dot | 2.47 GB |
| (2) kept on storing side | 2.46 GB |
| (3) mul-then-reduce | **3 301 892 980 864 bytes (3.3 TB)** |

Emission (3) compiles and then fails on first execution with
`RESOURCE_EXHAUSTED: Out of memory allocating 3301892980864 bytes`. At a large
enough multiply grid XLA stops fusing the product into the reduce and asks for
the whole grid as a buffer. This is 1300 times the incumbent's temp on the same
plan. No unconditional multiply-then-reduce can survive this.

**NeuralNetwork (mnist), four arms.** Drift floor 0.892 (0.722 to 1.658). The
target is too small for a latency number, so read the direction only.

| order:plan | (1) | (2) | (3) | legacy |
|---|---|---|---|---|
| reverse:exact | 0.996 | 1.002 | 1.001 | 0.996 |
| reverse:quant_all | 1.200 | 1.216 | **1.052** | 1.185 |
| reverse:compress_all | 0.910 | 1.021 | **0.906** | 0.926 |
| markowitz:exact | 4.532 | 7.248 | **2.939** | 4.727 |
| markowitz:quant_all | 4.134 | 4.605 | **3.386** | 4.178 |
| markowitz:compress_all | 1.529 | 1.527 | **1.386** | 1.558 |

Emission (3) is the fastest arm in five of six cells. Every temp is near
200 kB and the four arms are within a few hundred bytes of each other. Values
are bit-identical to the incumbent on every plan except Reduce, where they
differ by 1.8e-07.

NeuralNetwork is the small-grid target and TransformerLM Markowitz is the
large-grid one. The two point in opposite directions, which is the size rule.

### GPU (job 63830, pgi15-gpu15, RTX PRO 6000 Blackwell)

Ratios against the reverse-order exact plan of emission (1), measured
immediately before every arm, five pairs. Drift floor 1.0002 (0.9997 to
1.0008).

**TransformerLM at the campaign shape, reverse order.**

| plan | (1) bcast-in-dot | (2) kept on storing side | (3) mul-then-reduce |
|---|---|---|---|
| exact | 1.003 | 1.098 | **0.999** |
| quant_all | 0.976 | 1.078 | **0.976** |
| compress_all (Reduce) | 1.049 | 1.065 | **1.048** |

Emission (3) and emission (1) are the same executable in every practical
sense. Same static temp to the byte, same Triton temp to the byte, same fusion
count, same cuBLAS count, and the returned gradients are BIT-IDENTICAL on the
exact and Reduce plans. Only the HLO text differs.

| plan, exact | temp | Triton temp | cuBLAS calls | fusions | wrapped_slice |
|---|---|---|---|---|---|
| (1) bcast-in-dot | 33 715 696 B | 301 808 B | 9 | 44 | 0 |
| (2) kept on storing side | 33 881 840 B | 303 600 B | 28 | 33 | 54 |
| (3) mul-then-reduce | 33 715 696 B | 301 808 B | 9 | 44 | 0 |

This is grill2/F2 section 1 confirmed in the real engine. XLA on GPU rewrites
emission (1) into a multiply and a reduce. Writing that form by hand gives the
same program. Emission (2) is the one that changes the kernel mix: 28 cuBLAS
calls instead of 9, and 54 `wrapped_slice` copy kernels that did not exist,
because a library call needs contiguous buffers.

The 1.098 of emission (2) reproduces T28B-RESULT.md section 4's 1.1074 on the
same target, the same shape and the same order.

**TransformerLM, Markowitz order, exact plan. Compile only.**

| arm | temp | Triton temp | cuBLAS calls | fusions |
|---|---|---|---|---|
| (1) bcast-in-dot | 846 MB | 819 MB | 39 | 53 |
| (2) kept on storing side | **782 MB** | **750 MB** | 41 | 61 |
| (3) mul-then-reduce | 1 680 MB | 1 680 MB | 9 | 54 |

On the Markowitz order emission (3) costs twice the memory of the incumbent on
GPU. On CPU the same plan costs 3.3 TB and cannot run at all. Same order, same
plan, same code: the multiply grid is what changed.

**NeuralNetwork (mnist), four arms.** Drift floor 1.0008 (0.9958 to 1.0220).
Every arm of every cell lands inside the floor, and all four static temps are
equal to the byte (64 B on reverse exact, 1296 B on Reduce). The target is too
small for the emission to matter on GPU. The one visible effect is the plan,
not the emission: Reduce runs at 0.80 of exact under all four arms.

**What did not finish.** The TLM Markowitz latency loop hit the probe's
7200-second cap (job exit 124). Three arms times three plans times five paired
rounds, at 137 times the reverse-order cost per execution (finding 61 verdict
7), does not fit in two hours. The Markowitz compile records above did land.

## Deliverable (c): the rule for dsnn-3qm.28.2

**The rule: the multiply-grid rule, with one device bit above the threshold.**

> Compute the multiply grid of the frame contraction: the number of scalar
> products it performs.
>
> * At or below 2 to the 17 products (131 072): emit multiply-then-reduce.
>   Both devices agree here.
> * Above it: keep the axis on the storing operand. On GPU this costs 9.9
>   percent on the exact plan and 10.5 percent on Quant at the TLM campaign
>   shape, measured against multiply-then-reduce in the same rounds. If the
>   owner will not pay that, the one device bit goes here and only here: above
>   the threshold GPU keeps the broadcast form.
>
> Emission (1), broadcast-in-dot, is never emitted below the threshold and is
> byte-for-byte the same program as multiply-then-reduce on GPU above it. The
> engine needs two emission bodies, not three.

**The predicate.** In the frame's own variables:

```
mul_grid = prod_i total[i]                      # merged meta extent per pair
         * prod_i split[i]                      # contracted or carried split
         * prod_i pairs[i].lhs.block_len        # lhs free
         * prod_i pairs[i].rhs.shared_block_len # rhs free
         * prod(lhs_leftover) * prod(rhs_leftover)
```

**Its inputs, all available at the emission site today.**

* `shared, total, split = _contraction_factors(pairs)` (`ops/matmul.py:509`).
* The `Pair` list: `lhs.block_len` and `rhs.shared_block_len` per pair.
* The operand shapes: `lhs_val.shape[3N:]` and `rhs_val.shape[3N:]` are the
  two leftovers.

Nothing else is needed. The decision has to be taken inside `_lazy_frame`,
before `_prepare_contraction_views` builds the views, because keeping the axis
on the storing operand is the demote path and it changes what the views are.
`_contraction_factors` is already called there.
`probes/t284/t284_predicate_check.py` computes the predicate from those inputs
and compares it to the real multiply grid at every contraction of a two-layer
MLP: 5 of 5 exact.

**Where the threshold comes from.** The 80-cell sweep of section (a). For a
policy and a cell, define the regret as the latency of the policy's pick
divided by the latency of the best of the three emissions in that cell. Both
numbers are paired, in one process, in the same rounds. 2 to the 17 is the
smallest threshold at which BOTH devices reach a median regret of 1.000. Below
it the GPU median rises (1.038 at 2^15, 1.099 at 2^14). Above 2 to the 18 the
CPU median rises (1.055 at 2^19, 1.130 at 2^21). The optimum is a plateau over
2^17 to 2^18, so the threshold is not a knife edge.

**Why the threshold has to exist.** Multiply-then-reduce is not free and not
uniform. Below the threshold XLA fuses the product into the reduce on both
devices and nothing is written. Above it XLA stops, and the size of what it
then writes is the whole grid. The engine numbers show both ends of that.

| where | multiply-then-reduce temp | the alternatives |
|---|---|---|
| NeuralNetwork, either device | equal to every other arm | equal |
| TLM reverse exact, CPU | 18.00 MB | 56.85 MB (1), 394 kB (2) |
| TLM Markowitz exact, GPU | 1 680 MB | 846 MB (1), 782 MB (2) |
| TLM Markowitz exact, CPU | 3.3 TB, out of memory | 2.47 GB (1), 2.46 GB (2) |

**Why the devices still disagree above the threshold.** The engine numbers,
not the sweep, are the authority there, and they are opposite. On GPU at TLM
reverse the multiply-then-reduce and broadcast forms are the same program and
beat the 2-D dot by about 10 percent, because a cuBLAS call is a fusion
barrier that also forces 54 `wrapped_slice` copy kernels. On CPU at the same
plan the 2-D dot is 3.8 times faster than multiply-then-reduce and 17 times
faster than the broadcast form, and it is the only arm that survives the
Markowitz order. There is no shape term that reconciles those, because the
disagreement is not about shape: it is about whether a vendor GEMM is worth
its fusion barrier, and the two devices answer differently at the same shape.

**What this rule replaces.** Finding 62 and T28B-RESULT.md section 4 both
concluded that the choice is device-dependent at every shape. It is not. It is
device-independent below 2 to the 17 products, and the third emission is what
makes that true. The device term shrinks from "every single implicit sparse
axis" to "the ones whose contraction does more than 131 072 multiplies".

**A note for the deletion in .28.2 step 5.** Emission (1) can go. On GPU it
compiles to emission (3) (identical temp, identical Triton temp, identical
fusion and cuBLAS counts, bit-identical gradients on the exact and Reduce
plans). On CPU it is dominated: median regret 2.031 over the sweep, worst
83.30, and it is the only emission whose CPU static temp is never zero.

## What I did not do

**The TLM Markowitz latency never landed.** On CPU the multiply-then-reduce
arm compiled and then ran out of memory, and the exception escaped the probe's
try block, which only wraps the compile and the first execution. On GPU the
same stage hit the probe's 7200-second cap. Both are recorded above as compile
records, which is the datum that matters there. A rerun would need the shrunk
protocol and a guard around `measure`.

**I ran two jobs per device, not one.** The first pair (63828, 63829) carried
both halves; its sweep half succeeded and its engine half died at import on an
env var that ticket dsnn-3qm.64 deleted. The second pair (63830, 63831) re-ran
only the engine half. No extra measurement was taken. Only one of my jobs was
on the cluster per device at any time.

**The sweep is one seed and one pairing type**: the single implicit sparse
axis on an aligned grid. It does not cover the spatial-sparse pairing or the
double implicit axis. Finding 62 has the same scope on purpose.

**The runtime watermark is a weak channel here.** On CPU jax cannot clear
memory statistics, so the campaign instrument falls back to the static
estimate and says so in the log. On GPU the allocator pool only grows, so the
per-cell delta is zero once the pool is large. What it does show is one sided:
in the sweep, emission (3) never raised the GPU watermark in any of the 80
cells, while (1) raised it by up to 19 MB and (2) by up to 67 MB.

**I did not implement the rule.** That is ticket dsnn-3qm.28.2's job. What I
implemented is the emission it needs, its test, and a check that its predicate
is computable from what the emission site already holds.

**The threshold comes from one GPU model** (RTX PRO 6000 Blackwell) and one
CPU node. It is a plateau over 2^17 to 2^18, not an edge, so it should survive
a different machine. It should be re-read if the GPU changes.

**Prediction P1 was wrong in both directions and that is the finding.** I
predicted that multiply-then-reduce would materialize on CPU once P was above
1 and K was 64 or more. In the isolated sweep it materialized in none of the
80 CPU cells. In the real engine it materialized 18 MB on TLM reverse and 3.3
TB on TLM Markowitz. The isolated sweep is not a safe proxy for the engine at
large grids. P2, P3, P4 and P5 held. P6 held in direction (multiply-then-reduce
is 3.8 times the 2-D dot on CPU) and P7 held (its CPU temp sits between the
other two). P8 held: the predicate is the multiply grid, not the meta extent.
