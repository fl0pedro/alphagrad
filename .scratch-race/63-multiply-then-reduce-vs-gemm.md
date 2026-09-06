# 63 - Multiply-then-reduce against a GEMM (ticket dsnn-3qm.28.4)

Plain English. Short sentences. Every cost number is a ratio measured in one
process, back to back, with a drift floor beside it. No raw unpaired number is
a claim here.

## The question

A single implicit sparse axis is the meta axis of a diagonal pair that exactly
one operand stores. The engine has three ways to emit its contraction.

1. **Broadcast-in-dot.** Broadcast the side that does not store the axis, and
   call `dot_general` with the axis in the batch list. This is the incumbent
   tiled form.
2. **Kept on the storing side.** The axis rides as a free axis of the operand
   that stores it. The contraction becomes a clean 2-D dot. This is the
   planner's form and `GRAPHAX_TILED_LAZY=full`.
3. **Multiply-then-reduce.** Never emit a dot. `sum(lhs * rhs, axis=k)` over
   the broadcast shapes. This is the form XLA on GPU already rewrites (1) into.

The owner asked for the difference between (3) and a GEMM. The channels are
memory, latency, temp and every other one we have.

## Apparatus

Code: graphax `wip/t284-20260906` at `ecede37`, alphagrad `wip/t284-20260906`
at `b44128b6`. The graphax commit adds one race-only knob,
`GRAPHAX_TILED_MULREDUCE` (default OFF). It selects emission (3) at the one
frame-contraction site of `_execute_block_sparse_contraction`
(`src/graphax/sparse/ops/matmul.py`). It changes no default.

Probes: `alphagrad/.scratch-race/probes/t284/`.

| job | node | what |
|---|---|---|
| 63828 | pgi15-gpu15 | the synthetic sweep, GPU |
| 63829 | pgi15-cpu1 | the synthetic sweep, CPU |
| 63830 | pgi15-gpu15 | the engine probe, GPU |
| 63831 | pgi15-cpu1 | the engine probe, CPU |

Jobs 63828 and 63829 also tried the engine probe and failed at import: the
campaign SHARED_ENV still exports `ALPHAGRAD_FORCE_REV_ORDER`, which ticket
dsnn-3qm.64 deleted from the code. The probe now drops that variable. Their
sweep halves are the sweep data below.

## Deliverable (a): the synthetic sweep

One contraction. Meta extent M, block extent P, contracted extent K, output
extent Q fixed at 32. The grid of finding 62 (M in {4,16,64,256,1024}, P in
{1,4,16,64}, K in {4,16,64,256}), 80 cells, extended by emission (3). Every
cell compiles all three emissions plus a second executable of (2), which is
the drift floor. All four are timed back to back in the same rounds.

`SWEEP_TABLES`

## Deliverable (b): the three emissions inside the engine

`ENGINE_TABLES`

## Deliverable (c): the rule for dsnn-3qm.28.2

`RULE`

## What I did not do

`GAPS`
