# Lane B of the .28 race: the lazy tiled frame (ticket dsnn-3qm.67)

Branches: graphax `wip/t28b-20260905` and alphagrad `wip/t28b-20260905`.
Both are on truth. Base: graphax `1f3d404`, alphagrad `ae2852a9`.

## 1. What the tiled engine did wrong

The tiled executor builds a fixed physical frame. Each `Pair` gets three axes
per side: outer, block, shared_block. The outer axes merge into a grid of
`total[i] = lcm(outer_l, outer_r)` per pair. `_prepare_contraction_views` then
called `_as_shape(..., mode="broadcast")` on both operand views. That call
brought every frame slot up to its logical length. A slot that no operand
stores is physically size 1. So the call materialized the broadcast.

Three consequences follow. Finding 61 measured all three.

1. The CPU temp was 56.9 MB against the planner's 378 KB on TLM reverse. XLA on
   GPU fuses the broadcast into the GEMM. XLA on CPU allocates it.
2. Every implicit extent became a physical output axis. `_build_pair_dims`
   forces `pres_*` true for the whole pair as soon as any one of its six slots
   is stored. The `any_val` block is that rule.
3. The Reduce class paid both costs. Most of its extents are implicit.

A second defect is independent of the first. For the `spatial_primal_lhs`
pairing the extent lives in `PairData.shared_block_len`. `_finalize_output`
folds that value through `split` and `ss_out` into the rhs half of the grid.
`_build_pair_dims` read it from the lhs half instead. The lhs half is always 1
for that pairing. So both the logical size and the stored values collapsed.
That is finding 61, verdict 6.

## 2. What lane B changed

The file is `src/graphax/sparse/ops/matmul.py`.

`_lazy_frame(lhs_val, rhs_val, pairs)` reads the physical size of each of the
six slots off the prepared operand arrays. It returns an EFFECTIVE `Pair` list.
In that list every extent no operand stores is 1. The existing pipeline then
builds the small grid. No broadcast grows a buffer.

There are four rules. Each one is the tiled-frame form of one planner rule.

* The meta axis is stored by neither side. The grid axis stays 1. Both output
  dims of the pair keep their logical extent with `axis=None`. That is the
  planner's `out:implicit_kept` and `out:pair_retained`.
* The meta axis is stored by exactly one side. The axis leaves the
  `dot_general` batch list and rides as that side's free axis. The other side is
  never broadcast. Same values, same output shape, no metadata change.
* The contracted axis is stored by one side. That side is summed over it with
  `jnp.sum` and keepdims. The other side stays 1. This is the planner's rule
  that a contracted pair implicit on one side is a plain sum over the physical
  side.
* The contracted axis is stored by neither side. The extent folds into
  `scalar_mult` as an analytic scale. That is `rule:fold_scale`.

`CRes` now carries both frames. `shared_factors` and the two `*_block_lens`
describe the buffer. The `true_*` fields describe the logical topology. `lazy`
says which output slots stayed symbolic. `eff_pairs` is what the buffer was
built from. `_resolve_output_shape` reads the effective frame.
`_build_pair_dims` reads the true one. It clears `pres_*` for every lazy slot.

A safety net guards every rule. A slot stays symbolic only when its
effective grid axis really is 1. Three cases keep the incumbent frame slot for slot. The first is a genuine LCM
grid, where both outer lens are above 1 and unequal. The second is a
`spatial_sparse_*` pairing. The third is a partially stored extent. In those
three cases the code path is the incumbent one.

The code skips banded emission when the lazy frame fires. The band probes read
the frame geometry. They are not part of this change.

`spatial_primal_lhs` now reads its size and its axis from the rhs half of the
grid. That is the verdict 6 fix.

`src/graphax/sparse/ops/matmul_legacy_tiled.py` is a verbatim copy of the
incumbent executor of `1f3d404`. It holds nine functions. It is reachable only
under the race-only knob `GRAPHAX_TILED_LEGACY=1`. It exists so that the
landing test can pair candidate against incumbent inside one process. Step 3 of
the .28 design note deletes the file and the knob together.

`tests/core/sparse_tensor/lazy_tiled_test.py` is the toy test of the extent
loss. It has four cases. Three of them were red before the fix. All are green
after it.

## 3. Census

The table counts growing broadcasts in the tiled frame. The count comes from a
local probe that wraps `_as_shape`.

| target, plan | before | after |
|---|---|---|
| NeuralNetwork, exact | 15 calls, 860 to 50258 elements | none |
| NeuralNetwork, Reduce | 25 calls, 870 to 150574 elements | none |
| NeuralNetwork, Quant and Diag | as exact | none |
| TransformerLM small, exact | present on 151 pair slots | none |

The small TransformerLM is seq 8, dmodel 32, vocab 64.

## 4. The demote rule is a device-dependent choice

The demote rule is the one that moves a meta axis out of the `dot_general`
batch list when only one side stores it. A race-only knob `GRAPHAX_TILED_LAZY`
turns the rules on and off so that the cost can be attributed.

TransformerLM at the campaign shape, GPU, reverse order, exact plan. Every
number is the candidate over the incumbent, paired in one process.

| rules | candidate over incumbent | drift floor | job |
|---|---|---|---|
| full (every rule) | 1.1074 | 0.9996 | 63795 |
| nosum (only the contracted-axis sum off) | 1.1148 | 1.0008 | 63803 |
| nodemote (only the demotion off) | 1.0048 | 0.9994 | 63802 |

The whole cost is the demotion. XLA on GPU fuses the broadcast that the
demotion avoids into the batched dot. Once the axis leaves the batch list that
fusion is gone. XLA on CPU allocates the same broadcast instead, which is the
56.9 MB against 378 KB that finding 61 measured.

So a single engine cannot have both without a device-aware lowering choice.
The default of this lane is `nodemote`. `GRAPHAX_TILED_LAZY=full` restores the
leaner CPU frame. The owner decides which one the single engine keeps.
