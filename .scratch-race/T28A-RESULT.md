# Race lane A (.28): the planner base

Ticket dsnn-3qm.66. Branches: graphax `wip/t28a-20260905`, alphagrad
`wip/t28a-20260905`. Both are on truth. Bases: graphax `1f3d404`, alphagrad
`ae2852a9`.

Jobs on GPU: 63788 TLM, 63792 NN.
Jobs on CPU: 63785 NN, 63786 TLM Markowitz, 63787 TLM reverse.
Suites: 63783 graphax pytest, 63789 alphagrad per-module.

## 1. What landed

| file | lines | what |
|---|---|---|
| `graphax/src/graphax/sparse/lower/matmul.py` | 320-338 | two knobs: `GRAPHAX_PLANNER_DOT_GENERAL` (on), `GRAPHAX_PLANNER_PET` (off) |
| same | 340-406 | `_dot_pair`: one operand pair, explicit dimension numbers |
| same | 408-429 | `_dot_contract`: fold the operand list left to right |
| same | 928-953 | the emission site: `dot_general`, or `jnp.einsum` under the knob |
| `graphax/src/graphax/sparse/ops/matmul.py` | 1746-1766 | `_bump_lower`, and knob `GRAPHAX_PLANNER_DENSE` (on) |
| same | 1960-2022 | the gate census, and dense x dense to the planner |

Nothing else in graphax changed. The tiled executor is untouched. It is still
reachable under `GRAPHAX_EINSUM_GENERAL=0`.

## 2. The two readings that matter

**The einsum letters were not the cause of the GPU gap.** On TLM at the
campaign shape, reverse order, exact plan, the candidate is 1.1205 times the
incumbent. Finding 61 measured 1.134 for the old planner. The drift floor is
1.0001. Fusions are 36 for the incumbent and 31 for the candidate. HLO lines
are 1594 and 936. So the candidate emits less code, and is still slower. The
gap is structural, not a matter of the primitive.

**The plain dense class moved, and it paid on Markowitz.** On TLM Markowitz
exact the planner takes 128 contractions now. The tiled executor takes 1.
Finding 61 had 113 and 16. Fusions went from 62 to 55. Static temp went
from 781 MB to 712 MB. Markowitz Quant latency went from 0.833 to 0.730.

## 3. What still calls the tiled executor

One contraction, on every plan and both orders, on both targets. It is
`gate:rank0_operand`, a scalar operand. The two engines disagree there by
design. Every other gate counter reads zero.
