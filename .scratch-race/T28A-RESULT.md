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

## 4. Scorecard

Every ratio is candidate over incumbent, paired in one process. The drift
floor of that job stands beside it.

| cell | result | evidence |
|---|---|---|
| GPU latency, exact, reverse | FAIL | 1.1205, floor 1.0001. Finding 61: 1.134 |
| GPU latency, Quant and Diag, TLM | FAIL | reverse 1.1304 and 1.1263, Markowitz 0.7304 and 1.3581 |
| GPU latency, Quant and Diag, NN | PASS | 0.9935, 0.9958, 0.9990, 0.9782, floor 1.0018 |
| GPU Triton temp, every plan | FAIL | Quant reverse 1.818, Markowitz Diag 1.250 |
| CPU latency and temp, reverse | PASS on 5 of 6 | exact 0.0508 and 378 KB. Quant 0.0713 against 0.065 |
| CPU latency, Markowitz | FAIL | TLM exact 1.1978, NN exact 1.4348 |
| Reduce class, every device | PASS | 0.8378, 0.0592, 0.0339, 0.1058, extents kept |
| values, every plan | PASS | see below |
| tests | PASS | 1328 passed, 122 modules clean |

Values in detail. Sparse equals its own dense oracle bit for bit under both
engines on every plan of every job. The exact plan is 1.2e-4 from `jax.grad`
on TLM GPU, and 9.8e-7 on TLM CPU. All 15 TLM gradients come back in
parameter layout under the candidate.
