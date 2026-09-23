"""F = the REVERSE-MODE REFERENCE ORDER's face count (dsnn-dfw.74).

Owner ruling 2026-09-20. `--face-init-approx-per-plan a` and
`--face-init-skips-per-plan kappa` derive the face head's two init biases
from F, and agent/initnorm left F unwired because a plan's live-face count
is a property of an elimination ORDER, not of the graph: under
`--fixed-order free` every episode samples its own order with its own
count. The ruling names one order -- the REVERSE order, i.e. the rev-exact
reference the paired cost measures every candidate against -- so the init's
normalizer and the reward's denominator describe the same plan.

Pinned here:

1. THE COUNT IS THE ORDER'S. `face_count_on_order` on the NN256 graph gives
   a different number for the reverse walk than for the ascending one; a
   graph-only F would give one number for both.
2. THE NUMBER. NN256 (`--example NeuralNetwork`, the dsnn-dfw.74 rows) has
   25 valid vertices and F = 21 live faces on the reverse reference order.
   The hidden width is a SHAPE, so `ALPHAGRAD_NN_HIDDEN=256` does not move
   it -- the rung-1 launchers and this test see the same F.
3. THE START BLOCK. One block, tagged `[face-init]`, carrying every input
   to both formulas -- F and the order it came from, S, k, a, kappa, B, Bs,
   E[A], E[K] -- on the derived path and on the raw --face-none-bias path
   alike, plus the same numbers as a wandb-config dict.
4. THE WIRING. ppo.main counts F on the env, before the agent is built, and
   passes it to `resolve_face_init_bias`; the block's config reaches
   `wandb.init`.
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import jax.random as jrand                                       # noqa: E402
import pytest                                                    # noqa: E402

from alphagrad.approx.common.agent_factory import (              # noqa: E402
    FACE_INIT_TAG, derive_face_none_bias, derive_face_skip_bias,
    expected_face_counts, face_init_start_block, resolve_face_init_bias)
from alphagrad.approx.common.order import (                      # noqa: E402
    REFERENCE_ORDER_NAME, face_count_on_order, reference_order_face_count,
    reverse_order)
from alphagrad.approx.unified_face_head import (                 # noqa: E402
    FACE_SLOTS, NUM_APPROX_OPS)

S = FACE_SLOTS
K = NUM_APPROX_OPS - 1

#: THE OWNER'S NUMBERS, typed here on purpose (2026-09-20).
NN256_VALID_VERTICES = 25
NN256_F_REVERSE = 21
RUNG1_A = 3.0
RUNG1_KAPPA = 0.3


@pytest.fixture(scope="module")
def nn256_env():
    """The NN256 env of the dsnn-dfw.74 rows, built the way ppo.main builds
    it: get_fn / get_args / grad_target_setup / _traced_inlined, which is
    also what the measure actor and landscape_map use."""
    from alphagrad.approx.common.examples import (
        get_args, get_fn, grad_target_setup)
    from alphagrad.approx.env import EnvConfig, VertexEliminationEnv
    from alphagrad.approx.ppo import _traced_inlined, make_argparser

    example = "NeuralNetwork"
    args = make_argparser().parse_args(["--example", example])
    fn = get_fn(example)
    xs = get_args(example, jrand.PRNGKey(0), dataset="mnist",
                  grad_window=args.target_grad_window,
                  dataset_size=args.dataset_size,
                  temporal_rule=args.temporal_rule,
                  step_position=args.step_position)
    fn, xs, argnums = grad_target_setup(args, fn, xs, example)
    cj = _traced_inlined(fn, xs)
    cfg = EnvConfig(jaxpr=cj.jaxpr, argnums=argnums, has_aux=False,
                    sparse=False, cmp_type="flops", mem_type="peak_memory",
                    terminal_rewards_only=True, delta_obs=True)
    return VertexEliminationEnv(cfg, tuple(xs), list(cj.literals))


# ------------------------------------------- 1. the count is the order's

def test_the_face_count_is_a_property_of_the_order(nn256_env):
    """The premise of the whole ticket. If F were a graph property the two
    walks would agree; they do not, which is why the ruling has to name an
    order."""
    env = nn256_env
    cfg = env.config
    rev = reverse_order(env.valid_vertices)
    asc = sorted(int(v) for v in env.valid_vertices)
    f_rev = face_count_on_order(cfg.jaxpr, cfg.argnums, env.consts, env.args,
                                rev)
    f_asc = face_count_on_order(cfg.jaxpr, cfg.argnums, env.consts, env.args,
                                asc)
    assert list(rev) == list(reversed(asc))
    assert f_rev != f_asc, (f_rev, f_asc)


def test_the_reference_order_is_the_paired_references_own_order(nn256_env):
    """env.py measures every candidate against `sorted(o_list, reverse=True)`
    -- the reverse order over the vertices the plan eliminated -- and a
    complete plan eliminates every valid vertex, so the order counted here
    IS that walk."""
    env = nn256_env
    _, name, order = reference_order_face_count(env)
    assert name == REFERENCE_ORDER_NAME
    assert list(order) == sorted((int(v) for v in env.valid_vertices),
                                 reverse=True)


# ---------------------------------------------------------- 2. the number

def test_nn256_reference_order_face_count(nn256_env):
    env = nn256_env
    assert len(env.valid_vertices) == NN256_VALID_VERTICES
    F, _, order = reference_order_face_count(env)
    assert len(order) == NN256_VALID_VERTICES
    assert F == NN256_F_REVERSE


def _rung1_pair():
    return (derive_face_none_bias(NN256_F_REVERSE, S, K, RUNG1_A),
            derive_face_skip_bias(NN256_F_REVERSE, RUNG1_KAPPA))


def test_the_rung1_targets_derive_the_biases_the_launchers_run_with():
    """a = 3 approximations and kappa = 0.3 skips per plan at F = 21.

    Since the owner ruling of 2026-09-23 the head has k = 2 non-none ops per
    slot and ONE quant bit per face at -B, so B is the bisection inverse of
    E[A] rather than ln(F*S*k/a - k) (4.094345 on the four-way head); Bs is
    untouched."""
    B, Bs = _rung1_pair()
    assert Bs == pytest.approx(math.log(21 / 0.3 - 1))
    assert Bs == pytest.approx(4.234107, abs=1e-6)
    e_a, e_k = expected_face_counts(NN256_F_REVERSE, S, K, B, Bs)
    assert e_a == pytest.approx(RUNG1_A)
    assert e_k == pytest.approx(RUNG1_KAPPA)


def test_resolve_returns_those_biases_when_f_is_passed():
    from argparse import Namespace
    ns = Namespace(face_init_approx_per_plan=RUNG1_A,
                   face_init_skips_per_plan=RUNG1_KAPPA,
                   face_none_bias=0.0, face_skip_bias=None)
    B, Bs = resolve_face_init_bias(ns, F=float(NN256_F_REVERSE))
    B1, Bs1 = _rung1_pair()
    assert B == pytest.approx(B1, abs=1e-9)
    assert Bs == pytest.approx(Bs1, abs=1e-9)
    assert Bs == pytest.approx(4.234107, abs=1e-6)


# ------------------------------------------------------- 3. the start block

def _ns(**over):
    from argparse import Namespace
    ns = Namespace(face_init_approx_per_plan=None,
                   face_init_skips_per_plan=None,
                   face_none_bias=0.0, face_skip_bias=None)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_the_start_block_carries_every_input_on_the_derived_path():
    ns = _ns(face_init_approx_per_plan=RUNG1_A,
             face_init_skips_per_plan=RUNG1_KAPPA)
    B, Bs = resolve_face_init_bias(ns, F=float(NN256_F_REVERSE))
    order = reverse_order(range(1, NN256_VALID_VERTICES + 1))
    lines, cfg = face_init_start_block(ns, NN256_F_REVERSE,
                                       REFERENCE_ORDER_NAME, B, Bs,
                                       order=order)
    text = "\n".join(lines)
    for line in lines:
        assert line.startswith(FACE_INIT_TAG), line
    assert "path: derived" in text
    assert f"F = {NN256_F_REVERSE} live faces on the {REFERENCE_ORDER_NAME}" \
        in text
    assert f"S = {S} slots/face, k = {K} non-none ops/slot" in text
    assert "--face-init-approx-per-plan a = 3, " \
           "--face-init-skips-per-plan kappa = 0.3" in text
    assert f"B  = E[A]^-1(a) = {B:.6f}" in text
    assert "the -B on the face's QUANT logit" in text
    assert "Bs = ln(F/kappa - 1) = 4.234107" in text
    assert "E[A] = F*((1-q)*S*p + q*(1+(S-2)*p)), p = k/(exp(B)+k), " \
           "q = 1/(1+exp(B)) = 3.000000 " \
           "requested approximations/plan" in text
    assert "E[K] = F/(1+exp(Bs)) = 0.300000 requested skips/plan" in text
    assert f"the order: {NN256_VALID_VERTICES} vertices, " \
           "[25, 24, 23, 22, 21, 20] ... [3, 2, 1]" in text
    assert cfg == {
        "face_init_path": "derived",
        "face_init_F": NN256_F_REVERSE,
        "face_init_order": REFERENCE_ORDER_NAME,
        "face_init_slots_S": S,
        "face_init_ops_k": K,
        "face_init_a": RUNG1_A,
        "face_init_kappa": RUNG1_KAPPA,
        "face_init_B": pytest.approx(B, abs=1e-9),
        "face_init_Bs": pytest.approx(4.234107, abs=1e-6),
        "face_init_expected_approx_per_plan": pytest.approx(RUNG1_A),
        "face_init_expected_skips_per_plan": pytest.approx(RUNG1_KAPPA),
    }


def test_the_start_block_reports_the_same_counts_on_the_raw_path():
    """--face-none-bias 2 is what the collapsed rows ran. The block must
    say what that MEANS in plan counts, or the two paths cannot be compared
    at all."""
    ns = _ns(face_none_bias=2.0)
    lines, cfg = face_init_start_block(ns, NN256_F_REVERSE,
                                       REFERENCE_ORDER_NAME, 2.0, None)
    text = "\n".join(lines)
    assert "path: raw (--face-none-bias/--face-skip-bias)" in text
    assert "--face-init-approx-per-plan a = unset, " \
           "--face-init-skips-per-plan kappa = unset" in text
    assert "B  = --face-none-bias = 2.000000" in text
    assert "Bs = B (unset: SKIP gets -B) = 2.000000" in text
    e_a, e_k = expected_face_counts(NN256_F_REVERSE, S, K, 2.0, 2.0)
    assert (f"q = 1/(1+exp(B)) = {e_a:.6f} requested approximations/plan"
            in text)
    assert f"E[K] = F/(1+exp(Bs)) = {e_k:.6f}" in text
    assert cfg["face_init_path"] == "raw"
    assert cfg["face_init_a"] is None and cfg["face_init_kappa"] is None
    assert cfg["face_init_B"] == 2.0 and cfg["face_init_Bs"] == 2.0
    # the number a bias-2 start means on this head, named: the slot ops at
    # k/(e^B + k) and the face's quant bit at sigmoid(-B)
    p_q = 1.0 / (1.0 + math.exp(2.0))
    p_op = K / (math.exp(2.0) + K)
    assert e_a == pytest.approx(
        21 * ((1 - p_q) * 3 * p_op + p_q * (1 + 1 * p_op)))


def test_the_start_block_names_the_skip_bias_flag_when_it_is_set():
    ns = _ns(face_none_bias=2.0, face_skip_bias=5.0)
    lines, cfg = face_init_start_block(ns, NN256_F_REVERSE,
                                       REFERENCE_ORDER_NAME, 2.0, 5.0)
    text = "\n".join(lines)
    assert "Bs = --face-skip-bias = 5.000000" in text
    assert cfg["face_init_Bs"] == 5.0


# ----------------------------------------------------------- 4. the wiring

def test_ppo_main_counts_f_on_the_env_before_the_agent_and_passes_it():
    """F is counted on EVERY graph's env, before the agent, and reaches the
    init.

    THE WALK IS A LOOP NOW. Since the owner's ruling of 2026-09-22 a run may
    hold two graphs of one target, each with its own reference face count
    (measured: bptt 42, rtrl 65), so the count names the graph's env rather
    than a single `env`. The three facts this guarded are unchanged and are
    guarded here: the count happens before the agent exists, the derived bias
    reads it, and the bias is applied to the agent built after it.
    """
    import inspect
    from alphagrad.approx import ppo
    src = inspect.getsource(ppo.main)
    assert "reference_order_face_count as _ref_face_count" in src
    assert "resolve_face_init_bias(args, F=_face_ref_F)" in src
    i_count = src.index('_ref_face_count(_gg["env"])')
    i_build = src.index("_build_agent(args, total_v")
    i_resolve = src.index("resolve_face_init_bias(args, F=_face_ref_F)")
    i_apply = src.index("apply_face_none_bias(agent")
    assert i_count < i_build < i_resolve < i_apply
    # ONE WALK PER GRAPH, and the shared head's bias is derived from the
    # PRIMARY graph's count.
    i_guard = src.index("_face_bias_in_play = (")
    assert "for _k in _GRAPH_KEYS:" in src[i_guard:i_count]
    assert '_face_ref_F = _GRAPHS[_PRIMARY]["face_ref_F"]' in src
    # AND EVERY OTHER GRAPH GETS ITS OWN PAIR from its own F, after the agent
    # exists, because the offset is a vector over that agent's head.
    i_off = src.index('resolve_face_init_bias(args, F=_gg["face_ref_F"])')
    assert i_apply < i_off


def test_the_block_reaches_the_wandb_config():
    import inspect
    from alphagrad.approx import ppo
    src = inspect.getsource(ppo.main)
    assert "_wandb_config.update(_face_init_cfg)" in src
    assert src.index("_wandb_config.update(_face_init_cfg)") \
        < src.index("wandb.init(")


def test_no_bias_flag_means_no_walk_and_no_block():
    """A run that sets none of the four flags applies no bias, so there is
    nothing to normalize -- and it must not pay an elimination walk over the
    whole graph to be told so. A run with TWO graphs must not pay two."""
    import inspect
    from alphagrad.approx import ppo
    src = inspect.getsource(ppo.main)
    i_guard = src.index("_face_bias_in_play = (")
    i_count = src.index('_ref_face_count(_gg["env"])')
    assert i_guard < i_count
    assert "if _face_bias_in_play:" in src
    # The per-graph pair is inside the same guard, so a run with no bias flag
    # derives none of them either.
    assert "if not (_face_bias_in_play and _TWO_GRAPH):" in src
