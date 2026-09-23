"""Two input DAGs in one run (bead dsnn-dfw.116, owner ruling 2026-09-22).

``--temporal-rule bptt rtrl`` runs both graphs of one target and alternates
per episode. These tests pin the six things that makes true:

1. THE ARGUMENT. One rule in, one string out -- the namespace a one-rule run
   carries is the one it carried before two rules existed, which is what
   makes the old behaviour byte for byte the old behaviour.
2. THE SCHEDULE. Even episodes on the first rule named, odd on the second, as
   a pure function of the episode number so a resume lands on the right graph.
3. THE ISOLATION. An update on one graph's episode does not move the other
   graph's multiplier, PopArt or archive.
4. THE ROUND TRIP. The per-graph state survives a checkpoint, and a resume
   whose rule list moved is refused rather than run on the other graph.
5. THE SHARED VERTEX INDEX SPACE (owner ruling 1, 2026-09-22). One index
   space at the wider count, index i a SLOT and not a vertex identity, the
   narrower graph masked to its own slots and the boundary padded.
6. THE PER-GRAPH FACE INIT (owner ruling 2). The shared head keeps one
   (B, Bs) and each graph carries a fixed logit offset, so each DAG starts at
   the run's requested approximations and skips per plan from its own F.

Nothing here builds a graph: the measured graph shapes come from the probe
(job 67505) and what is under test is the bookkeeping that decides which
graph an episode is on. `tests/temporal_rule_test.py` owns the graphs.
"""

import copy

import pytest

from alphagrad.approx.common import checkpoint as ckpt
from alphagrad.approx.common.rsnn_shd import (
    RSNN_W2_TARGET,
    resolve_temporal_rule,
    resolve_temporal_rules,
    target_example,
    temporal_rule_for_episode,
    temporal_rule_list,
)
from alphagrad.approx.common.two_graph import (
    GraphStates,
    namespace_log,
    namespaced,
)

RSNN = "RSNN_SHD"


# ---------------------------------------------------------------------------
# 1. THE ARGUMENT
# ---------------------------------------------------------------------------

def test_one_rule_resolves_to_the_string_it_always_was():
    """A one-rule run's resolved value is the plain string, not a list.

    Everything downstream -- the wandb config, the dict the measure actors
    are built from, the checkpoint's argument namespace -- is
    `dict(vars(args))`, so a one-rule run whose value became a list would
    differ from the old run in every one of them.
    """
    for rule in ("tbptt", "bptt", "rtrl"):
        assert resolve_temporal_rules(RSNN, rule) == rule
        assert resolve_temporal_rules(RSNN, [rule]) == rule
        assert isinstance(resolve_temporal_rules(RSNN, [rule]), str)
    # and the default is still the baseline
    assert resolve_temporal_rules(RSNN, None) == "tbptt"
    assert resolve_temporal_rules("NeuralNetwork", None) is None
    # exactly what the single-rule resolver says, on every accepted form
    for rule in (None, "tbptt", "bptt", "rtrl", "window2"):
        assert (resolve_temporal_rules(RSNN, rule)
                == resolve_temporal_rule(RSNN, rule))


def test_two_rules_resolve_to_a_list_in_the_order_given():
    assert resolve_temporal_rules(RSNN, ["bptt", "rtrl"]) == ["bptt", "rtrl"]
    assert resolve_temporal_rules(RSNN, ["rtrl", "bptt"]) == ["rtrl", "bptt"]
    assert resolve_temporal_rules(RSNN, ["tbptt", "rtrl"]) == ["tbptt", "rtrl"]


def test_a_pair_that_is_not_two_graphs_of_one_target_is_refused():
    with pytest.raises(ValueError, match="same rule twice"):
        resolve_temporal_rules(RSNN, ["bptt", "bptt"])
    with pytest.raises(ValueError, match="window2"):
        resolve_temporal_rules(RSNN, ["bptt", "window2"])
    with pytest.raises(ValueError, match="at most 2"):
        resolve_temporal_rules(RSNN, ["tbptt", "bptt", "rtrl"])
    with pytest.raises(ValueError, match="empty list"):
        resolve_temporal_rules(RSNN, [])
    with pytest.raises(ValueError, match="NO time steps"):
        resolve_temporal_rules("NeuralNetwork", ["bptt", "rtrl"])


def test_two_rules_name_one_target():
    """`target_example` answers once for the pair, because the two graphs are
    two graphs OF one target."""
    assert target_example(RSNN, ["bptt", "rtrl"]) == RSNN
    assert target_example(RSNN, "window2") == RSNN_W2_TARGET
    with pytest.raises(ValueError, match="two TARGETS"):
        target_example(RSNN, ["bptt", "window2"])


def test_the_argparser_takes_one_or_two_rules():
    """THE PARSER ITSELF hands back the plain string for one rule.

    `nargs="+"` always produces a list, and collapsing it in `main` left the
    parsed namespace holding `['tbptt']`, which every launcher test that reads
    the parsed value compares against `'tbptt'`
    (`gen_fq_launchers_thesis_test`, suite job 67542). The collapse is in the
    action, so a one-rule namespace is byte for byte the old one at the
    argparse level too.
    """
    from alphagrad.approx.ppo import make_argparser

    p = make_argparser()
    base = ["--example", RSNN]
    for rule in ("tbptt", "bptt", "rtrl", "window2"):
        got = p.parse_args(base + ["--temporal-rule", rule]).temporal_rule
        assert got == rule and isinstance(got, str), got
    assert p.parse_args(
        base + ["--temporal-rule", "bptt", "rtrl"]).temporal_rule \
        == ["bptt", "rtrl"]
    assert p.parse_args(base).temporal_rule is None
    with pytest.raises(SystemExit):
        p.parse_args(base + ["--temporal-rule", "nonsense"])
    # a bad value in a PAIR is refused too, by `choices`
    with pytest.raises(SystemExit):
        p.parse_args(base + ["--temporal-rule", "bptt", "nonsense"])


def test_the_rule_list_helper_reads_every_form():
    assert temporal_rule_list(None) == ()
    assert temporal_rule_list("bptt") == ("bptt",)
    assert temporal_rule_list(["bptt", "rtrl"]) == ("bptt", "rtrl")
    assert temporal_rule_list(("bptt", "rtrl")) == ("bptt", "rtrl")


# ---------------------------------------------------------------------------
# 2. THE SCHEDULE
# ---------------------------------------------------------------------------

def test_even_episodes_take_the_first_rule_and_odd_the_second():
    pair = ["bptt", "rtrl"]
    got = [temporal_rule_for_episode(pair, ep) for ep in range(6)]
    assert got == ["bptt", "rtrl", "bptt", "rtrl", "bptt", "rtrl"]
    # the order named IS the schedule
    flipped = [temporal_rule_for_episode(["rtrl", "bptt"], ep)
               for ep in range(4)]
    assert flipped == ["rtrl", "bptt", "rtrl", "bptt"]


def test_one_rule_runs_on_every_episode():
    assert [temporal_rule_for_episode("rtrl", ep) for ep in range(5)] \
        == ["rtrl"] * 5
    assert temporal_rule_for_episode(None, 3) is None


def test_the_schedule_is_a_function_of_the_episode_so_a_resume_lands_right():
    """A resume restarts the loop at the checkpoint's episode, so the graph
    an episode runs on must depend on the episode NUMBER and on nothing the
    run carries. 1000 and 2000 are the checkpoint pins of the campaign row."""
    pair = ["bptt", "rtrl"]
    for ep in (0, 999, 1000, 1999, 2000, 3999):
        assert temporal_rule_for_episode(pair, ep) == pair[ep % 2]
    with pytest.raises(ValueError, match="negative"):
        temporal_rule_for_episode(pair, -1)


def test_the_store_schedules_the_same_way_the_function_does():
    st = GraphStates(["bptt", "rtrl"], lag_init=16.0)
    assert st.alternating and len(st) == 2
    assert [st.rule_for(ep) for ep in range(4)] == \
        ["bptt", "rtrl", "bptt", "rtrl"]
    one = GraphStates("rtrl", lag_init=16.0)
    assert not one.alternating
    assert [one.rule_for(ep) for ep in range(3)] == ["rtrl"] * 3


def test_a_store_refuses_a_rule_it_holds_no_graph_for():
    st = GraphStates(["bptt", "rtrl"], lag_init=16.0)
    with pytest.raises(KeyError, match="no graph"):
        st["tbptt"]
    with pytest.raises(ValueError, match="same graph twice"):
        GraphStates(["bptt", "bptt"], lag_init=1.0)
    with pytest.raises(ValueError, match="at least one"):
        GraphStates(None, lag_init=1.0)


# ---------------------------------------------------------------------------
# 3. THE ISOLATION
# ---------------------------------------------------------------------------

class _FakeArchive:
    """The two archive calls the store touches, and nothing else."""

    def __init__(self):
        self.added = []

    def add(self, point):
        self.added.append(point)


def test_an_update_on_one_graph_does_not_move_the_other():
    """The whole point of the per-graph half: a bptt episode's dual step, its
    PopArt step and its archive admission leave the rtrl graph where it was.
    """
    st = GraphStates(["bptt", "rtrl"], lag_init=16.0)
    for s in st:
        s.archive = _FakeArchive()
        s.popart = (0.0, 0.0, 0.0)

    b, r = st["bptt"], st["rtrl"]
    before = (r.lag_lambda, r.popart, list(r.archive.added), r.episodes)

    b.lag_lambda = 23.5
    b.popart = (1.0, 2.0, 3.0)
    b.archive.add((0.1, 0.2))
    b.episodes += 1

    assert (r.lag_lambda, r.popart, list(r.archive.added), r.episodes) \
        == before
    assert r.archive is not b.archive
    assert st["bptt"].lag_lambda == 23.5 and st["rtrl"].lag_lambda == 16.0


def test_every_graph_starts_from_the_same_lag_init():
    st = GraphStates(["bptt", "rtrl"], lag_init=16.0)
    assert [s.lag_lambda for s in st] == [16.0, 16.0]


# ---------------------------------------------------------------------------
# 4. THE ROUND TRIP
# ---------------------------------------------------------------------------

def _identity_archive_json(archive):
    return None if archive is None else {"added": list(archive.added)}


def _identity_archive_restore(archive, d):
    archive.added = list(d["added"])


def _filled_store():
    st = GraphStates(["bptt", "rtrl"], lag_init=16.0)
    for i, s in enumerate(st):
        s.archive = _FakeArchive()
        s.archive.add((float(i), float(i) + 1.0))
        s.lag_lambda = 12.0 + i
        s.episodes = 10 + i
        s.total_v = 59 + 10 * i
        s.num_valid = 58 + 10 * i
        s.face_F = 42 + 23 * i
        s.face_none_bias = 1.5 + i
        s.face_skip_bias = 0.5 + i
    return st


def test_the_per_graph_state_survives_a_round_trip():
    st = _filled_store()
    blob = st.to_json(_identity_archive_json)
    blob = copy.deepcopy(blob)          # a checkpoint is written and re-read

    back = GraphStates(["bptt", "rtrl"], lag_init=99.0)
    for s in back:
        s.archive = _FakeArchive()
    back.from_json(blob, _identity_archive_restore)

    for a, b in zip(st, back):
        assert b.rule == a.rule
        assert b.lag_lambda == a.lag_lambda
        assert b.episodes == a.episodes
        assert b.archive.added == a.archive.added
    assert back["bptt"].lag_lambda != back["rtrl"].lag_lambda


def test_a_resume_whose_rules_moved_is_refused():
    blob = _filled_store().to_json(_identity_archive_json)
    for rules in (["rtrl", "bptt"], ["bptt"], ["tbptt", "rtrl"]):
        other = GraphStates(rules, lag_init=1.0)
        with pytest.raises(ValueError, match="alternates over"):
            other.from_json(blob, _identity_archive_restore)


def test_a_checkpoint_missing_a_graph_is_refused():
    blob = _filled_store().to_json(_identity_archive_json)
    blob["graphs"].pop("rtrl")
    st = GraphStates(["bptt", "rtrl"], lag_init=1.0)
    with pytest.raises(ValueError, match="no state for the graph"):
        st.from_json(blob, _identity_archive_restore)


def test_check_resume_args_treats_the_two_rule_list_as_one_value():
    """The rule list is ONE argument. A resume that matches it passes; a
    resume that changed it is not a resume of that run."""
    import argparse

    def _ns(**kw):
        d = dict(episodes=100, resume="/x", seed=7,
                 temporal_rule=["bptt", "rtrl"])
        d.update(kw)
        return argparse.Namespace(**d)

    saved = ckpt.args_to_json(_ns())
    assert saved["temporal_rule"] == ["bptt", "rtrl"]
    ckpt.check_resume_args(saved, _ns())
    for bad in (["rtrl", "bptt"], ["bptt"], "bptt", None):
        with pytest.raises(ckpt.CheckpointError, match="temporal_rule"):
            ckpt.check_resume_args(saved, _ns(temporal_rule=bad))


# ---------------------------------------------------------------------------
# 5. THE NAMESPACED METRICS
# ---------------------------------------------------------------------------

def test_a_metric_key_gains_the_rule_after_its_first_element():
    assert namespaced("lagrangian/lambda", "bptt") == "lagrangian/bptt/lambda"
    assert namespaced("popart/mu_quality", "rtrl") == "popart/rtrl/mu_quality"
    assert namespaced("pareto/archive_size", "bptt") \
        == "pareto/bptt/archive_size"
    assert namespaced("episode", "bptt") == "episode/bptt"
    # a single-rule run's keys do not move
    assert namespaced("lagrangian/lambda", None) == "lagrangian/lambda"


def test_the_namespaced_log_keeps_the_plain_keys_too():
    log = {"lagrangian/lambda": 16.0, "pareto/archive_size": 3}
    got = namespace_log(log, "bptt")
    assert got["lagrangian/bptt/lambda"] == 16.0
    assert got["lagrangian/lambda"] == 16.0
    assert got["pareto/bptt/archive_size"] == 3
    assert namespace_log(log, None) == log


# ---------------------------------------------------------------------------
# 6. THE CARRY PLAN HOLDS ONE ENTRY PER GRAPH
# ---------------------------------------------------------------------------

class _FakeJaxpr:
    def __init__(self, tag):
        self.tag = tag


class _FakeConfig:
    def __init__(self, tag):
        self.jaxpr = _FakeJaxpr(tag)


def test_the_carry_plan_holds_one_entry_per_graph_and_the_graph_is_the_key():
    """One process may serve two graphs of one target, so the carry plan's
    state is a map. Registering the second graph must not forget the first,
    and the base jaxpr -- not the rule, which the measurement never sees -- is
    what selects between them."""
    from alphagrad.approx.common import carry_plan as CP

    CP.reset()
    try:
        cfg_b, cfg_r = _FakeConfig("bptt"), _FakeConfig("rtrl")
        CP.register(None, None, RSNN, "bptt", cfg_b, (), ())
        CP.register(None, None, RSNN, "rtrl", cfg_r, (), ())
        assert CP.armed(cfg_b) and CP.armed(cfg_r)
        assert CP._entry(cfg_b)["spec"]["rule"] == "bptt"
        assert CP._entry(cfg_r)["spec"]["rule"] == "rtrl"
        # a graph nobody registered is not armed, and neither is a rule with
        # no given edge
        assert not CP.armed(_FakeConfig("other"))
        CP.register(None, None, RSNN, "tbptt", _FakeConfig("tb"), (), ())
        assert CP.armed(cfg_b) and CP.armed(cfg_r)
        # with two graphs in the process, asking without naming one RAISES
        with pytest.raises(ValueError, match="named none"):
            CP._entry(None)
        # the variant caches are per graph
        CP._entry(cfg_b)["variants"]["diag"] = "B"
        assert CP._entry(cfg_r)["variants"] == {}
    finally:
        CP.reset()


# ---------------------------------------------------------------------------
# 7. THE SHARED VERTEX INDEX SPACE (owner ruling 1, 2026-09-22)
# ---------------------------------------------------------------------------
#
# ONE index space at the wider count. Index i is a SLOT, not a vertex
# identity: nothing assumes slot i of one graph is slot i of the other. The
# narrower graph is masked to its own slots and the env-to-agent boundary is
# padded up. The numbers are the measured ones (job 67505): bptt 59 equations
# and 58 valid vertices, rtrl 69 and 68.

BPTT_EQNS, BPTT_VALID = 59, 58
RTRL_EQNS, RTRL_VALID = 69, 68
SHARED = RTRL_EQNS


class _FakeState:
    """The two fields `vertex_avail_at_step` reads off a rollout state."""

    def __init__(self, order, step_count):
        import jax.numpy as jnp
        self.order = jnp.asarray(order, jnp.int32)
        self.step_count = jnp.asarray(step_count, jnp.int32)


def test_the_narrow_graph_is_masked_to_its_own_slots():
    """A bptt episode can never read or write a slot above its own count.

    `build_vertex_valid_static` is given the SHARED width and the graph's own
    valid vertices, so every slot the graph does not have is zero, and
    availability is that mask times the not-yet-chosen indicator. There is no
    step of any episode at which a padded slot is available.
    """
    import numpy as _np
    from alphagrad.approx.common.masks import (
        build_vertex_valid_static, vertex_avail_at_step)

    bptt_valid = list(range(1, BPTT_VALID + 1))
    rtrl_valid = list(range(1, RTRL_VALID + 1))
    vvs_b = build_vertex_valid_static(bptt_valid, SHARED)
    vvs_r = build_vertex_valid_static(rtrl_valid, SHARED)
    assert vvs_b.shape == vvs_r.shape == (SHARED,)
    assert float(_np.sum(_np.asarray(vvs_b))) == BPTT_VALID
    assert float(_np.sum(_np.asarray(vvs_r))) == RTRL_VALID
    # every slot the narrow graph does not have is dead in its mask
    assert _np.all(_np.asarray(vvs_b)[BPTT_VALID:] == 0.0)

    # and it stays dead at every step of a whole episode
    order = _np.zeros(BPTT_VALID, _np.int32)
    for step in range(BPTT_VALID + 1):
        if step:
            order[step - 1] = step          # eliminate 1, 2, 3, ...
        avail = _np.asarray(vertex_avail_at_step(
            _FakeState(order, step), vvs_b, SHARED, BPTT_VALID))
        assert avail.shape == (SHARED,)
        assert _np.all(avail[BPTT_VALID:] == 0.0), (step, avail[BPTT_VALID:])
        # and the slots it HAS are exactly the ones not yet chosen
        assert float(_np.sum(avail)) == float(BPTT_VALID - step)


def test_the_boundary_pad_carries_no_axis():
    """The padded per-vertex rows are inert.

    `axis_valid_static` is zero on every padded slot, so the agent reads no
    axis there. `compute_static_axis_state` and `build_pair_valid_mask` both
    take the width as an argument and fill only the rows their jaxpr has,
    which IS the pad.
    """
    import numpy as _np
    from alphagrad.approx.common.masks import vertex_axis_dims

    class _Aval:
        def __init__(self, shape):
            self.shape = shape

    class _Var:
        def __init__(self, shape):
            self.aval = _Aval(shape)

    class _Eqn:
        def __init__(self):
            self.outvars = [_Var((4, 5))]
            self.invars = [_Var((4, 5, 6))]

    class _Jaxpr:
        eqns = [_Eqn() for _ in range(BPTT_EQNS)]

    out_n, in_n = vertex_axis_dims(_Jaxpr(), SHARED)
    assert out_n.shape == in_n.shape == (SHARED,)
    # the graph's own rows carry its shapes ...
    assert _np.all(out_n[:BPTT_EQNS] == 2)
    assert _np.all(in_n[:BPTT_EQNS] == 3)
    # ... and every row past them is the pad, which carries nothing
    assert _np.all(out_n[BPTT_EQNS:] == 0)
    assert _np.all(in_n[BPTT_EQNS:] == 0)


def test_the_shared_space_is_the_maximum_and_never_less():
    """The shared width is the MAX over the graphs. A width below a graph's
    own count would cut that graph's vertices off, which is why ppo.main
    raises on a negative pad rather than clipping."""
    assert max(BPTT_EQNS, RTRL_EQNS) == SHARED
    assert SHARED >= BPTT_EQNS and SHARED >= RTRL_EQNS


# ---------------------------------------------------------------------------
# 8. THE PER-GRAPH FACE-INIT OFFSET (owner ruling 2, 2026-09-22)
# ---------------------------------------------------------------------------
#
# The shared face head keeps ONE (B, Bs). Each graph carries a fixed offset so
# that under a=1 and kappa=0.3 each DAG starts at one requested approximation
# and 0.3 requested skips per plan from its OWN reference face count F, which
# is 42 on bptt and 65 on rtrl (job 67505).

F_BPTT, F_RTRL = 42, 65
A_PER_PLAN, KAPPA = 1.0, 0.3


def _bias_pair(F):
    from alphagrad.approx.common.agent_factory import (
        derive_face_none_bias, derive_face_skip_bias)
    from alphagrad.approx.unified_face_head import (
        FACE_SLOTS, NUM_APPROX_OPS)
    return (derive_face_none_bias(F, FACE_SLOTS, NUM_APPROX_OPS - 1,
                                  A_PER_PLAN),
            derive_face_skip_bias(F, KAPPA))


def test_each_graph_starts_at_the_requested_rates_from_its_own_F():
    """The point of the offset: BOTH graphs start at a=1 and kappa=0.3."""
    from alphagrad.approx.common.agent_factory import expected_face_counts
    from alphagrad.approx.unified_face_head import (
        FACE_SLOTS, NUM_APPROX_OPS)

    for F in (F_BPTT, F_RTRL):
        B, Bs = _bias_pair(F)
        e_a, e_k = expected_face_counts(float(F), FACE_SLOTS,
                                        NUM_APPROX_OPS - 1, B, Bs)
        assert abs(e_a - A_PER_PLAN) < 1e-6, (F, e_a)
        assert abs(e_k - KAPPA) < 1e-6, (F, e_k)
    # and the two graphs genuinely need different numbers
    assert _bias_pair(F_BPTT) != _bias_pair(F_RTRL)


def _tiny_agent(key_seed=0):
    """An object with just the `face_path_policy.head` the offset reads."""
    import jax.random as jrand
    from alphagrad.approx.unified_face_head import UnifiedFaceHead

    class _FPP:
        pass

    class _A:
        pass

    head = UnifiedFaceHead(8, key=jrand.PRNGKey(key_seed))
    fpp = _FPP()
    fpp.head = head
    a = _A()
    a.face_path_policy = fpp
    return a, head


def test_the_offset_lands_on_the_same_logits_the_init_bias_writes():
    import numpy as _np
    from alphagrad.approx.common.agent_factory import face_logit_offset_vector
    from alphagrad.approx.unified_face_head import (
        FACE_SLOTS, OP_NONE, O_QUANT, O_SKIP, S_OP, slot_base)

    agent, head = _tiny_agent()
    vec = _np.asarray(face_logit_offset_vector(agent, 1.25, 0.75))
    assert vec.shape == (head.layout.width,)
    want = {slot_base(s, head.layout) + S_OP + OP_NONE: 1.25
            for s in range(FACE_SLOTS)}
    want[O_SKIP] = -0.75
    want[O_QUANT] = -1.25
    for i, v in enumerate(vec):
        assert abs(float(v) - want.get(i, 0.0)) < 1e-6, (i, v)
    # zero deltas mean no offset at all, which is a one-graph run
    assert face_logit_offset_vector(agent, 0.0, 0.0) is None


def test_the_offset_moves_the_logits_by_exactly_the_offset():
    """And it is added BEFORE the clamp, where the init bias sits.

    An offset added after the clamp would not be bounded by it, so the two
    ways of biasing the head would stop being the same parameterization.
    """
    import jax.numpy as jnp
    import numpy as _np
    from alphagrad.approx.common.agent_factory import face_logit_offset_vector
    from alphagrad.approx.unified_face_head import (
        set_face_logit_offset, set_logit_clamp, face_logit_offset)

    agent, head = _tiny_agent(1)
    ctx = jnp.arange(8, dtype=jnp.float32) / 8.0
    try:
        set_logit_clamp(0.0)
        set_face_logit_offset(None)
        base = _np.asarray(head.logits(ctx))
        off = face_logit_offset_vector(agent, 0.5, 0.25)
        set_face_logit_offset(off)
        moved = _np.asarray(head.logits(ctx))
        assert _np.allclose(moved - base, _np.asarray(off), atol=1e-5)

        # UNDER THE CLAMP it is c*tanh((z + off)/c), not c*tanh(z/c) + off.
        # `base` is the unclamped projection, so the expected value is built
        # from it directly.
        set_face_logit_offset(off)
        set_logit_clamp(15.0)
        clamped = _np.asarray(head.logits(ctx))
        expect = 15.0 * _np.tanh((base + _np.asarray(off)) / 15.0)
        assert _np.allclose(clamped, expect, atol=1e-4)
        wrong = 15.0 * _np.tanh(base / 15.0) + _np.asarray(off)
        assert not _np.allclose(clamped, wrong, atol=1e-4)
    finally:
        set_logit_clamp(0.0)
        set_face_logit_offset(None)
        assert face_logit_offset() is None


def test_the_sampler_and_the_replay_read_the_same_offset():
    """dsnn-dfw.95 FROM THE OTHER SIDE.

    When the sampler and the replay disagreed by a bias, the PPO ratio of
    every plan that approximated went to 1e-4 and those plans left the policy
    gradient. `logits` is the ONE funnel both paths come through, so the
    offset is read there and NOWHERE else -- any second reader could drift.
    The barrier that made the two agree in the first place stays.
    """
    import ast
    import inspect

    from alphagrad.approx import unified_face_head as ufh

    tree = ast.parse(inspect.getsource(ufh))
    readers = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id == "FACE_LOGIT_OFFSET":
                readers.add(node.name)
    assert readers == {"logits", "set_face_logit_offset",
                       "face_logit_offset"}, sorted(readers)

    # the .95 barrier is still in the clamp
    src = inspect.getsource(ufh.UnifiedFaceHead.logits)
    assert "optimization_barrier" in src, (
        "the dsnn-dfw.95 barrier left `logits`; without it the loss program "
        "fuses the projection into the bound and the replay scores a "
        "different number than the sampler")
    # and the offset is applied before it
    assert src.index("FACE_LOGIT_OFFSET") < src.index("optimization_barrier")

    # both entry points take z from `logits`, so neither can miss the offset
    for fn in (ufh.UnifiedFaceHead.sample,):
        assert "self.logits(" in inspect.getsource(fn)


def test_both_graphs_offsets_are_consistent_with_one_shared_head():
    """The head holds the PRIMARY graph's pair and the other graph carries
    the difference, so the EFFECTIVE pair on each graph is its own."""
    import numpy as _np
    from alphagrad.approx.common.agent_factory import face_logit_offset_vector
    from alphagrad.approx.unified_face_head import (
        OP_NONE, O_SKIP, S_OP, slot_base)

    agent, head = _tiny_agent(2)
    B_p, Bs_p = _bias_pair(F_BPTT)        # bptt named first -> the head's
    B_s, Bs_s = _bias_pair(F_RTRL)
    assert face_logit_offset_vector(agent, B_p - B_p, Bs_p - Bs_p) is None
    vec = _np.asarray(face_logit_offset_vector(agent, B_s - B_p, Bs_s - Bs_p))
    i_none = slot_base(0, head.layout) + S_OP + OP_NONE
    assert abs(float(vec[i_none]) - (B_s - B_p)) < 1e-6
    assert abs(float(vec[O_SKIP]) + (Bs_s - Bs_p)) < 1e-6
    # the head's own pair plus the offset IS the second graph's pair
    assert abs((B_p + float(vec[i_none])) - B_s) < 1e-6
    assert abs((Bs_p - float(vec[O_SKIP])) - Bs_s) < 1e-6
