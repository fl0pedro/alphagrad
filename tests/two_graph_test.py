"""Two input DAGs in one run (bead dsnn-dfw.116, owner ruling 2026-09-22).

``--temporal-rule bptt rtrl`` runs both graphs of one target and alternates
per episode. These tests pin the four things that makes true:

1. THE ARGUMENT. One rule in, one string out -- the namespace a one-rule run
   carries is the one it carried before two rules existed, which is what
   makes the old behaviour byte for byte the old behaviour.
2. THE SCHEDULE. Even episodes on the first rule named, odd on the second, as
   a pure function of the episode number so a resume lands on the right graph.
3. THE ISOLATION. An update on one graph's episode does not move the other
   graph's multiplier, PopArt or archive.
4. THE ROUND TRIP. The per-graph state survives a checkpoint, and a resume
   whose rule list moved is refused rather than run on the other graph.

Nothing here builds a graph: the measured graph shapes are in the report, and
what is under test is the bookkeeping that decides which graph an episode is
on. `tests/temporal_rule_test.py` owns the graphs themselves.
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
    from alphagrad.approx.ppo import make_argparser

    p = make_argparser()
    base = ["--example", RSNN]
    assert p.parse_args(base + ["--temporal-rule", "rtrl"]).temporal_rule \
        == ["rtrl"]
    assert p.parse_args(
        base + ["--temporal-rule", "bptt", "rtrl"]).temporal_rule \
        == ["bptt", "rtrl"]
    assert p.parse_args(base).temporal_rule is None
    with pytest.raises(SystemExit):
        p.parse_args(base + ["--temporal-rule", "nonsense"])


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
