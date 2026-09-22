"""The per-graph half of a run that alternates between two graphs.

OWNER RULING 2026-09-22 (bead dsnn-dfw.116). ``--temporal-rule bptt rtrl``
runs BOTH graphs of one target in one run and alternates per episode: even
episodes on the first rule named, odd on the second.

WHAT IS SHARED is the policy, the optimizer, the quality floor and the
preference conditioning -- one agent learns on both graphs and the token
stream, not a flag, tells it which one it is on.

WHAT IS PER GRAPH is in here: the band archive and its front, the reference
measurements the archive's bands are made of, the PopArt statistics and the
Lagrangian multiplier with its dual. Mixing any of them would compare two
graphs' costs as though they were one distribution; they are not, because the
graphs have different sizes (measured 2026-09-22 on RSNN_SHD: bptt 58 valid
vertices and 42 reference faces, rtrl 68 and 65).

This module holds no jax arrays of its own. The trainer hands it the values it
carries at a graph change and takes the other graph's back, so there is
exactly one place in the run that says which half of the state is which.
"""

from __future__ import annotations

from alphagrad.approx.common.rsnn_shd import (
    temporal_rule_for_episode,
    temporal_rule_list,
)


class GraphState:
    """Everything one graph of an alternating run owns.

    The three mutable members -- ``archive``, ``popart`` and ``lag_lambda`` --
    are the state a checkpoint is a state OF. The rest are facts about the
    graph, derived once at setup and never changed, kept here so that the
    episode that runs on this graph reads them from one place.
    """

    __slots__ = ("rule", "archive", "popart", "lag_lambda", "episodes",
                 "total_v", "num_valid", "face_F", "face_none_bias",
                 "face_skip_bias")

    def __init__(self, rule: str, *, lag_lambda: float):
        self.rule = str(rule)
        self.archive = None
        self.popart = None
        self.lag_lambda = float(lag_lambda)
        self.episodes = 0
        self.total_v = None
        self.num_valid = None
        self.face_F = None
        self.face_none_bias = None
        self.face_skip_bias = None

    def __repr__(self) -> str:
        return (f"GraphState(rule={self.rule!r}, episodes={self.episodes}, "
                f"lag_lambda={self.lag_lambda:g}, total_v={self.total_v}, "
                f"face_F={self.face_F})")


class GraphStates:
    """The store, one :class:`GraphState` per rule, in the order named.

    THE ORDER IS THE ALTERNATION. ``rules[0]`` runs on even episodes and
    ``rules[1]`` on odd ones, which is a pure function of the episode number,
    so a resume at episode N lands on the graph the uninterrupted run would
    have reached.
    """

    def __init__(self, rules, *, lag_init: float):
        rs = temporal_rule_list(rules)
        if not rs:
            raise ValueError(
                "a graph store needs at least one temporal rule; a target "
                "with no time steps has no graph to alternate between and "
                "must not build one.")
        if len(set(rs)) != len(rs):
            raise ValueError(
                f"the rules {list(rs)} name the same graph twice; each graph "
                f"owns its own archive, PopArt and multiplier, so two "
                f"entries under one name would be two states for one graph.")
        self.rules = rs
        self._states = {r: GraphState(r, lag_lambda=lag_init) for r in rs}

    # -- the schedule --------------------------------------------------
    def rule_for(self, episode: int) -> str:
        """The rule episode ``episode`` runs on."""
        return temporal_rule_for_episode(self.rules, episode)

    @property
    def alternating(self) -> bool:
        return len(self.rules) > 1

    # -- access --------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        for r in self.rules:
            yield self._states[r]

    def __getitem__(self, rule: str) -> GraphState:
        key = str(rule)
        state = self._states.get(key)
        if state is None:
            raise KeyError(
                f"this run holds no graph for the temporal rule {key!r}; it "
                f"runs {list(self.rules)}. A rule that is not one of them is "
                f"a graph nothing in the run was built for.")
        return state

    # -- the checkpoint ------------------------------------------------
    def to_json(self, archive_to_json) -> dict:
        """The per-graph state as plain JSON values.

        ``archive_to_json`` is ``common.checkpoint.pareto_archive_to_json``,
        passed in rather than imported so the checkpoint module may depend on
        this one and not the other way round.

        The PopArt accumulators are NOT written here: they are arrays, and the
        checkpoint's arithmetic half (``tree``) is where arrays go. This is
        the bookkeeping half.
        """
        return {
            "rules": list(self.rules),
            "graphs": {
                r: {
                    "rule": r,
                    "episodes": int(self._states[r].episodes),
                    "lag_lambda": float(self._states[r].lag_lambda),
                    "pareto_archive": archive_to_json(self._states[r].archive),
                    "total_v": self._states[r].total_v,
                    "num_valid": self._states[r].num_valid,
                    "face_F": self._states[r].face_F,
                    "face_none_bias": self._states[r].face_none_bias,
                    "face_skip_bias": self._states[r].face_skip_bias,
                }
                for r in self.rules
            },
        }

    def from_json(self, d: dict, archive_from_json) -> None:
        """Put a saved per-graph state back, in place.

        RAISES when the checkpoint's rules are not this run's, in that order:
        the alternation is indexed by the episode number, so a resume whose
        rules moved would put the other graph's archive and multiplier on
        every episode from here on.
        """
        saved = [str(r) for r in d.get("rules", ())]
        if saved != list(self.rules):
            raise ValueError(
                f"the checkpoint alternates over {saved} and this run over "
                f"{list(self.rules)}. The schedule is the episode number "
                f"modulo the number of rules, so resuming across a changed "
                f"rule list would run every episode on the other graph's "
                f"archive, PopArt and multiplier.")
        graphs = d.get("graphs", {})
        for r in self.rules:
            g = graphs.get(r)
            if g is None:
                raise ValueError(
                    f"the checkpoint holds no state for the graph {r!r} "
                    f"although it names it. A missing graph is a graph whose "
                    f"archive and multiplier would silently restart.")
            state = self._states[r]
            state.episodes = int(g["episodes"])
            state.lag_lambda = float(g["lag_lambda"])
            if state.archive is not None and g.get("pareto_archive"):
                archive_from_json(state.archive, g["pareto_archive"])

    # -- the log -------------------------------------------------------
    def describe(self) -> str:
        """One line per graph, for the start block."""
        return "\n".join(
            f"[two-graph] {r}: {'even' if i == 0 else 'odd'} episodes, "
            f"{self._states[r].total_v} vertices, "
            f"{self._states[r].num_valid} valid, "
            f"F={self._states[r].face_F} reference faces, "
            f"lambda0={self._states[r].lag_lambda:g}"
            for i, r in enumerate(self.rules))


def namespaced(key: str, rule: str | None) -> str:
    """``lagrangian/lambda`` -> ``lagrangian/bptt/lambda``.

    The rule goes after the first path element, so a dashboard that groups by
    the prefix keeps its groups and gains one level inside them. ``None`` --
    a run with one graph -- returns the key untouched, which is why a
    single-rule run's metric names do not move.
    """
    if rule is None:
        return key
    head, sep, tail = key.partition("/")
    if not sep:
        return f"{head}/{rule}"
    return f"{head}/{rule}/{tail}"


def namespace_log(log: dict, rule: str | None) -> dict:
    """``log`` with every key namespaced by ``rule``, plus the plain keys.

    BOTH FORMS ARE EMITTED. The namespaced key is the one that means
    something over a whole run -- two graphs write two series -- and the plain
    key is the episode's own value, which is what the existing dashboards
    read. A single-rule run emits the plain keys only, so nothing about its
    wandb record moves.
    """
    if rule is None:
        return dict(log)
    out = {namespaced(k, rule): v for k, v in log.items()}
    out.update(log)
    return out
