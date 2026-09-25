from __future__ import annotations

import inspect
import re
import sys
import types

import pytest

from alphagrad.approx.common.grad_oracle_async import AsyncGradOracle


class _Ref:
    def __init__(self, value):
        self.value = value


# Ray's calls as the oracle uses them. Every answer is ready at once.
_RAY = types.SimpleNamespace(
    wait=lambda refs, num_returns=None, timeout=None: (list(refs), []),
    get=lambda ref: ref.value,
    kill=lambda actor, no_restart=False: None)


class _Actor:
    # The oracle actor answers (status, rel_l2) and changes no trainer state.
    def __init__(self, rel_of):
        def remote(args_np, order, probe_seed, rule=None):
            rel = rel_of[order]
            return _Ref(("pass" if rel <= 1e-3 else "fail", rel))
        self.check = types.SimpleNamespace(remote=remote)


@pytest.mark.parametrize("episodes, shown", [
    ([[1.485e-16]], "1.485e-16"),
    ([[8.7e-15], [1.485e-16]], "8.700e-15"),
    ([[8.7e-15], [float("nan")]], "nan"),
    ([[]], "none"),
])
def test_the_exit_summary_shows_the_largest_rel_l2_of_the_actor_results(
        monkeypatch, episodes, shown):
    from alphagrad.approx.ppo import _grad_oracle_exit_summary
    monkeypatch.setitem(sys.modules, "ray", _RAY)
    rel_of = {(ep, k): rel for ep, rels in enumerate(episodes)
              for k, rel in enumerate(rels)}
    actor = _Actor(rel_of)
    oracle = AsyncGradOracle(None, timeout_s=60.0,
                             actor_factory=lambda: actor,
                             arg_resolver=lambda ep: ())
    try:
        for ep, rels in enumerate(episodes):
            if ep:
                oracle.take_results()
            oracle.submit(ep, 1234, [{"order": (ep, k), "plan_hashes": ["aa"]}
                                     for k in range(len(rels))])
        oracle.drain(5.0)
        line = _grad_oracle_exit_summary(oracle, 0.0)
    finally:
        oracle.close()
    c = oracle.counts()
    assert c["pending"] == 0 and c["submitted"] == len(rel_of), c
    m = re.search(r"\(max rel_l2 seen (\S+)\)$", line)
    assert m and m.group(1) == shown, line


def test_main_prints_the_exit_summary_and_no_process_local_maximum():
    import alphagrad.approx.ppo as ppo

    src = inspect.getsource(ppo.main)
    start = src.index("_GRAD_ORACLE.drain(")
    block = src[start:src.index("_GRAD_ORACLE.close()", start)]
    assert "_GRAD_ORACLE_STATS" not in block, block
    assert "_grad_oracle_exit_summary(" in block, block
