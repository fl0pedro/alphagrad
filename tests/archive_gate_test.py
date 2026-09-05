"""Pin the archive insertion floor: infeasible (destroyed) candidates never
reach the Pareto front when a quality floor is given.

Until 2026-09-04 the archive read ALPHAGRAD_QUALITY_GATE_MIN; the variable
was deleted with the quality gate (ticket dsnn-3qm.9) and the floor is the
``quality_floor`` constructor argument, fed by ppo.py's --quality-floor."""
import numpy as np
from alphagrad.approx.common.pareto_archive import ParetoArchive


def _arch(**kw):
    return ParetoArchive(["latency_ns", "peak_memory", "cosine_sim"], [0, 1, 3],
                         **kw)


def test_infeasible_candidate_rejected():
    a = _arch(quality_floor=0.05)
    assert not a.add(np.array([-37e3, -1e6, 0.0, 0.0]), [1], 1)   # destroyed
    assert a.add(np.array([-140e3, -60e6, 0.0, 0.885]), [2], 1)   # feasible
    assert len(a.pts) == 1


def test_floor_is_inclusive_at_tau():
    a = _arch(quality_floor=0.9)
    assert a.add(np.array([-140e3, -60e6, 0.0, 0.9]), [2], 1)     # at the floor
    assert not a.add(np.array([-100e3, -50e6, 0.0, 0.8999]), [3], 1)


def test_no_floor_keeps_history():
    a = _arch()
    assert a.quality_floor is None
    assert a.add(np.array([-37e3, -1e6, 0.0, 0.0]), [1], 1)
