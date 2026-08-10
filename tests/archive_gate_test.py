"""Pin the archive insertion gate: sub-gate (destroyed) candidates never
reach the Pareto front when ALPHAGRAD_QUALITY_GATE_MIN is set."""
import numpy as np
import pytest
from alphagrad.approx.common.pareto_archive import ParetoArchive


def _arch():
    return ParetoArchive(["latency_ns", "peak_memory", "cosine_sim"], [0, 1, 3])


def test_subgate_candidate_rejected(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0.05")
    a = _arch()
    assert not a.add(np.array([-37e3, -1e6, 0.0, 0.0]), [1], 1)   # destroyed
    assert a.add(np.array([-140e3, -60e6, 0.0, 0.885]), [2], 1)   # honest
    assert len(a.pts) == 1


def test_gate_off_keeps_history(monkeypatch):
    monkeypatch.delenv("ALPHAGRAD_QUALITY_GATE_MIN", raising=False)
    a = _arch()
    assert a.add(np.array([-37e3, -1e6, 0.0, 0.0]), [1], 1)
