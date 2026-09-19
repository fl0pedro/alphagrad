from __future__ import annotations

import pytest

from alphagrad.approx.common.core_budget import (
    check_disjoint, core_ids, layout_slices, node_core_layout)


def _eight_gpu():
    return node_core_layout(64, 7, trainer_cores=8, cores_per_actor=2,
                            oracle_cores=4)


def _four_gpu():
    return node_core_layout(64, 3, trainer_cores=8, cores_per_actor=2,
                            oracle_cores=4)


def test_eight_gpu_budget_is_the_ruling():
    lay = _eight_gpu()
    assert lay.trainer == (0, 8)
    assert len(lay.timing_actors) == 7
    assert all(w == 2 for _, w in lay.timing_actors)
    assert lay.oracle == (60, 4)
    assert lay.spare == tuple(range(22, 60))


def test_four_gpu_budget_is_the_ruling():
    lay = _four_gpu()
    assert lay.trainer == (0, 8)
    assert len(lay.timing_actors) == 3
    assert lay.oracle == (60, 4)
    assert lay.spare == tuple(range(14, 60))


@pytest.mark.parametrize("layout_fn", [_eight_gpu, _four_gpu])
def test_every_slice_is_disjoint(layout_fn):
    lay = layout_fn()
    check_disjoint(lay)
    held = set()
    for _name, base, width in layout_slices(lay):
        s = set(range(base, base + width))
        assert not (s & held)
        held |= s
    assert held <= set(range(lay.n_logical))


def test_a_budget_that_does_not_fit_raises():
    with pytest.raises(ValueError, match="does not fit"):
        node_core_layout(64, 30, trainer_cores=8, cores_per_actor=2,
                         oracle_cores=4)


def test_core_ids_are_absolute_cpus_of_the_job_mask():
    # The SLURM mask on the 8-GPU nodes is 0-31,128-159, so a position is not
    # a cpu id.
    cpus = tuple(range(0, 32)) + tuple(range(128, 160))
    lay = _eight_gpu()
    assert core_ids(cpus, *lay.trainer) == tuple(range(0, 8))
    assert core_ids(cpus, *lay.timing_actors[0]) == (8, 9)
    assert core_ids(cpus, *lay.oracle) == (156, 157, 158, 159)
    with pytest.raises(ValueError, match="leaves"):
        core_ids(cpus, 60, 8)


def test_launcher_constants_pin_the_budget():
    import importlib.util
    import pathlib
    p = (pathlib.Path(__file__).resolve().parents[1]
         / "tools" / "gen_fq_launchers.py")
    spec = importlib.util.spec_from_file_location("_gen_fq", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.THESIS_CORE_BUDGET_CPUS == 64
    assert mod.THESIS_CORE_BUDGET[8] == {
        "trainer": 8, "per_actor": 2, "oracle": 4}
    assert mod.THESIS_CORE_BUDGET[4] == {
        "trainer": 8, "per_actor": 2, "oracle": 4}
    for gpus in (4, 8):
        check_disjoint(mod.thesis_core_layout(gpus))
