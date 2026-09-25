import pathlib
import sys

import jax.random as jrand
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ppo_resume_equivalence_test as _eq                      # noqa: E402

_ONE_EPISODE = ("--episodes", "1", "--checkpoint-every", "0")
_PPO, _KL, _KL_APPROX = 4, 9, 5


def _episode(work, tag, epochs, *extra):
    _eq._run(work, tag, *_ONE_EPISODE, "--ppo-epochs", str(epochs), *extra)
    d = _eq._dump(work, tag, 0)
    m = d["metrics"]
    assert len(m) == 13 and np.asarray(m[_KL]).shape == (7,), (
        "the metrics tuple changed its layout: "
        + str([np.asarray(x).shape for x in m]))
    return d


def _slot(d, slot):
    return np.asarray(d["metrics"][slot], np.float32)


# dsnn-dfw.98: --target-kl T zeroes the policy term of a minibatch whose KL exceeds T.
@pytest.mark.slow
def test_the_gate_zeroes_the_policy_term_past_the_target(tmp_path):
    k = jrand.PRNGKey(0)
    assert (np.asarray(jrand.split(k, 1)[0])
            == np.asarray(jrand.split(k, 2)[0])).all(), (
        "epoch 0 of a 1-epoch and of a 2-epoch update would draw different "
        "minibatches, so the two runs below are not comparable")
    off = _episode(tmp_path, "off", 2)
    wide = _episode(tmp_path, "wide", 2, "--target-kl", "1e30")
    tight = _episode(tmp_path, "tight", 2, "--target-kl", "1e-30")
    first = _episode(tmp_path, "first", 1, "--target-kl", "1e-30")

    problems = _eq._diff(off, wide, 0)
    assert not problems, (
        "a target the KL never reaches changed the update:\n"
        + "\n".join(problems))
    assert _slot(tight, _KL)[_KL_APPROX] > np.float32(1e-30)
    # Epoch 1 runs after the policy moved, so its KL exceeds 1e-30 and its policy term must be 0.
    assert _slot(tight, _PPO) == _slot(first, _PPO) / np.float32(2), (
        _slot(tight, _PPO), _slot(first, _PPO))
    assert _slot(off, _PPO) != _slot(tight, _PPO), (
        _slot(off, _PPO), _slot(tight, _PPO))
