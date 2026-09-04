"""Pin how ALPHAGRAD_QUALITY_METRIC resolves after the 2026-08-28 swap.

The gradient cosine REPLACED the Jacobian cosine in the cosine slot:
`cosine` is a deprecated alias that resolves to `grad_cosine` wherever a
gradient is defined (any scalar-loss target) and to `jac_cosine` only for the
analytic AD benchmarks, which have no loss and no data generator.

`auto` was left on loss_drop by that swap. On 2026-09-02 the owner moved it
(ticket dsnn-3qm.39): `auto` is grad_cosine for scalar targets, jac_cosine for
the analytic benchmarks; loss_drop is selectable by name only.
"""
from __future__ import annotations

import os

import pytest

from alphagrad.approx.env import quality_metric


class _Scalar:
    """A trainable example: the traced target IS a scalar loss."""
    scalar_target = True


class _Analytic:
    """An analytic AD benchmark: the target is a full multi-output Jacobian."""
    scalar_target = False


# (env value, expected for a scalar-loss target, expected for an analytic one)
CASES = [
    ("auto",        "grad_cosine", "jac_cosine"),
    ("",            "grad_cosine", "jac_cosine"),
    ("loss_drop",   "loss_drop",   "loss_drop"),
    ("walk",        "loss_drop",   "loss_drop"),
    ("grad_cosine", "grad_cosine", "grad_cosine"),
    ("gradcos",     "grad_cosine", "grad_cosine"),
    ("jac_cosine",  "jac_cosine",  "jac_cosine"),
    ("jaccos",      "jac_cosine",  "jac_cosine"),
    # the deprecated alias: resolves by target kind
    ("cosine",      "grad_cosine", "jac_cosine"),
    ("cos",         "grad_cosine", "jac_cosine"),
    ("cosine_sim",  "grad_cosine", "jac_cosine"),
    ("none",        "none",        "none"),
    ("off",         "none",        "none"),
]


@pytest.mark.parametrize("want,scalar,analytic", CASES)
def test_quality_metric_resolution(monkeypatch, want, scalar, analytic):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", want)
    assert quality_metric(_Scalar()) == scalar
    assert quality_metric(_Analytic()) == analytic


def test_unknown_metric_is_a_hard_error(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "not_a_metric")
    with pytest.raises(ValueError, match="grad_cosine"):
        quality_metric(_Scalar())


def test_auto_is_grad_cosine_since_the_owner_ruling(monkeypatch):
    """With nothing selected the DEFAULT channel is the gradient cosine
    (owner ruling 2026-09-02); loss_drop is never the default."""
    monkeypatch.delenv("ALPHAGRAD_QUALITY_METRIC", raising=False)
    assert quality_metric(_Scalar()) == "grad_cosine"
    assert quality_metric(_Analytic()) == "jac_cosine"


def test_grad_cosine_k_defaults_to_one(monkeypatch):
    """K=1 is what keeps the channel at ONE exact execution per plan, and the
    2026-08-28 bake-off found K>1 strictly worse (0.874 -> 0.856 Pearson)."""
    from alphagrad.approx.env import _grad_cosine_k

    monkeypatch.delenv("ALPHAGRAD_GRAD_COSINE_K", raising=False)
    assert _grad_cosine_k() == 1
    monkeypatch.setenv("ALPHAGRAD_GRAD_COSINE_K", "4")
    assert _grad_cosine_k() == 4
    monkeypatch.setenv("ALPHAGRAD_GRAD_COSINE_K", "garbage")
    assert _grad_cosine_k() == 1


def test_probe_batch_index_zero_is_the_legacy_seed(monkeypatch):
    """index=0 must not move the loss-drop walk's batch."""
    from alphagrad.approx.env import _walk_seed

    assert _walk_seed("train", None) + 104729 * 0 == _walk_seed("train", None)
