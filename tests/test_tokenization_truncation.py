"""Tests for the jaxpr-tokenization truncation diagnostic.

The env's ``_callback`` slices the un-truncated jaxpr token sequence
to ``MAX_TOKENS`` and pads. Before the slice it calls
``_record_tokenization_truncation`` which (a) emits a one-time
``warnings.warn`` and (b) bumps a per-process counter. The
counter + max-observed-length are popped by the CPU actor each
rollout and forwarded to wandb via ``tokenization/truncated_count``
and ``tokenization/max_observed_len`` — same plumbing pattern as
``nan_skip_count``.
"""

from __future__ import annotations

import warnings

import pytest


def _reset_env_truncation_state():
    """Force-reset the module-level counters so tests are independent
    of import order / earlier truncations."""
    from alphagrad.approx import env

    env._TOKENIZATION_TRUNCATION_COUNT[0] = 0
    env._TOKENIZATION_TRUNCATION_MAX_LEN[0] = 0
    env._TOKENIZATION_TRUNCATION_OVERFLOW_SUM[0] = 0
    env._TOKENIZATION_TRUNCATION_WARNED[0] = False


def test_no_truncation_when_under_max_is_noop():
    _reset_env_truncation_state()
    from alphagrad.approx.env import (
        MAX_TOKENS,
        _record_tokenization_truncation,
        consume_tokenization_truncation_stats,
    )

    # Sequence within budget — no counter bump, no warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _record_tokenization_truncation(MAX_TOKENS - 1)
        _record_tokenization_truncation(MAX_TOKENS)  # exactly at boundary
    assert caught == []

    stats = consume_tokenization_truncation_stats()
    assert stats == {"count": 0, "max_observed_len": 0, "overflow_sum": 0}


def test_truncation_emits_warning_once_and_counts():
    _reset_env_truncation_state()
    from alphagrad.approx.env import (
        MAX_TOKENS,
        _record_tokenization_truncation,
        consume_tokenization_truncation_stats,
    )

    # Three over-budget calls — the first emits a UserWarning,
    # subsequent calls are silent but still counted.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _record_tokenization_truncation(MAX_TOKENS + 100)
        _record_tokenization_truncation(MAX_TOKENS + 5000)
        _record_tokenization_truncation(MAX_TOKENS + 1)

    # Exactly one warning fired.
    truncation_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "tokenization truncated" in str(w.message)
    ]
    assert len(truncation_warnings) == 1
    # Warning message carries the actual raw length so the user knows
    # how much headroom to add to MAX_TOKENS.
    assert f"raw_len={MAX_TOKENS + 100}" in str(truncation_warnings[0].message)

    # All three truncations counted; max_observed_len reflects the
    # largest; overflow_sum is the per-episode information loss (in
    # tokens): 100 + 5000 + 1 = 5101.
    stats = consume_tokenization_truncation_stats()
    assert stats["count"] == 3
    assert stats["max_observed_len"] == MAX_TOKENS + 5000
    assert stats["overflow_sum"] == 100 + 5000 + 1


def test_consume_resets_count_and_max_len_but_not_warned_flag():
    _reset_env_truncation_state()
    from alphagrad.approx import env
    from alphagrad.approx.env import (
        MAX_TOKENS,
        _record_tokenization_truncation,
        consume_tokenization_truncation_stats,
    )

    _record_tokenization_truncation(MAX_TOKENS + 10)
    first = consume_tokenization_truncation_stats()
    assert first["count"] == 1
    assert first["max_observed_len"] == MAX_TOKENS + 10
    assert first["overflow_sum"] == 10

    # After consume the counters are zero but ``warned`` stays sticky so
    # the warning never re-fires within the process.
    second = consume_tokenization_truncation_stats()
    assert second == {"count": 0, "max_observed_len": 0, "overflow_sum": 0}
    assert env._TOKENIZATION_TRUNCATION_WARNED[0] is True

    # Another truncation increments the counter but does NOT re-emit
    # the warning (sticky flag).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _record_tokenization_truncation(MAX_TOKENS + 20)
    truncation_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "tokenization truncated" in str(w.message)
    ]
    assert len(truncation_warnings) == 0
    third = consume_tokenization_truncation_stats()
    assert third["count"] == 1
    assert third["max_observed_len"] == MAX_TOKENS + 20
    assert third["overflow_sum"] == 20


def test_max_observed_len_tracks_largest_not_latest():
    _reset_env_truncation_state()
    from alphagrad.approx.env import (
        MAX_TOKENS,
        _record_tokenization_truncation,
        consume_tokenization_truncation_stats,
    )

    _record_tokenization_truncation(MAX_TOKENS + 9000)
    _record_tokenization_truncation(MAX_TOKENS + 100)
    _record_tokenization_truncation(MAX_TOKENS + 50)

    stats = consume_tokenization_truncation_stats()
    assert stats["count"] == 3
    # Largest of the three, not the latest.
    assert stats["max_observed_len"] == MAX_TOKENS + 9000
    # Overflow sum totals across ALL three truncations.
    assert stats["overflow_sum"] == 9000 + 100 + 50


def test_overflow_sum_zero_when_counter_at_max_boundary():
    """``raw_len == MAX_TOKENS`` is NOT a truncation — overflow stays
    zero. Boundary check so we don't accidentally inflate the
    information-loss metric by 1 on every successful tokenization."""
    _reset_env_truncation_state()
    from alphagrad.approx.env import (
        MAX_TOKENS,
        _record_tokenization_truncation,
        consume_tokenization_truncation_stats,
    )

    for _ in range(10):
        _record_tokenization_truncation(MAX_TOKENS)
    stats = consume_tokenization_truncation_stats()
    assert stats == {"count": 0, "max_observed_len": 0, "overflow_sum": 0}
