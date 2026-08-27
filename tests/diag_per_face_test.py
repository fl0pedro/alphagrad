"""``--diag-per-face``: the per-FACE legal factor set, and the verdict that
motivated it.

The claim under test in :func:`test_free_pair_is_not_rejected_by_construction`
is the one the design hinged on: "the head hardcodes ``factor = gcd(N_i, N_j)``
and ``diag_pair_factor_space`` returns that same gcd as ``base``, so
``d = factor // base == 1`` always and ``rule_is_legal``'s final ``d > 1`` test
rejects DIAG BY CONSTRUCTION." It is FALSE -- a free pair returns ``base = 1``,
not ``base = gcd`` -- and the rest of these tests pin what the real per-face
gap is instead.
"""
import contextlib
import math
import os

os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax import inline_call_primitives
from graphax.sparse.micro_actions import (
    Compress, Diag, Quant, apply_compress, apply_diag, apply_quant)

from alphagrad.approx.common import masks as M
from alphagrad.approx.common.masks import (
    LiveVertexMaskOracle, couple_quant_rules, diag_pair_factor_space,
    diag_pair_legal_factors, diag_valid_mask, make_live_masked_hook,
    project_diag_to_face, rule_is_legal, set_diag_per_face)

try:
    from jax.extend.core import ClosedJaxpr
except ImportError:                                        # pragma: no cover
    from jax._src.core import ClosedJaxpr

N_AX = 8


# --------------------------------------------------------------------------
# A graph whose ONE interesting vertex has faces with DIFFERENT gcds.
#
#   e = A @ x        A: (12, 6)   ->  e: (12,)
#   return B @ e, C @ e           B: (8, 12), C: (9, 12)
#
# Eliminating ``e`` visits two faces, one per consumer. gcd(8, 12) = 4 while
# gcd(9, 12) = 3, so the two faces do not share a single legal factor > 1 --
# which is precisely the situation a per-VERTEX factor cannot serve.
# --------------------------------------------------------------------------
_A = jnp.asarray(np.linspace(0.1, 0.9, 12 * 6, dtype=np.float32).reshape(12, 6))
_B = jnp.asarray(np.linspace(0.2, 0.8, 8 * 12, dtype=np.float32).reshape(8, 12))
_C = jnp.asarray(np.linspace(0.3, 0.7, 9 * 12, dtype=np.float32).reshape(9, 12))
_X = jnp.asarray(np.linspace(0.1, 0.9, 6, dtype=np.float32))


def _split_gcd(x):
    e = _A @ x
    return _B @ e, _C @ e


def _oracle_for(fn, xs):
    cj = jax.make_jaxpr(fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    closed = cj if jx is cj.jaxpr else ClosedJaxpr(jx, consts)
    o = LiveVertexMaskOracle(closed.jaxpr, list(closed.literals), list(xs),
                             tuple(range(len(xs))), max_axes=N_AX)
    return closed, o


def _all_faces(fn, xs):
    """``(oracle, [(vertex, face_index, live SparseTensor), ...])``.

    The tensors are JAX TRACERS owned by the oracle's IncrementalJaxpr trace.
    Reading their index metadata (``diag_valid_mask``,
    ``diag_pair_factor_space``) is safe anywhere, but APPLYING a transform
    touches ``val`` and must happen inside that trace or jax raises
    ``UnexpectedTracerError`` -- hence :func:`_in_trace`.
    """
    closed, o = _oracle_for(fn, xs)
    out = []
    for v in range(1, len(closed.jaxpr.eqns) + 1):
        try:
            faces = o.probe_faces(v, approx=True)
        except Exception:                                  # pragma: no cover
            faces = []
        for k, st in enumerate(faces):
            out.append((v, k, st))
        o.advance(v, rules=())
    return o, out


@contextlib.contextmanager
def _in_trace(o):
    """Re-enter the oracle's trace so probe tensors can be transformed."""
    from jax._src import core as _jcore
    with _jcore.set_current_trace(o._incrs[True].trace):
        yield


def _free_pairs(st):
    """Legal pairs on ``st`` that are NOT already coupled (``base == 1``)."""
    vm = diag_valid_mask(st, N_AX)
    got = []
    for i in range(N_AX):
        for j in range(N_AX):
            if not vm[i, j]:
                continue
            base, span = diag_pair_factor_space(st, i, j)
            if base == 1 and span > 1:
                got.append((i, j, base, span))
    return got


# ==========================================================================
# 1. THE VERDICT
# ==========================================================================
def test_free_pair_is_not_rejected_by_construction():
    """REFUTES ``d == 1 by construction``.

    For a FREE pair ``diag_pair_factor_space`` returns ``base = 1``, so the
    head's ``factor = gcd`` gives ``d = gcd // 1 = gcd > 1`` and
    ``rule_is_legal``'s ``d > 1 and span % d == 0`` test PASSES. The rejects
    measured in production are therefore NOT this.
    """
    _o, faces = _all_faces(_split_gcd, [_X])
    checked = 0
    for _v, _k, st in faces:
        dims = tuple(st.out_dims) + tuple(st.primal_dims)
        for i, j, base, span in _free_pairs(st):
            assert base == 1, "a free pair must report base == 1, not the gcd"
            assert span == math.gcd(int(dims[i].logical_size),
                                    int(dims[j].logical_size))
            # This is EXACTLY the request the live head emits for a free pair
            # whose nominal sizes match the live ones.
            assert rule_is_legal(st, Diag(i, j, span), max_dims=N_AX), (
                f"free pair {(i, j)} with factor=gcd={span} must be LEGAL")
            checked += 1
    assert checked, "no free pair found -- the fixture stopped exercising this"


def test_coupled_pair_is_the_one_that_can_reject():
    """The complement: an already-coupled pair has ``base = m > 1``, and a
    request of ``factor = m`` is the no-op ``span == 1`` case that
    ``diag_valid_mask`` masks out. This is the class that dominates the live
    skips -- an idempotent re-request, correctly refused."""
    o, faces = _all_faces(_split_gcd, [_X])
    st = None
    pair = None
    for _v, _k, cand in faces:
        fp = _free_pairs(cand)
        if fp:
            st, pair = cand, fp[0]
            break
    assert st is not None
    i, j, _base, span = pair
    with _in_trace(o):
        coupled = apply_diag(st, Diag(i, j, span))
    base2, span2 = diag_pair_factor_space(coupled, i, j)
    assert base2 == span, "after Diag(factor=g) the pair reports base == g"
    assert span2 == 1, "and nothing finer is legal -- span collapses to 1"
    assert not rule_is_legal(coupled, Diag(i, j, span), max_dims=N_AX)
    assert diag_pair_legal_factors(coupled, i, j) == [], (
        "an exhausted pair has an EMPTY legal set, so no projection can "
        "repair it -- the skip is correct, not a masking bug")


# ==========================================================================
# 2. THE PER-FACE LEGAL SET
# ==========================================================================
def test_faces_of_one_vertex_have_different_legal_factor_sets():
    """The premise of per-FACE masking: one vertex, two faces, disjoint legal
    factors -- so NO single per-vertex factor can be legal on both."""
    closed, o = _oracle_for(_split_gcd, [_X])
    found = None
    for v in range(1, len(closed.jaxpr.eqns) + 1):
        try:
            faces = o.probe_faces(v, approx=True)
        except Exception:                                  # pragma: no cover
            faces = []
        if len(faces) >= 2:
            sets = []
            for st in faces:
                s = set()
                for i, j, _b, _sp in _free_pairs(st):
                    s |= set(diag_pair_legal_factors(st, i, j))
                sets.append(s)
            pick = next(
                ((a, b) for a in range(len(sets)) for b in range(len(sets))
                 if sets[a] and sets[b] and (sets[a] - sets[b])), None)
            if pick is not None:
                found = (v, [faces[pick[0]], faces[pick[1]]],
                         [sets[pick[0]], sets[pick[1]]])
                break
        o.advance(v, rules=())
    assert found is not None, (
        "fixture no longer produces a vertex whose faces disagree on the "
        "legal factor set")
    v, faces, sets = found
    # A factor legal on one face but not the other: the old per-vertex choice
    # is illegal on at least one face, the projection is legal on BOTH.
    only_a = sets[0] - sets[1]
    assert only_a, f"faces of v={v} should disagree; got {sets}"
    f_a = sorted(only_a)[-1]
    for k, st in enumerate(faces[:2]):
        i, j = _free_pairs(st)[0][:2]
        req = Diag(i, j, f_a)
        alt = project_diag_to_face(st, req, max_dims=N_AX,
                                   factor_rule="largest")
        assert alt is not None, f"face {k} must afford SOME legal DIAG"
        assert rule_is_legal(st, alt, max_dims=N_AX), (
            f"the projection must be legal on face {k}")
        assert alt.factor in diag_pair_legal_factors(st, alt.i, alt.j)


def test_projection_only_ever_returns_a_legal_action():
    """Exhaustive over every face of the fixture and every requested factor in
    1..24: the projection is either ``None`` or legal. It must never invent an
    action the mask has not cleared."""
    _o, faces = _all_faces(_split_gcd, [_X])
    saw_alt = False
    for _v, _k, st in faces:
        for i in range(N_AX):
            for j in range(N_AX):
                if i == j:
                    continue      # graphax rejects i == j in the constructor
                for want in range(1, 25):
                    alt = project_diag_to_face(st, Diag(i, j, want),
                                               max_dims=N_AX)
                    if alt is None:
                        continue
                    saw_alt = True
                    assert rule_is_legal(st, alt, max_dims=N_AX), (
                        f"projection {alt} illegal on face {_v}/{_k}")
    assert saw_alt, "the fixture never produced a projection"


@pytest.mark.parametrize("rule", ("largest", "nearest", "smallest"))
def test_factor_rules_pick_from_the_legal_set(rule):
    _o, faces = _all_faces(_split_gcd, [_X])
    for _v, _k, st in faces:
        for i, j, _b, _sp in _free_pairs(st):
            legal = diag_pair_legal_factors(st, i, j)
            alt = project_diag_to_face(st, Diag(i, j, 3), max_dims=N_AX,
                                       factor_rule=rule)
            assert alt is not None and alt.factor in legal
            if rule == "largest":
                assert alt.factor == legal[-1]
            elif rule == "smallest":
                assert alt.factor == legal[0]
            else:
                assert alt.factor == min(legal, key=lambda f: (abs(f - 3), f))


def test_bad_rule_name_raises():
    with pytest.raises(ValueError):
        set_diag_per_face(True, rule="finest")
    set_diag_per_face(False)


# ==========================================================================
# 3. FLAG OFF == TODAY, EXACTLY
# ==========================================================================
def _reference_hook(rules, *, max_dims=8, max_axes=8, stats=None):
    """The pre-change ``make_live_masked_hook`` body, verbatim, as the oracle
    the default path is pinned against."""
    coupled, _ = couple_quant_rules(rules)

    def _kind_of(rule):
        if isinstance(rule, Diag):
            return "diag"
        if isinstance(rule, Compress):
            return "compress"
        if isinstance(rule, Quant):
            return "quant"
        return "other"

    def _bump(key):
        if stats is None:
            return
        stats[key] = stats.get(key, 0) + 1

    def _hook(st):
        cur = st
        for rule in coupled:
            _kind = _kind_of(rule)
            if not rule_is_legal(cur, rule, max_dims=max_dims,
                                 max_axes=max_axes):
                _bump("skipped")
                _bump(f"skipped_{_kind}")
                continue
            try:
                if isinstance(rule, Diag):
                    cur = apply_diag(cur, rule)
                elif isinstance(rule, Compress):
                    cur = apply_compress(cur, rule)
                elif isinstance(rule, Quant):
                    cur = apply_quant(cur, rule)
                else:
                    _bump("skipped")
                    _bump(f"skipped_{_kind}")
                    continue
                _bump("applied")
                _bump(f"applied_{_kind}")
            except ValueError:                             # pragma: no cover
                _bump("skipped_raised")
                _bump(f"skipped_{_kind}")
        return cur

    return _hook


def _rule_battery():
    out = []
    for i in range(3):
        for j in range(3):
            if i == j:
                continue          # graphax rejects i == j in the constructor
            for f in (2, 3, 4, 6, 12):
                out.append(Diag(i, j, f))
    out += [Compress(axes=(0,), kind="mean"), Quant(dtype="bfloat16")]
    return out


def test_flag_off_is_bit_identical_to_the_reference():
    set_diag_per_face(False)
    assert not M.diag_per_face_enabled()
    o, faces = _all_faces(_split_gcd, [_X])
    assert faces
    for _v, _k, st in faces:
        for rule in _rule_battery():
            s_new, s_ref = {}, {}
            with _in_trace(o):
                got = make_live_masked_hook((rule,), max_dims=N_AX,
                                            stats=s_new)(st)
                exp = _reference_hook((rule,), max_dims=N_AX,
                                      stats=s_ref)(st)
            assert s_new == s_ref, (
                f"flag-off telemetry drifted on {rule}: {s_new} != {s_ref}")
            # Same object when skipped; same structure when applied.
            if s_ref.get("applied"):
                assert type(got) is type(exp)
                assert [d.logical_size for d in got.out_dims] == [
                    d.logical_size for d in exp.out_dims]
                assert [d.size for d in got.out_dims] == [
                    d.size for d in exp.out_dims]
                assert [d.is_sparse for d in got.out_dims] == [
                    d.is_sparse for d in exp.out_dims]
                assert [d.is_sparse for d in got.primal_dims] == [
                    d.is_sparse for d in exp.primal_dims]
            else:
                assert got is st and exp is st
            assert "repaired_diag" not in s_new
            assert "skipped_diag_noop" not in s_new


def test_flag_off_never_emits_the_new_keys_even_when_a_projection_exists():
    set_diag_per_face(False)
    o, faces = _all_faces(_split_gcd, [_X])
    for _v, _k, st in faces:
        fp = _free_pairs(st)
        if not fp:
            continue
        i, j, _b, span = fp[0]
        bogus = Diag(i, j, span + 1) if (span + 1) % span else Diag(i, j, 7)
        if rule_is_legal(st, bogus, max_dims=N_AX):
            continue
        s = {}
        with _in_trace(o):
            out = make_live_masked_hook((bogus,), max_dims=N_AX, stats=s)(st)
        assert out is st
        assert s == {"skipped": 1, "skipped_diag": 1}


# ==========================================================================
# 4. FLAG ON: a rule the old path drops is applied, and never raises
# ==========================================================================
def test_flag_on_repairs_an_illegal_factor():
    o, faces = _all_faces(_split_gcd, [_X])
    repaired = 0
    try:
        for _v, _k, st in faces:
            fp = _free_pairs(st)
            if not fp:
                continue
            i, j, _b, span = fp[0]
            legal = diag_pair_legal_factors(st, i, j)
            bogus = next((f for f in range(2, 40) if f not in legal), None)
            if bogus is None:                              # pragma: no cover
                continue
            set_diag_per_face(False)
            s_off = {}
            with _in_trace(o):
                off = make_live_masked_hook(
                    (Diag(i, j, bogus),), max_dims=N_AX, stats=s_off)(st)
            assert off is st
            assert s_off.get("applied", 0) == 0

            set_diag_per_face(True, rule="largest")
            s_on = {}
            with _in_trace(o):
                got = make_live_masked_hook(
                    (Diag(i, j, bogus),), max_dims=N_AX, stats=s_on)(st)
            assert got is not st
            assert s_on.get("applied_diag") == 1
            assert s_on.get("repaired_diag") == 1
            assert s_on.get("skipped_raised", 0) == 0
            repaired += 1
    finally:
        set_diag_per_face(False)
    assert repaired, "no repairable request found in the fixture"


def test_flag_on_never_raises_downstream():
    """``skipped_raised`` must stay 0: the projection is cleared by
    ``rule_is_legal`` before it is applied, so ``apply_diag`` never throws."""
    o, faces = _all_faces(_split_gcd, [_X])
    try:
        for factor_rule in ("largest", "nearest", "smallest"):
            set_diag_per_face(True, rule=factor_rule)
            for _v, _k, st in faces:
                for rule in _rule_battery():
                    s = {}
                    with _in_trace(o):
                        make_live_masked_hook(
                            (rule,), max_dims=N_AX, stats=s)(st)
                    assert s.get("skipped_raised", 0) == 0, (
                        f"{rule} under rule={factor_rule} raised downstream")
    finally:
        set_diag_per_face(False)


def test_exhausted_pair_is_counted_as_a_noop_not_a_failure():
    """The 83% class. Flag on, the idempotent re-request is still skipped --
    but it is now reported as ``skipped_diag_noop`` so the applied fraction
    stops reading a correct no-op as a masking failure."""
    o, faces = _all_faces(_split_gcd, [_X])
    try:
        for _v, _k, st in faces:
            fp = _free_pairs(st)
            if not fp:
                continue
            i, j, _b, span = fp[0]
            set_diag_per_face(True, rule="largest", repair_pair=False)
            s = {}
            with _in_trace(o):
                coupled = apply_diag(st, Diag(i, j, span))
                out = make_live_masked_hook(
                    (Diag(i, j, span),), max_dims=N_AX, stats=s)(coupled)
            assert out is coupled
            assert s.get("skipped_diag") == 1
            assert s.get("skipped_diag_noop") == 1
            assert s.get("skipped_raised", 0) == 0
            return
    finally:
        set_diag_per_face(False)
    pytest.skip("fixture produced no coupleable pair")


def test_env_var_default_is_off():
    """A fresh import with no env var must be OFF -- the four upcoming runs
    that do NOT pass the flag must be unaffected."""
    assert os.environ.get("ALPHAGRAD_DIAG_PER_FACE", "0") in ("0", "")
    set_diag_per_face(False)
    assert not M.diag_per_face_enabled()
