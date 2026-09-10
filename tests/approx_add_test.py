"""Ticket .56 / finding 73 -- THE FACE ADD as a flag:
``--approx-add {lossy,lossless}``.

A face accumulation multiplies lhs (the predecessor edge) by rhs (the
successor edge) into ``new`` and, when the predecessor-to-successor edge
ALREADY exists, ADDS ``new`` onto it. Every approximation lands on the
CONTRACTION side -- that is what the three slots are -- so what this flag
governs is how the ADD's two addends are made to MEET, ``env.approx_add()``:

* ``lossy``    -- both addends are forced into ONE container, the one the
  approximated ``new`` slot landed on, with the old edge projected onto it. env
  emits ``((lhs, rhs, new), (None, MatchFreshJoin(), None))``: a
  ``graphax.sparse.ops.join.FaceJoinPolicy`` at the ``jr`` position, which
  graphax hands BOTH addends. The declared default.
* ``lossless`` -- the sum's support is the UNION of the two supports, so no
  non-zero of either addend is dropped. env emits the two-op form with the join
  triple ALL-None: graphax's sparse ``+`` already builds the union container
  (meta gcd, block lcm), so there is nothing for a policy to do.

Pinned here, on a toy graph (never TLM):

1. Under BOTH values the ``new`` slot's hook is installed at exactly ONE
   graphax site, ``res:new``. That is the root fix for finding 72's fault 1:
   the retired ``--approx-old same`` installed the SAME hook object at
   ``res:new`` AND at ``jr``, the pre-existing old edge, so one wire row ran on
   two tensors whose logical dims agreed and whose STORAGE did not -- a legal
   block subdivision on one, an idempotent no-op on the other -- and one
   legality mask answered for one of them.
2. ``lossy`` puts a POLICY, not the hook, on the old edge, and the policy
   returns two STRUCTURALLY IDENTICAL addends. ``lossless`` puts nothing there.
3. The retired names RAISE and name the replacement: ``same`` and ``exact``
   described a different computation, so an alias would silently change the
   measured object.
4. Every plan-log record carries the configuration that measured it, under the
   renamed ``approx_add`` key.
5. ``ppo.py`` declares ``--approx-add`` with the default ``lossy``, refuses
   ``--approx-old``, and ``landscape_map``'s restated choice list still agrees
   with env's.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.pop("ALPHAGRAD_PLAN_LOG", None)

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402
from graphax import IncrementalJaxpr                            # noqa: E402
from graphax.core import _stable_var_index                      # noqa: E402
from graphax.sparse.micro_actions import QUANT_DTYPES           # noqa: E402

import alphagrad.approx.env as envmod                           # noqa: E402
from alphagrad.approx.env import (                              # noqa: E402
    FACE_SLOTS,
    MAX_FACES,
    MAX_RULES_PER_VERTEX,
    QUANT_SENTINEL,
    StepAction,
    VertexEliminationEnv,
)

# Deliberately NOT bf16-representable, so a bf16 Quant on the old edge is a
# real change of the tensor, not a structural no-op.
_B = jnp.asarray(
    (np.arange(16, dtype=np.float64).reshape(4, 4) * 0.1234567 + 0.7654321)
    .astype(np.float32))
_W = jnp.asarray(np.arange(20, dtype=np.float32).reshape(5, 4) / 19.0)
_X4 = jnp.asarray(np.linspace(0.1, 0.9, 4, dtype=np.float32))


def _join_mat(x):
    """Two paths from ``x`` into ``w``: eliminating vertex 2 first CREATES the
    edge ``x -> w``; eliminating vertex 1 afterwards MERGES into it, so the face
    ``x -> u -> w`` has an OLD edge and the join runs."""
    u = jnp.sin(x)                   # vertex 1
    v = _B @ x                       # vertex 2
    w = u + v                        # vertex 3
    return jnp.sum(_W @ w)           # vertices 4, 5 (scalar loss)


def _bf16_index() -> int:
    for i, d in enumerate(QUANT_DTYPES):
        if jnp.dtype(d) == jnp.dtype(jnp.bfloat16):
            return i
    raise AssertionError(f"no bfloat16 in QUANT_DTYPES: {QUANT_DTYPES}")


def _toy():
    closed = jax.make_jaxpr(_join_mat)(_X4)
    cfg = SimpleNamespace(jaxpr=closed.jaxpr, argnums=(0,))
    ij = IncrementalJaxpr(closed.jaxpr, (0,), list(closed.literals), [_X4],
                          track_faces=False)
    vidx = _stable_var_index(closed.jaxpr)
    x_var = closed.jaxpr.invars[0]
    w_var = closed.jaxpr.eqns[2].outvars[0]
    return closed, cfg, ij, (vidx[x_var], vidx[w_var]), (x_var, w_var)


def _quant_new_row():
    """ONE face row: a bf16 QUANT in the ``new`` slot, nothing else."""
    face_row = np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    face_row[..., 2] = 0
    face_row[0, 2] = (QUANT_SENTINEL, _bf16_index(), 0)
    face_skip = np.zeros((MAX_FACES,), np.int32)
    return face_row, face_skip


def _recorder(hook, site, log):
    """Wrap ONE hook so its invocation is logged.

    A ``FaceJoinPolicy`` is returned UNWRAPPED: it is not callable (it takes
    the two addends, not one tensor), so a single-argument wrapper around it
    would not be the same object graphax dispatches on. Its firing is observed
    through the join counters and through the addends it returns instead.
    """
    from graphax.sparse.ops.join import FaceJoinPolicy
    if hook is None or isinstance(hook, FaceJoinPolicy):
        return hook

    def _h(st):
        out = hook(st)
        log.append((site, id(hook), out is not st))
        return out
    return _h


def _emit_and_eliminate(monkeypatch, approx_add):
    """Eliminate vertex 2, then vertex 1 with env's own face dict for the
    face ``x -> u -> w`` wrapped in site recorders. Returns (per_face entry,
    the transform log of the vertex-1 elimination, the old edge existed)."""
    monkeypatch.delenv("ALPHAGRAD_APPROX_OLD", raising=False)
    if approx_add is None:
        monkeypatch.delenv("ALPHAGRAD_APPROX_ADD", raising=False)
    else:
        monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", approx_add)
    closed, cfg, ij, key, (x_var, w_var) = _toy()
    ij.eliminate(2)
    old_exists = ij.graph[x_var][w_var] is not None
    face_row, face_skip = _quant_new_row()
    per_face = envmod._face_dict_for_vertex(cfg, ij, 1, face_row, face_skip)
    assert list(per_face) == [key], (list(per_face), key)
    entry = per_face[key]
    log: list = []
    if len(entry) == 2:
        (lhs, rhs, new), (jl, jr, jres) = entry
        wrapped = ((_recorder(lhs, "lhs", log), _recorder(rhs, "rhs", log),
                    _recorder(new, "new", log)),
                   (_recorder(jl, "jl", log), _recorder(jr, "old", log),
                    _recorder(jres, "new_old", log)))
    else:
        lhs, rhs, new = entry
        wrapped = (_recorder(lhs, "lhs", log), _recorder(rhs, "rhs", log),
                   _recorder(new, "new", log))
    ij.eliminate(1, (), {key: wrapped})
    return entry, log, old_exists


# --------------------------------------------------------------------------
# 1. ONE SITE per slot, under every value -- finding 72 fault 1 at the root
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", ["lossy", "lossless"])
def test_the_new_slot_hook_is_installed_at_exactly_one_site(monkeypatch, cfg):
    """The mask and the hook must answer for the SAME tensor.

    ``masks.slot_legality`` computes a slot's legality from ONE tensor and ANDs
    over the extra sites ``env.face_slot_sites()`` reports. With exactly one
    site per slot there is nothing to AND over and nothing that can go stale --
    which is the whole of finding 72's fault 1: ``--approx-old same`` put the
    ``new`` hook on the pre-existing old edge as well, and the mask never saw
    that tensor.
    """
    monkeypatch.delenv("ALPHAGRAD_APPROX_OLD", raising=False)
    monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", cfg)
    sites = envmod.face_slot_sites()
    assert sites == (("lhs",), ("rhs",), ("res:new",)), sites


# --------------------------------------------------------------------------
# 2. lossy puts a POLICY on the old edge; lossless puts nothing
# --------------------------------------------------------------------------

def test_lossy_puts_a_join_policy_at_jr_not_the_new_slot_hook(monkeypatch):
    from graphax.sparse.ops.join import FaceJoinPolicy
    entry, log, old_exists = _emit_and_eliminate(monkeypatch, "lossy")
    assert old_exists, "the toy face has no old edge -- the join never ran"
    assert len(entry) == 2, entry
    (lhs, rhs, new), (jl, jr, jres) = entry
    assert lhs is None and rhs is None and new is not None
    assert jl is None and jres is None
    assert isinstance(jr, FaceJoinPolicy), jr
    assert jr.mode == "lossy", jr
    # THE POINT: the old edge does NOT carry the new slot's hook any more.
    assert jr is not new
    # and the wire row fired on the fresh contraction only
    assert [site for site, _, _ in log] == ["new"], log


def test_lossless_puts_no_join_hook_at_all(monkeypatch):
    entry, log, old_exists = _emit_and_eliminate(monkeypatch, "lossless")
    assert old_exists, "the toy face has no old edge -- the join never ran"
    assert len(entry) == 2, entry
    (lhs, rhs, new), join3 = entry
    assert lhs is None and rhs is None and new is not None
    # graphax's sparse + already builds the union container, so a policy here
    # would compute it a second time and move no value.
    assert join3 == (None, None, None), join3
    assert [site for site, _, _ in log] == ["new"], log


def test_an_UNARMED_face_gets_no_join_policy(monkeypatch):
    """A face whose three slots are all None computes an EXACT contraction.

    The old edge it merges into may legitimately be WIDER than that exact
    contraction, because it accumulated approximated contributions at earlier
    steps. A `lossy` policy there would project it down to a container nobody
    asked to approximate -- information lost on a face the plan marked exact --
    and it would falsify the documented property that a SKIP-only or exact plan
    makes this flag inert.
    """
    monkeypatch.delenv("ALPHAGRAD_APPROX_OLD", raising=False)
    monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", "lossy")
    entry = envmod.face_entry_from_slots((None, None, None))
    assert entry == ((None, None, None), (None, None, None)), entry
    # one armed slot is enough to bring the policy back
    entry = envmod.face_entry_from_slots((None, None, lambda st: st))
    assert entry[1][1] is not None and entry[1][1].mode == "lossy"


def test_at_join_lets_the_legality_probe_drop_the_policy(monkeypatch):
    """The probe must not pay for a reconciliation nothing reads.

    Every site a slot hook is installed at is PRE-JOIN, so the policy cannot
    change a tensor the probe records, and the probe's elimination is undone by
    its snapshot. ``at_join`` is separate from ``at_site`` because the policy
    has no site: it takes two tensors, applies no wire row, and answers to no
    mask.
    """
    monkeypatch.delenv("ALPHAGRAD_APPROX_OLD", raising=False)
    monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", "lossy")
    hook = (lambda st: st)
    seen = []
    entry = envmod.face_entry_from_slots(
        (None, None, hook), at_join=lambda p: seen.append(p.mode) or None)
    assert seen == ["lossy"], seen
    assert entry == ((None, None, hook), (None, None, None)), entry
    # and the default keeps the live policy
    entry = envmod.face_entry_from_slots((None, None, hook))
    assert entry[1][1] is not None and entry[1][1].mode == "lossy"
    # every site a slot hook reaches is pre-join, which is what makes the
    # drop safe -- stated here so the two cannot drift
    assert envmod.face_slot_sites() == (("lhs",), ("rhs",), ("res:new",))


def test_the_default_is_lossy(monkeypatch):
    entry, log, _ = _emit_and_eliminate(monkeypatch, None)
    assert envmod.approx_add() == "lossy"
    assert envmod.APPROX_ADD_DEFAULT == "lossy"
    assert entry[1][1] is not None and entry[1][1].mode == "lossy"


def test_lossy_returns_two_structurally_identical_addends(monkeypatch):
    """The bar finding 72 set: ONE structure, therefore ONE mask.

    The policy is exercised through ``reconcile_addends`` on the SAME kind of
    operand pair the engine produces -- two contributions to one Jacobian block
    whose logical dims agree and whose storage does not.
    """
    from graphax.sparse.indexes import DenseIndex, DiagonalIndex
    from graphax.sparse.ops.join import container_of, reconcile_addends
    from graphax.sparse.tensor import SparseTensor

    # fresh: a 4x4 block pair held as a COUPLED diagonal (one physical axis)
    fresh = SparseTensor(
        (DiagonalIndex(0, 4, 0, 1),), (DiagonalIndex(1, 4, 0, 0),),
        jnp.asarray(np.arange(4, dtype=np.float32) + 1.0), fill_value=None)
    # old: the SAME logical dims held as two INDEPENDENT physical axes
    old = SparseTensor(
        (DenseIndex(0, 4, 0),), (DenseIndex(1, 4, 1),),
        jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4)),
        fill_value=None)
    assert container_of(fresh) != container_of(old)
    f2, o2, out = reconcile_addends(fresh, old, "lossy")
    assert container_of(f2) == container_of(o2), (
        f"lossy left the addends structurally different:\n"
        f"  fresh {container_of(f2)}\n  old   {container_of(o2)}")
    assert out.matched_target, (
        f"lossy did not reach the fresh contraction's container: "
        f"{out.container} vs {out.target}")


def test_lossless_drops_no_non_zero(monkeypatch):
    """``lossless`` must be lossless NUMERICALLY, not structurally.

    Two block-diagonal addends whose supports are DIFFERENT: a 4x4 pure
    diagonal and a 4x4 dense block. Their union support is the dense block, and
    every non-zero of both must survive into the sum.
    """
    from graphax.sparse.indexes import DenseIndex, DiagonalIndex
    from graphax.sparse.ops.join import container_of, reconcile_addends
    from graphax.sparse.tensor import SparseTensor

    a = SparseTensor(
        (DiagonalIndex(0, 4, 0, 1),), (DiagonalIndex(1, 4, 0, 0),),
        jnp.asarray([1.0, 2.0, 3.0, 4.0], jnp.float32), fill_value=None)
    b = SparseTensor(
        (DenseIndex(0, 4, 0),), (DenseIndex(1, 4, 1),),
        jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) + 100.0),
        fill_value=None)
    ref = np.asarray(a.dense(), np.float64) + np.asarray(b.dense(), np.float64)
    f2, o2, out = reconcile_addends(a, b, "lossless")
    assert container_of(f2) == container_of(o2)
    got = np.asarray((f2 + o2).dense(), np.float64)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert np.max(np.abs(got - ref)) == 0.0, np.max(np.abs(got - ref))
    # and the plain add -- which is what `lossless` actually emits -- agrees
    plain = np.asarray((a + b).dense(), np.float64)
    assert np.max(np.abs(plain - ref)) == 0.0, np.max(np.abs(plain - ref))


# --------------------------------------------------------------------------
# 3. the retired names RAISE and name the replacement
# --------------------------------------------------------------------------

def test_the_retired_hand_off_variable_raises(monkeypatch):
    monkeypatch.delenv("ALPHAGRAD_APPROX_ADD", raising=False)
    monkeypatch.setenv("ALPHAGRAD_APPROX_OLD", "same")
    with pytest.raises(ValueError, match="RETIRED"):
        envmod.approx_add()


@pytest.mark.parametrize("retired", ["same", "exact"])
def test_a_retired_value_on_the_new_variable_raises(monkeypatch, retired):
    monkeypatch.delenv("ALPHAGRAD_APPROX_OLD", raising=False)
    monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", retired)
    with pytest.raises(ValueError, match="RETIRED"):
        envmod.approx_add()


def test_the_deleted_env_var_moves_nothing(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_NEW_SLOT_JOIN", "0")
    entry, log, _ = _emit_and_eliminate(monkeypatch, "lossy")
    assert entry[1][1] is not None, \
        "ALPHAGRAD_NEW_SLOT_JOIN=0 must not switch the join policy off"


def test_a_value_outside_the_choices_is_an_error(monkeypatch):
    monkeypatch.delenv("ALPHAGRAD_APPROX_OLD", raising=False)
    monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", "1")
    with pytest.raises(ValueError, match="ALPHAGRAD_APPROX_ADD"):
        envmod.approx_add()


# --------------------------------------------------------------------------
# 4. the plan log records which configuration ran
# --------------------------------------------------------------------------

_M = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0 + 0.1)
_P = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 13.0 + 0.2)


def _square(x):
    e = _M @ x
    return jnp.sum(_P @ e)


def _one_terminal_record():
    closed = jax.make_jaxpr(_square)(_X4)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=[_X4], argnums=(0,), num_envs=0, target_fun=_square)
    state = env.reset()
    no_rules = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)
    no_rules = no_rules.at[..., 2].set(0)
    face_rows = jnp.full((MAX_FACES, FACE_SLOTS, 3), -1, jnp.int32)
    face_skip = jnp.zeros((MAX_FACES,), jnp.int32)
    for v in [int(x) for x in np.asarray(env.valid_vertices)]:
        state = env.step(
            state, StepAction(jnp.asarray(v, jnp.int32), no_rules,
                              face_rows, face_skip)).state
    recs = envmod.consume_plan_records()["records"]
    assert len(recs) == 1, [r.get("order") for r in recs]
    return recs[0]


@pytest.mark.parametrize("cfg", ["lossy", "lossless"])
def test_every_plan_record_carries_the_configuration(monkeypatch, cfg):
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.delenv("ALPHAGRAD_APPROX_OLD", raising=False)
    monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", cfg)
    envmod.consume_plan_records()
    try:
        rec = _one_terminal_record()
        # RENAMED, not aliased: a record written under the retired
        # ``approx_old`` column named a different computation and the two
        # columns must not be pooled.
        assert rec["approx_add"] == cfg, rec.get("approx_add")
        assert "approx_old" not in rec, rec.keys()
    finally:
        envmod.consume_plan_records()


# --------------------------------------------------------------------------
# 4. the flag on ppo.py
# --------------------------------------------------------------------------

def test_ppo_declares_the_flag_with_default_lossy():
    ppo = pytest.importorskip("alphagrad.approx.ppo")
    p = ppo.make_argparser()
    assert p.parse_args([]).approx_add == "lossy"
    assert p.parse_args(
        ["--approx-add", "lossless"]).approx_add == "lossless"
    with pytest.raises(SystemExit):
        p.parse_args(["--approx-add", "same"])
    with pytest.raises(SystemExit):
        p.parse_args(["--approx-add", "1"])


def test_ppo_still_accepts_the_retired_flag_so_it_can_refuse_it():
    """A launcher that names ``--approx-old`` believes it chose something.

    argparse must PARSE it (so the value reaches the check) and the run must
    then abort naming ``--approx-add``. Silently ignoring an unknown-but-
    parsed switch is the failure mode this guards.
    """
    ppo = pytest.importorskip("alphagrad.approx.ppo")
    p = ppo.make_argparser()
    assert p.parse_args(["--approx-old", "same"]).approx_old == "same"
    assert p.parse_args([]).approx_old is None


def test_landscape_maps_restated_choice_list_still_agrees():
    """``landscape_map`` restates the choice list because it builds its
    argparser BEFORE importing alphagrad (several env knobs are read at import
    of ``env``). The copy is allowed; DRIFT is not."""
    lm = pytest.importorskip("alphagrad.approx.tools.landscape_map")
    assert lm._APPROX_ADD_CHOICES == envmod.APPROX_ADD_CHOICES
    assert lm._APPROX_ADD_DEFAULT == envmod.APPROX_ADD_DEFAULT
