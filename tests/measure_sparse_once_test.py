"""ONE SPARSE EXECUTABLE PER MEASURED PLAN (owner rulings 2026-09-25, grill
round 1 Q18 c and round 2 Q2; dsnn-dfw.228, dsnn-dfw.234).

Every plan compiles one executable, the plan's sparse representation. The
cost channels and the static gate read it; the quality channels read its
outputs after a dense conversion outside the timed executions. At the
measurement boundary every gradient leaf leaves the executable as its value
buffer with the scalar folded in (dsnn-dfw.234), so the static output bytes
of an exact plan equal the reference's. Pinned here:

* exactly one measure compile per plan, and it is the sparse one;
* ``ALPHAGRAD_MEASURE_SPARSE=0`` is the only dense opt-out, any other value
  raises;
* the fold: a literal scalar of 1 leaves the val untouched, a scalar folds
  into it, a uniform tensor's buffer is its scalar, and the densified leaf
  equals the tensor's own dense form;
* the quality from the densified sparse outputs equals the dense executable's
  quality within float32 rounding, on the policy gate's recorded plan
  (``tests/golden/policy_gate_golden.json``: skips and quant rows on the
  gate's branched MLP) and on the exact plan, one diag and one compress plan
  of NN256 and TLM at a small batch, measured back to back in one process;
* on the exact plans the static output bytes of the sparse executable equal
  the reference's exactly (the toy scalar loss, NN256, TLM).

The gate module pins ten ALPHAGRAD_* variables at import, so its target is
copied here (``_mlp``, the same arguments) instead of imported.
"""
from __future__ import annotations

import json
import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from alphagrad.approx import env as E                           # noqa: E402
from alphagrad.approx.common import compile_cache as CC         # noqa: E402
from alphagrad.approx.env import (                              # noqa: E402
    FACE_SLOTS, MAX_RULES_PER_VERTEX, REWARD_INDEX)

GOLDEN = pathlib.Path(__file__).with_name("golden") / "policy_gate_golden.json"
# float32 rounding on a cosine and on a clipped relative residual: the dense
# executable materialises the same leaves inside XLA, the sparse one outside.
TOL = 1e-5
_GAPS: dict = {}


def _mlp(x, W1, W2, W3):
    h = jnp.tanh(x @ W1)
    a = jnp.tanh(h @ W2)
    b = jnp.tanh(h @ W3)
    return a * b


_ARGS = (jnp.ones((2, 8)), jnp.ones((8, 32)) * 0.1,
         jnp.ones((32, 16)) * 0.1, jnp.ones((32, 16)) * 0.1)
_ARGNUMS = (1, 2, 3)


@pytest.fixture(autouse=True)
def _restore_width():
    keep = E.MAX_FACES
    yield
    E.MAX_FACES = keep
    E._LIVE_CHAINS.clear()


@pytest.fixture
def measure(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
    monkeypatch.setenv("ALPHAGRAD_SKIP_COUNT_OPS", "1")
    monkeypatch.setenv("ALPHAGRAD_MEASURE_DEDUPE", "0")
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.setenv("ALPHAGRAD_FIDELITY", "1")
    for k in ("ALPHAGRAD_FIDELITY_WEIGHT", "ALPHAGRAD_COS_LOG_EVERY",
              "ALPHAGRAD_REV_EXACT_TELEMETRY", "ALPHAGRAD_MEASURE_SPARSE",
              "ALPHAGRAD_FACTORED_OUTPUTS", "GRAPHAX_FACTORED_OUTPUTS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(CC, "_LOCAL_CACHE", {})
    E.set_measure_timeout_s(120.0)
    seen = {"programs": [], "compiles": 0}
    real_program = E.measured_program

    def recording_program(config, order, consts, sparse=None, **kw):
        # The measured elimination is the one call that carries the plan's
        # transforms; the exact and rev-exact programs carry none.
        if "transforms" in kw and kw["transforms"] is not None:
            seen["programs"].append(sparse)
        return real_program(config, order, consts, sparse=sparse, **kw)

    monkeypatch.setattr(E, "measured_program", recording_program)
    real_compile = jax.stages.Lowered.compile

    def counting_compile(self, compiler_options=None, **kw):
        seen["compiles"] += 1
        return real_compile(self, compiler_options=compiler_options, **kw)

    monkeypatch.setattr(jax.stages.Lowered, "compile", counting_compile)
    E.consume_plan_records()
    E.consume_refused_counts()
    E.consume_last_refusal()

    def run(env, samples, order, specs, faces, skips, mode):
        if mode is None:
            monkeypatch.delenv("ALPHAGRAD_MEASURE_SPARSE", raising=False)
        else:
            monkeypatch.setenv("ALPHAGRAD_MEASURE_SPARSE", mode)
        seen["programs"].clear()
        seen["compiles"] = 0
        *_wire, reward = E._callback(
            env.config, env.args, env.consts, jnp.asarray(order),
            jnp.asarray(specs), jnp.asarray(faces), jnp.asarray(skips),
            int(len(order)), *samples)
        refusal = E.consume_last_refusal()
        assert refusal is None, refusal
        recs = E.consume_plan_records()["records"]
        assert len(recs) == 1, recs
        r = np.asarray(reward, dtype=np.float64)
        return {
            "quality": float(r[REWARD_INDEX["quality"]]),
            "fidelity": float(r[REWARD_INDEX["fidelity"]]),
            "latency_ns": float(recs[0].get("candidate_latency_ns") or 0.0),
            "ref_latency_ns": float(recs[0].get("ref_latency_ns") or 0.0),
            "mem_output_bytes": recs[0].get("mem_output_bytes"),
            "ref_output_bytes": recs[0].get("ref_output_bytes"),
            "mem_args_bytes": recs[0].get("mem_args_bytes"),
            "ref_args_bytes": recs[0].get("ref_args_bytes"),
            "mem_temp_bytes": recs[0].get("mem_temp_bytes"),
            "programs": list(seen["programs"]),
            "compiles": int(seen["compiles"]),
        }

    yield run
    E.set_measure_timeout_s(None)
    E.consume_plan_records()
    E.consume_refused_counts()


def _paired(run, env, samples, order, plan_arrays, label, exact=False):
    """The dense opt-out and the default, back to back; the default second so
    its reference and exact executables are already cached and its compile
    count is the candidate's alone."""
    specs, faces, skips = plan_arrays
    dense = run(env, samples, order, specs, faces, skips, "0")
    sparse = run(env, samples, order, specs, faces, skips, None)
    assert dense["programs"] == [False], dense["programs"]
    assert sparse["programs"] == [True], sparse["programs"]
    assert sparse["compiles"] == 1, (label, sparse["compiles"])
    gap_q = abs(sparse["quality"] - dense["quality"])
    gap_f = abs(sparse["fidelity"] - dense["fidelity"])
    _GAPS[label] = {"quality": gap_q, "fidelity": gap_f}
    print(f"[sparseonce] {label}: quality dense {dense['quality']:.7f} "
          f"sparse {sparse['quality']:.7f} (gap {gap_q:.3e}); fidelity dense "
          f"{dense['fidelity']:.7f} sparse {sparse['fidelity']:.7f} "
          f"(gap {gap_f:.3e}); static output bytes dense "
          f"{dense['mem_output_bytes']} sparse {sparse['mem_output_bytes']} "
          f"reference {sparse['ref_output_bytes']}; latency us dense "
          f"{dense['latency_ns'] / 1e3:.1f} sparse "
          f"{sparse['latency_ns'] / 1e3:.1f} reference "
          f"{sparse['ref_latency_ns'] / 1e3:.1f}; compiles dense "
          f"{dense['compiles']} sparse {sparse['compiles']}", flush=True)
    assert np.isfinite(dense["quality"]) and np.isfinite(sparse["quality"])
    assert gap_q <= TOL, (label, dense["quality"], sparse["quality"])
    assert gap_f <= TOL, (label, dense["fidelity"], sparse["fidelity"])
    # No scalar crosses the boundary (dsnn-dfw.234): the value buffers of
    # the sparse form never exceed the dense form's, and the exact plan's
    # equal the reference's to the byte.
    assert sparse["mem_output_bytes"] <= dense["mem_output_bytes"], label
    assert sparse["mem_output_bytes"] <= sparse["ref_output_bytes"], label
    if exact:
        assert (sparse["mem_output_bytes"] == sparse["ref_output_bytes"]
                == dense["mem_output_bytes"]), (
            label, sparse["mem_output_bytes"], sparse["ref_output_bytes"],
            dense["mem_output_bytes"])
    return dense, sparse


def _exact_arrays(n):
    specs = np.full((n, MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[..., 2] = 0
    faces = np.full((n, E.MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    skips = np.zeros((n, E.MAX_FACES), np.int32)
    return specs, faces, skips


def test_the_flag_accepts_the_two_values_only(monkeypatch):
    monkeypatch.delenv("ALPHAGRAD_MEASURE_SPARSE", raising=False)
    assert E.measure_sparse_enabled() is True
    monkeypatch.setenv("ALPHAGRAD_MEASURE_SPARSE", "1")
    assert E.measure_sparse_enabled() is True
    monkeypatch.setenv("ALPHAGRAD_MEASURE_SPARSE", "0")
    assert E.measure_sparse_enabled() is False
    for bad in ("yes", "true", "2", ""):
        monkeypatch.setenv("ALPHAGRAD_MEASURE_SPARSE", bad)
        with pytest.raises(ValueError, match="ALPHAGRAD_MEASURE_SPARSE"):
            E.measure_sparse_enabled()


def test_the_fold_leaves_value_buffers_only():
    from graphax.sparse.indexes import DenseIndex, DiagonalIndex
    from graphax.sparse.tensor import SparseTensor
    val = jnp.arange(2 * 3, dtype=jnp.float32).reshape(2, 3) + 1.0
    dims = ([DenseIndex(0, 2, 0)], [DenseIndex(1, 3, 1)])
    one = SparseTensor(*dims, val)
    folded = E._fold_leaf(one)
    assert isinstance(folded, E.CompactLeaf) and not folded.uniform
    assert folded.val is val
    scaled = SparseTensor(*dims, val, scalar_mult=jnp.asarray(2.5))
    folded = E._fold_leaf(scaled)
    np.testing.assert_array_equal(np.asarray(folded.val),
                                  np.asarray(val) * 2.5)
    np.testing.assert_array_equal(np.asarray(folded.dense()),
                                  np.asarray(scaled.dense()))
    uniform = SparseTensor(*dims, None, scalar_mult=jnp.asarray(3.0))
    folded = E._fold_leaf(uniform)
    assert folded.uniform and folded.val.shape == ()
    np.testing.assert_array_equal(np.asarray(folded.dense()),
                                  np.asarray(uniform.dense()))
    diag = SparseTensor([DiagonalIndex(0, 4, 0, 1)],
                        [DiagonalIndex(1, 4, 0, 0)],
                        jnp.arange(4, dtype=jnp.float32),
                        scalar_mult=jnp.asarray(0.5))
    folded = E._fold_leaf(diag)
    assert folded.val.shape == (4,)
    np.testing.assert_array_equal(np.asarray(folded.dense()),
                                  np.asarray(diag.dense()))
    # The pytree: one value child per leaf, the structure in the aux data.
    leaves, treedef = jax.tree_util.tree_flatten((folded, None, val))
    assert len(leaves) == 2
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(back[0], E.CompactLeaf) and back[1] is None
    # Traced inside a program: the scalar folds, the buffer is compact.
    out = jax.eval_shape(lambda v: E._fold_output(
        SparseTensor(*dims, v, scalar_mult=v[0, 0]), False), val)
    assert isinstance(out, E.CompactLeaf) and out.val.shape == (2, 3)
    assert len(jax.tree_util.tree_leaves(out)) == 1


def test_a_skipped_path_densifies_to_the_zeros_of_its_nominal_shape():
    from graphax.sparse.indexes import DenseIndex
    from graphax.sparse.tensor import SparseTensor
    closed = jax.make_jaxpr(_mlp)(*_ARGS)
    env = E.VertexEliminationEnv.from_jaxpr(
        closed, args=list(_ARGS), argnums=_ARGNUMS, num_envs=0,
        target_fun=_mlp)
    shapes = E._nominal_gradient_shapes(env.config)
    assert shapes == [(2, 16, 8, 32), (2, 16, 32, 16), (2, 16, 32, 16)]
    val = jnp.arange(2 * 16 * 8 * 32, dtype=jnp.float32).reshape(2, 16, 8, 32)
    st = SparseTensor([DenseIndex(0, 2, 0), DenseIndex(1, 16, 1)],
                      [DenseIndex(2, 8, 2), DenseIndex(3, 32, 3)], val)
    plain = jnp.ones((2, 16, 32, 16))
    for leaf in (st, E._fold_leaf(st)):
        dense = E._densify_gradient((leaf, None, plain), shapes)
        assert len(dense) == 3
        np.testing.assert_array_equal(np.asarray(dense[0]), np.asarray(val))
        assert dense[1].shape == (2, 16, 32, 16)
        assert float(jnp.abs(dense[1]).sum()) == 0.0
        assert dense[2] is plain
    with pytest.raises(RuntimeError, match="gradient leaves"):
        E._densify_gradient((st, None), shapes)
    with pytest.raises(RuntimeError, match="leaf 2"):
        E._densify_gradient((st, None, plain[..., :8]), shapes)


def _gate_plan():
    golden = json.loads(GOLDEN.read_text())
    steps = golden["steps"]
    order = [int(v) for v in steps[-1]["env_order_prefix"]]
    order += [int(v) for v in golden["config"]["valid_vertices"]
              if int(v) not in order]
    n = len(order)
    assert E.MAX_FACES >= max(len(s.get("face_skip") or []) for s in steps)
    specs, faces, skips = _exact_arrays(n)
    n_quant = 0
    for k, st in enumerate(steps):
        for i, row in enumerate(st["rule_specs"]):
            specs[k, i] = row
        for f, rows in enumerate(st.get("face_rows") or []):
            faces[k, f] = rows
            n_quant += sum(int(r[0]) == E.QUANT_SENTINEL for r in rows)
        for f, s in enumerate(st.get("face_skip") or []):
            skips[k, f] = int(s)
    # The recorded plan carries skips and quant rows; a golden without them
    # would pin the exact plan and nothing else.
    assert int(skips.sum()) > 0 and n_quant > 0
    return order, (specs, faces, skips)


def test_the_toy_scalar_loss_exact_plan_ties_the_reference_to_the_byte(
        measure, monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "jac_cosine")
    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((16, 16), dtype=np.float32) / 4.0)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 16, dtype=np.float32))

    def toy(v):
        return jnp.sum(jnp.tanh(W @ v) ** 2)

    closed = jax.make_jaxpr(toy)(x)
    env = E.VertexEliminationEnv.from_jaxpr(
        closed, args=[x], argnums=(0,), num_envs=0, target_fun=toy,
        measure_latency=True, terminal_rewards_only=True,
        latency_inner_reps=1)
    order = sorted(int(v) for v in np.asarray(env.valid_vertices))
    samples = (jnp.asarray(np.stack(
        [np.linspace(-1.0, 1.0, 16, dtype=np.float32)])),)
    dense, sparse = _paired(measure, env, samples, order,
                            _exact_arrays(len(order)), "toy-exact", exact=True)
    assert dense["quality"] > 0.99


def test_the_policy_gates_plan_prices_one_sparse_executable(measure,
                                                             monkeypatch):
    from alphagrad.approx.common.eval_samples import generate_eval_samples
    # The gate's target is a (2, 16) output with no data generator, so the
    # quality channel is the Jacobian cosine at the calibration samples.
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "jac_cosine")
    closed = jax.make_jaxpr(_mlp)(*_ARGS)
    env = E.VertexEliminationEnv.from_jaxpr(
        closed, args=list(_ARGS), argnums=_ARGNUMS, num_envs=0,
        target_fun=_mlp, per_face=True, delta_obs=True,
        measure_latency=True, terminal_rewards_only=True,
        latency_inner_reps=1, num_data_points=2)
    samples = generate_eval_samples(env, jax.random.PRNGKey(3), 2)
    order, (specs, faces, skips) = _gate_plan()
    # The exact plan on the gate's graph and order: a full Jacobian target.
    _paired(measure, env, samples, order, _exact_arrays(len(order)),
            "gate-exact", exact=True)
    # The recorded plan skips every live face, so its gradient is zero on
    # both executables (the sparse one returns no leaf at all) and the claim
    # is the compile count.
    _paired(measure, env, samples, order, (specs, faces, skips),
            "gate-plan-recorded")
    # The same wires with the skips lifted: the quant rows land and the
    # gradient is real.
    dense, _sparse = _paired(measure, env, samples, order,
                             (specs, faces, np.zeros_like(skips)),
                             "gate-plan-unskipped")
    assert dense["quality"] > 0.1


def _landscape(monkeypatch, tmp_path, example, dataset):
    import alphagrad.approx.tools.landscape_map as lm
    argv = ["--example", example, "--dataset", dataset,
            "--num-eval-samples", "2", "--num-data-points", "2",
            "--reps-per-point", "1", "--latency-inner-reps", "1",
            "--latency-warmup", "1", "--out-dir", str(tmp_path)]
    env, samples, _closed = lm.build_env(lm.make_argparser().parse_args(argv))
    order = lm.markowitz_order(env)
    inv = lm.face_inventory(env, order, capture_tensors=True)
    plans, _orders = lm.build_singleton_sweep_plans(
        env, order, inv, ops=("reduce", "diag"))
    picked = {"exact": ("exact", lm.get_plan_arrays({"wires": []},
                                                     len(order)))}
    for op in ("diag", "compress"):
        pid = next(p for p, pl in plans.items() if pl["op"] == op)
        picked[op] = (pid, lm.get_plan_arrays(plans[pid], len(order)))
    return env, samples, [int(v) for v in order], picked


def test_nn256_exact_diag_and_compress_plans_price_one_sparse_executable(
        measure, monkeypatch, tmp_path):
    from alphagrad.approx.common import examples as ex
    monkeypatch.setattr(ex, "_EQ_NN_HIDDEN", 8)
    monkeypatch.setattr(ex, "NN_VMAP_BATCH", 4)
    env, samples, order, picked = _landscape(
        monkeypatch, tmp_path, "VmappedNeuralNetwork", "mnist")
    assert env.config.scalar_target
    for op, (pid, arrays) in picked.items():
        dense, _sparse = _paired(measure, env, samples, order, arrays,
                                 f"nn256-{pid}", exact=(op == "exact"))
        assert dense["quality"] > 0.1, (pid, dense["quality"])


def test_tlm_exact_diag_and_compress_plans_price_one_sparse_executable(
        measure, monkeypatch, tmp_path):
    from alphagrad.approx.common import datasets as ds
    from alphagrad.approx.common import examples as ex
    monkeypatch.setenv("ALPHAGRAD_TLM_SEQ", "8")
    monkeypatch.setenv("ALPHAGRAD_TLM_DMODEL", "16")
    monkeypatch.setenv("ALPHAGRAD_TLM_VOCAB", "32")
    monkeypatch.setattr(ex, "NN_VMAP_BATCH", 4)
    corpus = np.asarray(
        jax.random.randint(jax.random.PRNGKey(7), (4096,), 0, 32), np.int32)
    monkeypatch.setattr(ds, "load_wikitext2",
                        lambda vocab, subset="train": corpus)
    env, samples, order, picked = _landscape(
        monkeypatch, tmp_path, "VmappedTransformerLM", "wikitext2")
    assert env.config.scalar_target
    for op, (pid, arrays) in picked.items():
        dense, _sparse = _paired(measure, env, samples, order, arrays,
                                 f"tlm-{pid}", exact=(op == "exact"))
        assert dense["quality"] > 0.1, (pid, dense["quality"])


def test_zz_report_the_worst_gap():
    if not _GAPS:
        pytest.skip("no paired measurement ran in this process")
    worst_q = max(v["quality"] for v in _GAPS.values())
    worst_f = max(v["fidelity"] for v in _GAPS.values())
    print(f"[sparseonce] worst quality gap {worst_q:.3e}, worst fidelity gap "
          f"{worst_f:.3e} over {sorted(_GAPS)}", flush=True)
    assert worst_q <= TOL and worst_f <= TOL
