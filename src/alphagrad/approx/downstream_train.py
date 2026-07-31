"""Downstream training of ``VmappedNeuralNetwork`` on MNIST with a
parametric gradient source.

The RL trainers (ppo / mu0 / gfn) learn an elimination order plus a
sequence of per-vertex approximation ops over the
``VmappedNeuralNetwork`` jaxpr (see ``examples.py:_neural_network``).
Per the research plan, the load-bearing measurement is *not* the
recorded ``cosine_sim`` of those gradients vs the exact reference, but
whether they actually train a downstream model: do the approximations
preserve downstream MNIST convergence?

This harness mirrors the training pattern of
``~/dsnn/graphax/tests/examples/cifar10_transformer.py``
(``optax.adam`` + ``jacve``-based gradients in a ``@jax.jit`` train
function + periodic test eval), but:

* The model is ``VmappedNeuralNetwork`` (the 2-layer MLP the RL agent
  actually trained against), not the CIFAR10 transformer.
* The dataset is MNIST (28×28 → 10).
* The ``--gradient-source`` flag selects which gradient callable to
  use: the exact ``jax.grad`` reference, ``graphax.jacve`` with a fixed
  ``order``, or a *learned* sequence loaded from a wandb run's
  ``best_sequences.json``.

Per-step metrics (train loss, test accuracy, wall time, peak HBM,
``cossim`` of the approximate gradient against ``jax.grad``) are
streamed to a CSV so the plotter can produce learning curves
comparing each candidate gradient source.

Usage:
    uv run alphagrad/src/alphagrad/approx/downstream_train.py \\
        --gradient-source jax_grad --train-steps 200 \\
        --output-csv out/downstream/jax_grad_42.csv

    uv run alphagrad/src/alphagrad/approx/downstream_train.py \\
        --gradient-source wandb:~/dsnn/wandb/run-XXXX:best_overall \\
        --train-steps 10000 --output-csv out/downstream/learned.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
import numpy as np
import optax
from graphax import jacve

from alphagrad.approx.common.datasets import load_dataset
from alphagrad.approx.common.examples import (
    data_gen,
    get_args,
    get_fn,
    infer_argnums,
)
from alphagrad.approx.common.seq_replay import (
    load_best_sequence,
    parse_recorded_seq,
)


# ---------------------------------------------------------------------------
# Loss / accuracy primitives — mirror the MLP-on-MNIST pattern from the
# RL env's reward harness (env.py:_callback uses the same `_neural_network`
# target and the same `data_gen` sampler). We separate `model_outputs`
# (returns per-sample squared-error vector, suitable for `jacve` which
# expects a non-scalar function for Jacobian-mode AD) from `loss_scalar`
# (mean across batch + output dims, suitable for `jax.grad`).
# ---------------------------------------------------------------------------


def _flatten_pytree(grads) -> jnp.ndarray:
    """Concatenate all leaves of a pytree of jnp arrays into a 1-D vector
    (used to compute cossim against the exact reference)."""
    leaves = jtu.tree_leaves(grads)
    return jnp.concatenate([jnp.ravel(jnp.asarray(l)) for l in leaves])


def _cosine(a: jnp.ndarray, b: jnp.ndarray) -> float:
    """Cosine similarity of two flattened gradient vectors. Returns NaN
    when either vector has zero norm — a zero gradient is degenerate
    (model output independent of inputs at that step), not a 0%-match
    against the exact reference. Treating it as NaN keeps the
    ``cossim_vs_exact`` column honest in downstream plots.
    """
    num = float(jnp.dot(a, b))
    den = float(jnp.linalg.norm(a) * jnp.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return num / den


def _make_loss_fns(target_fn: Callable):
    """Wrap ``target_fn`` (returns per-sample squared-error tensor) into
    matched scalar / sum reductions consumed by ``jax.grad`` and
    ``jacve`` respectively.

    ``target_fn(x, y, *weights)`` returns shape ``(batch, out_dim)`` for
    Vmapped variants. For scalar gradients we average; for Jacobian-mode
    callables we sum (so the resulting "Jacobian" of a scalar output
    matches the gradient).
    """

    def loss_scalar(x, y, *weights):
        return jnp.mean(target_fn(x, y, *weights))

    def loss_sum(x, y, *weights):
        return jnp.sum(target_fn(x, y, *weights))

    return loss_scalar, loss_sum


def _build_gradient_fn(
    spec: str,
    target_fn: Callable,
    argnums: tuple[int, ...],
    sample_args: tuple,
) -> tuple[Callable, bool]:
    """Construct the gradient callable for a given ``--gradient-source``.

    Returns ``(fn, provides_loss)`` where ``fn(x, y, *weights)`` returns:

    * ``(loss_value, grads_list)`` when ``provides_loss`` is True. This
      covers ``jax_grad`` (via ``jax.value_and_grad``) AND every
      jacve-backed path (``graphax_fwd`` / ``graphax_rev`` /
      ``wandb:*``) via ``jacve(..., has_aux=True)`` which returns
      ``(primal, jacobian)`` in a single forward pass — the primal IS
      the per-sample squared-error tensor, mean-contracted to give the
      scalar loss.
    * ``grads_list`` when ``provides_loss`` is False — currently only
      ``jax_jacrev`` and ``jax_jacfwd``, since ``jax.jacrev`` has no
      ``value_and_jacobian`` fusion (its ``has_aux`` parameter routes
      auxiliary outputs *from the wrapped function*, not the function's
      own primal). The caller falls back to a separate
      ``loss_scalar_jit`` call for those.

    All jacve-based paths build the full Jacobian of ``target_fn``
    (shape ``(batch, out_dim, *weight_shape)`` per arg) and contract it
    via ``mean(axis=(0, 1))`` so the resulting gradient matches what
    ``jax.grad(mean(target_fn))`` produces — this is the contract the
    RL agent trained against (it computed Jacobians of ``target_fn``
    and compared them against the exact reference).
    """
    loss_scalar, _ = _make_loss_fns(target_fn)

    if spec == "jax_grad":
        # value_and_grad → one forward pass produces both loss and grads,
        # vs separately calling loss_scalar_jit + jax.grad which forces
        # two passes per training step.
        return jax.jit(jax.value_and_grad(loss_scalar, argnums=argnums)), True

    if spec in ("jax_jacrev", "jax_jacfwd"):
        # ``jax.jacrev/jacfwd`` have no value+jacobian fusion — caller
        # recomputes loss via the standalone loss_scalar_jit.
        jac_fn = jax.jacrev if spec == "jax_jacrev" else jax.jacfwd
        return (
            _make_grad_from_jacobian(
                jax.jit(jac_fn(target_fn, argnums=argnums)),
                returns_primal=False,
            ),
            False,
        )

    if spec in ("graphax_rev", "graphax_fwd"):
        order_str = "rev" if spec == "graphax_rev" else "fwd"
        return (
            _make_grad_from_jacobian(
                jax.jit(
                    jacve(
                        target_fn, order=order_str,
                        argnums=argnums, has_aux=True,
                    )
                ),
                returns_primal=True,
            ),
            True,
        )

    if spec.startswith("wandb:"):
        rest = spec[len("wandb:") :]
        if ":" not in rest:
            raise ValueError(
                f"--gradient-source wandb:<run_dir>:<channel>, got {spec!r}"
            )
        run_dir, channel = rest.split(":", 1)
        seq = load_best_sequence(run_dir, channel)
        # Resolve gcd factors using the model's jaxpr shapes when possible.
        try:
            jaxpr = jax.make_jaxpr(target_fn)(*sample_args)
            axis_sizes: list[int] = []
            for eqn in jaxpr.jaxpr.eqns:
                for out in eqn.outvars:
                    if hasattr(out, "aval") and hasattr(out.aval, "shape"):
                        axis_sizes.extend(int(s) for s in out.aval.shape)
        except Exception:
            axis_sizes = []
        order, transforms = parse_recorded_seq(
            seq,
            axis_sizes=axis_sizes,
            skip_low_precision_quant=True,
        )
        dropped = sum(
            1 for r in seq
            if isinstance(r, (list, tuple)) and len(r) >= 6 and int(r[1]) == 2
        ) - sum(
            sum(1 for op in ops if op.__class__.__name__ == "Quant")
            for _, ops in transforms
        )
        if dropped:
            print(
                f"[downstream] dropped {dropped} low-precision Quant ops from "
                f"replay (float4/8/int2-4 cannot promote against float32 in "
                f"the current jacve pipeline)."
            )
        return (
            _make_grad_from_jacobian(
                jax.jit(
                    jacve(
                        target_fn,
                        order=order,
                        transforms=transforms,
                        argnums=argnums,
                        has_aux=True,
                    )
                ),
                returns_primal=True,
            ),
            True,
        )

    raise ValueError(
        f"Unknown --gradient-source {spec!r}. Valid: jax_grad, jax_jacrev, "
        f"jax_jacfwd, graphax_rev, graphax_fwd, wandb:<run_dir>:<channel>"
    )


def _make_grad_from_jacobian(
    jacobian_fn: Callable, *, returns_primal: bool = False,
) -> Callable:
    """Wrap a Jacobian-returning callable so it returns the *gradient* of
    the mean-loss (the convention the RL agent was trained against).

    The RL agent computed ``jacve(target_fn, ...)`` where ``target_fn``
    returns ``(batch, out_dim)`` per-sample squared-error. To use that
    same recorded ``order``/``transforms`` for a gradient step we need
    to contract the resulting Jacobian back to per-weight shape — and
    we want it to match ``jax.grad(mean(target_fn))`` so the
    ``cossim_vs_exact`` metric is interpretable. Contracting via
    ``mean(axis=(0, 1))`` reproduces ``jax.grad(mean(target_fn))``.

    Args:
        jacobian_fn: jit-compiled callable that returns the Jacobian.
            When ``returns_primal=False`` (jax.jacrev/jacfwd path) it
            returns the Jacobian alone. When ``returns_primal=True``
            (jacve has_aux=True path) it returns ``(primal, jacobian)``
            in one pass — the primal IS ``target_fn(*args)``
            (per-sample squared-error), mean-contracted to give the
            scalar loss with no extra forward pass.

    Returns:
        Callable that returns ``(loss_value, grads_list)`` when
        ``returns_primal=True``, else just ``grads_list``.
    """

    def _contract(j):
        # Final ``.astype(float32)`` so a Quant op that left the val in
        # a low-precision dtype (e.g. ``float4_e2m1fn``) is brought
        # back to a precision optax / the weight update expects —
        # otherwise JAX raises TypePromotionError on the
        # ``weights - lr * grads`` step. This is the only place we
        # widen; the gradient COMPUTATION still runs at whatever
        # precision the elimination chose.
        return jnp.mean(j, axis=(0, 1)).astype(jnp.float32)

    def _grads_only(*args):
        jac = jacobian_fn(*args)
        if isinstance(jac, (tuple, list)):
            return [_contract(j) for j in jac]
        return _contract(jac)

    def _loss_and_grads(*args):
        primal, jac = jacobian_fn(*args)
        loss_val = jnp.mean(primal).astype(jnp.float32)
        if isinstance(jac, (tuple, list)):
            grads = [_contract(j) for j in jac]
        else:
            grads = _contract(jac)
        return loss_val, grads

    return _loss_and_grads if returns_primal else _grads_only


def _eval_test_accuracy(
    target_fn: Callable,
    test_x: jnp.ndarray,
    test_y_onehot: jnp.ndarray,
    weights: Sequence[jnp.ndarray],
    batch_size: int = 256,
) -> float:
    """Sweep the test set, accumulate accuracy. ``target_fn`` is the
    Vmapped MLP; argmax over its output gives the predicted class. Test
    labels are one-hot; argmax recovers the true class.
    """
    n = int(test_x.shape[0])
    n_correct = 0
    for start in range(0, n, batch_size):
        x_b = test_x[start : start + batch_size]
        y_b = test_y_onehot[start : start + batch_size]
        out = target_fn(x_b, y_b, *weights)
        # target_fn returns 0.5 * (tanh(...) - y)**2 — predicted class is
        # the index where (tanh(...) - y_onehot_zero) is most negative.
        # Re-derive logits as ``y_pred_logits ≈ sqrt(2*out) + y_b`` is
        # ill-defined; instead compute predictions directly with the
        # weights using the bare network forward path.
        pred = _predict_class(x_b, weights)
        true = jnp.argmax(y_b, axis=1)
        n_correct += int(jnp.sum(pred == true))
    return n_correct / n


def _predict_class(x: jnp.ndarray, weights: Sequence[jnp.ndarray]) -> jnp.ndarray:
    """Manual forward of the 2-layer MLP defined in examples.py
    (``_neural_network``); duplicated here so the accuracy eval does not
    invoke the squared-error loss harness."""
    W1, b1, W2, b2 = weights
    a1 = jnp.tanh(x @ W1.T + b1)
    # argmax is invariant to the monotone output map, so this is correct for
    # both losses; written on the raw logits because under xent there is no
    # output tanh at all.
    return jnp.argmax(a1 @ W2.T + b2, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example", default="VmappedNeuralNetwork",
        help="Only ``VmappedNeuralNetwork`` is supported initially. Other "
        "names would require their own forward / accuracy primitives.",
    )
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Mirrors NN_VMAP_BATCH from datasets.py; the "
                        "model is vmapped over this axis.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=250197)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--gradient-source", required=True,
        help="One of: jax_grad, jax_jacrev, jax_jacfwd, graphax_rev, "
        "graphax_fwd, wandb:<run_dir>:<channel>",
    )
    parser.add_argument("--output-csv", default=None,
                        help="Per-step metrics CSV path. Defaults to "
                        "``out/downstream/<gradient-source>_<seed>.csv``.")
    parser.add_argument("--measure-cossim-vs-exact", action="store_true",
                        default=True,
                        help="At step 0 (and every --cossim-every steps), "
                        "compare the gradient pytree against jax.grad and "
                        "log the cossim of the flattened gradient.")
    parser.add_argument("--cossim-every", type=int, default=100)
    parser.add_argument("--no-measure-cossim-vs-exact",
                        dest="measure_cossim_vs_exact", action="store_false")
    parser.add_argument(
        "--wandb", choices=("online", "offline", "disabled"),
        default="offline",
        help="wandb mode. 'online' uploads as the run progresses; "
        "'offline' writes locally to ~/dsnn/wandb (sync later with "
        "`wandb sync`); 'disabled' skips wandb entirely (CSV only).",
    )
    parser.add_argument(
        "--wandb-project", default="dsnn-downstream",
        help="wandb project name when --wandb != disabled.",
    )
    parser.add_argument(
        "--wandb-entity", default="",
        help="wandb entity (team / user namespace). Empty = your personal "
        "namespace. Set to 'dll-streetview' to land in that team's project.",
    )
    parser.add_argument(
        "--name", default=None,
        help="wandb run name. Defaults to "
        "'downstream_<gradient-source>_seed<seed>'.",
    )
    args = parser.parse_args()

    # The research plan trains the RL gradient strictly for
    # ``VmappedNeuralNetwork + mnist`` — every recorded best sequence
    # under ``~/dsnn/wandb/`` is defined over that jaxpr. Replaying it
    # against a different (example, dataset) silently produces a
    # mismatched gradient (different jaxpr vertex count → out-of-range
    # order entries → zero gradient). Loud rejection prevents the
    # silent error.
    if args.example != "VmappedNeuralNetwork":
        raise SystemExit(
            f"--example={args.example!r} not supported. The research "
            "plan trains the RL gradient strictly for "
            "'VmappedNeuralNetwork'; replaying against another example "
            "produces zero gradients. Re-run with "
            "--example VmappedNeuralNetwork, or extend the harness "
            "deliberately (you'll also need to add the matching "
            "predict_class / dataset bookkeeping)."
        )
    if args.dataset != "mnist":
        raise SystemExit(
            f"--dataset={args.dataset!r} not supported. The research "
            "plan uses mnist throughout. Re-run with --dataset mnist, "
            "or extend `datasets.py:load_dataset` first."
        )

    key = jrand.PRNGKey(args.seed)

    # Build model + initial weights.
    target_fn = get_fn(args.example)  # vmapped 2-layer MLP
    argnums = tuple(infer_argnums(args.example))
    key, init_key = jrand.split(key)
    init_args = get_args(args.example, init_key, dataset=args.dataset)
    # init_args is [x, y, W1, b1, W2, b2] — the first two are dummies the
    # data sampler will replace. Capture the weight pytree.
    weights = list(init_args[2:])

    # Data sampler — yields a fresh (x, y) MNIST batch each call.
    sampler = data_gen(args.example, dataset=args.dataset)
    if sampler is None:
        raise RuntimeError(
            f"No data_gen for {args.example!r}; downstream training "
            "needs a sampler."
        )

    # One sample batch for shape-inference (jacve's `lower` needs concrete
    # shapes to build the elimination DAG; sampling once at startup is fine).
    key, sample_key = jrand.split(key)
    sample_x, sample_y = sampler(jrand.split(sample_key, 4))
    sample_args = (sample_x, sample_y, *weights)

    print(
        f"[downstream] example={args.example} dataset={args.dataset} "
        f"batch_size={args.batch_size} gradient_source={args.gradient_source}"
    )
    print(
        f"[downstream] weight shapes: "
        + ", ".join(f"{tuple(w.shape)}" for w in weights)
    )

    # wandb init — same convention the RL trainers use: offline by
    # default so a missing W&B login doesn't kill the run. `disabled`
    # short-circuits everything.
    wandb_run = None
    if args.wandb != "disabled":
        import os
        os.environ["WANDB_MODE"] = args.wandb
        import wandb as _wandb
        wandb_run_name = args.name or (
            f"downstream_{args.gradient_source.replace(':', '_').replace('/', '_')}"
            f"_seed{args.seed}"
        )
        wandb_run = _wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity or None),
            name=wandb_run_name,
            config={
                "example": args.example,
                "dataset": args.dataset,
                "train_steps": args.train_steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "eval_every": args.eval_every,
                "cossim_every": args.cossim_every,
                "gradient_source": args.gradient_source,
                "weight_shapes": [tuple(w.shape) for w in weights],
            },
            reinit=True,
        )
        print(f"[downstream] wandb run: {wandb_run.dir}")

    # Build the gradient callables. ``provides_loss`` is True for the
    # ``jax_grad`` path which uses ``jax.value_and_grad`` so a single
    # forward pass yields both loss and grads — the training loop
    # branches on this flag below. Always build jax_grad too so we can
    # measure cossim of the approximate gradient against the exact ref.
    grad_fn, grad_fn_provides_loss = _build_gradient_fn(
        args.gradient_source, target_fn, argnums, sample_args,
    )
    if args.measure_cossim_vs_exact:
        exact_grad_fn, _ = _build_gradient_fn(
            "jax_grad", target_fn, argnums, sample_args,
        )
    else:
        exact_grad_fn = None

    # Optimizer.
    optim = optax.adam(args.lr)
    opt_state = optim.init(weights)

    # Test set for periodic accuracy eval.
    test_x, test_y_onehot = load_dataset(args.dataset, dataset_size=None, subset="test")

    # CSV writer.
    out_path = Path(
        args.output_csv
        or f"out/downstream/{args.gradient_source.replace(':', '_').replace('/', '_')}_{args.seed}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_f = out_path.open("w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow([
        "step", "train_loss", "test_acc",
        "step_wall_ms", "peak_mem_bytes", "grad_cossim_vs_exact",
    ])

    loss_scalar, _ = _make_loss_fns(target_fn)
    loss_scalar_jit = jax.jit(loss_scalar)

    # ``ResourceMonitor`` from jax_memory_monitor — same primitive the
    # env's reward harness uses to read peak HBM. Optional: a missing
    # install (CPU-only dev box) records NaN for peak_mem so the
    # ``peak_mem.png`` plot can distinguish "unmeasured" from a
    # legitimate zero-byte step.
    try:
        from jax_memory_monitor import ResourceMonitor  # type: ignore
        _monitor_available = True
    except ImportError:
        ResourceMonitor = None  # type: ignore
        _monitor_available = False
        print(
            "[downstream] jax_memory_monitor unavailable — peak_mem_bytes "
            "will be NaN; install it for HBM measurements."
        )

    def _normalise_grads(grads_pytree):
        if isinstance(grads_pytree, dict):
            return [grads_pytree[a] for a in argnums]
        return list(grads_pytree)

    def _step_grads_with_loss(x_b, y_b, weights):
        """One call → ``(loss, grads_list)``. For ``jax_grad`` this is
        the fused ``value_and_grad`` path; for the jacve paths we
        recompute the scalar loss separately via ``loss_scalar_jit``.
        """
        if grad_fn_provides_loss:
            loss_val, grads_pytree = grad_fn(x_b, y_b, *weights)
            return loss_val, _normalise_grads(grads_pytree)
        loss_val = loss_scalar_jit(x_b, y_b, *weights)
        grads_pytree = grad_fn(x_b, y_b, *weights)
        return loss_val, _normalise_grads(grads_pytree)

    print(f"[downstream] CSV -> {out_path}")
    print(f"[downstream] running {args.train_steps} steps")
    start_wall = time.monotonic()
    for step in range(args.train_steps):
        key, batch_key = jrand.split(key)
        x_b, y_b = sampler(jrand.split(batch_key, 4))

        step_start = time.monotonic()
        if not _monitor_available:
            loss_val, grads_list = _step_grads_with_loss(x_b, y_b, weights)
            peak_mem_bytes = float("nan")
        else:
            with ResourceMonitor() as monitor:
                loss_val, grads_list = _step_grads_with_loss(x_b, y_b, weights)
                # Force materialisation inside the monitored region so
                # the peak HBM reading covers the full gradient compute.
                jax.block_until_ready(grads_list[0])
            peak_mem_bytes = float(monitor.stats.get("memory", 0.0))

        cossim_vs_exact = float("nan")
        if exact_grad_fn is not None and (
            step == 0 or (step % args.cossim_every == 0)
        ):
            # ``exact_grad_fn`` is always the ``jax_grad`` path (value_and_grad);
            # discard its loss and keep just the grads.
            _, exact_grads = exact_grad_fn(x_b, y_b, *weights)
            exact_list = _normalise_grads(exact_grads)
            cossim_vs_exact = _cosine(
                _flatten_pytree(grads_list),
                _flatten_pytree(exact_list),
            )

        updates, opt_state = optim.update(grads_list, opt_state, weights)
        weights = optax.apply_updates(weights, updates)
        jax.block_until_ready(weights[0])
        step_wall_ms = (time.monotonic() - step_start) * 1000.0

        test_acc = float("nan")
        if (step % args.eval_every == 0) or (step == args.train_steps - 1):
            test_acc = _eval_test_accuracy(
                target_fn, test_x, test_y_onehot, weights,
            )

        csv_w.writerow([
            step, float(loss_val), test_acc, step_wall_ms,
            peak_mem_bytes, cossim_vs_exact,
        ])
        csv_f.flush()

        if wandb_run is not None:
            wandb_run.log(
                {
                    "step": step,
                    "train_loss": float(loss_val),
                    "test_acc": test_acc,
                    "step_wall_ms": step_wall_ms,
                    "peak_mem_bytes": peak_mem_bytes,
                    "grad_cossim_vs_exact": cossim_vs_exact,
                },
                step=step,
            )

        if step % args.eval_every == 0 or step == args.train_steps - 1:
            print(
                f"[downstream] step={step:>6d}  loss={float(loss_val):+.5g}  "
                f"test_acc={test_acc:.4f}  wall_ms={step_wall_ms:.3f}  "
                f"peak_mem={peak_mem_bytes:.4g}  "
                f"cossim_vs_exact={cossim_vs_exact:.4f}"
            )

    csv_f.close()
    total_wall = time.monotonic() - start_wall
    print(f"[downstream] done in {total_wall:.2f}s — CSV at {out_path}")

    if wandb_run is not None:
        # Final summary scalars — easier to query from wandb-runs API
        # than re-aggregating the per-step history.
        final_test_acc = _eval_test_accuracy(
            target_fn, test_x, test_y_onehot, weights,
        )
        wandb_run.summary.update(
            {
                "final/test_acc": final_test_acc,
                "final/train_loss": float(loss_val),
                "final/total_wall_s": total_wall,
                "final/gradient_source": args.gradient_source,
            }
        )
        try:
            wandb_run.finish()
        except Exception as exc:
            print(f"[downstream] wandb.finish() failed: {exc}")


if __name__ == "__main__":
    main()
