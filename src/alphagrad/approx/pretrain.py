"""Stage B.5 — encoder pretraining.

Cheap supervised warm-start for the relational-bias encoder before any RL
spend. Implements the two tasks the architecture spec calls for:

* **Masked token prediction (BERT-style)** — pick 15% of non-padding tokens,
  replace with the MASK sentinel, train the encoder + a small prediction
  head to recover the original token ids. Labels are free.
* **Off-diagonal-energy prediction (B.5.next)** — for each equation, predict
  its local Jacobian's off-diagonal energy ratio. The encoder pools tokens
  by ``eqn_id`` to a per-vertex representation, projects to a scalar, and is
  supervised against the analytical / Hutchinson ratio that
  :func:`graphax.instrumentation.extract_jacobian_features` produces. This
  task auto-disables itself when ``GRAPHAX_JACOBIAN_INSTRUMENTATION`` is
  off — there are no labels in that case.

Both labels are free to generate, so the corpus extends as far as your
example library does. ``--pretrain-tasks`` chooses which losses to combine;
``both`` averages them with ``--energy-weight``.

After training, save the encoder weights with ``--save`` and
:func:`load_pretrained_encoder` will install them into a freshly-built
:class:`alphagrad.approx.ppo.Agent`. The checkpoint is via ``cloudpickle``;
production deployment would use orbax or similar.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cloudpickle
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import optax

from alphagrad.approx.common import (
    compute_eqn_ids_from_tokens,
    get_args,
    get_fn,
)
from alphagrad.approx.env import MAX_TOKENS
from alphagrad.transformer import Encoder, MLP, PositionalEncoder
from graphax.jaxpr import VEJaxpr, get_vocab

VOCAB_SIZE = 256
MASK_TOKEN_ID = 0       # padding-shared sentinel; we don't pretrain on padding anyway
MASK_PROB = 0.15
DEFAULT_CORPUS = (
    "Helmholtz",
    "Lighthouse",
    "RobotArm_6DOF",
    "RoeFlux_1d",
    "BlackScholes_Jacobian",
)
TOKEN_VOCAB, _, _ = get_vocab()


class PretrainModel(eqx.Module):
    """Encoder + per-token prediction head + per-vertex energy head."""

    embedding: eqx.nn.Embedding
    pos_enc: PositionalEncoder
    encoder: Encoder
    pred_head: eqx.nn.Linear
    energy_head: eqx.nn.Linear  # (embd_dim,) → scalar; sigmoid'd to [0, 1]

    def __init__(
        self, *, vocab_size, embd_dim, num_layers, num_heads, hidden_dim, key,
    ):
        keys = jrand.split(key, 4)
        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=keys[0])
        self.pos_enc = PositionalEncoder(embd_dim, MAX_TOKENS)
        self.encoder = Encoder(
            num_layers, num_heads, embd_dim, hidden_dim, key=keys[1],
        )
        self.pred_head = eqx.nn.Linear(embd_dim, vocab_size, key=keys[2])
        self.energy_head = eqx.nn.Linear(embd_dim, 1, key=keys[3])

    def encode_tokens(self, tokens, eqn_ids=None, *, key):
        """Run the shared encoder; returns the per-token hidden states."""
        x = jax.vmap(self.embedding)(tokens)
        x = self.pos_enc(x)
        return self.encoder(x, eqn_ids=eqn_ids, key=key)

    def __call__(self, tokens, eqn_ids=None, *, key):
        # Backward-compatible single-task interface — returns MLM logits.
        h = self.encode_tokens(tokens, eqn_ids=eqn_ids, key=key)
        return jax.vmap(self.pred_head)(h)

    def mlm_logits(self, tokens, eqn_ids=None, *, key):
        h = self.encode_tokens(tokens, eqn_ids=eqn_ids, key=key)
        return jax.vmap(self.pred_head)(h)

    def energy_per_vertex(self, tokens, eqn_ids, max_eqns: int, *, key):
        """Pool token hidden states by ``eqn_id`` (mean) and project to a
        scalar per vertex via :class:`energy_head`. Returns a
        ``(max_eqns,)`` array of *raw* logits — caller applies sigmoid for
        the loss against the off-diag-ratio target."""
        h = self.encode_tokens(tokens, eqn_ids=eqn_ids, key=key)
        # Compute per-eqn means via segment_sum + count.
        valid = (eqn_ids >= 0).astype(h.dtype)
        seg_ids = jnp.where(eqn_ids >= 0, eqn_ids, 0)  # safe gather index
        sums = jax.ops.segment_sum(
            h * valid[:, None], seg_ids, num_segments=max_eqns,
        )
        counts = jax.ops.segment_sum(valid, seg_ids, num_segments=max_eqns)
        means = sums / jnp.maximum(counts[:, None], 1.0)
        logits = jax.vmap(self.energy_head)(means).squeeze(-1)
        return logits  # (max_eqns,)


def _tokenize_one(target_fn, xs):
    """Run the same tokenization the env uses, end-to-end."""
    closed = jax.make_jaxpr(target_fn)(*xs)
    ve = VEJaxpr(closed.jaxpr)
    tokens = np.asarray(ve.tokenized()[:MAX_TOKENS])
    tokens = np.pad(tokens, (0, MAX_TOKENS - tokens.shape[0])).astype(np.int32)
    eqn_ids = compute_eqn_ids_from_tokens(tokens, TOKEN_VOCAB)
    return (
        jnp.asarray(tokens, dtype=jnp.int32),
        jnp.asarray(eqn_ids, dtype=jnp.int32),
        closed,
    )


def _energy_targets_for(closed_jaxpr, xs, max_eqns: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(targets, mask)`` of shape ``(max_eqns,)`` for the energy
    prediction task. ``targets`` are the off-diag-energy ratios per
    equation; ``mask`` is ``1`` where the entry is a real measurement and
    ``0`` for padding / non-finite results that we don't want to train on."""
    from graphax.instrumentation import (
        extract_jacobian_features,
        is_enabled as _instrument_enabled,
    )

    targets = np.zeros(max_eqns, dtype=np.float32)
    mask = np.zeros(max_eqns, dtype=np.float32)
    if not _instrument_enabled():
        return jnp.asarray(targets), jnp.asarray(mask)
    try:
        feats = extract_jacobian_features(
            closed_jaxpr.jaxpr, tuple(closed_jaxpr.literals), tuple(xs),
            n_probes=8,
        )
    except Exception:
        return jnp.asarray(targets), jnp.asarray(mask)
    if feats is None:
        return jnp.asarray(targets), jnp.asarray(mask)
    n = min(max_eqns, feats.off_diag_ratio.shape[0])
    finite = np.isfinite(feats.off_diag_ratio[:n])
    targets[:n] = np.where(finite, feats.off_diag_ratio[:n], 0.0).astype(np.float32)
    mask[:n] = finite.astype(np.float32)
    return jnp.asarray(targets), jnp.asarray(mask)


def build_corpus(example_names, base_seed: int = 0, *, with_energy: bool = False,
                 max_eqns: int = 256):
    """Tokenize each example. Optionally compute energy targets too.

    Drops examples that fail to tokenize so the corpus loop is robust to
    one-off graphax issues.
    """
    corpus = []
    for i, name in enumerate(example_names):
        try:
            target_fn = get_fn(name)
            xs = get_args(name, jrand.PRNGKey(base_seed + i))
            tokens, eqn_ids, closed = _tokenize_one(target_fn, xs)
            if with_energy:
                e_targets, e_mask = _energy_targets_for(closed, xs, max_eqns)
            else:
                e_targets = jnp.zeros(max_eqns, dtype=jnp.float32)
                e_mask = jnp.zeros(max_eqns, dtype=jnp.float32)
            corpus.append((name, tokens, eqn_ids, e_targets, e_mask))
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}: {e}")
    return corpus


def mask_tokens(tokens, key, mask_prob: float = MASK_PROB):
    """BERT-style masking. Returns ``(masked_tokens, mask_bool, target_tokens)``.

    Padding (``tokens == 0``) is never masked, so the predictor doesn't waste
    capacity on degenerate slots.
    """
    valid = tokens != 0
    rand = jrand.uniform(key, tokens.shape)
    mask = (rand < mask_prob) & valid
    masked = jnp.where(mask, MASK_TOKEN_ID, tokens)
    return masked, mask, tokens


def mlm_loss(model, tokens, eqn_ids, mask, target_tokens, key):
    logits = model.mlm_logits(tokens, eqn_ids=eqn_ids, key=key)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = log_probs[jnp.arange(tokens.shape[0]), target_tokens]
    masked_loss = -target_log_probs * mask.astype(jnp.float32)
    n_masked = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(masked_loss) / n_masked


def energy_loss(model, tokens, eqn_ids, energy_targets, energy_mask, max_eqns, key):
    """Sigmoid binary-cross-entropy on per-vertex off-diag-energy ratio.

    Off-diag ratio lives in [0, 1] so sigmoid + BCE is the natural fit; logits
    are unconstrained. The ``energy_mask`` zeros out per-vertex contributions
    that have no measurement (padding past the actual eqn count, or
    non-finite samples).
    """
    raw_logits = model.energy_per_vertex(tokens, eqn_ids, max_eqns, key=key)
    log_sig = jax.nn.log_sigmoid(raw_logits)
    log_one_minus_sig = jax.nn.log_sigmoid(-raw_logits)
    per_eqn = -(
        energy_targets * log_sig + (1.0 - energy_targets) * log_one_minus_sig
    )
    n_valid = jnp.maximum(jnp.sum(energy_mask), 1.0)
    return jnp.sum(per_eqn * energy_mask) / n_valid


def combined_loss(
    model, tokens, eqn_ids, mlm_mask, target_tokens,
    energy_targets, energy_mask, max_eqns,
    *, energy_weight: float, key,
):
    keys = jrand.split(key, 2)
    mlm = mlm_loss(model, tokens, eqn_ids, mlm_mask, target_tokens, keys[0])
    eng = energy_loss(
        model, tokens, eqn_ids, energy_targets, energy_mask, max_eqns, keys[1],
    )
    return mlm + energy_weight * eng, (mlm, eng)


def make_train_step(optimizer, *, do_mlm: bool, do_energy: bool, energy_weight: float,
                    max_eqns: int):
    if do_mlm and do_energy:
        @eqx.filter_jit
        def train_step(model, opt_state, tokens, eqn_ids, mlm_mask, target_tokens,
                       energy_targets, energy_mask, key):
            (loss, (mlm, eng)), grads = eqx.filter_value_and_grad(
                combined_loss, has_aux=True,
            )(
                model, tokens, eqn_ids, mlm_mask, target_tokens,
                energy_targets, energy_mask, max_eqns,
                energy_weight=energy_weight, key=key,
            )
            updates, new_opt_state = optimizer.update(
                grads, opt_state, eqx.filter(model, eqx.is_inexact_array),
            )
            new_model = eqx.apply_updates(model, updates)
            return new_model, new_opt_state, loss, mlm, eng
        return train_step
    if do_mlm:
        @eqx.filter_jit
        def train_step(model, opt_state, tokens, eqn_ids, mlm_mask, target_tokens,
                       energy_targets, energy_mask, key):
            del energy_targets, energy_mask
            loss, grads = eqx.filter_value_and_grad(mlm_loss)(
                model, tokens, eqn_ids, mlm_mask, target_tokens, key,
            )
            updates, new_opt_state = optimizer.update(
                grads, opt_state, eqx.filter(model, eqx.is_inexact_array),
            )
            new_model = eqx.apply_updates(model, updates)
            zero = jnp.array(0.0)
            return new_model, new_opt_state, loss, loss, zero
        return train_step

    @eqx.filter_jit
    def train_step(model, opt_state, tokens, eqn_ids, mlm_mask, target_tokens,
                   energy_targets, energy_mask, key):
        del mlm_mask, target_tokens
        loss, grads = eqx.filter_value_and_grad(energy_loss)(
            model, tokens, eqn_ids, energy_targets, energy_mask, max_eqns, key,
        )
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array),
        )
        new_model = eqx.apply_updates(model, updates)
        zero = jnp.array(0.0)
        return new_model, new_opt_state, loss, zero, loss
    return train_step


# ---------------------------------------------------------------------------
# Loading the pretrained encoder into a freshly-built Agent
# ---------------------------------------------------------------------------


def load_pretrained_encoder(path: str | Path, agent):
    """Replace `agent.embedding`, `agent.pos_enc`, `agent.encoder` in-place
    with the corresponding sub-modules from a saved :class:`PretrainModel`.

    Used by the RL trainer to warm-start the encoder before PPO. Other agent
    sub-modules (vertex_policy, rule_policy, value heads, ...) keep their
    fresh random init.
    """
    with open(path, "rb") as f:
        pretrained: PretrainModel = cloudpickle.load(f)
    agent = eqx.tree_at(lambda a: a.embedding, agent, pretrained.embedding)
    agent = eqx.tree_at(lambda a: a.pos_enc, agent, pretrained.pos_enc)
    agent = eqx.tree_at(lambda a: a.encoder, agent, pretrained.encoder)
    return agent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Encoder pretraining via masked-token-prediction (Stage B.5).",
    )
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--embd-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--mask-prob", type=float, default=MASK_PROB)
    p.add_argument("--examples", nargs="+", default=list(DEFAULT_CORPUS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default=None,
                   help="Where to pickle the trained PretrainModel for later "
                        "load_pretrained_encoder() consumption. Optional.")
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--pretrain-tasks", type=str, default="mlm",
                   choices=["mlm", "energy", "both"],
                   help="Which supervised tasks to train. `energy` requires "
                        "GRAPHAX_JACOBIAN_INSTRUMENTATION=1 to produce labels.")
    p.add_argument("--energy-weight", type=float, default=1.0,
                   help="Weight on the energy-prediction loss when "
                        "--pretrain-tasks=both.")
    p.add_argument("--max-eqns", type=int, default=256,
                   help="Static padding for per-vertex energy targets.")
    return p


def main():
    args = make_argparser().parse_args()
    key = jrand.PRNGKey(args.seed)

    do_mlm = args.pretrain_tasks in ("mlm", "both")
    do_energy = args.pretrain_tasks in ("energy", "both")

    if do_energy:
        from graphax.instrumentation import is_enabled as _instrument_enabled
        if not _instrument_enabled():
            print(
                "  warning: --pretrain-tasks={} but graphax instrumentation "
                "is OFF (GRAPHAX_JACOBIAN_INSTRUMENTATION not set). Energy "
                "labels will all be masked-out and the energy task degenerates."
                .format(args.pretrain_tasks)
            )

    print(f"Building corpus from: {args.examples}")
    corpus = build_corpus(
        args.examples, base_seed=args.seed,
        with_energy=do_energy, max_eqns=args.max_eqns,
    )
    if not corpus:
        raise RuntimeError("Empty corpus — every example failed to tokenize")
    print(f"Corpus: {len(corpus)} jaxprs  tasks={args.pretrain_tasks}")
    for name, tokens, eqn_ids, _e_targets, e_mask in corpus:
        n_real = int(jnp.sum(tokens != 0))
        max_id = int(jnp.max(eqn_ids))
        n_eqns = (max_id + 1) if max_id >= 0 else 0
        n_energy = int(jnp.sum(e_mask))
        print(
            f"  {name:<22s}  non-pad tokens={n_real:>5d}  num_eqns={n_eqns:>4d}"
            f"  energy_labels={n_energy:>4d}"
        )

    model_key, key = jrand.split(key)
    model = PretrainModel(
        vocab_size=VOCAB_SIZE,
        embd_dim=args.embd_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        key=model_key,
    )
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_step = make_train_step(
        optimizer, do_mlm=do_mlm, do_energy=do_energy,
        energy_weight=args.energy_weight, max_eqns=args.max_eqns,
    )

    print(f"\nTraining for {args.steps} steps, lr={args.lr}")
    losses: list[float] = []
    mlm_losses: list[float] = []
    eng_losses: list[float] = []
    t0 = time.time()
    for step in range(args.steps):
        idx = step % len(corpus)
        _, tokens, eqn_ids, e_targets, e_mask = corpus[idx]
        mask_key, step_key, key = jrand.split(key, 3)
        masked, mlm_mask, target = mask_tokens(tokens, mask_key, args.mask_prob)
        model, opt_state, loss, mlm_l, eng_l = train_step(
            model, opt_state, masked, eqn_ids, mlm_mask, target,
            e_targets, e_mask, step_key,
        )
        losses.append(float(loss))
        mlm_losses.append(float(mlm_l))
        eng_losses.append(float(eng_l))
        if step % args.print_every == 0 or step == args.steps - 1:
            recent = losses[-args.print_every:]
            recent_mlm = mlm_losses[-args.print_every:]
            recent_eng = eng_losses[-args.print_every:]
            print(
                f"  step {step:4d}/{args.steps}  loss={float(loss):7.4f}  "
                f"mlm={float(np.mean(recent_mlm)):6.3f}  "
                f"eng={float(np.mean(recent_eng)):6.3f}  "
                f"recent_avg={float(np.mean(recent)):7.4f}  "
                f"elapsed={time.time() - t0:5.1f}s"
            )

    print(f"\nFinal loss: {losses[-1]:.4f}  (started at {losses[0]:.4f})")
    print(f"Loss reduction: {(1 - losses[-1] / max(losses[0], 1e-9)) * 100:.1f}%")

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            cloudpickle.dump(model, f)
        print(f"Saved pretrained model to {save_path}")


if __name__ == "__main__":
    main()
