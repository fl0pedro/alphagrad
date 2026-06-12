"""mctx tree-traversal helpers — extract per-depth visit-count
distributions along the *chosen* action path so policy heads at every
depth can be distilled from the search.

The mctx ``policy_output.action_weights`` field gives the visit
distribution at the *root* only. For hierarchical MCTS where each "real"
elimination step decomposes into ``1 + 2·max_rules`` MCTS depth levels
(vertex / pair_k / factor_k), the deeper visit counts are buried inside
the search tree (``policy_output.search_tree.children_visits[batch, node,
action]`` with ``children_index[batch, node, action]`` connecting parent
to child). This module walks that tree.

Usage
-----
After the rollout's ``mctx.muzero_policy(...)`` call, pass the
``search_tree`` plus a fresh PRNG key to :func:`extract_path_visits`. It
returns:

  * ``visits_per_depth`` — ``(decision_depth, num_actions)`` normalised
    visit-count distributions, one per depth in the chosen path.
  * ``actions_per_depth`` — ``(decision_depth,) int32`` — the action
    sampled from each depth's visit-count distribution.

The caller can then:
  * Use ``actions_per_depth`` to construct the env action that gets
    played in the real rollout (replacing the previous "sample vertex
    from root, sample rules from agent prior" pattern).
  * Store ``visits_per_depth`` in the trajectory and use it as a
    cross-entropy target at every depth's policy head.

Edge cases
----------
* If a sub-tree along the chosen path is unexpanded
  (``sum(children_visits) == 0``), the helper falls back to a uniform
  distribution at that depth and onwards. Sampled actions in the
  fallback regime may be invalid (mctx's invalid-action mask only zeros
  out the *root*'s un-expanded children, not arbitrary deeper nodes); in
  practice this is rare when ``num_simulations >> decision_depth``.
* Tree traversal assumes the mctx tree's batch dim is 1 — i.e., the
  helper is called from inside the rollout's per-env vmap.
"""

from __future__ import annotations

import distrax
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand


# mctx's tree allocates index 0 to the root; "no parent / unvisited" is -1.
# Hard-coded to avoid a runtime dependency on ``mctx.Tree.ROOT_INDEX`` /
# ``mctx.Tree.UNVISITED`` symbols which aren't part of mctx's public surface.
_ROOT_INDEX = 0
_UNVISITED = -1


def extract_path_visits(tree, decision_depth: int, key):
    """Walk the chosen action path through ``tree``, collecting the
    visit-count distribution and sampled action at every depth.

    Parameters
    ----------
    tree
        ``mctx.policy_output.search_tree`` — leading batch dim must be 1.
    decision_depth
        Number of MCTS levels per "real" step (= ``1 + 2·max_rules``).
    key
        PRNG key for sampling at each depth.

    Returns
    -------
    visits_per_depth : (decision_depth, num_actions) float32
        Row ``d`` is the normalised visit-count distribution at depth
        ``d`` of the chosen path. Falls back to uniform when the
        sub-tree is unexpanded.
    actions_per_depth : (decision_depth,) int32
        Action sampled from row ``d`` of ``visits_per_depth``.
    """

    def step(carry, _):
        current_node, k = carry
        # tree's batch dim is 1 inside the rollout vmap.
        visits = tree.children_visits[0, current_node, :]
        visit_sum = jnp.sum(visits)

        sample_key, next_key = jrand.split(k)
        n_actions = visits.shape[-1]
        uniform = jnp.full(
            (n_actions,), 1.0 / n_actions, dtype=visits.dtype,
        )
        visit_dist = jnp.where(
            visit_sum > 0,
            visits / jnp.maximum(visit_sum, 1e-8),
            uniform,
        )
        action = distrax.Categorical(probs=visit_dist).sample(seed=sample_key)

        next_node = tree.children_index[0, current_node, action]
        # If the chosen child wasn't expanded, stay on ``current_node`` so
        # the next iteration's gather doesn't crash. The next iteration's
        # visit counts will all be zero → uniform fallback continues.
        next_node = jnp.where(next_node >= 0, next_node, current_node)

        return (next_node, next_key), (
            visit_dist.astype(jnp.float32),
            action.astype(jnp.int32),
        )

    init = (jnp.array(_ROOT_INDEX, dtype=jnp.int32), key)
    _, (visits, actions) = lax.scan(
        step, init, jnp.arange(decision_depth),
    )
    return visits, actions
