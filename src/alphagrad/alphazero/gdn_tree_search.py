"""
MCTS implementation for VertexEliminationEnv using token states.
"""

from functools import partial
from typing import Any, Callable, Tuple

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
import mctx

import alphagrad.utils as u

Array = jax.Array
PyTreeDef = Any
EnvStepFn = Callable[[Array, Array], Tuple[Array, float, bool]]
ValueTransform = Callable[[float], float]


def select_params(x, y):
    return y if eqx.is_inexact_array(x) else x


def _get_invalid_action_mask(state, total_v, valid_vertices_arr):
    vertices = jnp.arange(total_v) + 1

    # valid_vertices_arr can be either (num_valid,) or batched (..., num_valid)
    # We want to check for each vertex in 1..total_v if it exists in valid_vertices_arr
    # We expand vertices to broadcast with valid_vertices_arr's trailing dimension
    # vertices: (total_v, 1) to broadcast against valid_vertices_arr (..., num_valid)
    # Result of == is (..., total_v, num_valid), taking any(axis=-1) gives (..., total_v)
    is_valid = jnp.any(
        jnp.expand_dims(vertices, -1) == jnp.expand_dims(valid_vertices_arr, -2),
        axis=-1,
    )
    mask_valid = ~is_valid

    def _is_chosen(s):
        chosen = s.order
        # s.order might have shape (..., max_steps)
        # we want to return mask of shape (...) + (total_v,)
        # let's add a trailing dim to chosen and compare with vertices
        # chosen: (..., max_steps) -> (..., max_steps, 1)
        # vertices: (total_v,) -> (total_v,)
        return jnp.any(jnp.expand_dims(chosen, -1) == vertices, axis=-2)

    chosen_mask = _is_chosen(state)
    return mask_valid | chosen_mask


def make_recurrent_fn(
    model: PyTreeDef,
    step: EnvStepFn,
    inverse_value_transform: ValueTransform,
    total_v: int,
    valid_vertices: Array,
) -> Callable:
    """Creates a recurrent function for tree search as required by the MuZero
    algorithm. This function is used to expand the tree at the leaf node with a
    new node.
    """

    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    def recurrent_fn(params, rng_key, actions, state):
        vertex_actions = actions + 1  # action idx to vertex idx

        # Prefetch calculation: we don't have logits yet for the NEXT state,
        # but we can use the current state's information or wait.
        # However, step() is where prefetch is triggered.
        # In recurrent_fn, we might not have a good prefetch order yet.
        # But we can pass it if we want.

        next_state, reward, _ = step(state, vertex_actions)  # Env dynamics function

        _model = jtu.tree_map(select_params, model, params)

        try:
            model_input = next_state.tokens
        except (AttributeError, TypeError):
            model_input = next_state

        output = _model(model_input, rng_key)
        output = _model(model_input, rng_key)
        policy_logits = output[..., 1:]
        value = inverse_value_transform(output[..., 0])

        invalid_mask = _get_invalid_action_mask(next_state, total_v, valid_vertices)

        # Mask logits
        masked_logits = jnp.where(
            invalid_mask,
            -1e9,
            policy_logits,  # Or whatever large negative value
        )

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=reward, discount=1.0, prior_logits=masked_logits, value=value
        )
        return recurrent_fn_output, next_state

    return recurrent_fn


def make_tree_search(
    model: PyTreeDef,
    step: EnvStepFn,
    num_actions: int,
    total_v: int,
    valid_vertices: Array,
    inverse_value_transform: ValueTransform,
    num_considered_actions: int = 5,
    gumbel_scale: float = 1.0,
    num_simulations: int = 25,
    **qtransformkwargs,
) -> Tuple[Array, int, Array]:
    """Implementation of the environment interaction function for the MuZero
    algorithm.
    """
    qtransform = partial(mctx.qtransform_completed_by_mix_value, **qtransformkwargs)

    recurrent_fn = make_recurrent_fn(
        model, step, inverse_value_transform, total_v, valid_vertices
    )

    def environment_interaction(network, init_carry):
        states, num_muls, key = init_carry
        batchsize = states.tokens.shape[0]
        batched_network = eqx.filter_vmap(network)
        params = eqx.filter(network, eqx.is_inexact_array)

        def loop_fn(carry, _):
            state, num_muls, key = carry
            key, subkey = jrand.split(key, 2)

            keys = jrand.split(key, batchsize)

            try:
                model_input = state.tokens
            except (AttributeError, TypeError):
                model_input = state

            output = batched_network(model_input, keys)
            policy_logits = output[..., 1:]
            value = inverse_value_transform(output[..., 0])

            # Output is (B, ..., 1+total_v). We need policy_logits to be (B, ..., total_v)
            # and invalid_mask to be identically shaped.
            invalid_mask = _get_invalid_action_mask(state, total_v, valid_vertices)

            invalid_mask = jnp.broadcast_to(invalid_mask, policy_logits.shape)
            masked_logits = jnp.where(invalid_mask, -1e9, policy_logits)

            # --- Dirichlet Noise Injection ---
            dirichlet_alpha = 0.3
            dirichlet_fraction = 0.25

            noise_key, subkey = jrand.split(subkey)
            noise = jrand.dirichlet(
                noise_key, alpha=jnp.full(policy_logits.shape, dirichlet_alpha)
            )

            probs = jax.nn.softmax(masked_logits, axis=-1)
            mixed_probs = (
                1.0 - dirichlet_fraction
            ) * probs + dirichlet_fraction * noise

            # Re-mask and re-normalize to ensure valid distribution
            mixed_probs = jnp.where(invalid_mask, 0.0, mixed_probs)
            mixed_probs = mixed_probs / jnp.sum(mixed_probs, axis=-1, keepdims=True)

            noisy_logits = jnp.log(mixed_probs + 1e-9)
            noisy_logits = jnp.where(invalid_mask, -1e9, noisy_logits)

            treesearch_root = mctx.RootFnOutput(
                prior_logits=noisy_logits, value=value, embedding=state
            )

            policy_output = mctx.gumbel_muzero_policy(
                params,
                subkey,
                treesearch_root,
                recurrent_fn,
                num_simulations,
                invalid_actions=invalid_mask,
                qtransform=qtransform,
                gumbel_scale=gumbel_scale,
                max_num_considered_actions=num_considered_actions,
            )

            search_policy = policy_output.action_weights
            action = policy_output.action
            vertex_actions = action + 1  # index to vertex

            # Prefetch order from root policy or logits
            prefetch_order = jnp.argsort(masked_logits, axis=-1)[:, ::-1] + 1

            next_state, rewards, done = jax.vmap(step)(
                state, vertex_actions, prefetch_order=prefetch_order
            )

            num_muls += rewards

            try:
                obs = state.tokens
            except (AttributeError, TypeError):
                obs = state.reshape(batchsize, -1)

            aux = {
                "obs": obs,
                "policy": search_policy,
                "reward": rewards,
                "value": value,
                "done": done,
                "action": action,  # record actual action idx for tracking
            }

            return (next_state, num_muls, key), aux

        perf, output = lax.scan(
            loop_fn, (states, num_muls, key), None, length=num_actions
        )
        final_state, num_muls, _ = perf
        return final_state, num_muls, output

    return environment_interaction
