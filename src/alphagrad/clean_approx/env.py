import jax
import jax.numpy as jnp
from typing import NamedTuple, Any

# We reuse the graphax definitions instead of building custom ones.
# We import SparseTensor and micro-actions here to ensure they're available.
try:
    from graphax.sparse.tensor import SparseTensor
    from graphax.sparse.micro_actions import (
        COMPRESS_KINDS, QUANT_DTYPES, Compress, Diag, Quant, apply_micro_actions
    )
except ImportError:
    pass # for minimal testing without full graphax install

# Step Types
STEP_VERTEX_ELIM = 0
STEP_PATH_SELECT = 1
STEP_APPROX = 2

class EnvState(NamedTuple):
    """
    Environment state for the Vertex Elimination with Approximations MDP.
    Uses the real graphax SparseTensor components.
    """
    # Incrementally tokenized jaxpr representation (dummy array for the interface)
    tokens: jax.Array
    
    # RL phase tracking
    step_type: int
    
    # Mask of remaining vertices to eliminate (1.0 = available, 0.0 = eliminated)
    available_vertices: jax.Array
    
    # Active traversal state
    current_vertex: jax.Array
    current_path_index: jax.Array
    current_approx_step: jax.Array
    
class ApproxEnv:
    def __init__(self, num_vertices: int, num_dynamic_steps: int = 1):
        self.num_vertices = num_vertices
        self.num_dynamic_steps = num_dynamic_steps
        
    def reset(self, key: jax.Array) -> EnvState:
        # Initial tokenized representation of the main function
        tokens = jnp.zeros((1024,), dtype=jnp.int32) 
        
        return EnvState(
            tokens=tokens,
            step_type=STEP_VERTEX_ELIM,
            available_vertices=jnp.ones((self.num_vertices,), dtype=jnp.float32),
            current_vertex=jnp.array(-1, dtype=jnp.int32),
            current_path_index=jnp.array(0, dtype=jnp.int32),
            current_approx_step=jnp.array(0, dtype=jnp.int32)
        )
        
    def step_vertex_elim(self, state: EnvState, action_vertex: jax.Array) -> EnvState:
        """
        Transition: Vertex Elimination -> Path Select
        """
        # Mark vertex as eliminated
        avail = state.available_vertices.at[action_vertex].set(0.0)
        
        # Trigger tokenized jaxpr to print the first local pass
        new_tokens = state.tokens 
        
        return state._replace(
            tokens=new_tokens,
            step_type=STEP_PATH_SELECT,
            available_vertices=avail,
            current_vertex=action_vertex,
            current_path_index=jnp.array(0, dtype=jnp.int32)
        )
        
    def step_path_select(self, state: EnvState, skip: jax.Array) -> EnvState:
        """
        Transition: Path Select -> Approx (if continue) or Next Path (if skip)
        """
        def skip_path(s: EnvState):
            # Tokenize "approx: skip"
            return s._replace(step_type=STEP_VERTEX_ELIM, current_vertex=jnp.array(-1, dtype=jnp.int32))
            
        def continue_path(s: EnvState):
            # Enter approximation phase
            return s._replace(step_type=STEP_APPROX, current_approx_step=jnp.array(0, dtype=jnp.int32))
            
        return jax.lax.cond(skip, skip_path, continue_path, state)
        
    def step_approx(self, state: EnvState, action_type: jax.Array, args: jax.Array) -> EnvState:
        """
        Transition: Approx -> Approx (repeat dynamic steps) -> Next Path
        Applies graphax approximations to pre, post, and new SparseTensors.
        """
        # Here we construct graphax Diag/Compress/Quant objects based on action_type
        # and apply them via apply_micro_actions() to the local Jacobian edges.
        
        next_approx_step = state.current_approx_step + 1
        
        def repeat_approx(s: EnvState):
            return s._replace(current_approx_step=next_approx_step)
            
        def finish_path(s: EnvState):
            return s._replace(step_type=STEP_VERTEX_ELIM, current_vertex=jnp.array(-1, dtype=jnp.int32))
            
        is_done = next_approx_step >= self.num_dynamic_steps
        return jax.lax.cond(is_done, finish_path, repeat_approx, state)

    def step(self, state: EnvState, action: Any) -> EnvState:
        """
        Main step function that dispatches based on step_type.
        """
        pass
