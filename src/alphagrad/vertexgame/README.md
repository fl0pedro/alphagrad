# VertexEliminationEnv

`VertexEliminationEnv` is a JAX-compatible environment for optimizing Automatic Differentiation (AD) via vertex elimination (cross-country elimination) in a Computational Graph (generated natively via JAXpr). Allowing an agent to interatively choose an elimination order to minimize the total number of computation (e.g. estimated by multiplications and additions) required to compute the Jacobian.

### Cross-Country Elimination for AD

#### State Representation ($S_t$)

The state maps the evolving topological and algebraic structure of the intermediate representation (IR) as nodes are contracted.

* **Tokenized IR (`graphax`):** A vectorized token sequence representing the active JAXpr. As vertices are eliminated, the IR compresses absorbed operations and expands to explicitly model the newly formed Jacobian accumulations.
* **Structural Tensor (`core`):** A 3D adjacency tensor with the third dimension defining the sparsity type (21 types), input shape and output shape. The tensor is then tokenized columnwise as an input for the model.

#### Action Space ($A_t$)

* $a_t \in \mathcal{V}_{\text{active}}$: A discrete scalar identifying the intermediate vertex ID to eliminate.
* Input and terminal output nodes are rigidly masked. Invalid ID selections trigger a no-op transition.

#### Transition Dynamics (`step`)

Executing `step` applies the cross-country elimination rule, absorbing $a_t$ by front-eliminating incoming edges and back-eliminating outgoing edges.

* **CPU-Bounded Tracing:** Dynamically mutating and re-evaluating the symbolic JAXpr via `graphax` necessitates wrapping the logic in `jax.pure_callback`. Because graph tracing cannot be lowered to XLA, this structural evaluation forms a hard CPU bottleneck during environment rollouts.
* **Accelerator-Native Alternative:** Relying exclusively on the structural tensor (`core`) bypasses runtime tracing. Graph updates become static tensor updates driven by predefined `MUL_SPARSITY_MAP` and `ADD_SPARSITY_MAP` lookups, executing entirely on the accelerator. However losing primitive information and restricting arrays to two dimensions.

#### Reward Formulation ($R_t$)

* $R_t = -(N_{\text{mul}} + N_{\text{add}})$
* The scalar penalty is defined strictly by the local fused multiply and add (fmas) cost of multiplying the incoming edge Jacobians by the outgoing edge Jacobians at $a_t$.
* Sparse diagonal tensors (often seen in Jacobians) efficiently implemented reduce this cost. (TODO, implement this correct count for blocks)

### Batched Rollouts and Parallelism

To efficiently generate trajectories for reinforcement learning algorithms (e.g. PPO), the environment must natively support hardware-accelerated batching.

#### Vectorization Requirement

The environment's transition functions (`reset` and `step`) must be compatible with `jax.vmap`. This allows the rollout loop to execute $N$ environments in perfectly synchronous parallel lockstep, avoiding the overhead of multi-processing wrappers standard in traditional RL libraries.

#### Batching Mechanics in Practice: Spatial and Temporal Axes:
A batched rollout combines `jax.vmap` across the environment dimension $N$ and `jax.lax.scan` across the temporal dimension $T$. The policy evaluates a state tensor of shape $(N, \dots)$ to yield $N$ parallel actions, which are then passed to the vectorized `step` function.

## Verification Suite

The environment is thoroughly verified through a set of `pytest`-compatible tests in `alphagrad/tests/`. Specifically:
```bash
./.venv/bin/pytest alphagrad/tests/test_env_validity.py alphagrad/tests/test_env_robustness.py alphagrad/tests/test_jit_vmap.py alphagrad/tests/test_parallel.py alphagrad/tests/test_env_length.py --no-header --no-cov
```

# TODO 

### 1. Functional Validity
- **[test_env_validity.py](../../tests/test_env_validity.py)**:
    - Verifies full rollouts for different functions (e.g., element-wise vs. matrix multiplication).
    - Confirms that element-wise functions result in zero elimination rewards (diagonal Jacobian).
    - Ensures that every step's tokenized state internally matches an explicit manual `extract_jaxpr` call.

### 2. Optimization and Parallelism
- **[test_jit_vmap.py](../../tests/test_jit_vmap.py)**: Confirms that `reset` and `step` can be JIT-compiled and vmapped without side effects or tracers leaking.
- **[test_parallel.py](../../tests/test_parallel.py)**: Ensures that a `vmap`-ed rollout across a batch of environments stays perfectly in sync with a sequential rollout.

### 3. Robustness
- **[test_env_robustness.py](../../tests/test_env_robustness.py)**:
    - Tests that the environment handles arbitrary valid elimination orders (reverse, random, etc.).
    - Verifies "safe" behavior for invalid actions: out-of-bounds IDs or redundant eliminations return a no-op (zero reward) instead of crashing.

### 4. JAXpr State Evolution
- **[test_env_length.py](../../tests/test_env_length.py)**: Verifies that the underlying JAXpr actually evolves (typically gets longer/more complex) as elimination steps are taken.

## Usage Example

```python
import jax
import jax.numpy as jnp
from alphagrad.vertexgame.vertex_game_w_tokens import VertexEliminationEnv

def f(x):
    return jnp.sum(jnp.sin(x) * x)

x = jnp.ones((5,))
jaxpr = jax.make_jaxpr(f)(x)
env = VertexEliminationEnv.from_jaxpr(jaxpr, args=(x,))

state = env.reset()
out = envstep(state, action=1)

print(f"Reward: {out.reward}, Terminated: {out.terminated}")
```
```out
Reward: -0.0, Terminated: False
```
