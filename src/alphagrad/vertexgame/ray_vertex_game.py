import jax
import jax.numpy as jnp
from flax import nnx
from typing import Any, Sequence, NamedTuple

class EnvState(NamedTuple):
    tokens: jnp.ndarray
    order: jnp.ndarray
    step_count: jnp.ndarray
    fmas: jnp.ndarray
    terminated: jnp.ndarray

class EnvOut(NamedTuple):
    state: EnvState
    reward: jnp.ndarray
    terminated: jnp.ndarray

MAX_TOKENS = 1024 #32768
VOCAB_SIZE = 256  # Must match the embedding table size in the PPO agent

# --- Trie Cache (remains Python-based, wrapped in callbacks) ---

class TrieNode:
    __slots__ = ["children", "tokens", "counts"]
    def __init__(self):
        self.children: dict[int, TrieNode] = {}
        self.tokens: jnp.ndarray | None = None
        self.counts: int | None = None


class TrieCache:
    def __init__(self):
        self.root = TrieNode()

    def _walk(self, order_prefix: Sequence[int]) -> TrieNode | None:
        node = self.root
        for v in order_prefix:
            v_int = int(v)
            child = node.children.get(v_int)
            if child is None:
                return None
            node = child
        return node

    def _walk_or_create(self, order_prefix: Sequence[int]) -> TrieNode:
        node = self.root
        for v in order_prefix:
            v_int = int(v)
            child = node.children.get(v_int)
            if child is None:
                child = TrieNode()
                node.children[v_int] = child
            node = child
        return node

    def get_tokens(self, order_prefix: Sequence[int]) -> jnp.ndarray | None:
        node = self._walk(order_prefix)
        return node.tokens if node else None

    def set_tokens(self, order_prefix: Sequence[int], value: jnp.ndarray):
        self._walk_or_create(order_prefix).tokens = value

    def get_counts(self, order_prefix: Sequence[int]) -> int | None:
        node = self._walk(order_prefix)
        return node.counts if node else None

    def set_counts(self, order_prefix: Sequence[int], value: int):
        self._walk_or_create(order_prefix).counts = value


# --- Callbacks for non-pure Graphax calls ---

def _tokenize_callback(jaxpr, argnums, sparse, order, stop, args_np, consts_np, cache: TrieCache):
    from graphax.core import extract_jaxpr
    import numpy as np
    
    order_prefix = order[:int(stop)].tolist()
    cached = cache.get_tokens(order_prefix)
    if cached is not None:
        return cached

    ve = extract_jaxpr(
        jaxpr, argnums, order_prefix, sparse, args_np, consts_np
    )
    result = np.zeros(MAX_TOKENS, dtype=np.int32)
    tokens = ve.tokenized()
    n = min(len(tokens), MAX_TOKENS)
    result[:n] = tokens[:n]
    np.clip(result, 0, VOCAB_SIZE - 1, out=result)
    
    res_jnp = jnp.array(result)
    cache.set_tokens(order_prefix, res_jnp)
    return res_jnp


def _get_counts_callback(jaxpr, argnums, sparse, order, stop, args_np, consts_np, cache: TrieCache):
    from graphax.core import vertex_elimination_jaxpr
    import numpy as np
    
    order_prefix = order[:int(stop)].tolist()
    cached = cache.get_counts(order_prefix)
    if cached is not None:
        return cached

    outs, aux = vertex_elimination_jaxpr(
        jaxpr,
        order_prefix,
        consts_np,
        *args_np,
        argnums=argnums,
        count_ops=True,
        sparse_representation=sparse,
    )
    counts = int(aux["fmas"])
    cache.set_counts(order_prefix, counts)
    return jnp.array(counts, dtype=jnp.int32)


class RayVertexGame(nnx.Module):
    """
    JAX-compatible vertex elimination environment using flax.nnx.
    """
    def __init__(self, config: dict[str, Any]):
        if "target_fn" in config:
            target_fn_val = config["target_fn"]
            if isinstance(target_fn_val, str):
                import graphax.examples as examples
                target_fn = getattr(examples, target_fn_val)
            else:
                target_fn = target_fn_val

            args_raw = config["args"]
            jax_args = tuple(jnp.array(x) for x in args_raw)
            closed_jaxpr = jax.make_jaxpr(target_fn)(*jax_args)
            self.jaxpr = closed_jaxpr.jaxpr
            self.consts = closed_jaxpr.literals
            self.argnums = tuple(range(len(jax_args)))
            # Wrap data values with nnx.data or keep them as Params/Variables if needed.
            # Here we use nnx.Variable to hold these tuples of arrays.
            self.args_np = nnx.Variable(tuple(jax.device_get(x) for x in jax_args))
            self.consts_np = nnx.Variable(tuple(jax.device_get(c) for c in self.consts))
        else:
            self.jaxpr = config["jaxpr"]
            self.argnums = tuple(config["argnums"])
            self.args_np = nnx.Variable(tuple(jax.device_get(a) for a in config["args"]))
            self.consts_np = nnx.Variable(tuple(jax.device_get(c) for c in config["consts"]))

        self.sparse = config.get("sparse", False)

        # Precompute valid vertices
        valid_vertices = config.get("valid_vertices")
        if valid_vertices is None:
            from graphax.core import _build_graph
            _, _, _, _, vo_vertices = _build_graph(self.jaxpr, self.args_np, self.consts_np)
            valid = []
            for i, eqn in enumerate(self.jaxpr.eqns, 1):
                if eqn.outvars[0] not in self.jaxpr.outvars or i in vo_vertices:
                    valid.append(i)
            valid_vertices = tuple(valid)
        
        self.valid_vertices = jnp.array(valid_vertices, dtype=jnp.int32)
        self.total_v = len(self.jaxpr.eqns)
        self.num_valid = len(valid_vertices)

        # Trie cache (isolated per instance)
        self.cache = TrieCache()

    def reset(self):
        return self.get_observation(self.valid_vertices, jnp.array(0, dtype=jnp.int32))

    def get_observation(self, order, step_count):
        def callback(order, step_count, args_np, consts_np):
            return _tokenize_callback(
                self.jaxpr, self.argnums, self.sparse,
                order, step_count, args_np, consts_np, self.cache
            )

        tokens = jax.pure_callback(
            callback,
            jnp.zeros(MAX_TOKENS, dtype=jnp.int32),
            order, step_count,
            self.args_np.value, self.consts_np.value
        )
        return EnvState(
            tokens=tokens,
            order=order,
            step_count=step_count,
            fmas=jnp.array(0, dtype=jnp.int32),
            terminated=jnp.array(False, dtype=jnp.bool_)
        )

    def step(self, state: EnvState, action: jnp.ndarray):
        # action is a scalar jnp.int32
        action = action.astype(jnp.int32)
        
        # Swapping the chosen vertex into position `idx` (JAX version)
        idx = state.step_count
        order = state.order
        
        # Find position of action in order
        pos = jnp.argmax(order == action)
        
        # Update order: order[idx] = action, and shift elements if necessary
        mask_shift = (jnp.arange(len(order)) > idx) & (jnp.arange(len(order)) <= pos)
        shifted_order = jnp.where(mask_shift, jnp.roll(order, 1), order)
        new_order = shifted_order.at[idx].set(action)
        
        new_step_count = state.step_count + 1
        
        # Get FMA counts via callback
        def callback(order, step_count, args_np, consts_np):
            return _get_counts_callback(
                self.jaxpr, self.argnums, self.sparse,
                order, step_count, args_np, consts_np, self.cache
            )

        new_fmas = jax.pure_callback(
            callback,
            jnp.array(0, dtype=jnp.int32),
            new_order, new_step_count,
            self.args_np.value, self.consts_np.value
        )
        
        reward = (state.fmas - new_fmas).astype(jnp.float32)
        terminated = new_step_count >= self.num_valid
        
        def tokenize_cb(order, step_count, args_np, consts_np):
            return _tokenize_callback(
                self.jaxpr, self.argnums, self.sparse,
                order, step_count, args_np, consts_np, self.cache
            )

        new_tokens = jax.pure_callback(
            tokenize_cb,
            jnp.zeros(MAX_TOKENS, dtype=jnp.int32),
            new_order, new_step_count,
            self.args_np.value, self.consts_np.value
        )

        new_state = EnvState(
            tokens=new_tokens,
            order=new_order,
            step_count=new_step_count,
            fmas=new_fmas,
            terminated=terminated
        )
        
        return EnvOut(new_state, reward, terminated)