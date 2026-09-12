"""Check the multiply-grid predicate is computable from _contraction_factors,
the Pair list and the operand shapes, and equals the real multiply grid."""
import os, math
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
from graphax import jacve
import importlib
mm = importlib.import_module("graphax.sparse.ops.matmul")

rows = []
_orig_contract = mm._frame_contract

# capture, per contraction: the grid from the PAIRS (the predicate) and the
# grid from the VIEWS (the truth).
_pending = {}

_orig_exec = mm._execute_block_sparse_contraction
def exec_wrap(lhs_val, rhs_val, pairs, ctx):
    N = len(pairs)
    eff, lazy, demote = mm._lazy_frame(lhs_val, rhs_val, pairs)
    shared, total, split, scalar = mm._contraction_factors(eff)
    lhs_leftover = list(lhs_val.shape[3 * N:])
    rhs_leftover = list(rhs_val.shape[3 * N:])
    pred = 1
    for i, p in enumerate(eff):
        pred *= total[i] * split[i] * int(p.lhs.block_len) * int(p.rhs.shared_block_len)
    pred *= math.prod(lhs_leftover) * math.prod(rhs_leftover)
    _pending["pred"] = pred
    return _orig_exec(lhs_val, rhs_val, pairs, ctx)
mm._execute_block_sparse_contraction = exec_wrap

def contract_wrap(a, b, dims):
    (cl, cr), (bl, br) = dims
    b_free = [i for i in range(b.ndim) if i not in cr and i not in br]
    truth = math.prod(a.shape) * math.prod(b.shape[i] for i in b_free)
    rows.append((_pending.get("pred"), truth, tuple(a.shape), tuple(b.shape)))
    return _orig_contract(a, b, dims)
mm._frame_contract = contract_wrap

def f(x, W1, W2):
    return jnp.sum(jnp.tanh(jnp.tanh(x @ W1) @ W2))

x = jnp.asarray(np.random.default_rng(0).standard_normal((8, 12)), jnp.float32)
W1 = jnp.asarray(np.random.default_rng(1).standard_normal((12, 10)), jnp.float32)
W2 = jnp.asarray(np.random.default_rng(2).standard_normal((10, 5)), jnp.float32)
jax.make_jaxpr(jacve(f, "rev", argnums=(1, 2)))(x, W1, W2)
ok = sum(1 for p, t, _, _ in rows if p == t)
print(f"contractions {len(rows)}, predicate == real multiply grid in {ok}")
for p, t, sa, sb in rows[:12]:
    print(f"  pred {p:>12} truth {t:>12} {'OK' if p == t else 'MISMATCH'} lhs {sa} rhs {sb}")
