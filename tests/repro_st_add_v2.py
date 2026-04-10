import jax
import jax.numpy as jnp
from graphax.sparse.tensor import SparseTensor, DenseDimension, SparseDimension

def test_sparse_add_mapping_bug():
    # Target shape (2, 2, 3, 2)
    # Logical IDs: 0, 1, 2, 3
    # LHS:
    # ID 0: size 2, val_dim 0
    # ID 1: size 2, val_dim None
    # ID 3: size 2, val_dim 1 (broadcastable size 1)
    # ID 2: size 3, val_dim 2
    
    out_dims_lhs = [
        DenseDimension(0, 2, 0),
        DenseDimension(1, 2, None)
    ]
    primal_dims_lhs = [
        DenseDimension(2, 3, 2), # Note val_dim 2
        DenseDimension(3, 2, 1)  # Note val_dim 1, size 2 (will use val.shape[1]=1)
    ]
    # val.shape must be (2, 1, 3) 
    # axis 0: ID 0 (size 2)
    # axis 1: ID 3 (size 1 -> broad to 2)
    # axis 2: ID 2 (size 3)
    val_lhs = jnp.arange(6).reshape((2, 1, 3)).astype(jnp.float32)
    lhs = SparseTensor(out_dims_lhs, primal_dims_lhs, val_lhs)
    
    # RHS: Same structure but materialized differently or just same to trigger add
    rhs = lhs.copy()
    
    print(f"LHS structure: {lhs}")
    print(f"LHS val shape: {lhs.val.shape}")
    
    try:
        print("Attempting addition...")
        res = lhs + rhs
        print("Success!")
    except Exception as e:
        print(f"Failed with: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sparse_add_mapping_bug()
