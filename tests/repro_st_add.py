import jax
import jax.numpy as jnp
from graphax.sparse.tensor import SparseTensor, DenseDimension, SparseDimension

def test_sparse_add_bug():
    # 4D tensor structure
    # out_dims: ID 0 (Dense), ID 1 (Sparse paired with 3)
    # primal_dims: ID 2 (Dense), ID 3 (Sparse paired with 1)
    
    # LHS: only DenseDimension 0 has value. 1, 2, 3 are None or Sparse
    out_dims_lhs = [
        DenseDimension(0, 2, 0), 
        SparseDimension(1, 3, None, 3)
    ]
    primal_dims_lhs = [
        DenseDimension(2, 4, None),
        SparseDimension(3, 3, None, 1)
    ]
    val_lhs = jnp.zeros((2,)) # Only dim 0 materialized
    lhs = SparseTensor(out_dims_lhs, primal_dims_lhs, val_lhs)
    
    # RHS: Same structure, but maybe different materialized dims
    out_dims_rhs = [
        DenseDimension(0, 2, 0),
        SparseDimension(1, 3, None, 3)
    ]
    primal_dims_rhs = [
        DenseDimension(2, 4, None),
        SparseDimension(3, 3, None, 1)
    ]
    val_rhs = jnp.zeros((2,))
    rhs = SparseTensor(out_dims_rhs, primal_dims_rhs, val_rhs)
    
    print("Testing Sparse addition...")
    try:
        res = lhs + rhs
        print("Success!")
    except Exception as e:
        print(f"Failed with: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sparse_add_bug()
