import jax
import jax.numpy as jnp
import jax.random as jr
from graphax.sparse.tensor import SparseDimension, SparseTensor, _arr2st

key = jr.PRNGKey(42)

# --- 1. block @ dense ---
a, b, c, d = 2, 3, 4, 5
key, k1, k2 = jr.split(key, 3)

A = jr.uniform(k1, (a, b, c))
B = jr.uniform(k2, (a * c, d))

A_st = SparseTensor(
    (SparseDimension(0, a, 0, 1, b, 1),), (SparseDimension(1, a, 0, 0, c, 2),), A
)
B_st = _arr2st(B)

B_prime = B.reshape(a, c, d)

R_inter = jax.lax.dot_general(A, B_prime, (((2,), (1,)), ((0,), (0,))))
R = R_inter.reshape(a * b, d)

assert jnp.allclose(R, jnp.array(A_st) @ jnp.array(B_st))
print("block @ dense")

# --- 2. dense @ block ---
a, b, c, d = 2, 3, 4, 5
key, k1, k2 = jr.split(key, 3)

A = jr.uniform(k1, (a, b * c))
B = jr.uniform(k2, (b, c, d))

A_st = _arr2st(A)
B_st = SparseTensor(
    (SparseDimension(0, b, 0, 1, c, 1),), (SparseDimension(1, b, 0, 0, d, 2),), B
)

A_prime = A.reshape(a, b, c)

R_inter = jax.lax.dot_general(A_prime, B, (((2,), (1,)), ((1,), (0,))))
R = R_inter.transpose(1, 0, 2).reshape(a, b * d)

assert jnp.allclose(R, jnp.array(A_st) @ jnp.array(B_st))
print("dense @ block")

# --- 3. block @ pure ---
a, b, c = 2, 3, 4
key, k1, k2 = jr.split(key, 3)

A = jr.uniform(k1, (a, b, c))
B = jr.uniform(k2, (a * c,))

A_st = SparseTensor(
    (SparseDimension(0, a, 0, 1, b, 1),), (SparseDimension(1, a, 0, 0, c, 2),), A
)
B_st = SparseTensor(
    (SparseDimension(0, a * c, 0, 1),), (SparseDimension(1, a * c, 0, 0),), B
)

B_prime = B.reshape(a, c)

R = A * jnp.expand_dims(B_prime, 1)

R_st = SparseTensor(
    (SparseDimension(0, a, 0, 1, b, 1),), (SparseDimension(1, a, 0, 0, c, 2),), R
)

assert jnp.allclose(jnp.array(R_st), jnp.array(A_st) @ jnp.array(B_st))
print("block @ pure")

# --- 4. pure @ block ---
a, b, c = 2, 3, 4
key, k1, k2 = jr.split(key, 3)

A = jr.uniform(k1, (a * b,))
B = jr.uniform(k2, (a, b, c))

A_st = SparseTensor(
    (SparseDimension(0, a * b, 0, 1),), (SparseDimension(1, a * b, 0, 0),), A
)
B_st = SparseTensor(
    (SparseDimension(0, a, 0, 1, b, 1),), (SparseDimension(1, a, 0, 0, c, 2),), B
)

A_prime = A.reshape(a, b)

R = jnp.expand_dims(A_prime, 2) * B

R_st = SparseTensor(
    (SparseDimension(0, a, 0, 1, b, 1),), (SparseDimension(1, a, 0, 0, c, 2),), R
)

assert jnp.allclose(jnp.array(R_st), jnp.array(A_st) @ jnp.array(B_st))
print("pure @ block")

# --- 5. block @ block (A | B) ---
a, b, c, d, e, f = 2, 3, 4, 5, 6, 7
k = e // b  # 2
key, k1, k2 = jr.split(key, 3)

A = jr.uniform(k1, (a, d, e))
B = jr.uniform(k2, (c, b, f))

A_st = SparseTensor(
    (SparseDimension(0, a, 0, 1, d, 1),), (SparseDimension(1, a, 0, 0, e, 2),), A
)
B_st = SparseTensor(
    (SparseDimension(0, c, 0, 1, b, 1),), (SparseDimension(1, c, 0, 0, f, 2),), B
)

A_prime = A.reshape(a, d, k, b).transpose(0, 2, 1, 3).reshape(c, d, b)

R_inter = jax.lax.dot_general(A_prime, B, (((2,), (1,)), ((0,), (0,))))
R = R_inter.reshape(a, k, d, f).transpose(0, 2, 1, 3).reshape(a, d, k * f)

R_st = SparseTensor(
    (SparseDimension(0, a, 0, 1, d, 1),), (SparseDimension(1, a, 0, 0, k * f, 2),), R
)

assert jnp.allclose(jnp.array(R_st), jnp.array(A_st) @ jnp.array(B_st))
print("block @ block (A | B)")

# --- 6. block @ block (B | A) ---
a, b, c, d, e, f = 4, 6, 2, 5, 3, 7
k = b // e  # 2
key, k1, k2 = jr.split(key, 3)

A = jr.uniform(k1, (a, d, e))
B = jr.uniform(k2, (c, b, f))

A_st = SparseTensor(
    (SparseDimension(0, a, 0, 1, d, 1),), (SparseDimension(1, a, 0, 0, e, 2),), A
)
B_st = SparseTensor(
    (SparseDimension(0, c, 0, 1, b, 1),), (SparseDimension(1, c, 0, 0, f, 2),), B
)

B_prime = B.reshape(c, k, e, f).reshape(a, e, f)

R_inter = jax.lax.dot_general(A, B_prime, (((2,), (1,)), ((0,), (0,))))
R = R_inter.reshape(c, k * d, f)

R_st = SparseTensor(
    (SparseDimension(0, c, 0, 1, k * d, 1),), (SparseDimension(1, c, 0, 0, f, 2),), R
)

assert jnp.allclose(jnp.array(R_st), jnp.array(A_st) @ jnp.array(B_st))
print("block @ block (B | A)")

# --- 7. block @ block (gcd) ---
a, b, c, d, e, f = 4, 6, 2, 5, 3, 7
k = jnp.gcd(a, b)  # 2
l = jnp.lcm(a, b) // k  # 6
key, k1, k2 = jr.split(key, 3)

A = jr.uniform(k1, (a, d, e))
B = jr.uniform(k2, (b, c, f))

A_st = SparseTensor(
    (SparseDimension(0, a, 0, 1, d, 1),), (SparseDimension(1, a, 0, 0, e, 2),), A
)
B_st = SparseTensor(
    (SparseDimension(0, b, 0, 1, c, 1),), (SparseDimension(1, b, 0, 0, f, 2),), B
)

A_prime = A.reshape(k, a // k, d, e).transpose(0, 2, 1, 3).reshape(k, d, l)
B_prime = B.reshape(k, b // k, c, f).transpose(0, 2, 1, 3).reshape(k, l, f)

R_inter = jax.lax.dot_general(A_prime, B_prime, (((), ()), ((0, 2), (0, 1))))
R = R_inter.reshape(k, c, e, d, f).transpose(0, 1, 3, 2, 4).reshape(k, c * d, e * f)

R_st = SparseTensor(
    (SparseDimension(0, k, 0, 1, c * d, 1),),
    (SparseDimension(1, k, 0, 0, e * f, 2),),
    R,
)

assert jnp.allclose(jnp.array(R_st), jnp.array(A_st) @ jnp.array(B_st))
print("block @ block (gcd)")

# --- 8. block @ block (coprime, A > B) ---

a, b, c, d, e, f = 2, 3, 4, 5, 6, 7
key = jr.PRNGKey(42)
key, k1, k2 = jr.split(key, 3)

A = jr.uniform(k1, (a, d, e))
B = jr.uniform(k2, (b, c, f))

A_st = SparseTensor(
    (SparseDimension(0, a, 0, 1, d, 1),), (SparseDimension(1, a, 0, 0, e, 2),), A
)
B_st = SparseTensor(
    (SparseDimension(0, b, 0, 1, c, 1),), (SparseDimension(1, b, 0, 0, f, 2),), B
)

A_dense = jnp.array(A_st)

A_prime = A_dense.reshape(a * d, b, c)

R_inter = jax.lax.dot_general(A_prime, B, (((2,), (1,)), ((1,), (0,))))

R = R_inter.transpose(1, 0, 2).reshape(a * d, b * f)

assert jnp.allclose(R, jnp.array(A_st) @ jnp.array(B_st))
print("block @ block (coprime, A > B)")

# --- 9. block @ block (coprime, A < B) ---

key = jr.PRNGKey(0)
a, b, c, d, e, f = 2, 3, 4, 5, 6, 7
key, k1, k2 = jr.split(key, 3)

A = jr.uniform(k1, (b, f, c))
B = jr.uniform(k2, (a, e, d))

A_st = SparseTensor(
    (SparseDimension(0, b, 0, 1, f, 1),), (SparseDimension(1, b, 0, 0, c, 2),), A
)
B_st = SparseTensor(
    (SparseDimension(0, a, 0, 1, e, 1),), (SparseDimension(1, a, 0, 0, d, 2),), B
)

B_dense = jnp.array(B_st)

B_prime = B_dense.reshape(b, c, a * d)

R_inter = jax.lax.dot_general(A, B_prime, (((2,), (1,)), ((0,), (0,))))

R = R_inter.reshape(b * f, a * d)

assert jnp.allclose(R, jnp.array(A_st) @ jnp.array(B_st))
print("block @ block (coprime, A < B)")
