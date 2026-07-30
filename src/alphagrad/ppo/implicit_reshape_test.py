from itertools import chain

import jax
import jax.numpy as jnp
import jax.random as jr

key = jr.PRNGKey(42)

a, b, c, d = 2, 3, 4, 5
x, y = 0.24, 1.32


def dense_ref(A_full, B_full):
    return jax.lax.dot_general(A_full, B_full, (((2,), (1,)), ((0,), (0,))))


# --- 1. ([a], b, c) @ (a, c, d) ---
key, k1, k2 = jr.split(key, 3)
A = jr.uniform(k1, (b, c))
B = jr.uniform(k2, (a, c, d))

A_dense = jnp.broadcast_to(A, (a, b, c))
R_ref = dense_ref(A_dense * x, B * y)

z = x * y
R = jax.lax.dot_general(A, B, (((1,), (1,)), ((), ()))).transpose(1, 0, 2)
print(R.shape)

R_dense = jnp.broadcast_to(R, (a, b, d)) * z
assert jnp.allclose(R_dense, R_ref)
print("([a], b, c) @ (a, c, d)")

# --- 2. (a, [b], c) @ (a, c, d) ---
key, k1, k2 = jr.split(key, 3)
A = jr.uniform(k1, (a, c))
B = jr.uniform(k2, (a, c, d))

A_dense = jnp.broadcast_to(jnp.expand_dims(A, 1), (a, b, c))
R_ref = dense_ref(A_dense * x, B * y)

z = x * y
R = jax.lax.dot_general(A, B, (((1,), (1,)), ((0,), (0,))))
print(R.shape)

R_dense = jnp.broadcast_to(jnp.expand_dims(R, 1), (a, b, d)) * z
assert jnp.allclose(R_dense, R_ref)
print("(a, [b], c) @ (a, c, d)")

# --- 3. (a, b, [c]) @ (a, c, d) ---
key, k1, k2 = jr.split(key, 3)
A = jr.uniform(k1, (a, b))
B = jr.uniform(k2, (a, c, d))

A_dense = jnp.broadcast_to(jnp.expand_dims(A, 2), (a, b, c))
R_ref = dense_ref(A_dense * x, B * y)

A_prime = jnp.expand_dims(A, 2)
B_prime = jnp.expand_dims(B.sum(axis=1), 1)

z = x * y
R = A_prime * B_prime
print(R.shape)

R_dense = jnp.broadcast_to(R, (a, b, d)) * z
assert jnp.allclose(R_dense, R_ref)
print("(a, b, [c]) @ (a, c, d)")

# --- 4. (a, b, c) @ (a, [c], d) ---
key, k1, k2 = jr.split(key, 3)
A = jr.uniform(k1, (a, b, c))
B = jr.uniform(k2, (a, d))

B_dense = jnp.broadcast_to(jnp.expand_dims(B, 1), (a, c, d))
R_ref = dense_ref(A * x, B_dense * y)

A_prime = jnp.expand_dims(A.sum(axis=2), 2)
B_prime = jnp.expand_dims(B, 1)

z = x * y
R = A_prime * B_prime
print(R.shape)

R_dense = jnp.broadcast_to(R, (a, b, d)) * z
assert jnp.allclose(R_dense, R_ref)
print("(a, b, c) @ (a, [c], d)")

# --- 5. ([a], [b], c) @ (a, c, d) ---
key, k1, k2 = jr.split(key, 3)
A = jr.uniform(k1, (c,))
B = jr.uniform(k2, (a, c, d))

A_dense = jnp.broadcast_to(A, (a, b, c))
R_ref = dense_ref(A_dense * x, B * y)

z = x * y
R = jax.lax.dot_general(A, B, (((0,), (1,)), ((), ())))
print(R.shape)

R_dense = jnp.broadcast_to(jnp.expand_dims(R, 1), (a, b, d)) * z
assert jnp.allclose(R_dense, R_ref)
print("([a], [b], c) @ (a, c, d)")

# --- 6. ([a], [b], [c]) @ (a, [c], [d]) ---
key, k1, k2 = jr.split(key, 3)
B = jr.uniform(k2, (a,))

A_dense = jnp.ones((a, b, c))
B_dense = jnp.broadcast_to(jnp.expand_dims(B, (1, 2)), (a, c, d))
R_ref = dense_ref(A_dense * x, B_dense * y)

z = c * x * y
R = B
print(R.shape)

R_dense = jnp.broadcast_to(jnp.expand_dims(R, (1, 2)), (a, b, d)) * z
print(R.shape)
assert jnp.allclose(R_dense, R_ref)
print("([a], [b], [c]) @ (a, [c], [d])")

# --- 7. ([a], [b], [c]) @ ([a], [c], [d]) ---
A_dense = jnp.ones((a, b, c))
B_dense = jnp.ones((a, c, d))
R_ref = dense_ref(A_dense * x, B_dense * y)

z = c * x * y
R = None
print(R)

R_dense = jnp.broadcast_to(z, (a, b, d))
assert jnp.allclose(R_dense, R_ref)
print("([a], [b], [c]) @ ([a], [c], [d])")
