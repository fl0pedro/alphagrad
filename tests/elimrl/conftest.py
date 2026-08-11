import os

# CPU-fast tests: force the CPU backend before jax is imported anywhere.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
