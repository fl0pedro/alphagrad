import multiprocessing as mp
from concurrent import futures
import os

# Create the pool as early as possible with 'spawn' 
# to avoid any interference with JAX's internal threading/GPU states.
_TOKEN_POOL = None

def get_token_pool():
    global _TOKEN_POOL
    if _TOKEN_POOL is None:
        try:
            ctx = mp.get_context('spawn')
            _TOKEN_POOL = futures.ProcessPoolExecutor(
                max_workers=os.cpu_count(),
                mp_context=ctx
            )
        except Exception as e:
            # If spawn fails, we'll hit this
            pass
    return _TOKEN_POOL

# Optional: eager initialization
if os.environ.get("EAGER_TOKEN_POOL", "0") == "1":
    get_token_pool()
