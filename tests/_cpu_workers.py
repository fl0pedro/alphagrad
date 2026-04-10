import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_DEFAULT_DEVICE'] = 'cpu'

import numpy as np, cloudpickle
from multiprocessing import shared_memory
import jax

_WORK_JAXPR_CACHE = {}

def _get_jaxpr(s, d):
    if s.fn not in _WORK_JAXPR_CACHE: _WORK_JAXPR_CACHE[s.fn] = jax.make_jaxpr(s.fn)(*d.args).jaxpr
    return _WORK_JAXPR_CACHE[s.fn]

def _tokenize(s, d, order, stop):
    from graphax.core import extract_jaxpr
    jaxpr, ord_np = _get_jaxpr(s, d), np.asarray(order)
    arg_np = jax.tree_util.tree_map(np.asarray, d.args)
    con_np = jax.tree_util.tree_map(np.asarray, d.consts)
    ve = extract_jaxpr(jaxpr, s.argnums, s.has_aux, ord_np, int(stop), s.sparse, arg_np, con_np)
    res = np.zeros(1024, dtype=np.int32)
    res[:min(len(ve.tokenized), 1024)] = ve.tokenized[:1024]
    return res

def pickle_worker(data, order, stop):
    s, d = cloudpickle.loads(data); return _tokenize(s, d, order, stop)

def shm_worker(ctx_name, ord_name, ord_shape, stop):
    ctx = shared_memory.SharedMemory(name=ctx_name)
    s, d = cloudpickle.loads(bytes(ctx.buf))
    ctx.close()
    ord_shm = shared_memory.SharedMemory(name=ord_name)
    order = np.ndarray(ord_shape, dtype=np.int32, buffer=ord_shm.buf)
    res = _tokenize(s, d, order, stop)
    ord_shm.close()
    return res
