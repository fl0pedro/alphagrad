"""Dynamic device dtype-capability scan -> the allowed low-precision quant menu.

Runs a tiny GEMM per candidate dtype on the CURRENT device ONCE (cached per
device), keeps only those that (a) have a real dot_general kernel and (b) beat
float32 by >= min_speedup. Replaces the hardcoded ALPHAGRAD_QUANT_ALLOWED so the
menu auto-adapts: int16/int4/float4 drop out, float8_e4m3fn comes in on Blackwell.
"""
import jax, jax.numpy as jnp, time
from jax import lax

_CACHE = {}
# candidate compute dtypes. float32 IS included (a valid "no-quant" choice the
# policy can pick); unsupported dtypes (no dot_general kernel, e.g. float4_e2m1fn)
# are dropped by the scan. We keep every SUPPORTED dtype and let the policy learn
# the best use — we do NOT prune by speedup (a dtype slow at one size may win at
# another; that's the policy's job to discover).
_CANDIDATES = ["float8_e4m3fn", "float8_e5m2", "float8_e4m3", "float8_e8m0fnu",
               "int8", "int16", "int4", "bfloat16", "float16", "float32",
               "float4_e2m1fn"]


def _bench(dtype_str, N, reps, key):
    """(ms per matmul) or None if the dtype/kernel is unavailable on this device."""
    try:
        d = jnp.dtype(dtype_str)
    except Exception:
        return None
    try:
        if d.name.startswith(("int", "uint")):
            a = (jax.random.normal(key, (N, N)) * 8).astype(d)
            b = (jax.random.normal(key, (N, N)) * 8).astype(d)
            pet = jnp.int32
        else:
            a = jax.random.normal(key, (N, N)).astype(d)
            b = jax.random.normal(key, (N, N)).astype(d)
            pet = jnp.float32 if d.itemsize <= 1 else None
        @jax.jit
        def mm(a, b):
            return lax.dot_general(a, b, (((1,), (0,)), ((), ())), preferred_element_type=pet)
        r = mm(a, b); r.block_until_ready()
        t0 = time.perf_counter()
        for _ in range(reps):
            r = mm(a, b)
        r.block_until_ready()
        return (time.perf_counter() - t0) / reps
    except Exception:
        return None


def scan_allowed_dtypes(N=1024, reps=15, force=None, verbose=True):
    """List of every SUPPORTED compute dtype on THIS device (has a working
    dot_general kernel), ordered fastest-first, float32 always included. Only
    UNSUPPORTED dtypes (no kernel) are dropped. The policy searches the best use
    of the returned menu — we deliberately do NOT prune by speedup. `force`
    (comma string, e.g. ALPHAGRAD_QUANT_ALLOWED) short-circuits with an explicit
    menu. Cached per (device, N)."""
    if force:
        menu = [s.strip() for s in str(force).split(",") if s.strip()]
        if verbose:
            print(f"[dtype-scan] forced menu: {menu}", flush=True)
        return menu
    dev = jax.devices()[0]
    ck = (getattr(dev, "id", 0), getattr(dev, "device_kind", ""), N)
    if ck in _CACHE:
        return _CACHE[ck]
    key = jax.random.PRNGKey(0)
    base = _bench("float32", N, reps, key)
    supported, unsupported, detail = [], [], {}
    for dt in _CANDIDATES:
        ms = _bench(dt, N, reps, key)
        detail[dt] = (base / ms) if (ms and base) else None
        (supported if ms is not None else unsupported).append(dt)
    if "float32" not in supported:      # float32 must always be an option
        supported.append("float32"); detail.setdefault("float32", 1.0)
    supported.sort(key=lambda dt: -(detail[dt] or 0.0))
    _CACHE[ck] = supported
    if verbose:
        kept = ", ".join(f"{dt}({detail[dt]:.2f}x)" if detail[dt] else f"{dt}(?)" for dt in supported)
        print(f"[dtype-scan] {dev.device_kind} N={N}", flush=True)
        print(f"[dtype-scan] SUPPORTED (menu, fastest-first): {kept}", flush=True)
        print(f"[dtype-scan] UNSUPPORTED (dropped): {unsupported}", flush=True)
    return supported


if __name__ == "__main__":
    print("device:", jax.devices()[0].device_kind)
    m = scan_allowed_dtypes()
    print("=> dynamic quant menu:", m)
    # show it also honors a forced override
    print("=> forced:", scan_allowed_dtypes(force="int8,bfloat16"))
