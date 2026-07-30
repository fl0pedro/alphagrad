#!/usr/bin/env python3
"""Flag-guarded AZ objective alignment for E2 (user-directed):
   ALPHAGRAD_AZ_ALIGN=1  ->  (a) DROP flops from the scalarize weights,
                             (b) PopArt per-channel z-score of [lat,peak,flops,cos]
   so AZ optimises the SAME objective as PPO {cosine:1.0, lat:0.06, peak:0.06}.
   Default OFF => MU=0,SD=1,W=[-.06,-.06,-.06,1.0] == byte-identical legacy.
Idempotent-ish: aborts if already patched.
"""
import pathlib, sys
p = pathlib.Path("src/alphagrad/approx/autoscheduler_loop.py")
s = p.read_text()
if "_AZ_ALIGN" in s:
    print("ALREADY PATCHED — abort"); sys.exit(0)

# (A) globals + refresh, right after the symlog def
A_old = "def symlog(v): return np.sign(v) * np.log1p(np.abs(v))\n"
A_new = A_old + '''
# ---- E2 OBJECTIVE ALIGNMENT (user-directed 2026-07-11): PopArt per-channel
# normalisation + DROP FLOPS so AZ optimises the SAME objective as PPO
# {cosine_sim:1.0, latency:0.06, peak:0.06}. Flag-guarded: default OFF =
# byte-identical legacy raw scalarize (MU=0, SD=1, W=[-.06,-.06,-.06,1.0]).
_AZ_ALIGN = os.environ.get("ALPHAGRAD_AZ_ALIGN", "0") == "1"
_AZ_W4 = np.array([-0.06, -0.06, 0.0 if _AZ_ALIGN else -0.06, 1.0], dtype=np.float64)
_AZ_MU4 = np.zeros(4, dtype=np.float64)   # PopArt per-channel mean over [lat,peak,flops,cos]
_AZ_SD4 = np.ones(4, dtype=np.float64)    # PopArt per-channel std (1 => legacy raw scale)
def _refresh_az_norm(_bufY):
    """Recompute PopArt (mu,sigma) over the measured 4-tuple buffer. No-op unless
    ALPHAGRAD_AZ_ALIGN=1 (keeps MU=0,SD=1 => scalarize == legacy raw)."""
    global _AZ_MU4, _AZ_SD4
    if not _AZ_ALIGN:
        return
    Y = np.asarray(_bufY, dtype=np.float64)
    if Y.ndim == 2 and len(Y):
        Y = Y[np.all(np.isfinite(Y), axis=1)]
    if Y.ndim == 2 and len(Y) >= 8:
        _AZ_MU4 = Y.mean(0)
        _AZ_SD4 = Y.std(0) + 1e-8
'''
assert A_old in s, "anchor A (symlog) not found"
s = s.replace(A_old, A_new, 1)

# (B) rewrite scalarize
B_old = ('def scalarize(raw4):\n'
         '    lat, peak, flops, cos = raw4\n'
         '    return float(-0.06 * lat - 0.06 * peak - 0.06 * flops + 1.0 * cos)\n')
B_new = ('def scalarize(raw4):\n'
         '    # higher=better. Legacy(AZ_ALIGN off): -0.06*(lat+peak+flops)+1.0*cos.\n'
         '    # Aligned: PopArt z-score, flops weight 0, W=[-.06,-.06,0,1.0].\n'
         '    r = np.asarray(raw4, dtype=np.float64)\n'
         '    return float(np.sum(_AZ_W4 * (r - _AZ_MU4) / _AZ_SD4))\n')
assert B_old in s, "anchor B (scalarize) not found"
s = s.replace(B_old, B_new, 1)

# (C1) train_ranking weight line
C1_old = "    W = jnp.asarray(np.array([-0.06, -0.06, -0.06, 1.0]))  # scalarize weights over [lat,peak,flops,cos] raw\n"
C1_new = ("    W = jnp.asarray(_AZ_W4)                                 # aligned: flops weight 0; legacy: [-.06]*3+[1]\n"
          "    _MUj = jnp.asarray(_AZ_MU4); _SDj = jnp.asarray(_AZ_SD4)  # PopArt (0,1 => legacy raw)\n")
assert C1_old in s, "anchor C1 (train_ranking W) not found"
s = s.replace(C1_old, C1_new, 1)

# (C2) train_ranking scalar_pred return
C2_old = ("        raw = jnp.sign(sl) * jnp.expm1(jnp.abs(sl))       # (B,4) raw\n"
          "        return jnp.sum(raw * W, axis=-1)                  # (B,) scalar higher=better\n")
C2_new = ("        raw = jnp.sign(sl) * jnp.expm1(jnp.abs(sl))       # (B,4) raw\n"
          "        return jnp.sum(W * (raw - _MUj) / _SDj, axis=-1)  # (B,) scalar higher=better (PopArt when aligned)\n")
assert C2_old in s, "anchor C2 (train_ranking return) not found"
s = s.replace(C2_old, C2_new, 1)

# (D) refresh before seed-buffer bufS init
D_old = 'bufS = [scalarize(y) for y in bufY]                    # scalarized cost (rank target)\n'
D_new = ('_refresh_az_norm(bufY)                                 # PopArt stats from seed buffer (no-op unless AZ_ALIGN)\n'
         'bufS = [scalarize(y) for y in bufY]                    # scalarized cost (rank target)\n')
assert D_old in s, "anchor D (bufS init) not found"
s = s.replace(D_old, D_new, 1)

# (E) refresh + recompute bufS before per-round ranking retrain
E_old = ("    # RETRAIN on the growing buffer (ranking loss)\n"
         "    _tr0 = time.time()\n")
E_new = ("    # E2-ALIGN: refresh PopArt (mu,sigma) on the grown buffer + recompute rank\n"
         "    # targets so train_ranking sees a consistent normalisation (no-op when OFF:\n"
         "    # MU=0,SD=1 => scalarize unchanged => bufS recompute is value-identical).\n"
         "    _refresh_az_norm(bufY)\n"
         "    bufS[:] = [scalarize(y) for y in bufY]\n"
         "    # RETRAIN on the growing buffer (ranking loss)\n"
         "    _tr0 = time.time()\n")
assert E_old in s, "anchor E (ranking retrain) not found"
s = s.replace(E_old, E_new, 1)

p.write_text(s)
print("PATCHED autoscheduler_loop.py (AZ_ALIGN, flag-guarded)")
