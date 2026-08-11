"""alphagrad.elimrl -- M0: unified face/vertex elimination MDP + isolated
real-measurement worker. EXACT accumulation only; no approximations anywhere
in this package.

Submodules (import explicitly -- this __init__ stays import-free so that
JAX-free parents, e.g. the MeasureClient host process, never pull jax):

    env             one MDP: FACE(i,j,k) micro-actions + VERTEX(j) macro-actions
    measure_worker  restartable subprocess worker + MeasureClient (stdlib only)
    baselines       jacve/jacfe/jax.grad/jax.jacrev through the same worker
    symmetry        Foata trace keys + coloured automorphism orbits (M1)
    features        typed static/dynamic feature extraction for the GNN (M2)
    encoder         bucketed + incremental edge-conditioned GNN encoder (M2)
    encoder_bench   M2 acceptance-gate benchmark (per-step end-to-end ms)
"""
