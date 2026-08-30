import io, sys

P = "/Users/assmuth/dsnn/alphagrad/src/alphagrad/approx/ppo.py"
E = "/Users/assmuth/dsnn/alphagrad/src/alphagrad/approx/env.py"

# ---------------- FIX 2: remat the K-loop -----------------------------
src = io.open(P, encoding="utf-8").read()

OLD = '''        def _carry_heads(M, I, ch, nv, pos, owner, part, vs, vc,
                         pref, dtok, deqn, dcnt):
            carry2 = EncCarry(M=M, I=I, cumhist=ch, nvalid=nv, pos=pos)
            vs2, vc2 = vs, vc
            # The deltas are STORED, with their lengths. The loss re-derives
            # no window from anything, so there is no length to get wrong.
            # PYTHON loop, not a scan: K is static, and at K=1 this is
            # literally the single call it replaced -- same ops, same order,
            # bit-identical.
            for _k in range(dtok.shape[0]):
                carry2, vs2, vc2 = _carry_stream.advance(
                    agent, carry2, vs2, vc2,
                    dtok[_k], deqn[_k], dcnt[_k], owner[_k],
                    window=MAX_DELTA_TOKENS, participants=part[_k],
                    # The loss is reverse-differentiated through this extend,
                    # so it cannot use the rollout's while_loop -- it passes
                    # the batch-wide `budget` instead and gets the scan/cond
                    # form, which has a transpose rule and skips exactly the
                    # same pad steps.
                    chunk=None, budget=_delta_budget,
                )
'''

NEW = '''        def _advance_k(carry2, vs2, vc2, dtok_k, deqn_k, dcnt_k, own_k,
                       part_k):
            return _carry_stream.advance(
                agent, carry2, vs2, vc2,
                dtok_k, deqn_k, dcnt_k, own_k,
                window=MAX_DELTA_TOKENS, participants=part_k,
                # The loss is reverse-differentiated through this extend,
                # so it cannot use the rollout's while_loop -- it passes
                # the batch-wide `budget` instead and gets the scan/cond
                # form, which has a transpose rule and skips exactly the
                # same pad steps.
                chunk=None, budget=_delta_budget,
            )

        # REMAT THE K-LOOP BODY. `advance` is a whole `encode_extend` over
        # the delta window, and the K of them were an UNROLLED PYTHON LOOP:
        # reverse-mode AD stored every step's rows -- a (window, E) array per
        # K per sample -- so peak memory tracked K almost linearly (measured
        # on the TLM: 8971 MiB at K=1, 9037 at 2, 13197 at 4, 13401 at 8,
        # 21785 at 16). With remat each of the K steps stores only its
        # boundary `(carry, vmem_sums, vmem_counts)` and recomputes its own
        # forward when the cotangent arrives -- the same trade `_block` and
        # `_chunk_d` already make one level down, and the reason nesting is
        # correct rather than doubly wasteful: the inner remat bounds what a
        # single recomputed step costs.
        #
        # NOTHING COMPUTED CHANGES, only where the activations live: the
        # recomputation replays the identical jaxpr on the identical inputs,
        # so the forward is bitwise identical and the cotangents are the
        # cotangents of the same function (proved in
        # tests/carry_heads_remat_equiv_test.py: outputs AND gradients
        # bit-identical at K=1..4, remat on vs off).
        # ALPHAGRAD_CARRY_HEADS_REMAT=0 restores the stored-residual form.
        _advance_step = (
            jax.checkpoint(_advance_k)
            if os.environ.get("ALPHAGRAD_CARRY_HEADS_REMAT", "1") != "0"
            else _advance_k
        )

        def _carry_heads(M, I, ch, nv, pos, owner, part, vs, vc,
                         pref, dtok, deqn, dcnt):
            carry2 = EncCarry(M=M, I=I, cumhist=ch, nvalid=nv, pos=pos)
            vs2, vc2 = vs, vc
            # The deltas are STORED, with their lengths. The loss re-derives
            # no window from anything, so there is no length to get wrong.
            # PYTHON loop, not a scan: K is static, and at K=1 this is
            # literally the single call it replaced -- same ops, same order,
            # bit-identical.
            for _k in range(dtok.shape[0]):
                carry2, vs2, vc2 = _advance_step(
                    carry2, vs2, vc2,
                    dtok[_k], deqn[_k], dcnt[_k], owner[_k], part[_k],
                )
'''

assert src.count(OLD) == 1, src.count(OLD)
src = src.replace(OLD, NEW)
io.open(P, "w", encoding="utf-8").write(src)
print("ppo.py patched")

# ---------------- FIX 1: delta window 32768 -> 4096 -------------------
esrc = io.open(E, encoding="utf-8").read()

OLDE = '''# THE ONLY REAL BOUND IN THE OBSERVATION PATH, and the only one JAX's static
# shapes actually require. SIZED FROM THE MEASURED DISTRIBUTION:
# ``decode3_data.py`` over 192 TLM trajectories measured the largest SINGLE
# delta at 25,737 tokens (mean whole-stream length 43,678, max 122,910), so
# 32768 is the next power of two with headroom. The previous 1024 dropped
# ~96% of the flagship's worst delta EVERY step, and the drop was silent
# (issue #81: the ``tokenization/*`` counters are process-blind -- they read 0
# from the driver while the callback process clips).
MAX_DELTA_TOKENS = int(os.environ.get("ALPHAGRAD_MAX_DELTA_TOKENS", "32768"))'''

NEWE = '''# THE ONLY REAL BOUND IN THE OBSERVATION PATH, and the only one JAX's static
# shapes actually require. SIZED FROM THE MEASURED DISTRIBUTION -- and it is
# a MEMORY FLOOR, not just a padding bound: every reverse-differentiated
# `encode_extend` in the loss materialises a ``(window, E)`` row block per
# sample per K-step whatever the actual delta length is, which is why the
# ``ALPHAGRAD_EXTEND_CHUNK`` sweep found peak memory FLAT at 2051 MB across
# every chunk size. Blocking cannot shrink a fixed window; only the window
# can.
#
# 32768 came from a single 25,737-token worst case reported by
# ``decode3_data.py``. A later per-step measurement of the actual delta
# distribution on BOTH the 2- and 3-block TransformerLM (380 / 540 steps)
# does not reproduce it: median 0, mean 75-114, p95 537-642, p99 1001-1411,
# MAX 1173-2833 -- i.e. the observed worst case is 0.23-0.35% of a 32768
# window and the buffer was 8-16x oversized on the very target it was sized
# for. 4096 clears the largest delta ever measured here (2833) with ~45%
# headroom and is still the next power of two above it.
#
# This is safe to shrink because overflow is LOUD: it RAISES by default
# (``ALPHAGRAD_DELTA_OVERFLOW``), so a target whose deltas really do exceed
# 4096 stops rather than silently desyncing the recurrence -- the failure
# mode that made the old bound feel like it had to be generous. Raise the
# env var if a new target trips it; do NOT switch to clip to hide it.
MAX_DELTA_TOKENS = int(os.environ.get("ALPHAGRAD_MAX_DELTA_TOKENS", "4096"))'''

assert esrc.count(OLDE) == 1, esrc.count(OLDE)
esrc = esrc.replace(OLDE, NEWE)
io.open(E, "w", encoding="utf-8").write(esrc)
print("env.py patched")
