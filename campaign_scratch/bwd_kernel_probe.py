"""IS THE PALLAS KERNEL'S OWN ``chunk_size`` ON THE PPO PATH AT ALL?

``PalimpsaMixer.chunk_size`` (default 16) is a static field nothing in
alphagrad overrides. Before sweeping it, establish empirically whether the
kernel is ever entered under the production incremental-encode topology --
the reading of ppo.py is that it is NOT (``encode_extend`` re-implements the
recurrence with ``associative_scan``/``scan``, and every ``self.encode`` call
site is behind ``if precomputed is None``), but a counter is proof and a
reading is not.

Run: this file with the SAME argv the production launcher uses.
"""
import os
import sys

_CNT = {"palimpsa": 0, "attention": 0, "chunk_sizes": set()}


def _install():
    import alphagrad.transformer.palimpsa_pallas as PP
    import alphagrad.transformer.palimpsa_encoder as PE

    _p = PP.palimpsa
    _a = PP.palimpsa_attention

    def wrap_p(q, k, v, b, gt, g, Ip, scale=None, chunk_size=16):
        _CNT["palimpsa"] += 1
        _CNT["chunk_sizes"].add((int(chunk_size), tuple(q.shape)))
        return _p(q, k, v, b, gt, g, Ip, scale=scale, chunk_size=chunk_size)

    def wrap_a(q, k, v, b, gt, g, Ip, scale=None, chunk_size=16):
        _CNT["attention"] += 1
        return _a(q, k, v, b, gt, g, Ip, scale, chunk_size)

    PP.palimpsa = wrap_p
    PP.palimpsa_attention = wrap_a
    # palimpsa_encoder did `from ... import palimpsa`, so it holds its OWN
    # reference; patching the defining module alone would miss every call.
    PE.palimpsa = wrap_p


def main():
    _install()
    from alphagrad.approx import ppo
    try:
        ppo.main()
    except SystemExit:
        pass
    finally:
        print("[kernel-probe] palimpsa() traces      :", _CNT["palimpsa"],
              flush=True)
        print("[kernel-probe] palimpsa_attention()   :", _CNT["attention"],
              flush=True)
        for cs, shp in sorted(_CNT["chunk_sizes"]):
            print(f"[kernel-probe]   chunk_size={cs} q.shape={shp}",
                  flush=True)
        if _CNT["palimpsa"] == 0:
            print("[kernel-probe] VERDICT: the Pallas kernel is NEVER entered "
                  "on this path -- chunk_size is dead code here.", flush=True)
        else:
            print("[kernel-probe] VERDICT: the kernel IS on the path.",
                  flush=True)


if __name__ == "__main__":
    main()
