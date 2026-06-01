"""Pin the typed best_sequences.json round-trip.

The legacy on-disk format is a list of int rows with sentinels
(``COMPRESS_SENTINEL=-2``, ``QUANT_SENTINEL=-3``, end=-1, factor=-1 for
gcd). The new format is a list of typed dicts like::

    [{"vertex": 5, "ops": [
        {"op": "Diag", "i": 0, "j": 1, "factor": 2},
        {"op": "Compress", "axes": [2], "kind": "mean"}
    ]}, ...]

Both formats decode to the same ``(order, transforms)`` via
``parse_recorded_seq`` — these tests pin that invariant.
"""
from __future__ import annotations

from graphax.sparse.micro_actions import (
    Compress, Diag, Quant, COMPRESS_KINDS, QUANT_DTYPES,
)

from alphagrad.approx.common.seq_replay import (
    OP_COMPRESS, OP_DIAG, OP_END, OP_QUANT,
    parse_recorded_seq, to_typed_records,
)


def test_to_typed_records_handles_ppo_6tuple():
    """PPO 6-tuples are the most common legacy format. Round-trip into
    typed dicts must preserve vertex / op / args. Includes one of each
    op type plus an END (which becomes an empty ops list)."""
    print("\n[typed] PPO 6-tuple → typed dict")
    legacy = [
        [5, OP_DIAG, 0, 1, 2, 0],
        [3, OP_COMPRESS, 2, 0, 0, 0],
        [2, OP_QUANT, 0, 0, 0, 3],
        [1, OP_END, 0, 0, 0, 0],
    ]
    typed = to_typed_records(legacy)
    assert typed == [
        {"vertex": 5, "ops": [{"op": "Diag", "i": 0, "j": 1, "factor": 2}]},
        {"vertex": 3, "ops": [{"op": "Compress", "axes": [2],
                                "kind": COMPRESS_KINDS[0]}]},
        {"vertex": 2, "ops": [{"op": "Quant", "dtype": QUANT_DTYPES[3]}]},
        {"vertex": 1, "ops": []},
    ], f"unexpected typed records: {typed}"
    print(f"  {len(typed)} records produced, all typed")


def test_to_typed_records_coalesces_same_vertex_rows():
    """Multiple PPO 6-tuple rows for the same vertex must coalesce into
    one typed record with multiple ops — that's the on-disk shape we want
    (one entry per vertex; ops in their applied order)."""
    print("\n[typed] coalesce repeated vertex rows")
    legacy = [
        [5, OP_DIAG, 0, 1, 2, 0],
        [5, OP_COMPRESS, 2, 0, 0, 0],
        [3, OP_DIAG, 0, 1, 4, 0],
    ]
    typed = to_typed_records(legacy)
    assert len(typed) == 2, f"expected 2 vertices, got {len(typed)}: {typed}"
    assert typed[0]["vertex"] == 5
    assert len(typed[0]["ops"]) == 2, (
        f"vertex 5 should carry 2 ops, got {len(typed[0]['ops'])}"
    )
    assert typed[0]["ops"][0]["op"] == "Diag"
    assert typed[0]["ops"][1]["op"] == "Compress"
    print(f"  vertex 5 carries {len(typed[0]['ops'])} ops; vertex 3 carries {len(typed[1]['ops'])}")


def test_to_typed_records_idempotent_on_typed_input():
    """A second pass through ``to_typed_records`` must be a no-op. This
    is important for callers that don't know whether the snapshot's
    ``seq`` field is already typed."""
    print("\n[typed] idempotent on already-typed input")
    typed = [
        {"vertex": 5, "ops": [{"op": "Diag", "i": 0, "j": 1, "factor": 2}]},
        {"vertex": 3, "ops": []},
    ]
    typed2 = to_typed_records(typed)
    assert typed2 == typed, f"not idempotent: {typed2!r} vs {typed!r}"
    print("  pass-through preserved")


def test_parse_recorded_seq_dispatches_on_typed_format():
    """Loading the typed format yields the same (order, transforms) as
    the legacy PPO 6-tuple format — this is the load-side back-compat
    contract: callers don't need to branch."""
    print("\n[typed] parse_recorded_seq accepts typed format")
    legacy_ppo = [
        [4, OP_DIAG, 0, 1, 2, 0],         # 0-indexed vertex 4 → 1-indexed 5
        [2, OP_COMPRESS, 2, 0, 0, 0],     # 0-indexed vertex 2 → 1-indexed 3
        [1, OP_QUANT, 0, 0, 0, 3],        # 0-indexed vertex 1 → 1-indexed 2
    ]
    typed = to_typed_records(legacy_ppo)
    # The typed format records 0-indexed vertices verbatim (matches how
    # the snapshot writer feeds the int recordings through).
    order_typed, t_typed = parse_recorded_seq(typed, one_indexed=False)
    order_legacy, t_legacy = parse_recorded_seq(legacy_ppo, one_indexed=False)
    assert order_typed == order_legacy, (
        f"order mismatch: typed={order_typed} vs legacy={order_legacy}"
    )
    assert t_typed == t_legacy, (
        f"transforms mismatch:\n  typed={t_typed}\n  legacy={t_legacy}"
    )
    print(f"  both formats decode to order={order_typed} "
          f"transforms={[(v, [type(t).__name__ for t in ts]) for v, ts in t_typed]}")


def test_parse_typed_handles_empty_ops_pure_ve():
    """A vertex with empty ``ops`` is the pure-VE case — it must appear
    in ``order`` but not in ``transforms`` (matches the legacy behaviour
    for OP_END rows)."""
    print("\n[typed] empty ops → vertex in order, not in transforms")
    typed = [
        {"vertex": 5, "ops": [{"op": "Diag", "i": 0, "j": 1, "factor": 2}]},
        {"vertex": 3, "ops": []},
        {"vertex": 1, "ops": []},
    ]
    order, transforms = parse_recorded_seq(typed, one_indexed=False)
    assert order == [5, 3, 1]
    assert len(transforms) == 1
    assert transforms[0][0] == 5
    assert isinstance(transforms[0][1][0], Diag)
    print(f"  order={order}, transforms={[(v, [type(t).__name__ for t in ts]) for v, ts in transforms]}")


def test_parse_typed_compress_and_quant_strings_round_trip():
    """COMPRESS/QUANT records use kind/dtype STRINGS (not int indices) so
    the JSON is self-documenting. Decoder must look them up against
    COMPRESS_KINDS / QUANT_DTYPES and emit the right typed micro-action."""
    print("\n[typed] Compress.kind / Quant.dtype as strings")
    typed = [
        {"vertex": 5, "ops": [{"op": "Compress", "axes": [2], "kind": "abs_max"}]},
        {"vertex": 3, "ops": [{"op": "Quant", "dtype": "bfloat16"}]},
    ]
    _, transforms = parse_recorded_seq(typed, one_indexed=False)
    compress = transforms[0][1][0]
    assert isinstance(compress, Compress)
    assert compress.kind == "abs_max"
    assert compress.axes == (2,)
    quant = transforms[1][1][0]
    assert isinstance(quant, Quant)
    assert quant.dtype == "bfloat16"
    print(f"  Compress(kind='abs_max', axes=(2,)) and Quant(dtype='bfloat16') decoded")


def main():
    print("=== typed best_sequences.json round-trip tests ===")
    test_to_typed_records_handles_ppo_6tuple()
    test_to_typed_records_coalesces_same_vertex_rows()
    test_to_typed_records_idempotent_on_typed_input()
    test_parse_recorded_seq_dispatches_on_typed_format()
    test_parse_typed_handles_empty_ops_pure_ve()
    test_parse_typed_compress_and_quant_strings_round_trip()
    print("\nALL TYPED-SEQ FORMAT TESTS OK")


if __name__ == "__main__":
    main()
