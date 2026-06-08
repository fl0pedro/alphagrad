"""Unit tests for :mod:`alphagrad.approx.common.render_sequence`."""

from alphagrad.approx.common.render_sequence import (
    OP_COMPRESS,
    OP_DIAG,
    OP_END,
    OP_QUANT,
    render_action_row,
    render_sequence,
)


def test_decode_diag():
    assert render_action_row([7, OP_DIAG, 1, 2, 4, 0]) == (
        7,
        "diag(i=1, j=2, factor=4)",
    )


def test_decode_compress():
    assert render_action_row([3, OP_COMPRESS, 0, 0, 0, 1]) == (
        3,
        'compress(axis=0, kind="min")',
    )


def test_decode_quant_float4():
    from graphax.sparse.micro_actions import QUANT_DTYPE_INDEX
    idx = QUANT_DTYPE_INDEX["float4_e2m1fn"]
    assert render_action_row([5, OP_QUANT, 0, 0, 0, idx]) == (
        5,
        'quant("float4_e2m1fn")',
    )


def test_decode_op_end():
    assert render_action_row([2, OP_END, 0, 0, 0, 0]) == (2, None)


def test_decode_legacy_single_int():
    assert render_action_row([4]) == (4, None)
    assert render_action_row(4) == (4, None)


def test_render_pure_ve_collapse():
    """The cossim=1.0 failure mode: every row is OP_END."""
    seq = [
        [0, OP_END, 1, 2, 5, 18],
        [10, OP_END, 1, 2, 5, 18],
        [1, OP_END, 1, 2, 5, 18],
    ]
    assert render_sequence(seq) == [(0, []), (10, []), (1, [])]


def test_render_grouped_with_approx():
    from graphax.sparse.micro_actions import QUANT_DTYPE_INDEX
    f4 = QUANT_DTYPE_INDEX["float4_e2m1fn"]
    seq = [
        [5, OP_QUANT, 0, 0, 0, f4],
        [5, OP_COMPRESS, 0, 0, 0, 0],
        [5, OP_DIAG, 1, 2, 4, 0],
        [5, OP_END, 0, 0, 0, 0],
        [3, OP_DIAG, 2, 2, 1, 0],
        [3, OP_END, 0, 0, 0, 0],
        [1, OP_END, 0, 0, 0, 0],
    ]
    assert render_sequence(seq) == [
        (
            5,
            [
                'quant("float4_e2m1fn")',
                'compress(axis=0, kind="mean")',
                "diag(i=1, j=2, factor=4)",
            ],
        ),
        (3, ["diag(i=2, j=2, factor=1)"]),
        (1, []),
    ]


def test_render_implicit_commit_on_vertex_change():
    """If the recorder omits OP_END and the vertex changes, we still emit
    the prior group."""
    seq = [
        [5, OP_DIAG, 1, 2, 4, 0],
        [3, OP_DIAG, 0, 1, 2, 0],
    ]
    assert render_sequence(seq) == [
        (5, ["diag(i=1, j=2, factor=4)"]),
        (3, ["diag(i=0, j=1, factor=2)"]),
    ]


def test_render_unknown_op_falls_through():
    assert render_action_row([0, 99, 0, 0, 0, 0]) == (0, "<unknown op=99>")
    assert render_action_row([0, OP_QUANT, 0, 0, 0, 9999])[1] == 'quant("<idx=9999>")'


# ---------------------------------------------------------------------------
# parse_recorded_seq — separate from the renderer above (no overlap of
# code paths) but lives in the same test file because the input wire
# format is shared.
# ---------------------------------------------------------------------------


def test_parse_recorded_seq_pure_ve():
    """A cossim=1 sequence (all OP_END) produces empty transforms and
    1-indexed order."""
    from alphagrad.approx.common.seq_replay import parse_recorded_seq

    seq = [
        [0, OP_END, 0, 0, 0, 0],
        [5, OP_END, 0, 0, 0, 0],
        [3, OP_END, 0, 0, 0, 0],
    ]
    order, transforms = parse_recorded_seq(seq)
    assert order == [1, 6, 4]  # +1 offset
    assert transforms == []


def test_parse_recorded_seq_with_quant_and_axis_sizes():
    """Diag with factor=-1 is resolved against axis_sizes; Quant flows
    through; OP_END commits without adding an op."""
    from alphagrad.approx.common.seq_replay import parse_recorded_seq
    from graphax.sparse.micro_actions import QUANT_DTYPE_INDEX

    f16 = QUANT_DTYPE_INDEX["float16"]
    seq = [
        [4, OP_DIAG, 0, 1, -1, 0],   # gcd(8, 4) = 4 → factor 4
        [4, OP_QUANT, 0, 0, 0, f16],
        [4, OP_END, 0, 0, 0, 0],
        [2, OP_END, 0, 0, 0, 0],
    ]
    order, transforms = parse_recorded_seq(seq, axis_sizes=[8, 4, 12])
    assert order == [5, 3]
    assert len(transforms) == 1
    v, ops = transforms[0]
    assert v == 5
    # First op is Diag with resolved factor 4, second is Quant float16.
    assert ops[0].__class__.__name__ == "Diag"
    assert ops[0].factor == 4
    assert ops[1].__class__.__name__ == "Quant"
    assert ops[1].dtype == "float16"


def test_parse_recorded_seq_skip_low_precision_quant():
    """``skip_low_precision_quant=True`` drops float4 / float8 / int<8 ops."""
    from alphagrad.approx.common.seq_replay import parse_recorded_seq
    from graphax.sparse.micro_actions import QUANT_DTYPE_INDEX

    f4 = QUANT_DTYPE_INDEX["float4_e2m1fn"]
    bf16 = QUANT_DTYPE_INDEX["bfloat16"]
    seq = [
        [0, OP_QUANT, 0, 0, 0, f4],     # dropped under skip flag
        [0, OP_QUANT, 0, 0, 0, bf16],   # kept
        [0, OP_END, 0, 0, 0, 0],
    ]
    order, transforms = parse_recorded_seq(
        seq, skip_low_precision_quant=True,
    )
    assert order == [1]
    assert len(transforms) == 1
    v, ops = transforms[0]
    assert len(ops) == 1
    assert ops[0].dtype == "bfloat16"


def test_parse_recorded_seq_zero_indexed_mode():
    """``one_indexed=False`` keeps the raw vertex IDs for debugging."""
    from alphagrad.approx.common.seq_replay import parse_recorded_seq

    seq = [[7, OP_END, 0, 0, 0, 0]]
    order, _ = parse_recorded_seq(seq, one_indexed=False)
    assert order == [7]
    order2, _ = parse_recorded_seq(seq, one_indexed=True)
    assert order2 == [8]


def test_parse_recorded_seq_malformed_rows():
    """Truncated / empty rows degrade to OP_END at vertex 0; mixed-length
    rows in the same sequence don't crash the parser."""
    from alphagrad.approx.common.seq_replay import parse_recorded_seq

    seq = [
        [],                     # empty row → vertex 0, OP_END
        [4],                    # bare-vertex row (len=1) → vertex 4, OP_END
        [2, OP_DIAG, 1, 0],     # truncated row (len=4) → vertex 2, OP_END (factor info lost)
        [5, OP_DIAG, 0, 1, 2, 0],  # full row → vertex 5, Diag(0, 1, 2)
        7,                      # bare int → vertex 7, OP_END
    ]
    order, transforms = parse_recorded_seq(seq)
    # All vertices are 1-indexed in the output. seen-set dedupes vertex 0
    # (the empty row's 0 + offset = 1 collides with no other entry here).
    assert order == [1, 5, 3, 6, 8]
    assert len(transforms) == 1  # only the full-row OP_DIAG entry
    v, ops = transforms[0]
    assert v == 6
    assert ops[0].__class__.__name__ == "Diag"
    assert ops[0].factor == 2


def test_parse_recorded_seq_diag_gcd_coprime_silently_drops():
    """``factor=-1`` (gcd-auto) against coprime axis sizes resolves to
    factor=1 which is degenerate; the rule is silently dropped — the
    vertex elim still happens, just without block-diagonalisation."""
    from alphagrad.approx.common.seq_replay import parse_recorded_seq

    # gcd(7, 5) = 1 → degenerate → Diag dropped.
    seq = [
        [3, OP_DIAG, 0, 1, -1, 0],
        [3, OP_END, 0, 0, 0, 0],
    ]
    order, transforms = parse_recorded_seq(seq, axis_sizes=[7, 5])
    assert order == [4]  # vertex 3 + 1 offset
    assert transforms == []  # Diag dropped, no other ops


def test_parse_recorded_seq_diag_factor_doesnt_divide_axes():
    """``factor=3`` against axis sizes (8, 4) doesn't divide either side
    → drop the rule rather than have apply_diag raise at jacve time."""
    from alphagrad.approx.common.seq_replay import parse_recorded_seq

    seq = [[0, OP_DIAG, 0, 1, 3, 0]]
    order, transforms = parse_recorded_seq(seq, axis_sizes=[8, 4])
    # 8 % 3 != 0, so Diag dropped.
    assert transforms == []


def test_render_action_row_handles_empty_and_truncated():
    """The renderer's wire-format tolerance should match the parser's."""
    # Empty list — no vertex info; we report 0 for stability.
    v, repr_str = render_action_row([])
    assert v == 0 and repr_str is None
    # Truncated row — first element is vertex, rest treated as OP_END.
    v, repr_str = render_action_row([5, OP_DIAG, 1])
    assert v == 5 and repr_str is None
