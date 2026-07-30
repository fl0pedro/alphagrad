#!/usr/bin/env python3
"""Add ProposerTokenizer.order_token_ids_micro([(action_idx, rules)]) — append-only
tokens for a MICRO-bearing partial order (same pre-edges/new_edges/emit_micro_blocks
pattern as env._callback's 0dfe7ef path). Additive; existing API untouched."""
import pathlib, sys
p = pathlib.Path("src/alphagrad/approx/append_only_jaxpr.py")
s = p.read_text()
if "order_token_ids_micro" in s:
    print("ALREADY PATCHED"); sys.exit(0)
anchor = ('    def order_token_ids(self, action_idxs):\n'
          '        """Full append-only token IDS (persistent vocab) for the order."""\n'
          '        return self.vocab.encode(self.order_tokens(action_idxs))')
add = anchor + ('\n\n'
    '    def order_token_ids_micro(self, actions):\n'
    '        """Append-only token IDS for a MICRO-bearing partial order.\n'
    '\n'
    '        ``actions`` = [(action_idx, rules)] with ``rules`` a tuple of graphax\n'
    '        Diag/Compress/Quant objects (or None/()). Mirrors env._callback\'s\n'
    '        append-only path: eliminate, diff the edge set, emit micro blocks for\n'
    '        the newly created edges. Pure Python — no re-trace.\n'
    '        """\n'
    '        s = self._template\n'
    '        s.reset_to_base()\n'
    '        for a, rules in actions:\n'
    '            pre = set(s._edges.keys())\n'
    '            s.eliminate(int(self.valid[int(a)]), symbolic=True)\n'
    '            if rules:\n'
    '                new_edges = [k for k in s._edges.keys() if k not in pre]\n'
    '                s.emit_micro_blocks(new_edges, tuple(rules))\n'
    '        return self.vocab.encode(s.tokens())')
assert anchor in s, "order_token_ids anchor not found"
s = s.replace(anchor, add, 1)
p.write_text(s)
print("order_token_ids_micro added")
