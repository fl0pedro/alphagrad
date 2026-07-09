"""Append-only jaxpr tokenization for elimination-order search.

Spec: ``~/dsnn/model_doc/append_only_jaxpr_tokenization.md``.

The AZ / surrogate proposer re-tokenizes each candidate order by re-invoking
graphax ``_callback``, which re-traces the loss jaxpr and recompiles
``jit(_loss)`` (~1230 recompiles / round -- the real round-time bottleneck).

This module removes that cost. It builds the token stream ONCE for the base
graph, then APPENDS one standalone jaxpr *block* per action (eliminate a vertex
/ DIAG / COMPRESS / QUANT / accumulate). No re-trace, no recompile, works for
ANY jaxpr.

Design (mirrors the spec exactly):

* **Base** = the value jaxpr of ``fun``, with the Jacobian output var(s)
  PRE-ALLOCATED (reserved). Eliminations / approximations BUILD the Jacobian
  incrementally, filling those reserved outputs.

* **Each action** is appended as a standalone jaxpr block = local fn-defs (for
  any op the block newly introduces) + the LITERAL jaxpr equations that action
  computes.  Eliminations emit the real chain-rule contraction (``dot_general``
  + ``add``) on the graph's edge tensors; approximations emit the exact lossy
  op (COMPRESS = ``mean`` reduction + ``broadcast``; QUANT = ``convert``).
  Every equation is ``var = fn_name(args)`` where ``fn_name`` was defined by
  some block -- **DEFINE ALL FUNCTIONS, no inlining, even single-use ops.**

* **Two append-only registries** in the bookkeeper: (a) variable names,
  (b) function definitions ((op, exact-params) -> fn_name, defined first-time
  seen, referenced thereafter). Nothing re-defined, nothing re-traced.

The emitted stream is BOTH a token/text stream (for the palimpsa encoder) AND a
runnable numeric program: :func:`AppendOnlyStream.build_jax_fn` compiles the
accumulated blocks into a Python function that evaluates the emitted jaxpr and
returns the reserved Jacobian output(s) -- this is the semantic-correctness
gate (test B in the demo): the emitted append-only jaxpr must be a valid program
computing the right Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import jax._src.core as jax_core
from graphax.core import _build_graph, _force
from graphax.sparse.micro_actions import Diag, Compress, Quant


def _is_literal(v) -> bool:
    return isinstance(v, jax_core.Literal)


# ---------------------------------------------------------------------------
# Token vocabulary
# ---------------------------------------------------------------------------
# A small, fixed vocabulary shared by the base tokenizer and every action
# block, so base + blocks live in ONE coherent stream (spec section 5.1).
# Structural tokens get low fixed ids; identifiers (vars / fn names) and
# integer literals are pooled append-only above the structural range.

_STRUCTURAL = [
    "<pad>", "<base>", "<block>", "<end>",
    "inputs:", "output:", "fndef:", "eqn:",
    "=", "(", ")", ",", "->", ":",
    # primitive op names emitted by the block emitter / base tokenizer
    "integer_pow", "dot_general", "add", "mul", "sub", "tanh", "exp", "log",
    "sin", "cos", "reduce_sum", "reduce_mean", "broadcast", "convert",
    "transpose", "reshape", "neg", "div", "sqrt", "identity", "const",
    # dtype tokens for QUANT
    "f32", "f16", "bf16", "f8", "i8", "i4",
]
_STRUCT_ID = {t: i for i, t in enumerate(_STRUCTURAL)}
_N_STRUCT = len(_STRUCTURAL)


class Vocabulary:
    """Append-only vocabulary: structural ids fixed, identifiers pooled above."""

    def __init__(self) -> None:
        self._id: Dict[str, int] = dict(_STRUCT_ID)
        self._next = _N_STRUCT

    def token_id(self, tok: str) -> int:
        i = self._id.get(tok)
        if i is None:
            i = self._next
            self._id[tok] = i
            self._next += 1
        return i

    def encode(self, toks: Sequence[str]) -> List[int]:
        return [self.token_id(t) for t in toks]

    @property
    def size(self) -> int:
        return self._next


# ---------------------------------------------------------------------------
# Append-only registries (spec section 2.4, invariant 5)
# ---------------------------------------------------------------------------


class VarRegistry:
    """Append-only variable-name registry.

    Maps opaque graph objects (graphax ``core.Var`` / any hashable key) to
    stable, deterministic string names.  Names are assigned first-come and
    NEVER reused or reassigned -- the naming set only ever grows, exactly like
    the token stream.  Reserved Jacobian-output names are pre-allocated up
    front so eliminations / approximations assign into them.
    """

    def __init__(self) -> None:
        self._name: Dict[Any, str] = {}
        self._order: List[str] = []
        self._counter = 0

    def _fresh(self) -> str:
        # a, b, ..., z, a1, b1, ... -- deterministic, jaxpr-like.
        n = self._counter
        self._counter += 1
        letters = "abcdefghijklmnopqrstuvwxyz"
        base = letters[n % 26]
        suffix = n // 26
        return base if suffix == 0 else f"{base}{suffix}"

    def name(self, key: Any) -> str:
        """Return the (stable) name for ``key``, allocating on first sight.

        ``key`` may be any object; unhashable graph objects (jax Literals /
        Vars in some builds) are keyed by ``id`` so the registry never raises.
        """
        try:
            hash(key)
        except TypeError:
            key = ("id", id(key))
        nm = self._name.get(key)
        if nm is None:
            nm = self._fresh()
            self._name[key] = nm
            self._order.append(nm)
        return nm

    def reserve(self, key: Any) -> str:
        """Pre-allocate a reserved output name (same mechanism as ``name``)."""
        return self.name(key)

    def has(self, key: Any) -> bool:
        return key in self._name


@dataclass(frozen=True)
class FnDef:
    """A defined function: an op + its exact params, referenced by name."""

    name: str
    op: str
    params: Tuple[Tuple[str, Any], ...]  # sorted (k, repr(v)) for hashing

    def text(self) -> str:
        if self.params:
            ps = ", ".join(f"{k}={v}" for k, v in self.params)
            return f"fndef: {self.name} = {self.op}[{ps}]"
        return f"fndef: {self.name} = {self.op}"


class FnRegistry:
    """Append-only function-definition registry.

    ``(op, exact-params) -> fn_name``.  A distinct (op, params) is DEFINED the
    first time it is seen anywhere in the stream (append-only); thereafter every
    use references the existing definition.  The defined-function set only ever
    GROWS -- nothing is re-defined.  Applies to ALL ops incl. single-use ones
    (spec invariant 3: no inlining, ever).
    """

    def __init__(self) -> None:
        self._by_key: Dict[Tuple[str, Tuple], str] = {}
        self._defs: List[FnDef] = []
        self._counter = 0

    def _fresh(self) -> str:
        nm = f"F{self._counter}"
        self._counter += 1
        return nm

    def get_or_define(self, op: str, params: Optional[Dict[str, Any]] = None
                      ) -> Tuple[str, Optional[FnDef]]:
        """Return ``(fn_name, new_fndef_or_None)``.

        ``new_fndef`` is non-None ONLY the first time this (op, params) is seen
        -- the block that first uses it appends that def; all later uses append
        nothing (they just reference the name).
        """
        params = params or {}
        key = (op, tuple(sorted((k, repr(v)) for k, v in params.items())))
        name = self._by_key.get(key)
        if name is not None:
            return name, None
        name = self._fresh()
        fndef = FnDef(name=name, op=op, params=key[1])
        self._by_key[key] = name
        self._defs.append(fndef)
        return name, fndef


# ---------------------------------------------------------------------------
# Emitted equation IR (the literal jaxpr equations of a block)
# ---------------------------------------------------------------------------


@dataclass
class Eqn:
    """One literal jaxpr equation: ``out = fn_name(args)``.

    ``fn`` is the FnDef (op + params) selected from the append-only registry.
    ``args`` are input var names OR ('const', ndarray) constant leaves.
    ``out`` is the output var name.
    """

    out: str
    fn: FnDef
    args: Tuple[Any, ...]

    def text(self) -> str:
        parts = []
        for a in self.args:
            if isinstance(a, tuple) and a and a[0] == "const":
                parts.append(f"<const {tuple(a[1].shape)}>")
            else:
                parts.append(str(a))
        return f"eqn: {self.out} = {self.fn.name}({', '.join(parts)})"


@dataclass
class Block:
    """A standalone jaxpr block appended for one action.

    * ``new_fndefs``: fn-defs this block introduces (append-only contribution).
    * ``eqns``: the literal jaxpr equations the action computes.
    * ``reserved_out``: reserved Jacobian-output var name filled by this block
      (None for a pure-intermediate block).
    * ``tag``: human tag (eliminated vertex / approx name) -- verbose intent.
    """

    tag: str
    new_fndefs: List[FnDef] = field(default_factory=list)
    eqns: List[Eqn] = field(default_factory=list)
    reserved_out: Optional[str] = None
    header: str = ""

    def text_lines(self) -> List[str]:
        lines = [f"--- block: {self.tag}"]
        for d in self.new_fndefs:
            lines.append("    " + d.text())
        for e in self.eqns:
            lines.append("    " + e.text())
        if self.reserved_out is not None:
            lines.append(f"    <fills output: {self.reserved_out}>")
        lines.append("    <end>")
        return lines


# ---------------------------------------------------------------------------
# Numeric edge-Jacobian graph (mirror of graphax cross-country elimination)
# ---------------------------------------------------------------------------
#
# We synthesize the block equations from the graph edge structure.  Each edge
# carries a dense partial-derivative tensor of shape (out_dims..., primal_dims..)
# (graphax SparseTensor.dense()).  Cross-country elimination of a vertex
# contracts every (in_edge -> vertex) with every (vertex -> out_edge) over the
# vertex's own dims and accumulates parallel paths -- exactly a chain of
# dot_general + add.  We reproduce that on the dense edge tensors so the emitted
# jaxpr, when evaluated, computes the CORRECT Jacobian (semantic gate, test B).


@dataclass
class EdgeInfo:
    """A live edge in the numeric graph: name of its current tensor + shape.

    ``n_out`` / ``n_prim`` are the counts of leading output-index axes and
    trailing primal-index axes (so a contraction knows which axes to sum).
    """

    var_name: str          # name of the tensor holding this edge Jacobian
    n_out: int             # number of leading output axes
    n_prim: int            # number of trailing primal axes
    out_sizes: Tuple[int, ...]
    prim_sizes: Tuple[int, ...]


class AppendOnlyStream:
    """The append-only tokenizer + numeric emitter.

    Build once for the base graph (:meth:`__init__`), then call
    :meth:`eliminate` / :meth:`compress` / :meth:`quant` / :meth:`diag` per
    action to APPEND a block.  :meth:`tokens` returns the running token stream;
    :meth:`build_jax_fn` compiles the emitted equations into a numeric Jacobian
    function (semantic-correctness check).
    """

    def __init__(self, fun: Callable, args: Sequence[Any], argnums=(0,)):
        self.vocab = Vocabulary()
        self.vars = VarRegistry()
        self.fns = FnRegistry()
        self.blocks: List[Block] = []
        self._base_tokens: List[str] = []

        flat_args = list(args)
        cj = jax.make_jaxpr(fun)(*flat_args)
        self.jaxpr = cj.jaxpr
        self.consts = cj.literals
        self.argnums = tuple(argnums)

        # numeric graph (edges carry dense partial-derivative tensors)
        env, graph, tgraph, vo = _build_graph(
            self.jaxpr, flat_args, self.consts, argnums=argnums)
        self.graph = graph
        self.tgraph = tgraph

        # -- Base tokens: value jaxpr, Jacobian output vars PRE-ALLOCATED --
        toks: List[str] = ["<base>", "inputs:"]
        # differentiable inputs
        self._input_vars = [self.jaxpr.invars[i] for i in self.argnums]
        for v in self._input_vars:
            toks += [self.vars.name(v), ","]
        # value equations (the primal computation) -- define-all-functions
        toks.append("eqn:")
        for eqn in self.jaxpr.eqns:
            op = eqn.primitive.name
            fn_name, fndef = self.fns.get_or_define(op, dict(eqn.params))
            if fndef is not None:
                toks += ["fndef:", fndef.name, "=", op]
            out_name = self.vars.name(eqn.outvars[0])
            toks += [out_name, "=", fn_name, "("]
            for iv in eqn.invars:
                if _is_literal(iv):
                    toks += ["const", ","]
                else:
                    toks += [self.vars.name(iv), ","]
            toks.append(")")
        # outputs + RESERVED Jacobian output vars
        toks.append("output:")
        for v in self.jaxpr.outvars:
            toks += [self.vars.name(v), ","]
        # reserve one Jacobian output var per (output, differentiable-input)
        self._reserved: Dict[Tuple[Any, Any], str] = {}
        toks.append("Jacobians-reserved:")
        for ov in self.jaxpr.outvars:
            for iv in self._input_vars:
                key = ("J", id(ov), id(iv))
                nm = self.vars.reserve(key)
                self._reserved[(id(ov), id(iv))] = nm
                toks += [nm, ","]
        self._base_tokens = toks

        # -- numeric emit bookkeeping --
        # each live edge (sv, dv) -> EdgeInfo. Constant leaves are stored in
        # self._consts keyed by edge var name.
        self._edges: Dict[Tuple[Any, Any], EdgeInfo] = {}
        self._consts: Dict[str, np.ndarray] = {}
        self._live_verts = set()  # central vars not yet eliminated
        for sv in graph:
            for dv in graph[sv]:
                st = _force(graph[sv][dv])
                dense = np.asarray(st.dense())
                n_out = len(st.out_dims)
                n_prim = len(st.primal_dims)
                # edge constant leaf var
                ename = self.vars.name(("edge", id(sv), id(dv)))
                self._consts[ename] = dense
                self._edges[(id(sv), id(dv))] = EdgeInfo(
                    var_name=ename, n_out=n_out, n_prim=n_prim,
                    out_sizes=tuple(dense.shape[:n_out]),
                    prim_sizes=tuple(dense.shape[n_out:]),
                )
        # adjacency by graph var object id -> the actual Var objects
        self._var_by_id = {}
        for sv in graph:
            self._var_by_id[id(sv)] = sv
            for dv in graph[sv]:
                self._var_by_id[id(dv)] = dv
        for dv in tgraph:
            self._var_by_id[id(dv)] = dv
        # predecessors / successors as id-keyed sets
        self._succ: Dict[int, set] = {id(sv): {id(dv) for dv in graph[sv]}
                                      for sv in graph}
        self._pred: Dict[int, set] = {id(dv): {id(sv) for sv in tgraph[dv]}
                                      for dv in tgraph}

        self._input_ids = {id(v) for v in self._input_vars}
        self._output_ids = {id(v) for v in self.jaxpr.outvars}

        # per-instance mutable bookkeeping (see class annotations)
        self._jac_edge_of: Dict[Tuple, str] = {}
        self._compress_ops: Dict[str, Tuple] = {}
        self._quant_ops: Dict[str, Tuple] = {}
        self._computed: Dict[str, Tuple] = {}

    # ---- token / text views ------------------------------------------
    def token_text(self) -> List[str]:
        lines = ["=== BASE ==="]
        lines.append(" ".join(self._base_tokens))
        for b in self.blocks:
            lines += b.text_lines()
        return lines

    def tokens(self) -> List[str]:
        toks = list(self._base_tokens)
        for b in self.blocks:
            toks.append("<block>")
            for d in b.new_fndefs:
                toks += ["fndef:", d.name, "=", d.op]
            for e in b.eqns:
                toks += ["eqn:", e.out, "=", e.fn.name, "("]
                for a in e.args:
                    if isinstance(a, tuple) and a and a[0] == "const":
                        toks += ["const", ","]
                    else:
                        toks += [str(a), ","]
                toks.append(")")
            toks.append("<end>")
        return toks

    def token_ids(self) -> List[int]:
        return self.vocab.encode(self.tokens())

    # ---- block emitters ----------------------------------------------
    def _emit_eqn(self, block: Block, op: str, params: Dict[str, Any],
                  args: Tuple[Any, ...], out_name: str) -> None:
        fn_name, fndef = self.fns.get_or_define(op, params)
        if fndef is not None:
            block.new_fndefs.append(fndef)
        # resolve fndef object for the eqn (get the canonical one)
        canonical = next(d for d in self.fns._defs if d.name == fn_name)
        block.eqns.append(Eqn(out=out_name, fn=canonical, args=args))

    # symbolic elementary partials for the direct input->output edge case
    # (spec worked example: eliminate x**2 -> c = 2*a). Emits the REAL jaxpr of
    # d(op)/d(input) as defined-function equations so the model sees the
    # derivative symbolically rather than as a baked constant.
    def _emit_elementary(self, block: Block, eqn, in_var, in_name: str) -> str:
        prim = eqn.primitive.name
        out = self.vars.name(("elem", id(eqn.outvars[0]), id(in_var),
                              len(block.eqns)))
        if prim == "integer_pow":
            y = int(eqn.params["y"])
            # d(a**y)/da = y * a**(y-1)
            p = self.vars.name(("pw", out))
            self._emit_eqn(block, "integer_pow", {"y": y - 1}, (in_name,), p)
            self._emit_eqn(block, "mul", {}, (("const", np.float32(y)), p), out)
        elif prim == "tanh":
            t = self.vars.name(("th", out))
            self._emit_eqn(block, "tanh", {}, (in_name,), t)
            t2 = self.vars.name(("th2", out))
            self._emit_eqn(block, "mul", {}, (t, t), t2)
            self._emit_eqn(block, "sub", {}, (("const", np.float32(1.0)), t2), out)
        elif prim == "exp":
            self._emit_eqn(block, "exp", {}, (in_name,), out)
        elif prim in ("add",):
            self._emit_eqn(block, "identity", {}, (("const", np.float32(1.0)),), out)
        elif prim == "mul":
            other = [v for v in eqn.invars if v is not in_var]
            oname = self.vars.name(other[0]) if other else in_name
            self._emit_eqn(block, "identity", {}, (oname,), out)
        else:
            self._emit_eqn(block, "identity", {}, (in_name,), out)
        return out

    def eliminate(self, vertex: int, symbolic: bool = False) -> Block:
        """Eliminate ``vertex`` (1-based; central var = jaxpr.eqns[vertex-1]).

        Appends a block with the real chain-rule contraction equations that
        eliminating the vertex computes -- one ``dot_general`` per (in_edge,
        out_edge) path plus an ``add`` accumulation onto any pre-existing edge.
        Fills the reserved Jacobian output when a contraction lands directly on
        an (input -> output) edge.

        ``symbolic``: when the central vertex is itself an OUTPUT with an
        incoming edge straight from a differentiable input (the ``x**2`` worked
        example), emit that edge's elementary partial SYMBOLICALLY (``c = 2*a``)
        instead of as a constant leaf -- matching the spec's worked example.
        """
        eqn = self.jaxpr.eqns[vertex - 1]
        block = Block(tag=f"eliminate v{vertex} ({eqn.primitive.name})")

        # OUTPUT-vertex / direct input->output edge case: the incoming edge from
        # a differentiable input IS the Jacobian. Emit it (symbolically if
        # requested, else as its constant leaf) into the reserved output. This
        # is the ``x**2`` worked example: eliminate -> ``c = 2*a``.
        for central in eqn.outvars:
            cid = id(central)
            if cid in self._output_ids and not self._succ.get(cid):
                for pid in list(self._pred.get(cid, set())):
                    if pid not in self._input_ids:
                        continue
                    in_e = self._edges.get((pid, cid))
                    if in_e is None:
                        continue
                    iv = self._var_by_id[pid]
                    ov = self._var_by_id[cid]
                    rk = (id(ov), id(iv))
                    if rk not in self._reserved:
                        continue
                    if symbolic:
                        nm = self._emit_elementary(
                            block, eqn, iv, self.vars.name(iv))
                    else:
                        nm = in_e.var_name  # constant leaf already emitted
                    block.reserved_out = self._reserved[rk]
                    self._jac_edge_of[rk] = nm
                    self._edges.pop((pid, cid), None)
                    self._pred.get(cid, set()).discard(pid)
                    self._succ.get(pid, set()).discard(cid)

        for central in eqn.outvars:
            cid = id(central)
            if cid not in self._succ:
                continue
            preds = list(self._pred.get(cid, set()))
            succs = list(self._succ.get(cid, set()))
            new_edges: Dict[Tuple[int, int], EdgeInfo] = {}
            for pid in preds:
                in_e = self._edges.get((pid, cid))
                if in_e is None:
                    continue
                for sid in succs:
                    out_e = self._edges.get((cid, sid))
                    if out_e is None:
                        continue
                    # contract out_e (out_dims_c, prim_dims_via_central) with
                    # in_e (out_dims_central, prim_dims_pred): sum over central.
                    prod_name = self.vars.name(("prod", pid, cid, sid,
                                                len(block.eqns)))
                    # dot_general contracting out_e's central primal axes with
                    # in_e's central output axes.
                    n_c = in_e.n_out  # central appears as in_e's output axes
                    dims = self._contract_dims(out_e, in_e, n_c)
                    self._emit_eqn(block, "dot_general",
                                   {"dimension_numbers": dims},
                                   (out_e.var_name, in_e.var_name), prod_name)
                    n_out_new = out_e.n_out
                    n_prim_new = in_e.n_prim
                    out_sizes = out_e.out_sizes
                    prim_sizes = in_e.prim_sizes
                    self._record_computed(prod_name, out_sizes, prim_sizes)
                    key = (pid, sid)
                    existing = self._edges.get(key)
                    if key in new_edges:
                        existing = new_edges[key]
                    if existing is not None:
                        # accumulate parallel path
                        acc_name = self.vars.name(("acc", pid, sid,
                                                   len(block.eqns)))
                        self._emit_eqn(block, "add", {},
                                       (existing.var_name, prod_name), acc_name)
                        self._record_computed(acc_name, out_sizes, prim_sizes)
                        new_edges[key] = EdgeInfo(acc_name, n_out_new, n_prim_new,
                                                  out_sizes, prim_sizes)
                    else:
                        new_edges[key] = EdgeInfo(prod_name, n_out_new,
                                                  n_prim_new, out_sizes,
                                                  prim_sizes)
            # commit new edges, remove central-incident edges
            for pid in preds:
                self._edges.pop((pid, cid), None)
                self._succ.get(pid, set()).discard(cid)
            for sid in succs:
                self._edges.pop((cid, sid), None)
                self._pred.get(sid, set()).discard(cid)
            self._succ.pop(cid, None)
            self._pred.pop(cid, None)
            for (pid, sid), info in new_edges.items():
                self._edges[(pid, sid)] = info
                self._succ.setdefault(pid, set()).add(sid)
                self._pred.setdefault(sid, set()).add(pid)
                # if this is an (input -> output) edge, it fills a reserved out
                if pid in self._input_ids and sid in self._output_ids:
                    iv = self._var_by_id[pid]
                    ov = self._var_by_id[sid]
                    rk = (id(ov), id(iv))
                    if rk in self._reserved:
                        block.reserved_out = self._reserved[rk]
                        self._jac_edge_of[rk] = info.var_name
        self.blocks.append(block)
        return block

    def _contract_dims(self, out_e: EdgeInfo, in_e: EdgeInfo, n_c: int):
        """dimension_numbers for out_e @ in_e contracting the central var axes.

        out_e layout: [out_axes(out_e.n_out)] + [central_axes(n_c)]
        in_e layout : [central_axes(n_c)] + [pred_prim(in_e.n_prim)]
        Contract out_e's trailing central axes with in_e's leading central axes.
        Result: out_e out_axes (batch-free) ++ in_e pred_prim.
        """
        out_contract = tuple(range(out_e.n_out, out_e.n_out + n_c))
        in_contract = tuple(range(0, n_c))
        return ((out_contract, in_contract), ((), ()))

    def compress(self, edge: Tuple[Any, Any], axis: int, kind: str = "mean"
                 ) -> Block:
        """Apply a COMPRESS approximation to a live edge.

        Emits the EXACT lossy jaxpr: ``reduce_mean`` over ``axis`` then
        ``broadcast`` back to shape (spec 2.3).  Both ops are defined functions.
        ``edge`` is (id(sv), id(dv)); ``axis`` is a physical axis of the edge
        tensor.
        """
        info = self._edges[edge]
        block = Block(tag=f"COMPRESS edge axis={axis} kind={kind}")
        shape = info.out_sizes + info.prim_sizes
        # mean over axis
        red_name = self.vars.name(("compress_mean", edge, len(self.blocks)))
        op = "reduce_mean" if kind == "mean" else f"reduce_{kind}"
        self._emit_eqn(block, op, {"axes": (axis,)},
                       (info.var_name,), red_name)
        # broadcast back
        bc_name = self.vars.name(("compress_bc", edge, len(self.blocks)))
        self._emit_eqn(block, "broadcast",
                       {"shape": shape, "axis": axis},
                       (red_name,), bc_name)
        self._compress_ops[bc_name] = (info.var_name, axis, kind, shape)
        self._edges[edge] = EdgeInfo(bc_name, info.n_out, info.n_prim,
                                     info.out_sizes, info.prim_sizes)
        self.blocks.append(block)
        return block

    def quant(self, edge: Tuple[Any, Any], dtype: str = "float16") -> Block:
        """Apply a QUANT approximation: cast the edge tensor to ``dtype``.

        Emits the exact lossy jaxpr ``convert_element_type`` (round-trip
        through the narrow dtype and back to float32 so the numeric loss is
        modelled, matching graphax Quant semantics).
        """
        info = self._edges[edge]
        block = Block(tag=f"QUANT edge dtype={dtype}")
        q_name = self.vars.name(("quant", edge, dtype, len(self.blocks)))
        self._emit_eqn(block, "convert", {"new_dtype": dtype},
                       (info.var_name,), q_name)
        self._quant_ops[q_name] = (info.var_name, dtype)
        self._edges[edge] = EdgeInfo(q_name, info.n_out, info.n_prim,
                                     info.out_sizes, info.prim_sizes)
        self.blocks.append(block)
        return block

    # book-keeping helpers -------------------------------------------------
    def _record_computed(self, name, out_sizes, prim_sizes):
        self._computed[name] = (out_sizes, prim_sizes)

    # ---- numeric evaluation of the emitted jaxpr ---------------------
    def build_numeric(self) -> Dict[Tuple, np.ndarray]:
        """Evaluate the ACCUMULATED emitted equations and return the reserved
        Jacobian outputs (semantic-correctness gate).

        This is a straight interpreter over the emitted ``Eqn`` list: it starts
        from the edge constant leaves, executes every emitted op in order (each
        keyed by its defined function), and reads off the reserved Jacobian
        vars.  If the emitted jaxpr is a valid program computing the right
        Jacobian, this equals graphax's jacve / jax.jacrev to float tolerance.
        """
        vals: Dict[str, Any] = {}
        for name, arr in self._consts.items():
            vals[name] = jnp.asarray(arr)
        for block in self.blocks:
            for e in block.eqns:
                args = []
                for a in e.args:
                    if isinstance(a, tuple) and a and a[0] == "const":
                        args.append(jnp.asarray(a[1]))
                    else:
                        args.append(vals[a])
                vals[e.out] = _apply_op(e.fn.op, dict(e.fn.params), args)
        out = {}
        for rk, name in self._jac_edge_of.items():
            out[rk] = np.asarray(vals[name])
        return out

    def build_jax_fn(self) -> Callable[[], Any]:
        """Return a jittable fn that computes the reserved Jacobian outputs.

        Used for the recompile-free demo: building the stream for a new order
        produces a DIFFERENT set of constant leaves + equations, but the traced
        program shape is identical across orders that reach the same Jacobian,
        so ``jit`` does not recompile (test C).
        """
        edge_names = list(self._consts.keys())
        edge_vals = [jnp.asarray(self._consts[n]) for n in edge_names]
        eqns = [(e.out, e.fn.op, dict(e.fn.params), e.args)
                for b in self.blocks for e in b.eqns]
        jac_out = dict(self._jac_edge_of)

        def fn(consts):
            vals = dict(zip(edge_names, consts))
            for out, op, params, args in eqns:
                a = []
                for x in args:
                    if isinstance(x, tuple) and x and x[0] == "const":
                        a.append(jnp.asarray(x[1]))
                    else:
                        a.append(vals[x])
                vals[out] = _apply_op(op, params, a)
            return {k: vals[v] for k, v in jac_out.items()}

        return jax.jit(fn), edge_vals


# ---------------------------------------------------------------------------
# op interpreter (maps emitted (op, params) -> the real jax primitive call)
# ---------------------------------------------------------------------------


def _apply_op(op: str, params: Dict[str, Any], args: List[Any]):
    if op == "dot_general":
        dn = params["dimension_numbers"]
        if isinstance(dn, str):
            dn = eval(dn)  # params were repr'd for hashing
        return jax.lax.dot_general(args[0], args[1], dn)
    if op == "add":
        return args[0] + args[1]
    if op == "mul":
        return args[0] * args[1]
    if op == "sub":
        return args[0] - args[1]
    if op == "neg":
        return -args[0]
    if op == "reduce_mean":
        axes = params["axes"]
        if isinstance(axes, str):
            axes = eval(axes)
        return jnp.mean(args[0], axis=axes, keepdims=True)
    if op == "reduce_sum":
        axes = params["axes"]
        if isinstance(axes, str):
            axes = eval(axes)
        return jnp.sum(args[0], axis=axes, keepdims=True)
    if op == "broadcast":
        shape = params["shape"]
        if isinstance(shape, str):
            shape = eval(shape)
        return jnp.broadcast_to(args[0], shape)
    if op == "convert":
        dt = params["new_dtype"]
        if isinstance(dt, str) and dt.startswith("'"):
            dt = eval(dt)
        orig = args[0].dtype
        return args[0].astype(jnp.dtype(dt)).astype(orig)
    if op == "identity":
        return args[0]
    if op == "integer_pow":
        y = params["y"]
        if isinstance(y, str):
            y = eval(y)
        return args[0] ** int(y)
    if op == "tanh":
        return jnp.tanh(args[0])
    if op == "exp":
        return jnp.exp(args[0])
    raise NotImplementedError(f"emitted op not interpretable: {op}")
