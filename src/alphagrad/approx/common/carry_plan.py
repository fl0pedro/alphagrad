"""THE CARRY CONTAINER, RESOLVED PER PLAN, ON THE MEASUREMENT SIDE.

OWNER RULINGS, 2026-09-16.

  A. The plan-produced carry is the ONLY mode. The exact carry is the plan
     with no approximation on the carried-Jacobian face. There is no
     run-level container flag any more.
  B. The carry's CONTAINER FOLLOWS THE PLAN's approximation on that face, for
     all four classes and their combinations: none is exact real time
     recurrent learning, Diag is e-prop, Reduce is a coarser trace with an
     axis collapsed, Quant is a low precision trace, and Skip is no carry at
     all, which is truncated backpropagation through time. The same for the
     future adjoint's path over the suffix under ``bptt``.

WHERE IT IS REALIZED, AND WHY HERE.

The POLICY always sees ONE graph: the step body with the DENSE carry edge.
Its observation, its vertex set, its face bound and its action space never
move, so one policy acts on one action space and the container is a decision
the policy makes on a face like any other.

The MEASUREMENT is where the decision becomes real. A plan's classes on the
carried-Jacobian face name a rule; that rule is run over the whole prefix (or
suffix), the value it produces is stored in the container the rule implies,
and the program that is compiled and timed is the one that reads THAT
container. Every plan compiles its own program anyway, so this costs the
measurement nothing it was not already paying.

THE APPROXIMATION MOVES OUT OF THE GRAPH AND INTO THE ARGUMENT. On the
measured program the carried faces are accumulated EXACTLY, because the
approximation is already in the value and in its container. Applying it twice
would price it twice, and the rtrl2 agent measured what that looks like: a
Diag on top of an already block diagonal carry costs 15 percent more latency
and buys nothing.

WHAT THIS MODULE DOES, IN ORDER.

1. :func:`register` is called at each of the three sites that build a target
   (the trainer, the measure actor and ``tools/landscape_map``) and records
   what the target was built from, so a variant of it can be built later.
2. :func:`container_for_plan` reads the plan's wires on the carried-Jacobian
   vertices and names the container they imply.
3. :func:`measurement_env` builds that container's program ONCE and caches it.
4. :func:`transport_wires` carries the plan from the policy's graph onto that
   program: the step body's vertices and order come across unchanged, every
   face decision lands on the face that IS its face there, and the carry
   block's vertices are replaced by the ones the container's own attachment
   has, accumulated exactly.

THE TRANSPORT IS VERIFIED, NOT ASSUMED. The two programs differ ONLY inside
``graphax.examples.neuromorphic.snn_carry_scope``; everything outside it is
the same step body, equation for equation. :func:`_alignment` asserts that --
same count, same primitive, same output shape and dtype, in order -- and
raises if it ever stops being true, rather than measuring a plan that has
quietly landed on different faces. :func:`_counterparts` names what else of
the policy's graph is on the measured one, and a face decision whose face
has no counterpart there raises with the face named (owner ruling
2026-09-25, dsnn-rbjh).
"""

from __future__ import annotations

import numpy as np

#: temporal rule -> what :func:`register` was told for THAT GRAPH. One
#: process may serve two graphs of one target (owner ruling 2026-09-22, the
#: alternating run), so this is a map and not a single spec. Each entry is
#: ``{"spec": …, "base": …, "variants": {}, "eval_samples": {}}``:
#:
#:   spec          what the base target was built from
#:   base          the policy-side program: config, its args and its consts
#:   variants      container name -> the built variant, a dict with the keys
#:                 ``config``, ``args``, ``consts``, ``vertex_map``,
#:                 ``alt_carry``, ``valid``, ``chain``, ``var_map``,
#:                 ``carry_map``, ``links`` and ``base`` (see
#:                 :func:`_variant_of`)
#:   eval_samples  (container, base-draw digest, count) -> that variant's
#:                 eval samples; bounded, and cleared when it grows
#:
#: A rule with no given edge (tbptt, window2) holds NO entry: there is no
#: carried value, so there is no container to choose.
_ENTRIES: dict = {}


def reset() -> None:
    """Forget everything. For tests, and for a process that rebuilds."""
    _ENTRIES.clear()


def _entry(config=None) -> dict | None:
    """The entry whose BASE PROGRAM is ``config``'s, or ``None``.

    THE GRAPH IS THE KEY. Two rules of one target have identical step bodies
    and differ only in the given edge, so nothing but the base jaxpr tells
    their entries apart; selecting by anything weaker would build the other
    graph's container and measure a program the policy never acted on.

    ``config`` is ``None`` only where there is nothing to select by -- a test
    or a tool with one graph in the process. That RAISES when the process
    holds two, rather than picking one of them.
    """
    if config is None:
        if len(_ENTRIES) > 1:
            raise ValueError(
                f"this process holds {len(_ENTRIES)} carry-plan graphs "
                f"({sorted(_ENTRIES)}) and the caller named none. Pass the "
                f"config of the graph the plan was acted on; picking one "
                f"would measure the other graph's container.")
        return next(iter(_ENTRIES.values()), None)
    for entry in _ENTRIES.values():
        if entry["base"]["config"].jaxpr is config.jaxpr:
            return entry
    return None


def armed(config=None) -> bool:
    """Is a plan's container a question for ``config``'s graph at all?"""
    if config is None:
        return bool(_ENTRIES)
    return _entry(config) is not None


def register(args_like, key, example, temporal_rule, config, args, consts,
             *, dataset=None, dataset_size=-1, step_position=None):
    """Record what the base target was built from, for ONE graph.

    Called at every site that builds a recurrent SHD env, once per graph the
    process holds. On any other target it is a no-op, so the call can sit
    unconditionally beside the build.

    ``args_like`` is the argparse namespace (or the actor's args dict) that
    ``grad_target_setup`` reads; ``key`` is the SAME key ``get_args`` was
    given, because the variant has to draw the same recording and the same
    weights or its given values are the carry of another weight set.

    Registering a rule REPLACES that rule's entry and leaves every other
    rule's alone: a run that alternates registers each of its graphs in turn,
    and a second call must not forget the first.
    """
    from alphagrad.approx.common.rsnn_shd import GIVEN_EDGE_RULES, is_rsnn
    rule = str(temporal_rule)
    _ENTRIES.pop(rule, None)
    if not is_rsnn(example) or rule not in GIVEN_EDGE_RULES:
        # No given temporal edge means no carried value, so there is no
        # container to choose and nothing here has anything to do.
        return
    _ENTRIES[rule] = {
        "spec": {
            "args_like": args_like,
            "key": key,
            "example": str(example),
            "rule": rule,
            "dataset": dataset,
            "dataset_size": dataset_size,
            "step_position": step_position,
        },
        "base": {"config": config, "args": tuple(args), "consts": consts},
        "variants": {},
        "eval_samples": {},
    }


def _eval_key(eval_samples):
    """The key a variant draws ITS eval samples from.

    THE BASE DRAW DECIDES IT. A variant's eval samples cannot be the base
    ones -- the shapes of the given values move with the container -- so the
    question is only how the two are tied together. They are tied by a DIGEST
    of the base draw: the trainer, every Ray measure actor and
    ``tools/landscape_map`` then derive the SAME variant samples from the same
    base samples, without the episode's key having to travel to an actor that
    is handed arrays. The digest changes with the episode exactly as the base
    draw does, so a variant's samples are redrawn per episode too.

    The recording and the weights are the run's own in every container (the
    generator draws them from the run key), so what differs between a
    container's samples and the base ones is the sampled STEP POSITIONS -- and
    the graph, its vertex count, its face count and every cost channel are
    step-independent by construction on this target.
    """
    import jax.random as jrand
    return jrand.PRNGKey(int.from_bytes(_eval_tag(eval_samples)[:4], "little"))


#: Arrays this big or bigger are hashed by their SHAPE and DTYPE only. The
#: carried Jacobian is 226 MB per sample and hashing its content on every
#: callback would cost more than the measurement; what moves between two
#: episodes' draws is the STEP POSITION, and every small slot -- the input
#: frame and the five carried state components -- moves with it.
_TAG_CONTENT_MAX = 1 << 20


def _eval_tag(eval_samples) -> bytes:
    """A cheap content digest of one episode's eval draw.

    A big slot is never pulled off the device: its shape and dtype go into
    the digest and its content does not.
    """
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    for a in eval_samples:
        shape = tuple(int(d) for d in getattr(a, "shape", ()))
        dtype = np.dtype(getattr(a, "dtype", np.float32))
        h.update(repr(shape).encode())
        h.update(dtype.str.encode())
        nbytes = int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize
        if nbytes < _TAG_CONTENT_MAX:
            h.update(np.asarray(a).tobytes())
    return h.digest()


# ---------------------------------------------------------------------------
# 1. WHICH CONTAINER A PLAN IMPLIES
# ---------------------------------------------------------------------------
def carry_scope_mask(jaxpr) -> np.ndarray:
    """``(len(jaxpr.eqns),)`` bool: is equation ``i`` in the carry block?

    Vertex ``v`` is equation ``v - 1`` (``common/order.py``), so entry
    ``v - 1`` answers for vertex ``v``.
    """
    from graphax.examples.neuromorphic import SNN_CARRY_SCOPE
    out = np.zeros(len(jaxpr.eqns), dtype=bool)
    for i, e in enumerate(jaxpr.eqns):
        stack = getattr(e.source_info, "name_stack", None)
        if stack is None:
            continue
        out[i] = SNN_CARRY_SCOPE in str(stack).split("/")
    return out


def classes_on_carry_faces(jaxpr, o_list, face_specs, face_skips,
                           rule_specs=None) -> set:
    """The ACTION CLASSES the plan put on the carried-Jacobian faces.

    Reads the wire directly, by the same sentinels ``decode_rule_specs_in_frame``
    reads: a row leading with ``QUANT_SENTINEL`` is a Quant, one leading with
    ``COMPRESS_SENTINEL`` is a Reduce (the code spells Reduce ``compress``),
    a row leading with a non-negative axis index is a Diag, and a set bit in
    ``face_skips`` is a Skip.
    """
    from alphagrad.approx.env import COMPRESS_SENTINEL, QUANT_SENTINEL
    mask = carry_scope_mask(jaxpr)
    faces = np.asarray(face_specs)
    skips = np.asarray(face_skips)
    rules = None if rule_specs is None else np.asarray(rule_specs)
    found: set = set()
    for pos, v in enumerate(o_list):
        v = int(v)
        if not (1 <= v <= len(mask)) or not mask[v - 1]:
            continue
        if pos < skips.shape[0] and bool(np.any(skips[pos] == 1)):
            found.add("skip")
        rows = []
        if pos < faces.shape[0]:
            rows.append(faces[pos].reshape(-1, faces.shape[-1]))
        if rules is not None and pos < rules.shape[0]:
            rows.append(rules[pos].reshape(-1, rules.shape[-1]))
        for block in rows:
            lead = block[:, 0]
            if bool(np.any(lead == QUANT_SENTINEL)):
                found.add("quant")
            if bool(np.any(lead == COMPRESS_SENTINEL)):
                found.add("reduce")
            if bool(np.any(lead >= 0)):
                found.add("diag")
    return found


def container_for_plan(config, o_list, face_specs, face_skips,
                       rule_specs=None) -> str | None:
    """The container this plan implies, or ``None`` when nothing applies.

    ``None`` means the process holds no recurrent target with a given edge,
    or the plan put nothing on the carried faces AND the base program is
    already the exact one -- in both cases the measurement runs as it is.

    A SKIP ON ANY CARRIED FACE MEANS NO CARRY. The ruling's table reads
    "Skip = no carry = t-BPTT", and the carry is ONE value produced by ONE
    rule over the recording: declining part of its contraction is declining
    the rule. So a Skip anywhere on the block dominates every other class and
    the program that is measured is the truncated one.
    """
    from alphagrad.approx.common.rsnn_shd import (EXACT_CONTAINER,
                                                  container_from_classes)
    if _entry(config) is None:
        return None
    classes = classes_on_carry_faces(config.jaxpr, o_list, face_specs,
                                     face_skips, rule_specs)
    if not classes:
        return EXACT_CONTAINER
    return container_from_classes(classes)


# ---------------------------------------------------------------------------
# 2. THE PROGRAM THAT CONTAINER IMPLIES
# ---------------------------------------------------------------------------
def _traced_inlined(target_fn, xs):
    """``jax.make_jaxpr(target_fn)(*xs)`` on the form that is eliminated.

    The same three lines the trainer, the measure actor and the sweep tool
    each carry, for the same reason: the elimination runs on the INLINED
    jaxpr, so the vertex numbering has to come from that form.
    """
    import jax
    from graphax import inline_call_primitives
    cj = jax.make_jaxpr(target_fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    if jx is cj.jaxpr:
        return cj
    try:                                    # jax >= 0.4.31
        from jax.extend.core import ClosedJaxpr
    except ImportError:                     # older / internal layout
        from jax._src.core import ClosedJaxpr
    return ClosedJaxpr(jx, consts)


def valid_vertices(jaxpr, args, consts, argnums) -> tuple:
    """The eliminable vertices of ``jaxpr``.

    The same rule ``VertexEliminationEnv.__init__`` applies, called through
    the same graph builder so the two can never drift.
    """
    from alphagrad.approx.env import _build_graph
    _, _, _, vo_vertices = _build_graph(jaxpr, args, consts, argnums)
    out = []
    for i, eqn in enumerate(jaxpr.eqns, 1):
        if eqn.outvars[0] not in jaxpr.outvars or eqn.outvars[0] in vo_vertices:
            out.append(i)
    return tuple(out)


def _build_variant(container: str, entry: dict) -> dict:
    """Build the program ``container`` implies, and its map from the base."""
    from alphagrad.approx.common.examples import (data_gen, get_args, get_fn,
                                                  grad_target_setup)
    from alphagrad.approx.common.rsnn_shd import (EXACT_CONTAINER,
                                                  SKIP_CONTAINER,
                                                  target_example)
    spec = entry["spec"]
    base = entry["base"]
    base_cfg = base["config"]
    if container == SKIP_CONTAINER:
        # NO CARRY AT ALL. The rule the measurement compiles is the truncated
        # one, which is the graph with no given edge.
        rule, cont = "tbptt", EXACT_CONTAINER
    else:
        rule, cont = spec["rule"], container
    example = target_example(spec["example"], rule)
    xs = get_args(example, spec["key"], dataset=spec["dataset"],
                  dataset_size=spec["dataset_size"], temporal_rule=rule,
                  step_position=spec["step_position"],
                  carry_container=cont)
    gen = data_gen(example, dataset=spec["dataset"],
                   dataset_size=spec["dataset_size"], key=spec["key"],
                   temporal_rule=rule, carry_container=cont)
    fn, xs, argnums = grad_target_setup(spec["args_like"], get_fn(example),
                                        xs, example)
    cj = _traced_inlined(fn, xs)
    from alphagrad.approx.common.rsnn_shd import is_full_rollout
    # Under the full rollout the paired reference is jax.grad of the sequence
    # loss for every container, so the skip variant needs no oracle of its own.
    if container == SKIP_CONTAINER and not is_full_rollout(base_cfg):
        # THE QUALITY REFERENCE IS THE ARM'S RULE, NOT THIS PROGRAM'S. A
        # truncated program's own exact gradient is the truncated gradient, so
        # scoring against it would read 1.0 for a plan that threw the whole
        # prefix away. The reference is the base program -- the arm's rule
        # with the exact carry -- and the generator says so.
        from alphagrad.approx.env import _loss_target
        base_gen = base_cfg.data_gen
        gen.reference_oracle = {
            "draw": base_gen,
            # The base rtrl target returns (loss, *state); the oracle is
            # jax.grad of the LOSS (owner ruling 2026-09-24, Q27b).
            "target": _loss_target(base_cfg),
            "argnums": tuple(base_cfg.argnums),
            "args": tuple(base["args"]),
            "slots": tuple(getattr(base_gen, "data_slots",
                                   range(len(base["args"])))),
        }
    # THE VARIANT'S OWN OUTPUT COUNT: the skip variant is the truncated
    # graph with the loss alone, the others carry the five state rows.
    cfg = base_cfg._replace(jaxpr=cj.jaxpr, argnums=tuple(argnums),
                            target_fun=fn, data_gen=gen,
                            carried_outputs=max(len(cj.jaxpr.outvars) - 1, 0))
    return _variant_of(container, cfg, tuple(xs), tuple(cj.literals), base)


def _variant_of(container, cfg, args, consts, base) -> dict:
    """The variant dict of ``container``'s program ``cfg`` against ``base``.

    ``base`` is the policy's program, ``{"config", "args", "consts"}``. The
    keys: ``vertex_map`` and ``alt_carry`` (:func:`_alignment`), ``valid``
    (the measured graph's eliminable vertices), ``chain``
    (:func:`contraction_chains`), ``var_map`` and ``carry_map``
    (:func:`_counterparts`), ``links`` (:func:`_carry_links`), and ``base``
    itself, whose graph :func:`transport_wires` replays to read the faces
    the policy decided on.
    """
    jx = cfg.jaxpr
    base_jx = base["config"].jaxpr
    vmap, alt_carry = _alignment(base_jx, jx)
    var_map, carry_map = _counterparts(base_jx, jx, vmap)
    valid = set(valid_vertices(jx, args, consts, tuple(cfg.argnums)))
    return {
        "container": container,
        "config": cfg,
        "args": tuple(args),
        "consts": tuple(consts),
        "vertex_map": vmap,
        "alt_carry": alt_carry,
        "valid": valid,
        "chain": (contraction_chains(jx, [v for v in alt_carry if v in valid])
                  if chain_applies(container) else set()),
        "var_map": var_map,
        "carry_map": carry_map,
        "links": _carry_links(base_jx, jx, var_map, carry_map, alt_carry),
        "base": base,
    }


def measurement_env(container: str, config=None) -> dict | None:
    """The built variant for ``container`` on ``config``'s graph.

    ``None`` when no variant is needed, or when this process holds no entry
    for that graph. ``config`` names WHICH graph in a process that serves two.
    """
    from alphagrad.approx.common.rsnn_shd import EXACT_CONTAINER
    entry = _entry(config)
    if entry is None:
        return None
    if container == EXACT_CONTAINER:
        # The base program IS the exact container: the policy's graph carries
        # the dense edge by construction, so there is nothing to build.
        return None
    v = entry["variants"].get(container)
    if v is None:
        v = _build_variant(container, entry)
        entry["variants"][container] = v
    return v


def eval_samples_for(container: str, eval_samples, config=None,
                     generator=None):
    """This episode's eval samples, redrawn in ``container``.

    Keyed by a digest of the base draw (:func:`_eval_key`), so every process
    that measures this plan builds the same ones, and cached per (container,
    digest) INSIDE the graph's own entry, so two graphs in one process never
    read each other's draw. ``config`` names which graph.

    With ``generator`` -- the plan's own draw through its measured
    executable -- the samples are drawn through it on the container's program
    (the base program for ``exact``) and are NOT cached: they belong to one
    plan.
    """
    if not eval_samples:
        return None
    entry = _entry(config)
    if entry is None:
        return None
    from alphagrad.approx.common.rsnn_shd import is_full_rollout
    if is_full_rollout(entry["base"]["config"]):
        # Whole recordings do not depend on the container.
        return eval_samples
    var = measurement_env(container, config)
    if var is None and generator is None:
        return None
    n = int(len(eval_samples[0]))
    from alphagrad.approx.common.eval_samples import generate_eval_samples

    class _Shim:
        config = entry["base"]["config"] if var is None else var["config"]
        args = entry["base"]["args"] if var is None else var["args"]

    if generator is not None:
        _Shim.config = _Shim.config._replace(data_gen=generator)
        return generate_eval_samples(_Shim, _eval_key(eval_samples), n)
    cache = entry["eval_samples"]
    tag = (container, _eval_tag(eval_samples), n)
    hit = cache.get(tag)
    if hit is not None:
        return hit
    out = generate_eval_samples(_Shim, _eval_key(eval_samples), n)
    if len(cache) > 64:
        cache.clear()
    cache[tag] = out
    return out


# ---------------------------------------------------------------------------
# 3. THE ALIGNMENT AND THE TRANSPORT
# ---------------------------------------------------------------------------
def _eqn_signature(eqn):
    outs = tuple((tuple(v.aval.shape), str(v.aval.dtype))
                 for v in eqn.outvars)
    return (eqn.primitive.name, outs)


def _alignment(base_jaxpr, alt_jaxpr):
    """``(vertex_map, alt_carry_vertices)``, or raise.

    The two programs differ only inside the carry block. Everything outside
    it is aligned by POSITION and CHECKED equation by equation, so a change to
    the model that moved a step-body equation would raise here instead of
    silently landing a plan on a different step body.
    """
    b_mask = carry_scope_mask(base_jaxpr)
    a_mask = carry_scope_mask(alt_jaxpr)
    b_body = [i for i, m in enumerate(b_mask) if not m]
    a_body = [i for i, m in enumerate(a_mask) if not m]
    if len(b_body) != len(a_body):
        raise ValueError(
            f"the two carry containers disagree about the STEP BODY: "
            f"{len(b_body)} equations outside the carry scope against "
            f"{len(a_body)}. The container may only change the attachment "
            f"block; a difference here means the plan cannot be carried "
            f"across and the measurement would be of another graph.")
    for i, j in zip(b_body, a_body):
        bs = _eqn_signature(base_jaxpr.eqns[i])
        as_ = _eqn_signature(alt_jaxpr.eqns[j])
        if bs != as_:
            raise ValueError(
                f"step-body equation {i + 1} of the policy's graph is {bs} "
                f"and equation {j + 1} of the measured graph is {as_}. The "
                f"two must be the same step body.")

    vertex_map = {i + 1: j + 1 for i, j in zip(b_body, a_body)}
    alt_carry = tuple(j + 1 for j, m in enumerate(a_mask) if m)
    return vertex_map, alt_carry


def _producers(jaxpr) -> dict:
    """variable -> the 1-based vertex whose equation produces it."""
    return {ov: i for i, e in enumerate(jaxpr.eqns, 1) for ov in e.outvars}


def _same_aval(b, a) -> bool:
    return (tuple(b.aval.shape) == tuple(a.aval.shape)
            and str(b.aval.dtype) == str(a.aval.dtype))


def _same_operand(b, a, var_map) -> bool:
    """Is operand ``a`` of the measured graph the counterpart of ``b``?"""
    from jax._src.core import Literal
    if isinstance(b, Literal) or isinstance(a, Literal):
        if not (isinstance(b, Literal) and isinstance(a, Literal)):
            return False
        vb, va = np.asarray(b.val), np.asarray(a.val)
        return (vb.dtype == va.dtype and vb.shape == va.shape
                and bool(np.array_equal(vb, va)))
    return var_map.get(b) is a


def _same_equation(be, ae, var_map) -> bool:
    """The same primitive, with the same parameters, on counterparts."""
    if (_eqn_signature(be) != _eqn_signature(ae)
            or len(be.invars) != len(ae.invars)):
        return False
    try:
        if be.params != ae.params:
            return False
    except (TypeError, ValueError):
        # A parameter that does not compare (an array) is not known equal.
        return False
    return all(_same_operand(b, a, var_map)
               for b, a in zip(be.invars, ae.invars))


def _counterparts(base_jaxpr, alt_jaxpr, vertex_map):
    """``(var_map, carry_map)``: what of the policy's graph IS on the other.

    A face is the triple of its eliminated vertex, its predecessor and its
    successor, and those are ids of the graph at hand (CONTEXT.md, Face;
    finding dsnn-xox9), so a face decision can only move between the two
    graphs by what the three vertices ARE (owner ruling 2026-09-25,
    dsnn-rbjh). ``var_map`` sends every variable of the policy's graph that
    has a counterpart to it, and nothing else:

      * an INPUT is the argument in the same slot, typed the same;
      * a STEP-BODY variable is its equation's, through ``vertex_map``;
      * an ATTACHED STATE -- a carry variable the step body reads -- is what
        the aligned step equation of the measured graph reads in the same
        operand slot: the one thing of the block the step sees;
      * a carry OUTPUT (the attached loss under ``bptt``) is the measured
        graph's output in the same position, when that is the carry's too;
      * any other carry variable is the carry equation of the measured graph
        that applies the same primitive, with the same parameters, to the
        counterparts of its operands: the weight differences on every
        container, the whole block where the container leaves it alone.

    What remains has no counterpart: the inside of a block the container
    rewrote (the dense rows' stacked tensordot against the diag container's
    row sums), and what the truncated program of the skip container drops
    (the attached states, the state rows, the attached loss). ``carry_map``
    is the carry vertices among the counterparts, vertex -> vertex.

    RAISES when the two step bodies read different values in one operand
    slot, or when two variables would share one counterpart: either would
    move a decision onto a face that is not its own.
    """
    from jax._src.core import Literal
    b_mask = carry_scope_mask(base_jaxpr)
    a_mask = carry_scope_mask(alt_jaxpr)
    b_prod, a_prod = _producers(base_jaxpr), _producers(alt_jaxpr)
    alt_inputs = set(alt_jaxpr.invars)
    var_map: dict = {}
    taken: dict = {}
    carry_map: dict = {}

    def bind(b, a, where):
        if var_map.get(b, a) is not a or taken.get(a, b) is not b:
            raise ValueError(
                f"{where}: {b} of the policy's graph would have two "
                f"counterparts, or {a} of the measured graph two originals. "
                f"A face decision would land on a face that is not its own.")
        var_map[b] = a
        taken[a] = b
        i = b_prod.get(b)
        if i is not None and b_mask[i - 1]:
            carry_map[i] = a_prod[a]

    for group in ("constvars", "invars"):
        for n, (b, a) in enumerate(zip(getattr(base_jaxpr, group),
                                       getattr(alt_jaxpr, group))):
            if _same_aval(b, a):
                bind(b, a, f"{group[:-1]} {n}")
    for i, j in vertex_map.items():
        for b, a in zip(base_jaxpr.eqns[i - 1].outvars,
                        alt_jaxpr.eqns[j - 1].outvars):
            bind(b, a, f"step-body equation {i}")
    # THE ATTACHED STATES, read off the aligned step equations' operands.
    for i, j in vertex_map.items():
        be, ae = base_jaxpr.eqns[i - 1], alt_jaxpr.eqns[j - 1]
        where = (f"step-body equation {i} of the policy's graph (equation "
                 f"{j} of the measured graph)")
        if len(be.invars) != len(ae.invars):
            raise ValueError(
                f"{where} reads {len(be.invars)} operands against "
                f"{len(ae.invars)}. The two must be the same step body.")
        for p, (b, a) in enumerate(zip(be.invars, ae.invars)):
            if isinstance(b, Literal) or b in var_map:
                if not _same_operand(b, a, var_map):
                    raise ValueError(
                        f"{where} reads {b} in operand {p} and the measured "
                        f"graph reads {a}. The two must be the same step "
                        f"body.")
                continue
            i_b, j_a = b_prod.get(b), a_prod.get(a)
            if i_b is None or not b_mask[i_b - 1]:
                raise ValueError(
                    f"{where} reads {b} in operand {p}, which is neither the "
                    f"carry block's nor a value the measured graph has.")
            if j_a is None and a in alt_inputs:
                # THE CARRY IS DECLINED: the truncated program reads the raw
                # state where the policy's reads the attached one, and the
                # attached state has no counterpart there.
                continue
            if j_a is None or not a_mask[j_a - 1] or not _same_aval(b, a):
                raise ValueError(
                    f"{where} reads the attached {b} (vertex {i_b}) in "
                    f"operand {p} and the measured graph reads {a}, which is "
                    f"not an attached value of the same type.")
            bind(b, a, where)
    # THE CARRY'S OUTPUTS: the attached loss of a rule whose block closes
    # the step. Under the skip container the output is the step's own loss,
    # the counterpart of the policy's loss vertex, and the attached loss has
    # none.
    for o, (b, a) in enumerate(zip(base_jaxpr.outvars, alt_jaxpr.outvars)):
        if isinstance(b, Literal) or isinstance(a, Literal):
            continue
        if b in var_map:
            if var_map[b] is not a:
                raise ValueError(
                    f"output {o} is {b} on the policy's graph and {a} on the "
                    f"measured graph, which is not its counterpart.")
            continue
        i_b, j_a = b_prod.get(b), a_prod.get(a)
        if (i_b is not None and b_mask[i_b - 1] and j_a is not None
                and a_mask[j_a - 1] and _same_aval(b, a)):
            bind(b, a, f"output {o}")
    # THE REST OF THE BLOCK, forward: an equation is its counterpart's when
    # it is the same operation on counterparts. Ambiguity names none.
    a_carry = [j for j in range(1, len(alt_jaxpr.eqns) + 1) if a_mask[j - 1]]
    for i in range(1, len(base_jaxpr.eqns) + 1):
        if not b_mask[i - 1] or i in carry_map:
            continue
        be = base_jaxpr.eqns[i - 1]
        hits = [j for j in a_carry
                if not any(ov in taken for ov in alt_jaxpr.eqns[j - 1].outvars)
                and _same_equation(be, alt_jaxpr.eqns[j - 1], var_map)]
        if len(hits) == 1:
            for b, a in zip(be.outvars, alt_jaxpr.eqns[hits[0] - 1].outvars):
                bind(b, a, f"carry equation {i}")
    return var_map, carry_map


def _carry_links(base_jaxpr, alt_jaxpr, var_map, carry_map, alt_carry):
    """How the carry vertices WITHOUT a counterpart join what has one.

    The faces of a step-body vertex name what is live around it: an attached
    state not eliminated yet, or, once it is, whatever its eliminated
    ancestors lead back to -- a weight, a weight difference, a vertex inside
    the block. So beyond where the vertices with a counterpart go, an order
    carried across has to keep WHEN each link through the rewritten inside
    of the block closes: when the last vertex on some path from a SOURCE (a
    variable with a counterpart that the inside reads) to a SINK (a carry
    variable with a counterpart that reads the inside: an attached state, a
    carry output) is eliminated. :func:`transport_order` reads that off the
    policy's order and closes the same link in the same segment of the
    step body on the measured graph.

    Returns a dict:

      pairs      measured carry vertex without a counterpart -> the
                 (source, sink) links through it, named by the POLICY's
                 variables
      base       the policy's carry vertices without a counterpart, in
                 equation order, each with its operands: a policy variable
                 with a counterpart, or such a vertex
      sinks      sink -> (its vertex on the policy's graph, its operands)
      consumers  measured carry vertex -> the carry vertices that read it
    """
    from jax._src.core import Literal
    b_mask = carry_scope_mask(base_jaxpr)
    b_prod, a_prod = _producers(base_jaxpr), _producers(alt_jaxpr)
    back = {a: b for b, a in var_map.items()}
    b_free = {i for i in range(1, len(base_jaxpr.eqns) + 1)
              if b_mask[i - 1] and i not in carry_map}

    def operands(eqn):
        out = []
        for iv in eqn.invars:
            if isinstance(iv, Literal):
                continue
            p = b_prod.get(iv)
            if p in b_free:
                out.append(p)
            elif iv in var_map:
                out.append(iv)
        return tuple(out)

    base = [(i, operands(base_jaxpr.eqns[i - 1])) for i in sorted(b_free)]
    sinks = {ov: (i, operands(base_jaxpr.eqns[i - 1]))
             for i in carry_map for ov in base_jaxpr.eqns[i - 1].outvars}
    carry = set(int(v) for v in alt_carry)
    a_free = sorted(j for j in carry if j not in set(carry_map.values()))
    consumers: dict = {j: [] for j in carry}
    for c, e in enumerate(alt_jaxpr.eqns, 1):
        if c not in carry:
            continue
        for iv in e.invars:
            p = None if isinstance(iv, Literal) else a_prod.get(iv)
            if p in carry and c not in consumers[p]:
                consumers[p].append(c)
    src: dict = {}
    for j in a_free:                 # equation order is topological
        s = set()
        for iv in alt_jaxpr.eqns[j - 1].invars:
            if isinstance(iv, Literal):
                continue
            p = a_prod.get(iv)
            if p in src:
                s |= src[p]
            elif iv in back:
                s.add(back[iv])
        src[j] = s
    snk: dict = {}
    for j in reversed(a_free):
        t = set()
        for c in consumers[j]:
            if c in snk:
                t |= snk[c]
            else:
                # a carry vertex with a counterpart: the link ends here
                t.update(back[ov] for ov in alt_jaxpr.eqns[c - 1].outvars
                         if ov in back)
        snk[j] = t
    return {
        "pairs": {j: frozenset((s, t) for s in src[j] for t in snk[j])
                  for j in a_free},
        "base": base,
        "sinks": sinks,
        "consumers": {j: tuple(c) for j, c in consumers.items()},
    }


def chain_applies(container) -> bool:
    # The chain rule serves the readout trace's factored attachment, which
    # only the diag container has; the other containers' contractions are
    # the dense rows' tensordots, and those run cheaper forward.
    return "diag" in str(container).split("+")


def contraction_chains(jaxpr, carry) -> set:
    """The carry-scope vertices in the chain of an attachment contraction.

    A ``dot_general`` inside the carry scope, its descendants inside the
    scope, and the ancestors inside the scope whose every consumer is in
    the chain. On RSNN_SHD's diag container that is the readout trace's
    chain up to the attached readout membrane; the weight's shared
    difference vertex, which the hidden chains read too, stays outside.
    """
    from jax._src.core import Literal
    scope = set(int(v) for v in carry)
    producer = {}
    for i, e in enumerate(jaxpr.eqns, 1):
        for ov in e.outvars:
            producer[ov] = i
    consumers: dict = {v: set() for v in scope}
    parents: dict = {v: set() for v in scope}
    for i, e in enumerate(jaxpr.eqns, 1):
        for iv in e.invars:
            if isinstance(iv, Literal):
                continue
            p = producer.get(iv)
            if p in scope:
                consumers[p].add(i)
                if i in scope:
                    parents[i].add(p)
    chain: set = set()
    for d in scope:
        if jaxpr.eqns[d - 1].primitive.name != "dot_general":
            continue
        stack = [d]
        while stack:
            v = stack.pop()
            if v in chain:
                continue
            chain.add(v)
            stack.extend(c for c in consumers[v] if c in scope)
    grew = True
    while grew:
        grew = False
        for v in list(chain):
            for p in parents[v]:
                if p not in chain and consumers[p] <= chain:
                    chain.add(p)
                    grew = True
    return chain


def transport_order(o_list, variant) -> list:
    """The elimination order, carried onto the variant's graph.

    The step body's vertices keep their relative order exactly, because that
    is the order the policy chose. They cut the order into SEGMENTS, the
    runs of carry vertices between two of them, and the carry block's
    vertices -- a different set on the two graphs -- go into the segments
    where every step-body vertex meets the same faces as on the policy's
    graph (owner ruling 2026-09-25, dsnn-rbjh):

      * a carry vertex with a counterpart (:func:`_counterparts`: an
        attached state, a weight difference, all of a block the container
        leaves alone) goes into the segment its counterpart went into, so a
        step-body vertex meets it live, or eliminated, as the policy's did;
      * one without goes into the segment in which the policy's order closed
        the earliest of the links through the rewritten inside of the block
        that it lies on (:func:`_carry_links`), so a step-body vertex reaches
        a weight through an eliminated attached state from the same segment
        on;
      * one on no such link goes no later than the carry vertices it feeds.

    Inside a segment the carry vertices run in the block's own order. When
    the policy eliminated its carry block in one run -- the reverse order,
    and every order that does not interleave the block with the step body --
    all of them go into that run, in that order, which is what the fraction
    rule before this one gave.

    WHY NOT A FRACTION. The carry vertices used to be spread by FRACTION:
    after the k-th of the policy's carry vertices, the same fraction of the
    variant's. On an order that interleaves, that put the attached state
    before the recurrent contraction on the diag program where the policy's
    graph kept it live (RSNN_SHD rtrl, the static Markowitz order and the
    forward order, job 68087): the recurrent face's state edge did not
    exist there, and e-prop's Diag on it landed on another face.
    """
    vmap = variant["vertex_map"]
    valid = variant["valid"]
    # A contraction's chain runs from the attached state back to its
    # operands, ahead of the rest of the block, which runs from the weights
    # to the states: the readout factor meets the loss before the trace,
    # and a hidden trace meets the step's coefficients once, at the state.
    carry = [v for v in variant["alt_carry"] if v in valid]
    chain = variant.get("chain")
    if chain is None:
        chain = (contraction_chains(variant["config"].jaxpr, carry)
                 if carry and chain_applies(variant["container"]) else set())
    alt_carry = (sorted((v for v in carry if v in chain), reverse=True)
                 + sorted(v for v in carry if v not in chain))
    rank = {v: r for r, v in enumerate(alt_carry)}
    seg_of: dict = {}
    n_body = 0
    for v in o_list:
        v = int(v)
        j = vmap.get(v)
        if j is None:
            seg_of[v] = n_body
        elif j in valid:
            n_body += 1
    runs = set(seg_of.values())
    if len(runs) <= 1:
        where = dict.fromkeys(carry, runs.pop() if runs else n_body)
    else:
        where = _carry_segments(variant, seg_of, carry)
    at: dict = {}
    for x in carry:
        at.setdefault(where[x], []).append(x)
    out: list = []
    s = 0
    for v in o_list:
        j = vmap.get(int(v))
        if j is None or j not in valid:
            continue
        out.extend(sorted(at.get(s, ()), key=rank.__getitem__))
        s += 1
        out.append(j)
    out.extend(sorted(at.get(s, ()), key=rank.__getitem__))
    # A VERTEX ELIMINABLE ON THE VARIANT ONLY. The eliminable set is a
    # property of the graph: on the rtrl graph the ``a`` output's add is a
    # pure output and not eliminable, while on the truncated (skip) variant
    # the same equation is a dead vertex and is. The policy never chose a
    # position for it, so it carries no decision and goes last.
    seen = set(out)
    out.extend(j for j in sorted(valid) if j not in seen)
    if sorted(out) != sorted(valid):
        raise ValueError(
            f"the transported order has {len(out)} vertices and the measured "
            f"graph has {len(valid)} eliminable ones. A transported order "
            f"must be a permutation of them; it is not, so the plan would be "
            f"measured on a partial elimination.")
    return out


def _carry_segments(variant, seg_of, carry) -> dict:
    """Measured carry vertex -> the segment of the step body it goes into.

    ``seg_of`` is the segment each carry vertex of the POLICY's order went
    into; :func:`transport_order` says what the three rules are.
    """
    inf = float("inf")
    links = variant["links"]
    origin = {j: i for i, j in variant["carry_map"].items()}
    # WHEN THE POLICY'S ORDER CLOSED EACH LINK: the segment of the last
    # vertex on the earliest-closed path from the source, -1 when the sink
    # reads the source itself.
    reach: dict = {}
    for s in {s for pairs in links["pairs"].values() for s, _t in pairs}:
        closed: dict = {}
        for i, operands in links["base"]:
            closed[i] = min((_closes(o, s, closed, seg_of) for o in operands),
                            default=inf)
        reach[s] = closed
    tau: dict = {}
    for pairs in links["pairs"].values():
        for s, t in pairs:
            if (s, t) in tau:
                continue
            vertex, operands = links["sinks"][t]
            c = min((_closes(o, s, reach[s], seg_of) for o in operands),
                    default=inf)
            # an attached state eliminated later than its link closes still
            # opens the link to the step body only when it goes
            tau[(s, t)] = c if c == inf else max(c, seg_of.get(vertex, -1))
    last = max(seg_of.values())
    out: dict = {}
    for j in sorted(carry, reverse=True):
        i = origin.get(j)
        if i is not None and i in seg_of:
            out[j] = seg_of[i]
            continue
        best = min((tau[p] for p in links["pairs"].get(j, ())), default=inf)
        if best == inf:
            best = min((out[c] for c in links["consumers"].get(j, ())
                        if c in out), default=inf)
        out[j] = last if best == inf else max(int(best), 0)
    return out


def _closes(o, s, closed, seg_of):
    # The segment from which operand ``o`` of a carry equation is joined to
    # the source ``s`` through eliminated vertices of the policy's order.
    if o is s:
        return -1
    if isinstance(o, int):
        return max(closed[o], seg_of.get(o, float("inf")))
    return float("inf")


def transport_wires(o_list, variant, rule_specs, face_specs, face_skips,
                    face_joins=None):
    """The plan's WIRE ARRAYS, carried onto the variant's graph.

    A FACE DECISION MOVES BY THE FACE'S IDENTITY, NOT BY ITS POSITION (owner
    ruling 2026-09-25, dsnn-rbjh). The wires are indexed by (elimination
    step, face position), and a position names a face only on the graph it
    was enumerated on: a face is the triple of its eliminated vertex, its
    predecessor and its successor, and which triples a step-body vertex shows
    depends on what is live around it, which the container changes. Measured,
    job 66101: a real policy plan asked for face ``(52, 85)`` of body vertex
    51, and the compact container's graph has no such pair. Job 68087:
    carried by position, e-prop's Diag on the recurrent face's state edge
    was applied to its V in-edge face under the static Markowitz order, and
    nothing raised.

    So each step-body vertex keeps its per-vertex rows at its new position,
    and each of its face decisions -- a face row that is not the end row, a
    Skip, and the join bit with them -- is read as the triple it was made on,
    on a replay of the POLICY's graph, carried to the triple of the
    counterparts (:func:`_counterparts`), and written at that triple's
    position on a replay of the MEASURED graph along the transported order.
    Both replays apply the rows the way ``env._face_transforms_for_order``
    does, so a Skip that removes a path removes it on both. A decision whose
    face has no counterpart RAISES with the face named, as does a decision on
    a vertex the measured graph does not eliminate; nothing falls back to a
    position.

    The variant's carry vertices keep ONE decision of the plan's carry faces:
    the Quant bit (owner ruling 2026-09-23). A Quant on the carried face is
    the narrow container AND the narrow contraction, so every face of every
    carry vertex gets the plan's own QUANT row on lhs and rhs -- the row is
    copied, dtype column and all -- whenever the plan put a Quant on any
    carried face. Diag and Reduce keep travelling in the value only: their
    rows on the carry vertices stay exact, because they are what chose the
    container and the container realizes them in the store. The caller then
    enumerates the faces on the VARIANT's own replay, which is what
    ``_face_transforms_for_order`` does on the policy's graph.

    Returns ``(order, rule_specs, face_specs, face_skips, face_joins)``, the
    last four as arrays of the same widths they came in with.
    """
    from alphagrad.approx.env import QUANT_SENTINEL
    from alphagrad.approx.unified_face_head import QUANT_SLOTS
    order = transport_order(o_list, variant)
    pos = {}
    for k, v in enumerate(order):
        pos.setdefault(int(v), k)
    vmap = variant["vertex_map"]
    valid = variant["valid"]

    rs = np.asarray(rule_specs)
    fs = np.asarray(face_specs)
    sk = np.asarray(face_skips)
    jn = None if face_joins is None else np.asarray(face_joins)
    T = len(order)
    rs2 = np.full((T,) + rs.shape[1:], -1, dtype=np.int32)
    rs2[:, :, 2] = 0
    fs2 = np.full((T,) + fs.shape[1:], -1, dtype=np.int32)
    sk2 = np.zeros((T,) + sk.shape[1:], dtype=np.int32)
    jn2 = None if jn is None else np.zeros((T,) + jn.shape[1:], dtype=np.int32)

    quant_row = None
    steps: dict = {}
    for k, v in enumerate(o_list):
        j = vmap.get(int(v))
        if j is None:
            # A CARRY VERTEX OF THE POLICY'S GRAPH. Its Diag / Reduce / Skip
            # chose the container and travel in the value; its Quant is the
            # narrow contraction too, and the plan's own row carries it.
            if k < fs.shape[0]:
                for f in range(fs.shape[1]):
                    if k < sk.shape[0] and int(sk[k][f]) == 1:
                        continue
                    lhs = fs[k][f][QUANT_SLOTS[0]]
                    if int(lhs[0]) == QUANT_SENTINEL and quant_row is None:
                        quant_row = tuple(int(x) for x in lhs)
            continue
        k2 = pos.get(j)
        if k2 is None:
            # A STEP-BODY VERTEX THE VARIANT DOES NOT ELIMINATE. The
            # eliminable set is a property of the GRAPH, not of the equation
            # alone: `_build_graph` decides it from the arguments too, and
            # the containers carry different ones. An all-exact row has
            # nothing to carry and is dropped; a row that carries a decision
            # would have it vanish, so it raises.
            # Every row but the end row (-1) is a decision; the Quant and
            # Reduce sentinels lie below it, so a maximum does not see them.
            if (bool(np.any(fs[k][..., 0] != -1)) or bool(np.any(sk[k] != 0))
                    or bool(np.any(rs[k][:, 0] != -1))):
                raise ValueError(
                    f"vertex {int(v)} of the policy's graph carries an "
                    f"approximation and its image {j} is not eliminable on "
                    f"the measured graph, so the decision would be dropped "
                    f"in silence.")
            continue
        rs2[k2] = rs[k]
        steps[k] = k2
    if quant_row is not None:
        for j in variant["alt_carry"]:
            if j not in valid:
                continue
            k2 = pos[int(j)]
            for s in QUANT_SLOTS:
                fs2[k2, :, s] = quant_row
    _move_face_decisions(o_list, order, variant, steps, (rs, fs, sk, jn),
                         (rs2, fs2, sk2, jn2))
    return order, rs2, fs2, sk2, jn2


def _decided(rows, skips) -> np.ndarray:
    """``(F,)`` bool: the faces of one step that carry a decision.

    Every row but the end row (-1) is a decision; the Quant and Reduce
    sentinels lie below it. A join bit alone is not one: it only says how a
    face that carries an approximation joins.
    """
    rows = np.asarray(rows)
    return ((np.asarray(skips).reshape(-1) == 1)
            | np.any(rows[..., 0] != -1, axis=-1))


def _move_face_decisions(o_list, order, variant, steps, wires, moved):
    """Write every face decision of the step body at its face on the variant.

    ``steps`` is policy step -> measured step for the step-body vertices both
    graphs eliminate; ``wires`` the policy's ``(rule_specs, face_specs,
    face_skips, face_joins)`` and ``moved`` the measured arrays, written in
    place. See :func:`transport_wires`.
    """
    rs, fs, sk, jn = wires
    rs2, fs2, sk2, jn2 = moved
    decided = [k for k in sorted(steps) if bool(np.any(_decided(fs[k], sk[k])))]
    if not decided:
        return
    var_map = variant["var_map"]
    base = variant["base"]
    base_jx = base["config"].jaxpr
    program = f"the {variant['container']} program"
    wanted: dict = {}

    def read(k, v, specs):
        if k in steps:
            got = {}
            for f in np.flatnonzero(_decided(fs[k], sk[k])):
                if f >= len(specs):
                    # past the live faces: the policy's own graph never
                    # reads this row either
                    continue
                spec = specs[f]
                triple = tuple(var_map.get(x) for x in
                               (spec.central, spec.in_edge, spec.out_edge))
                if None in triple:
                    lost = [n for n, x in zip(
                        ("vertex", "predecessor", "successor"), triple)
                        if x is None]
                    raise ValueError(
                        f"vertex {v} of the policy's graph decides face {f}, "
                        f"{_face_name(base_jx, v, spec)}, and its "
                        f"{' and '.join(lost)} has no counterpart on "
                        f"{program}: it is part of what that container "
                        f"rewrites or drops. A face decision moves by the "
                        f"face's identity, never by its position (owner "
                        f"ruling 2026-09-25, dsnn-rbjh), so this plan cannot "
                        f"be measured there.")
                got[triple] = (int(f), v, spec)
            wanted[k] = got
        return fs[k], sk[k], (None if jn is None else jn[k])

    _walk_faces(base["config"], base["consts"], base["args"], o_list, rs,
                decided[-1] + 1, read)
    at_step = {steps[k]: k for k in decided}
    width = fs2.shape[1]

    def place(k2, j, specs):
        k = at_step.get(k2)
        if k is not None:
            want = dict(wanted[k])
            for f2, spec in enumerate(specs):
                hit = want.pop((spec.central, spec.in_edge, spec.out_edge),
                               None)
                if hit is None or f2 >= width:
                    # past the wire: the face dict below refuses the vertex
                    continue
                f = hit[0]
                fs2[k2, f2] = fs[k, f]
                sk2[k2, f2] = sk[k, f]
                if jn2 is not None:
                    jn2[k2, f2] = jn[k, f]
            if want:
                f, v, spec = next(iter(want.values()))
                raise ValueError(
                    f"vertex {v} of the policy's graph decides face {f}, "
                    f"{_face_name(base_jx, v, spec)}, and vertex {j} of "
                    f"{program} has no such face when the transported order "
                    f"eliminates it (its faces there: "
                    f"{[_face_name(variant['config'].jaxpr, j, s) for s in specs]}"
                    f"). A face decision moves by the face's identity, never "
                    f"by its position (owner ruling 2026-09-25, dsnn-rbjh), "
                    f"so this plan cannot be measured there.")
        return fs2[k2], sk2[k2], (None if jn2 is None else jn2[k2])

    _walk_faces(variant["config"], variant["consts"], variant["args"], order,
                rs2, max(at_step) + 1, place)


def _walk_faces(config, consts, args, order, rule_rows, upto, rows_at):
    """Eliminate ``order[:upto]`` on ``config``'s graph as the measurement's
    face enumeration does, handing each step's faces to ``rows_at``.

    ``rows_at(k, v, specs)`` gets graphax's ``FaceSpec`` list of vertex ``v``
    on the live graph and returns the wire rows ``(face_row, face_skip,
    face_join)`` step ``k`` applies. They are applied as
    ``env._face_transforms_for_order`` applies them -- through
    ``env._face_dict_for_vertex`` on the same keys, with the per-vertex rules
    wrapped the same way -- so every later step shows the faces that replay
    shows: a Skip removes its path here too.
    """
    from graphax import face_specs_of
    from graphax.incremental import IncrementalJaxpr
    from alphagrad.approx.common.masks import make_live_masked_hook
    from alphagrad.approx.env import (_face_dict_for_vertex,
                                      decode_vertex_rule_specs)
    ij = IncrementalJaxpr(config.jaxpr, tuple(config.argnums), list(consts),
                          list(args), track_faces=False)
    for k in range(upto):
        v = int(order[k])
        specs = face_specs_of(ij.graph, ij.tgraph, v, config.jaxpr)
        row, skip, join = rows_at(k, v, specs)
        per_face = _face_dict_for_vertex(config, ij, v, row, skip,
                                         face_join=join,
                                         keys=[s.key for s in specs])
        rules = decode_vertex_rule_specs(config.jaxpr, v,
                                         np.asarray(rule_rows[k]).tolist())
        ij.eliminate(v, (make_live_masked_hook(tuple(rules)),)
                     if rules else (), per_face or None)


def _face_name(jaxpr, v, spec) -> str:
    """A face as ``predecessor -> vertex -> successor``, by vertex id and
    primitive (an input by its argument slot), and its key."""
    prod = _producers(jaxpr)

    def name(x):
        i = prod.get(x)
        if i is not None:
            return f"{i} ({jaxpr.eqns[i - 1].primitive.name})"
        if x in jaxpr.invars:
            return f"input {jaxpr.invars.index(x)}"
        return str(x)
    return (f"the face {name(spec.in_edge)} -> {v} -> {name(spec.out_edge)} "
            f"(key {tuple(spec.key)})")


def carry_at_rest_bytes(variant_or_entry, rule: str) -> int:
    """The bytes the carried value occupies in its container: the given
    blocks of the measured program, the way they arrive at the step.

    Under ``rtrl`` the given tuple is the five stacked tensors
    ``rsnn_shd.carry_from_executable`` builds, nothing else; under ``bptt`` the
    five adjoints.
    """
    from alphagrad.approx.common.rsnn_shd import RSNN_HEAD_SLOTS
    args = tuple(variant_or_entry["args"])
    given = args[RSNN_HEAD_SLOTS:]
    return int(sum(int(np.asarray(x).nbytes) for x in given))
