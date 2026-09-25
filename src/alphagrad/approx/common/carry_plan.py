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
4. :func:`transport` carries the plan from the policy's graph onto that
   program: the step body's vertices, faces and order come across unchanged,
   and the carry block's vertices are replaced by the ones the container's own
   attachment has, accumulated exactly.

THE TRANSPORT IS VERIFIED, NOT ASSUMED. The two programs differ ONLY inside
``graphax.examples.neuromorphic.snn_carry_scope``; everything outside it is
the same step body, equation for equation. :func:`_alignment` asserts that --
same count, same primitive, same output shape and dtype, in order -- and
raises if it ever stops being true, rather than measuring a plan that has
quietly landed on different faces.
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
#:                 ``alt_carry`` and ``valid``
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
    consts = tuple(cj.literals)
    args = tuple(xs)
    vmap, alt_carry = _alignment(base_cfg.jaxpr, cj.jaxpr)
    valid = set(valid_vertices(cj.jaxpr, args, consts, tuple(argnums)))
    return {
        "container": container,
        "config": cfg,
        "args": args,
        "consts": consts,
        "vertex_map": vmap,
        "alt_carry": alt_carry,
        "valid": valid,
        "chain": contraction_chains(cj.jaxpr,
                                    [v for v in alt_carry if v in valid]),
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
    is the order the policy chose. The carry block's vertices are a different
    set on the two graphs, so they are spread through the body at the SAME
    relative positions the policy's carry vertices held: after the k-th of the
    policy's carry vertices, the same FRACTION of the variant's carry vertices
    has been eliminated. That keeps the one thing about them the policy
    decided -- how early the carried value is contracted against the step --
    and invents nothing else.
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
                 if carry else set())
    alt_carry = (sorted((v for v in carry if v in chain), reverse=True)
                 + sorted(v for v in carry if v not in chain))
    base_carry = [int(v) for v in o_list if int(v) not in vmap]
    n_a, n_b = len(alt_carry), len(base_carry)
    out: list = []
    seen_b = 0
    emitted = 0
    for v in o_list:
        v = int(v)
        j = vmap.get(v)
        if j is None:
            seen_b += 1
            want = n_a if n_b == 0 else int(round(seen_b * n_a / n_b))
            while emitted < want:
                out.append(alt_carry[emitted])
                emitted += 1
            continue
        if j in valid:
            out.append(j)
    while emitted < n_a:
        out.append(alt_carry[emitted])
        emitted += 1
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


def transport_wires(o_list, variant, rule_specs, face_specs, face_skips,
                    face_joins=None):
    """The plan's WIRE ARRAYS, carried onto the variant's graph.

    A PLAN IS INDEXED BY (ELIMINATION STEP, FACE POSITION), NOT BY FACE KEY.
    That is what the policy emits and it is the only thing that survives the
    move: a face KEY is ``(in_edge, out_edge)`` under the live graph's stable
    var index, and every elimination REWIRES the graph, so the keys a body
    vertex shows after the carry block has been eliminated are the keys THAT
    carry block left behind. Two containers leave different ones. Measured,
    job 66101: a real policy plan asked for face ``(52, 85)`` of body vertex
    51, and the compact container's graph has no such pair.

    So the wires travel by POSITION. Each step-body vertex keeps its own rows
    at its new position in the transported order. The variant's carry
    vertices keep ONE decision of the plan's carry faces: the Quant bit
    (owner ruling 2026-09-23). A Quant on the carried face is the narrow
    container AND the narrow contraction, so every face of every carry vertex
    gets the plan's own QUANT row on lhs and rhs -- the row is copied, dtype
    column and all -- whenever the plan put a Quant on any carried face. Diag
    and Reduce keep travelling in the value only: their rows on the carry
    vertices stay exact, because they are what chose the container and the
    container realizes them in the store. The caller then enumerates the
    faces on the VARIANT's own replay, which is what
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
        fs2[k2] = fs[k]
        sk2[k2] = sk[k]
        if jn2 is not None:
            jn2[k2] = jn[k]
    if quant_row is not None:
        for j in variant["alt_carry"]:
            if j not in valid:
                continue
            k2 = pos[int(j)]
            for s in QUANT_SLOTS:
                fs2[k2, :, s] = quant_row
    return order, rs2, fs2, sk2, jn2


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
