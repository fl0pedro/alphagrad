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

#: What :func:`register` was told, or ``None`` when no recurrent target is in
#: this process. One process serves one target, which is why this is module
#: state and not a field of every call.
_SPEC: dict | None = None

#: The base (policy-side) program: the config, its args and its consts.
_BASE: dict | None = None

#: container name -> the built variant. A variant is a dict with the keys
#: ``config``, ``args``, ``consts``, ``vertex_map``, ``var_map``,
#: ``carry_vertices`` and ``valid``.
_VARIANTS: dict = {}

#: (container, base-draw digest, count) -> the eval samples that variant
#: draws. Bounded, and cleared when it grows.
_EVAL_SAMPLES: dict = {}


def reset() -> None:
    """Forget everything. For tests, and for a process that rebuilds."""
    global _SPEC, _BASE
    _SPEC = None
    _BASE = None
    _VARIANTS.clear()
    _EVAL_SAMPLES.clear()


def armed() -> bool:
    """Is a plan's container a question in this process at all?"""
    return _SPEC is not None


def register(args_like, key, example, temporal_rule, config, args, consts,
             *, dataset=None, dataset_size=-1, step_position=None):
    """Record what the base target was built from.

    Called at every site that builds a recurrent SHD env. On any other target
    it is a no-op, so the call can sit unconditionally beside the build.

    ``args_like`` is the argparse namespace (or the actor's args dict) that
    ``grad_target_setup`` reads; ``key`` is the SAME key ``get_args`` was
    given, because the variant has to draw the same recording and the same
    weights or its reference weights stop matching the run's own.
    """
    global _SPEC, _BASE
    from alphagrad.approx.common.rsnn_shd import GIVEN_EDGE_RULES, is_rsnn
    _VARIANTS.clear()
    _EVAL_SAMPLES.clear()
    if not is_rsnn(example) or str(temporal_rule) not in GIVEN_EDGE_RULES:
        # No given temporal edge means no carried value, so there is no
        # container to choose and nothing here has anything to do.
        _SPEC = None
        _BASE = None
        return
    _SPEC = {
        "args_like": args_like,
        "key": key,
        "example": str(example),
        "rule": str(temporal_rule),
        "dataset": dataset,
        "dataset_size": dataset_size,
        "step_position": step_position,
    }
    _BASE = {"config": config, "args": tuple(args), "consts": consts}


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
    if _SPEC is None:
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
        if eqn.outvars[0] not in jaxpr.outvars or i in vo_vertices:
            out.append(i)
    return tuple(out)


def _build_variant(container: str) -> dict:
    """Build the program ``container`` implies, and its map from the base."""
    from alphagrad.approx.common.examples import (data_gen, get_args, get_fn,
                                                  grad_target_setup)
    from alphagrad.approx.common.rsnn_shd import (EXACT_CONTAINER,
                                                  SKIP_CONTAINER,
                                                  target_example)
    spec = _SPEC
    base_cfg = _BASE["config"]
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
    if container == SKIP_CONTAINER:
        # THE QUALITY REFERENCE IS THE ARM'S RULE, NOT THIS PROGRAM'S. A
        # truncated program's own exact gradient is the truncated gradient, so
        # scoring against it would read 1.0 for a plan that threw the whole
        # prefix away. The reference is the base program -- the arm's rule
        # with the exact carry -- and the generator says so.
        base_gen = base_cfg.data_gen
        gen.reference_oracle = {
            "draw": base_gen,
            "target": base_cfg.target_fun,
            "argnums": tuple(base_cfg.argnums),
            "args": tuple(_BASE["args"]),
            "slots": tuple(getattr(base_gen, "data_slots",
                                   range(len(_BASE["args"])))),
        }
    cfg = base_cfg._replace(jaxpr=cj.jaxpr, argnums=tuple(argnums),
                            target_fun=fn, data_gen=gen)
    consts = tuple(cj.literals)
    args = tuple(xs)
    vmap, var_map, alt_carry = _alignment(base_cfg.jaxpr, cj.jaxpr)
    return {
        "container": container,
        "config": cfg,
        "args": args,
        "consts": consts,
        "vertex_map": vmap,
        "var_map": var_map,
        "alt_carry": alt_carry,
        "valid": set(valid_vertices(cj.jaxpr, args, consts, tuple(argnums))),
    }


def measurement_env(container: str) -> dict | None:
    """The built variant for ``container``, or ``None`` when none is needed."""
    from alphagrad.approx.common.rsnn_shd import EXACT_CONTAINER
    if _SPEC is None:
        return None
    if container == EXACT_CONTAINER:
        # The base program IS the exact container: the policy's graph carries
        # the dense edge by construction, so there is nothing to build.
        return None
    v = _VARIANTS.get(container)
    if v is None:
        v = _build_variant(container)
        _VARIANTS[container] = v
    return v


def eval_samples_for(container: str, eval_samples):
    """This episode's eval samples, redrawn in ``container``.

    Keyed by a digest of the base draw (:func:`_eval_key`), so every process
    that measures this plan builds the same ones, and cached per (container,
    digest) so an episode draws them once however many plans it measures.
    """
    if not eval_samples:
        return None
    var = measurement_env(container)
    if var is None:
        return None
    n = int(len(eval_samples[0]))
    tag = (container, _eval_tag(eval_samples), n)
    hit = _EVAL_SAMPLES.get(tag)
    if hit is not None:
        return hit
    from alphagrad.approx.common.eval_samples import generate_eval_samples

    class _Shim:
        config = var["config"]
        args = var["args"]

    out = generate_eval_samples(_Shim, _eval_key(eval_samples), n)
    if len(_EVAL_SAMPLES) > 64:
        _EVAL_SAMPLES.clear()
    _EVAL_SAMPLES[tag] = out
    return out


# ---------------------------------------------------------------------------
# 3. THE ALIGNMENT AND THE TRANSPORT
# ---------------------------------------------------------------------------
def _eqn_signature(eqn):
    outs = tuple((tuple(v.aval.shape), str(v.aval.dtype))
                 for v in eqn.outvars)
    return (eqn.primitive.name, outs)


def _alignment(base_jaxpr, alt_jaxpr):
    """``(vertex_map, var_map, alt_carry_vertices)``, or raise.

    The two programs differ only inside the carry block. Everything outside
    it is aligned by POSITION and CHECKED equation by equation, so a change to
    the model that moved a step-body equation would raise here instead of
    silently landing a plan's faces on other faces.

    ``var_map`` carries the FACE KEYS across. A face key is a pair of stable
    var indices (``graphax.core._stable_var_index``), so the map is built from
    the operands and the outputs of the aligned equations -- which also maps
    the five values the carry block hands to the step body, whatever produced
    them: an ``add`` of the attachment in one program, a graph input in the
    truncated one.
    """
    try:                                    # jax >= 0.4.31
        from jax.extend.core import Literal
    except ImportError:                     # older / internal layout
        from jax._src.core import Literal
    from graphax.core import _stable_var_index

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
    bidx = _stable_var_index(base_jaxpr)
    aidx = _stable_var_index(alt_jaxpr)
    var_map: dict = {}

    def _pair(bv, av):
        if isinstance(bv, Literal) or isinstance(av, Literal):
            return
        b, a = bidx.get(bv), aidx.get(av)
        if b is None or a is None:
            return
        seen = var_map.setdefault(b, a)
        if seen != a:
            raise ValueError(
                f"variable {b} of the policy's graph maps to both {seen} and "
                f"{a} of the measured graph. The step-body alignment is not "
                f"a function and the plan cannot be carried across.")

    for i, j in zip(b_body, a_body):
        be, ae = base_jaxpr.eqns[i], alt_jaxpr.eqns[j]
        for bv, av in zip(be.invars, ae.invars):
            _pair(bv, av)
        for bv, av in zip(be.outvars, ae.outvars):
            _pair(bv, av)

    alt_carry = tuple(j + 1 for j, m in enumerate(a_mask) if m)
    return vertex_map, var_map, alt_carry


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
    alt_carry = [v for v in variant["alt_carry"] if v in valid]
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
    if sorted(out) != sorted(valid):
        raise ValueError(
            f"the transported order has {len(out)} vertices and the measured "
            f"graph has {len(valid)} eliminable ones. A transported order "
            f"must be a permutation of them; it is not, so the plan would be "
            f"measured on a partial elimination.")
    return out


def transport_faces(ft_by_vertex, variant):
    """The per-face transforms, carried onto the variant's graph.

    The carry block's own faces are NOT carried: their approximation is what
    chose the container and it is realized in the value and in the store. What
    comes across is the step body's faces, under the face keys the variant's
    own stable var index gives them.
    """
    if not ft_by_vertex:
        return None
    vmap = variant["vertex_map"]
    var_map = variant["var_map"]
    out: dict = {}
    for v, per_face in ft_by_vertex.items():
        j = vmap.get(int(v))
        if j is None:
            continue
        moved = {}
        for key, entry in per_face.items():
            a, b = int(key[0]), int(key[1])
            a2, b2 = var_map.get(a), var_map.get(b)
            if a2 is None or b2 is None:
                raise ValueError(
                    f"face {key} of step-body vertex {v} names a variable "
                    f"the two graphs do not share. Every operand of an "
                    f"aligned step-body equation is mapped, so this is a "
                    f"face the alignment does not describe and the plan "
                    f"cannot be carried across.")
            moved[(a2, b2)] = entry
        if moved:
            out[j] = moved
    return out or None


def transport_transforms(transforms, variant):
    """The per-vertex transform list, carried onto the variant's graph."""
    if not transforms:
        return []
    vmap = variant["vertex_map"]
    out = []
    for v, hooks in transforms:
        j = vmap.get(int(v))
        if j is not None:
            out.append((j, hooks))
    return out
