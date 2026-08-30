import sys
P = "/Users/assmuth/dsnn/alphagrad/tests/rule_replay_test.py"
OLD = '''    from graphax import jacve
    from graphax.core import _build_graph
    from alphagrad.approx.common.masks import make_live_masked_hook

    fn = lambda x, y: jnp.tanh(jnp.sin(x) * y) + jnp.exp(jnp.sin(x) * y)
    args = (jnp.ones((4, 4)) * 0.5, jnp.ones((4, 4)) * 0.4)
    cj = jax.make_jaxpr(fn)(*args)
    _, _, _, vo = _build_graph(cj.jaxpr, args, list(cj.literals), (0, 1))
    valid = [i for i, eqn in enumerate(cj.jaxpr.eqns, 1)
             if eqn.outvars[0] not in cj.jaxpr.outvars or i in vo]
    order = list(reversed(valid))

    def _flat(t):
        return np.concatenate(
            [np.asarray(x, np.float64).ravel()
             for x in jax.tree_util.tree_leaves(t)])

    ref = _flat(jax.jit(jacve(fn, order, argnums=(0, 1)))(*args))
    moved = 0
    for idx, v in enumerate(order[:-1]):          # NON-final positions only
        rules = decode_vertex_rule_specs(
            cj.jaxpr, int(v), _rows([COMPRESS_SENTINEL, 0, 0]))
        if not rules:
            continue
        out = _flat(jax.jit(jacve(                # must not raise
            fn, order, argnums=(0, 1),
            transforms=[(int(v),
                         (make_live_masked_hook(tuple(rules)),))]))(*args))
        cos = float(out @ ref / (np.linalg.norm(out) * np.linalg.norm(ref)))
        if cos < 0.999:
            moved += 1
    assert moved > 0, (
        "no NON-FINAL COMPRESS changed the measured Jacobian -- the position "
        "gate is back, or the rules are being dropped somewhere downstream")'''
NEW = '''    from graphax import jacve
    from graphax.core import _build_graph
    from alphagrad.approx.common.masks import make_live_masked_hook
    from alphagrad.approx.common.examples import get_args, get_fn

    # A REAL target (the small 4-8-4 MLP), not the 2-op toy above: on the toy
    # every axis-0 COMPRESS happens to land on an edge that is already uniform
    # along that axis, so the reduction is the identity and the test could not
    # tell a working COMPRESS from a dropped one.
    fn = get_fn("NeuralNetwork")
    args = get_args("NeuralNetwork", jax.random.PRNGKey(0))
    argnums = tuple(range(len(args)))
    cj = jax.make_jaxpr(fn)(*args)
    _, _, _, vo = _build_graph(cj.jaxpr, args, list(cj.literals), argnums)
    valid = [i for i, eqn in enumerate(cj.jaxpr.eqns, 1)
             if eqn.outvars[0] not in cj.jaxpr.outvars or i in vo]
    order = list(reversed(valid))

    def _flat(t):
        return np.concatenate(
            [np.asarray(x, np.float64).ravel()
             for x in jax.tree_util.tree_leaves(t)])

    ref = _flat(jax.jit(jacve(fn, order, argnums=argnums))(*args))
    tried = moved = 0
    for v in order[:-1]:                          # NON-final positions only
        rules = ()
        for axis in range(4):
            rules = decode_vertex_rule_specs(
                cj.jaxpr, int(v), _rows([COMPRESS_SENTINEL, axis, 0]))
            if rules:
                break
        if not rules:
            continue
        tried += 1
        out = _flat(jax.jit(jacve(                # must not raise
            fn, order, argnums=argnums,
            transforms=[(int(v),
                         (make_live_masked_hook(tuple(rules)),))]))(*args))
        cos = float(out @ ref / (np.linalg.norm(out) * np.linalg.norm(ref)))
        if cos < 0.999:
            moved += 1
    assert tried > 5, f"only {tried} non-final vertices took a COMPRESS at all"
    assert moved > 0, (
        "no NON-FINAL COMPRESS changed the measured Jacobian -- the position "
        "gate is back, or the rules are being dropped somewhere downstream")'''
s = open(P).read()
if NEW in s:
    print("already"); sys.exit(0)
assert OLD in s, "anchor missing"
open(P, "w").write(s.replace(OLD, NEW, 1))
print("ok")
