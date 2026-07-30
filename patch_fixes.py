#!/usr/bin/env python3
"""Three fixes: (1) cossim formula (cos(v,v)=1 for tiny v — floor the PRODUCT of
norms, not each vector by sqrt(1e-7)); (2) GA empty-population IndexError guard;
(3) _measure1 broaden crash-catch + clear_caches to survive graphax GPU errors."""
import pathlib, sys
p = pathlib.Path("src/alphagrad/approx/autoscheduler_loop.py")
env = pathlib.Path("src/alphagrad/approx/env.py")

# --- (1) cossim in env.py ---
es = env.read_text()
old_cs = (
    'def cossim(target, preds):\n'
    '    target = target / jnp.maximum(\n'
    '        jnp.linalg.norm(target, keepdims=True), jnp.sqrt(1e-7)\n'
    '    )\n'
    '    preds = preds / jnp.maximum(jnp.linalg.norm(preds, keepdims=True), jnp.sqrt(1e-7))\n'
    '    return jnp.sum(target * preds)'
)
new_cs = (
    'def cossim(target, preds):\n'
    '    # Floor the PRODUCT of norms at a tiny eps (not each vector by sqrt(1e-7)).\n'
    '    # The old per-vector floor made cos(v,v) = ||v||^2/1e-7 < 1 for a tiny (but\n'
    '    # IDENTICAL) gradient -> exact/pure-order solutions wrongly scored cos~0.02.\n'
    '    denom = jnp.maximum(jnp.linalg.norm(target) * jnp.linalg.norm(preds), 1e-30)\n'
    '    return jnp.sum(target * preds) / denom'
)
if 'jnp.linalg.norm(target) * jnp.linalg.norm(preds)' in es:
    print("cossim ALREADY fixed")
else:
    assert old_cs in es, "cossim anchor"
    es = es.replace(old_cs, new_cs, 1); env.write_text(es); print("(1) cossim fixed")

# --- (2) GA empty-pop guard + (3) measure catch in autoscheduler_loop.py ---
s = p.read_text()

# (2) after building pop from the initial GA measure, re-seed until >=2 valid
old_ga = (
    '        mr = _mf_measure(pop0)\n'
    '        pop = [c for c, r in mr if r is not None]; praws = [r for c, r in mr if r is not None]\n'
)
new_ga = (
    '        mr = _mf_measure(pop0)\n'
    '        pop = [c for c, r in mr if r is not None]; praws = [r for c, r in mr if r is not None]\n'
    '        _ga_re = 0\n'
    '        while len(pop) < 2 and _n[0] < _budget and _ga_re < 20:\n'
    '            _mr2 = _mf_measure([_mf_rand_cand(_orng) for _ in range(P)])\n'
    '            pop += [c for c, r in _mr2 if r is not None]; praws += [r for c, r in _mr2 if r is not None]\n'
    '            _ga_re += 1\n'
)
if '_ga_re' in s:
    print("GA guard ALREADY present")
else:
    assert old_ga in s, "GA anchor"; s = s.replace(old_ga, new_ga, 1); print("(2) GA empty-pop guard added")

# also make the whole GA loop skip if pop still empty (defensive)
old_gaw = '        while _n[0] < _budget:\n            kids = []\n            for _ in range(min(P, max(_budget - _n[0], 1))):'
new_gaw = '        while _n[0] < _budget and len(pop) >= 1:\n            kids = []\n            for _ in range(min(P, max(_budget - _n[0], 1))):'
if old_gaw in s and 'len(pop) >= 1' not in s.split('elif _opt == "ga"')[1][:800]:
    s = s.replace(old_gaw, new_gaw, 1); print("    GA loop empty-guard added")

# (3) broaden _measure1 crash-catch + clear_caches
old_m = (
    '            _callback(env.config, env.args, env.consts, jnp.asarray(order_arr),\n'
    '                      jnp.asarray(specs), len(order_arr), *ev, raw_sink=rs)\n'
    '        except Exception:\n'
    '            return None'
)
new_m = (
    '            _callback(env.config, env.args, env.consts, jnp.asarray(order_arr),\n'
    '                      jnp.asarray(specs), len(order_arr), *ev, raw_sink=rs)\n'
    '        except BaseException:\n'
    '            try:\n'
    '                jax.clear_caches(); import gc as _gcm; _gcm.collect()\n'
    '            except Exception:\n'
    '                pass\n'
    '            return None'
)
if 'except BaseException:' in s and old_m not in s:
    print("measure catch ALREADY broadened")
else:
    assert old_m in s, "measure catch anchor"; s = s.replace(old_m, new_m, 1); print("(3) _measure1 catch broadened + clear_caches")

p.write_text(s)
print("done")
