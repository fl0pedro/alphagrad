"""perf2: ADD-ONLY instrumentation for the face-enum prefix cache.

Counters + one host timer. No behaviour change: nothing here is read by the
cache logic, and every added statement is a counter increment or a
perf_counter delta into the existing _PROF sink.
"""
import io, re, sys

ENV = "src/alphagrad/approx/env.py"
PPO = "src/alphagrad/approx/ppo.py"

def sub1(s, old, new, tag):
    assert s.count(old) == 1, f"{tag}: {s.count(old)} matches"
    return s.replace(old, new)

src = io.open(ENV).read()

src = sub1(src,
    '_FACE_ENUM_STATS = {"ext": 0, "cold": 0}',
    '_FACE_ENUM_STATS = {"ext": 0, "cold": 0, "compress": 0, "elims": 0,\n'
    '                    "calls": 0, "build": 0}',
    "stats-dict")

# time the _sigs construction (the cache KEY), which is O(prefix x MAX_FACES)
src = sub1(src,
    "        _sigs = tuple(\n"
    "            (int(o_list[k]),",
    "        _t_key = time.perf_counter()\n"
    "        _sigs = tuple(\n"
    "            (int(o_list[k]),",
    "sigs-start")

src = sub1(src,
    "            for k in range(len(o_list)))\n"
    "        _base = (id(config.jaxpr), tuple(config.argnums))",
    "            for k in range(len(o_list)))\n"
    "        _prof_add(\"cb.face_enum_key\", time.perf_counter() - _t_key)\n"
    "        _FACE_ENUM_STATS[\"calls\"] += 1\n"
    "        _base = (id(config.jaxpr), tuple(config.argnums))",
    "sigs-end")

src = sub1(src,
    "        if not _last_compress:\n"
    "            _cache_key = _base + (_sigs,)",
    "        if _last_compress:\n"
    "            _FACE_ENUM_STATS[\"compress\"] += 1\n"
    "        if not _last_compress:\n"
    "            _cache_key = _base + (_sigs,)",
    "compress-count")

src = sub1(src,
    "    if ij is None:\n"
    "        ij = IncrementalJaxpr(config.jaxpr, tuple(config.argnums),\n"
    "                              list(consts), list(args), track_faces=False)\n"
    "    last = len(o_list) - 1",
    "    if ij is None:\n"
    "        _FACE_ENUM_STATS[\"build\"] += 1\n"
    "        ij = IncrementalJaxpr(config.jaxpr, tuple(config.argnums),\n"
    "                              list(consts), list(args), track_faces=False)\n"
    "    _FACE_ENUM_STATS[\"elims\"] += len(o_list) - _start\n"
    "    last = len(o_list) - 1",
    "elims-count")

# split the caller's dense .tolist() out of the call so it gets its own phase
src = sub1(src,
    "    if _have_face_actions and not _unified_fe:\n"
    "        ft_by_vertex = _face_transforms_for_order(\n"
    "            config, consts, args, o_list, specs_list,\n"
    "            _faces_np.tolist(), _skips_np.tolist(),\n"
    "            honor_last_compress=_honor_mid_compress,\n"
    "        )\n"
    "    _pf(\"cb.face_enum\")",
    "    if _have_face_actions and not _unified_fe:\n"
    "        _fr_list = _faces_np.tolist()\n"
    "        _fs_list = _skips_np.tolist()\n"
    "        _pf(\"cb.face_tolist\")\n"
    "        ft_by_vertex = _face_transforms_for_order(\n"
    "            config, consts, args, o_list, specs_list,\n"
    "            _fr_list, _fs_list,\n"
    "            honor_last_compress=_honor_mid_compress,\n"
    "        )\n"
    "    _pf(\"cb.face_enum\")",
    "tolist-phase")

io.open(ENV, "w").write(src)

p = io.open(PPO).read()
p = sub1(p,
    '                            f"  face_enum(ext/cold)={_fs[\'ext\']}/"\n'
    '                            f"{_fs[\'cold\']}")\n'
    "                        _ss.update(hit=0, ext=0, cold=0, nostore=0)\n"
    "                        _fs.update(ext=0, cold=0)",
    '                            f"  face_enum(ext/cold/compress)="\n'
    '                            f"{_fs[\'ext\']}/{_fs[\'cold\']}/"\n'
    '                            f"{_fs[\'compress\']}"\n'
    '                            f"  face_enum(calls/elims/build)="\n'
    '                            f"{_fs[\'calls\']}/{_fs[\'elims\']}/"\n'
    '                            f"{_fs[\'build\']}")\n'
    "                        _ss.update(hit=0, ext=0, cold=0, nostore=0)\n"
    "                        _fs.update(ext=0, cold=0, compress=0, elims=0,\n"
    "                                   calls=0, build=0)",
    "ppo-print")
io.open(PPO, "w").write(p)
print("perf2_instr: OK")
