"""perf2 FIX: make the face-enum prefix cache incremental IN PRACTICE.

Two changes, both to the CACHE KEY / the wire marshalling around it -- the
replay itself was already incremental:

 1. `_callback` stops materialising the dense face wires as Python lists.
 2. `_face_transforms_for_order` keys on `_face_wire_keys` (the sparse,
    injective per-vertex encoding the stream cache already uses) instead of
    re-`np.asarray(...).tobytes()`-ing every vertex of the prefix on every
    step.
"""
import io

ENV = "src/alphagrad/approx/env.py"


def sub1(s, old, new, tag):
    assert s.count(old) == 1, f"{tag}: {s.count(old)} matches"
    return s.replace(old, new)


src = io.open(ENV).read()

# ---- 1. caller: hand the face wires over as NUMPY -------------------------
src = sub1(src,
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
    "    if _have_face_actions and not _unified_fe:\n"
    "        # NUMPY, not `.tolist()`. `_face_dict_for_vertex` pulls the three\n"
    "        # wire ints out of `face_row[f][s]` explicitly and reads\n"
    "        # `face_skip[f]` scalar-wise, so it never needed Python lists --\n"
    "        # and materialising T x MAX_FACES x FACE_SLOTS x 3 Python ints per\n"
    "        # callback is O(T x MAX_FACES) per step, O(T^2 x MAX_FACES) per\n"
    "        # episode, of which >99% is -1 padding (measured live occupancy is\n"
    "        # ~1.24 faces per vertex). Same argument, same fix as the\n"
    "        # `_fe_inline` wires below already use.\n"
    "        ft_by_vertex = _face_transforms_for_order(\n"
    "            config, consts, args, o_list, specs_list,\n"
    "            _faces_np, _skips_np,\n"
    "            honor_last_compress=_honor_mid_compress,\n"
    "        )\n"
    "    _pf(\"cb.face_enum\")",
    "caller-numpy")

# ---- 2. the KEY: sparse per-vertex wire signatures ------------------------
src = sub1(src,
    "        _t_key = time.perf_counter()\n"
    "        _sigs = tuple(\n"
    "            (int(o_list[k]),\n"
    "             np.asarray(specs_list[k], dtype=np.int64).tobytes(),\n"
    "             np.asarray(face_rows_list[k], dtype=np.int64).tobytes(),\n"
    "             np.asarray(face_skips_list[k], dtype=np.int64).tobytes())\n"
    "            for k in range(len(o_list)))\n",
    "        _t_key = time.perf_counter()\n"
    "        # THE KEY WAS THE PATHOLOGY, NOT THE REPLAY. The dense form this\n"
    "        # replaces rebuilt `np.asarray(<nested list>).tobytes()` for every\n"
    "        # vertex of the prefix on every step: MAX_FACES x FACE_SLOTS x 3\n"
    "        # ints per vertex (22842 at the flagship bound), i.e.\n"
    "        # O(T x MAX_FACES) per step and O(T^2 x MAX_FACES) per episode --\n"
    "        # the exact term the cache exists to remove, left standing in its\n"
    "        # own lookup. `_face_wire_keys` is the sparse injective encoding\n"
    "        # `_INCR_STREAM_CACHE` already keys on: ONE vectorised pass over\n"
    "        # the prefix, a few bytes per vertex, and identical discriminating\n"
    "        # power (every entry not listed is exactly the -1 / 0 pad).\n"
    "        _fw = _face_wire_keys(np.asarray(face_rows_list),\n"
    "                              np.asarray(face_skips_list), len(o_list))\n"
    "        _sp = np.ascontiguousarray(\n"
    "            np.asarray(specs_list, dtype=np.int64))\n"
    "        _sigs = tuple(\n"
    "            (int(o_list[k]), _sp[k].tobytes()) + _fw[k]\n"
    "            for k in range(len(o_list)))\n",
    "sparse-key")

io.open(ENV, "w").write(src)
print("perf2_fix: OK")
