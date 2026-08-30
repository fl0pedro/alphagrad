import io, re
src = io.open("src/alphagrad/approx/env.py").read()
ppo = io.open("src/alphagrad/approx/ppo.py").read()
anchors = [
 ("env-def", "def _face_transforms_for_order(config, consts, args, o_list, specs_list,\n                               face_rows_list, face_skips_list):"),
 ("env-dispatch", "    from graphax import SKIP_FACE, faces_of\n    from graphax.incremental import IncrementalJaxpr\n    from alphagrad.approx.common.masks import make_live_masked_hook\n\n    ij = None\n"),
 ("env-caller", "    if _have_face_actions and not _unified_fe:\n        _fr_list = _faces_np.tolist()\n        _fs_list = _skips_np.tolist()\n        _pf(\"cb.face_tolist\")\n        ft_by_vertex = _face_transforms_for_order(\n            config, consts, args, o_list, specs_list,\n            _fr_list, _fs_list,\n        )\n    _pf(\"cb.face_enum\")"),
 ("env-facekey", "            face_key=_face_wire_keys(_faces_np, _skips_np, len(o_list))\n            if (ft_by_vertex is not None or _fe_inline) else None,"),
]
for name, a in anchors:
    print(f"{name}: {src.count(a)}")
old = ("                            f\"  face_enum(calls/elims/build)=\"\n"
       "                            f\"{_fs['calls']}/{_fs['elims']}/\"\n"
       "                            f\"{_fs['build']}\")\n")
print("ppo-print:", ppo.count(old))
