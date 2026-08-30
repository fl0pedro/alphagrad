import io
p = "tests/face_prefix_cache_test.py"
s = io.open(p).read()
reps = [
 ('    assert stats_on == {"ext": T - 1, "cold": 1}, stats_on',
  '    assert {k: stats_on[k] for k in ("ext", "cold")} == {\n'
  '        "ext": T - 1, "cold": 1}, stats_on'),
 ('    assert stats_on == {"ext": T - 2, "cold": 1}, stats_on',
  '    assert {k: stats_on[k] for k in ("ext", "cold")} == {\n'
  '        "ext": T - 2, "cold": 1}, stats_on'),
 ('    assert E._FACE_ENUM_STATS == {"ext": 0, "cold": 0}, E._FACE_ENUM_STATS',
  '    assert {k: E._FACE_ENUM_STATS[k] for k in ("ext", "cold")} == {\n'
  '        "ext": 0, "cold": 0}, E._FACE_ENUM_STATS'),
]
for a, b in reps:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
io.open(p, "w").write(s)
print("tests patched")
