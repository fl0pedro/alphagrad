import io
p = "perf2_faceenum_bench2.py"
s = io.open(p).read()
old = """                faces[e][:k], skips[e][:k], honor_last_compress=True)"""
new = """                faces[e][:k], skips[e][:k])"""
assert s.count(old) == 1
io.open(p, "w").write(s.replace(old, new))
print("ok")
