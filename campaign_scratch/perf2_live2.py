import io
p = "src/alphagrad/approx/ppo.py"
s = io.open(p).read()
old = ('                            f"  face_enum(calls/elims/build)="\n'
       '                            f"{_fs[\'calls\']}/{_fs[\'elims\']}/"\n'
       '                            f"{_fs[\'build\']}")\n')
new = ('                            f"  face_enum(calls/elims/build)="\n'
       '                            f"{_fs[\'calls\']}/{_fs[\'elims\']}/"\n'
       '                            f"{_fs[\'build\']}"\n'
       '                            f"  live_chain={_envmod.'
       'consume_live_chain_stats()}")\n')
assert s.count(old) == 1, s.count(old)
io.open(p, "w").write(s.replace(old, new))
print("ok")
