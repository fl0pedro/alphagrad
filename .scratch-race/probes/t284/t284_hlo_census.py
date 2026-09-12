#!/usr/bin/env python3
"""Classify every launched kernel of an XLA/GPU HLO module."""
import re, sys, os, collections

CALL = re.compile(r"^\s+%?([\w.\-]+) = (\(.*?\)|\S+) (fusion|custom-call|copy|copy-start|"
                  r"copy-done|all-reduce|reduce|dot|transpose|concatenate|slice|"
                  r"bitcast|constant|parameter|tuple|get-tuple-element|"
                  r"bitcast-convert|convert|reshape|broadcast)\(")
KIND = re.compile(r"kind=(k\w+)")
CT = re.compile(r'custom_call_target="([^"]+)"')


def census(path):
    txt = open(path).read()
    entry = txt[txt.index("\nENTRY "):] if "\nENTRY " in txt else txt
    # size of a shape like f32[32,128]{1,0}
    def nbytes(shape):
        if shape.startswith("("):
            return sum(nbytes(m2.group(0)) for m2 in
                       re.finditer(r"\w+\[[\d,]*\]", shape))
        m = re.match(r"(\w+?)(\d+)\[([\d,]*)\]", shape)
        if not m:
            return 0
        w = int(m.group(2)) // 8 or 1
        dims = [int(d) for d in m.group(3).split(",") if d]
        n = 1
        for d in dims:
            n *= d
        return n * w
    rows = []
    for line in entry.split("\n"):
        m = CALL.match(line)
        if not m:
            continue
        name, shape, op = m.group(1), m.group(2), m.group(3)
        if op in ("bitcast", "constant", "parameter", "tuple",
                  "get-tuple-element", "bitcast-convert"):
            continue          # not a launched kernel
        cls = op
        if op == "fusion":
            k = KIND.search(line)
            kind = k.group(1) if k else "kUnknown"
            cls = f"fusion:{kind}"
            if "wrapped_slice" in line:
                cls = "copy:wrapped_slice"
            elif "wrapped_concatenate" in line:
                cls = "copy:wrapped_concatenate"
            elif "wrapped_transpose" in line:
                cls = "copy:wrapped_transpose"
        elif op == "custom-call":
            t = CT.search(line)
            tgt = t.group(1) if t else "?"
            cls = "cublas" if "cublas" in tgt else f"custom:{tgt}"
        rows.append((cls, nbytes(shape)))
    c = collections.Counter(r[0] for r in rows)
    b = collections.Counter()
    for cls, nb in rows:
        b[cls] += nb
    return c, b, len(rows)


files = sys.argv[1:]
allcls = set()
data = {}
for f in files:
    c, b, tot = census(f)
    data[f] = (c, b, tot)
    allcls |= set(c)
names = [os.path.basename(f).replace("hlo_", "").replace(".txt", "") for f in files]
w = max(len(n) for n in names) + 2
print(f"{'kernel class':28s}" + "".join(f"{n[:30]:>32s}" for n in names))
for cls in sorted(allcls):
    line = f"{cls:28s}"
    for f in files:
        c, b, _ = data[f]
        line += f"{c.get(cls, 0):>10d} ({b.get(cls, 0)/1024:>8.0f} kB)  " if c.get(cls) else f"{'-':>10s}{'':>12s}  "
    print(line)
print(f"{'TOTAL launched kernels':28s}" + "".join(f"{data[f][2]:>10d}{'':>22s}" for f in files))
