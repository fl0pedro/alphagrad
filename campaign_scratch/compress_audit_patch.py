"""TEMPORARY instrumentation: count COMPRESS rows decoded with is_last True/False.

Applied to src/alphagrad/approx/env.py, reverted by --revert.
"""
import re
import sys

PATH = "/Users/assmuth/dsnn/alphagrad/src/alphagrad/approx/env.py"

BEGIN = "# --- BEGIN COMPRESS AUDIT (temporary) ---"
END = "# --- END COMPRESS AUDIT (temporary) ---"

BLOCK = f'''{BEGIN}
_COMPRESS_AUDIT = os.environ.get("ALPHAGRAD_COMPRESS_AUDIT", "0") == "1"
_COMPRESS_AUDIT_STATS = {{"decodes": 0, "compress_decodes": 0,
                         "compress_honored": 0, "compress_dropped": 0}}
_COMPRESS_AUDIT_PLANS = {{"honored": set(), "dropped": set()}}


def _compress_audit(vertex, spec_rows, is_last):
    _COMPRESS_AUDIT_STATS["decodes"] += 1
    try:
        has = any(int(r[0]) == COMPRESS_SENTINEL for r in spec_rows)
    except Exception:
        return
    if not has:
        return
    _COMPRESS_AUDIT_STATS["compress_decodes"] += 1
    key = (int(vertex), tuple(tuple(int(x) for x in r) for r in spec_rows))
    if is_last:
        _COMPRESS_AUDIT_STATS["compress_honored"] += 1
        _COMPRESS_AUDIT_PLANS["honored"].add(key)
    else:
        _COMPRESS_AUDIT_STATS["compress_dropped"] += 1
        _COMPRESS_AUDIT_PLANS["dropped"].add(key)


if _COMPRESS_AUDIT:
    import atexit as _atexit
    import json as _json

    def _compress_audit_dump():
        d = dict(_COMPRESS_AUDIT_STATS)
        d["distinct_compress_rows_honored"] = len(
            _COMPRESS_AUDIT_PLANS["honored"])
        d["distinct_compress_rows_dropped"] = len(
            _COMPRESS_AUDIT_PLANS["dropped"])
        d["distinct_dropped_only"] = len(
            _COMPRESS_AUDIT_PLANS["dropped"] - _COMPRESS_AUDIT_PLANS["honored"])
        out = os.environ.get("ALPHAGRAD_COMPRESS_AUDIT_DIR", "/tmp")
        try:
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, f"audit_{{os.getpid()}}.json"), "w") as f:
                _json.dump(d, f)
        except Exception:
            pass
        sys.stderr.write(f"[compress-audit pid={{os.getpid()}}] {{d}}\\n")

    _atexit.register(_compress_audit_dump)
{END}


'''

HOOK = """    if _COMPRESS_AUDIT:
        _compress_audit(vertex, spec_rows, is_last)
"""


def apply():
    src = open(PATH).read()
    if BEGIN in src:
        print("already applied")
        return
    anchor = "def decode_vertex_rule_specs(jaxpr, vertex, spec_rows, is_last: bool) -> tuple:"
    assert anchor in src, "anchor not found"
    src = src.replace(anchor, BLOCK + anchor, 1)
    # insert the hook right after the function's docstring
    i = src.index(anchor)
    j = src.index('    eqn = jaxpr.eqns[vertex - 1]', i)
    src = src[:j] + HOOK + src[j:]
    if "\nimport sys\n" not in src.split("def ")[0]:
        src = src.replace("\nimport os\n", "\nimport os\nimport sys\n", 1)
    open(PATH, "w").write(src)
    print("applied")


def revert():
    src = open(PATH).read()
    if BEGIN not in src:
        print("not applied")
        return
    src = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n\n\n",
                 "", src, flags=re.S)
    src = src.replace(HOOK, "", 1)
    open(PATH, "w").write(src)
    print("reverted")


if __name__ == "__main__":
    (revert if "--revert" in sys.argv else apply)()
