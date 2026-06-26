import json, os, numpy as np, collections
F = os.path.expanduser("~/dsnn/search_space_full_grad/results_seed12345.jsonl")
names = ["muls_adds_fmas", "flops", "latency_ns", "max_io_sum", "bytes_accessed",
         "peak_memory", "cosine_sim", "frob_residual", "xla_peak_memory"]
valid = []
errs = collections.Counter()
ninv = 0
n = 0
for line in open(F):
    try:
        r = json.loads(line)
    except Exception:
        continue
    n += 1
    if r.get("invalid"):
        ninv += 1
        e = str(r.get("err", "?"))
        key = e.split(":", 1)[0]  # exception type
        errs[key] += 1
        continue
    rv = r.get("reward_vec")
    if rv and len(rv) == 9:
        valid.append(rv)
V = np.array(valid)
print("lines=%d invalid=%d valid_lines=%d" % (n, ninv, len(valid)))
print("\n=== per-channel coverage over valid records ===")
for i, nm in enumerate(names):
    col = V[:, i]
    nz = float(np.mean(col != 0.0))
    print("  [%d] %-16s nonzero=%.3f  |min|=%.3g med=%.3g max=%.3g"
          % (i, nm, nz, np.min(np.abs(col)), np.median(np.abs(col)), np.max(np.abs(col))))
print("\n=== invalid error types (top 10) ===")
for e, c in errs.most_common(10):
    print("  %6d  %s" % (c, e))
# peek a couple full invalid messages
print("\n=== sample invalid messages ===")
shown = 0
for line in open(F):
    r = json.loads(line)
    if r.get("invalid"):
        print("  -", str(r.get("err"))[:160])
        shown += 1
        if shown >= 5:
            break
