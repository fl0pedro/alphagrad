import json, sys, glob, os

def latest(pat):
    fs = [f for f in glob.glob(os.path.expanduser(pat)) if os.path.getsize(f) > 0]
    return max(fs, key=os.path.getmtime) if fs else None

targets = [
    ("C-MORL pareto", "~/dsnn/wandb/offline-run-*/files/cmorl_pareto_front.json"),
    ("MOGFN pareto", "~/dsnn/wandb/offline-run-*/files/mogfn_pareto_front.*.json"),
    ("C-MORL all-cand", "~/dsnn/wandb/offline-run-*/files/cmorl_all_front_candidates.json"),
    ("MOGFN all-cand", "~/dsnn/wandb/offline-run-*/files/mogfn_all_front_candidates.*.json"),
]
for label, pat in targets:
    path = latest(pat)
    if not path:
        print(label, "-> no file"); continue
    d = json.load(open(path))
    rows = d.get("front") or d.get("candidates") or []
    n = len(rows)
    withseq = sum(1 for r in rows if r.get("seq") not in (None, [], ""))
    print("== %s (%d rows) ==  with_seq=%d/%d" % (label, n, withseq, n))
    if rows:
        s = rows[0].get("seq")
        slen = len(s) if hasattr(s, "__len__") else "-"
        print("   seq type=%s len=%s" % (type(s).__name__, slen))
        head = s[:2] if isinstance(s, list) else str(s)[:140]
        print("   seq head:", head)
        print("   obj:", rows[0].get("obj"))
