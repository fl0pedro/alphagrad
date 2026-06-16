"""Signal F: learned combiner. Loads every qsig_*.json produced by the signal
finders, joins them per-rule on (src, idx), and asks whether a linear combiner of
the rank-transformed signals beats the best single signal. Reports:
  - per-signal Spearman vs accuracy (both / cmorl / mogfn / lat-controlled)
  - 5-fold CV Spearman of a ridge combiner over all signals
  - leave-one-source-out (train cmorl -> predict mogfn and vice versa)
  - greedy forward-selection of the best small subset
  uv run python src/alphagrad/approx/qsig_F_combiner.py
"""
import os, glob, json, itertools
import numpy as np

EXP = os.path.expanduser("~/dsnn/train_exp")


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float)
def pear(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
def spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    return pear(rank(x[m]), rank(y[m]))


# ---- load every signal, join on (src, idx) ----------------------------------
# EXCLUDE: signals listed in QSIG_EXCLUDE (comma-sep) are dropped from the
# feature set. Used to produce the CHEAP-ONLY picture by dropping B_kstep
# (expensive + tautological). e.g. QSIG_EXCLUDE=B_kstep
EXCLUDE = set(x for x in os.environ.get("QSIG_EXCLUDE", "").split(",") if x)
sigs = {}
for f in sorted(glob.glob(f"{EXP}/qsig_*.json")):
    name = os.path.basename(f)[5:-5]
    if name == "F_combiner" or name in EXCLUDE:
        continue
    try:
        recs = json.load(open(f))
    except Exception:
        continue
    sigs[name] = {(r["src"], r["idx"]): r for r in recs}

if not sigs:
    raise SystemExit("no qsig_*.json signal files found")

names = sorted(sigs)
# keys present in ALL signals (and with finite acc)
common = set.intersection(*[set(d) for d in sigs.values()])
keys = sorted(common)
src = np.array([k[0] for k in keys])
acc = np.array([sigs[names[0]][k]["acc"] for k in keys], float)
lat = np.array([sigs[names[0]][k]["lat"] for k in keys], float)
X = np.column_stack([[sigs[n][k]["sig"] for k in keys] for n in names]).astype(float)

# impute NaN feature values to that feature's median (rare skipped rules)
for j in range(X.shape[1]):
    col = X[:, j]; med = np.nanmedian(col); col[~np.isfinite(col)] = med; X[:, j] = col

print(f"signals: {names}")
print(f"joined rules: n={len(keys)} (cmorl={int((src=='cmorl').sum())}, mogfn={int((src=='mogfn').sum())})\n")

# ---- per-signal Spearman ----------------------------------------------------
print("== per-signal Spearman vs accuracy ==")
both_scores = {}
for j, n in enumerate(names):
    cm = (src == "cmorl"); mo = (src == "mogfn")
    sb = spear(X[:, j], acc); sc = spear(X[cm, j], acc[cm]); sm = spear(X[mo, j], acc[mo])
    both_scores[n] = sb
    print(f"  {n:14s} BOTH={sb:+.3f}  C-MORL={sc:+.3f}  MOGFN={sm:+.3f}")
best_single = max(both_scores, key=both_scores.get)
print(f"\nbest single (BOTH): {best_single} = {both_scores[best_single]:+.3f}\n")


# ---- ridge combiner with k-fold CV (rank features) --------------------------
# NOTE: ridge is fit on mean-centered X and y with the intercept handled
# analytically (b0 = y_mean - x_mean @ beta). The previous version appended a
# ones-column and zeroed its regularizer, which produced a near-singular system
# and exploding (1e17) weights -> CV collapsed to ~0. Standardization stats are
# computed on the TRAIN fold only (no test-fold leakage).
def ridge_fit(Xtr, ytr, lam=1.0):
    mx = Xtr.mean(0); my = ytr.mean()
    Xc = Xtr - mx; yc = ytr - my
    beta = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ yc)
    b0 = my - mx @ beta
    return (beta, b0)


def ridge_pred(w, Xte):
    beta, b0 = w
    return Xte @ beta + b0


def _fit_predict(Xtr, ytr, Xte, lam=1.0):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9          # train-only standardization
    w = ridge_fit((Xtr - mu) / sd, ytr, lam)
    return ridge_pred(w, (Xte - mu) / sd)


def cv_spearman(cols, n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    Xr = np.column_stack([rank(X[:, c]) for c in cols])
    yr = rank(acc)
    n = len(yr); idx = rng.permutation(n); preds = np.zeros(n)
    for fold in range(n_folds):
        te = idx[fold::n_folds]; tr = np.setdiff1d(idx, te)
        preds[te] = _fit_predict(Xr[tr], yr[tr], Xr[te])
    return spear(preds, acc)


all_cols = list(range(len(names)))
print("== combiner ==")
print(f"  all-signals 5-fold CV Spearman (BOTH) = {cv_spearman(all_cols):+.3f}")

# leave-one-source-out: train on one source, predict the other
def loso():
    out = {}
    Xr = np.column_stack([rank(X[:, c]) for c in all_cols]); yr = rank(acc)
    for hold in ("cmorl", "mogfn"):
        te = (src == hold); tr = ~te
        pr = _fit_predict(Xr[tr], yr[tr], Xr[te])
        out[hold] = spear(pr, acc[te])
    return out
lo = loso()
print(f"  leave-one-source-out: train->predict cmorl={lo['cmorl']:+.3f}  mogfn={lo['mogfn']:+.3f}")

# greedy forward selection
chosen, best = [], -1.0
remaining = list(all_cols)
while remaining:
    scored = [(cv_spearman(chosen + [c]), c) for c in remaining]
    s, c = max(scored)
    if s <= best + 1e-3:
        break
    best = s; chosen.append(c); remaining.remove(c)
print(f"  greedy subset: {[names[c] for c in chosen]}  CV Spearman={best:+.3f}")

summary = dict(signals=names, per_signal_both=both_scores, best_single=best_single,
               best_single_both=both_scores[best_single],
               combiner_all_cv=cv_spearman(all_cols), loso=lo,
               greedy_subset=[names[c] for c in chosen], greedy_cv=best,
               n=len(keys))
json.dump(summary, open(f"{EXP}/qsig_F_combiner.json", "w"), indent=2)
print(f"\nsaved -> {EXP}/qsig_F_combiner.json")
