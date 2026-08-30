"""decode2_dscheck: numpy-only sanity facts about the face dataset."""
import sys, numpy as np
d = np.load(sys.argv[1], allow_pickle=True)
FN = [str(x) for x in d["f_names"]]; T = d["f_tgt"]
c = {n: i for i, n in enumerate(FN)}
print("faces", T.shape, "trajectories", d["tok"].shape)
print("ln_out == stat_ln_j elementwise:",
      bool(np.allclose(T[:, c["ln_out"]], T[:, c["stat_ln_j"]])),
      " max|diff|", float(np.abs(T[:, c["ln_out"]] - T[:, c["stat_ln_j"]]).max()))
print("ln_prim == stat_ln_i elementwise:",
      bool(np.allclose(T[:, c["ln_prim"]], T[:, c["stat_ln_i"]])),
      " max|diff|", float(np.abs(T[:, c["ln_prim"]] - T[:, c["stat_ln_i"]]).max()))
for n in FN:
    v = T[:, c[n]]
    print(f"  {n:13s} std {v.std():7.3f}  n_uniq {len(np.unique(v)):5d}")
# how much of each target is explained by the static face key alone?
key = d["f_si"].astype(np.int64) * 10000 + d["f_sj"]
_, inv = np.unique(key, return_inverse=True)
print("\nvariance explained by the STATIC face key (in-sample upper bound):")
for n in FN:
    v = T[:, c[n]]
    if v.std() < 1e-9:
        print(f"  {n:13s} constant"); continue
    m = np.bincount(inv, weights=v) / np.maximum(np.bincount(inv), 1)
    r2 = 1 - ((v - m[inv]) ** 2).mean() / v.var()
    print(f"  {n:13s} R2 {r2:6.3f}")
# faces per (traj, step)
q = d["f_q"]; tr = d["f_traj"]
k2 = tr.astype(np.int64) * 1000 + q
u, cnt = np.unique(k2, return_counts=True)
print(f"\nfaces per (traj,step): mean {cnt.mean():.2f} max {cnt.max()} "
      f"groups>=2 {(cnt >= 2).mean()*100:.0f}%  groups {len(u)}")
