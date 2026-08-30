"""Apply the warm-up health-line fix to a copy of ppo.py.

usage: python patch_health.py <path-to-ppo.py>
Exact-string replacements only; errors loudly if an anchor is missing or
ambiguous, so it can never half-apply to a file another agent has edited.
"""
import sys

OLD_1 = '''        _HEALTH_N[0] += 1
        if _HEALTH_N[0] <= int(os.environ.get("ALPHAGRAD_HEALTH_EPISODES", "3")):
'''

NEW_1 = '''        # PopArt WARM-START episodes run NO gradient step: the caller passes an
        # all-NaN `_wmets` purely to keep the tuple shape uniform (see
        # --popart-init-episodes) and host_log drops every loss-derived key
        # ~100 lines below. This line ran BEFORE that drop and counted warm-up
        # rows against the budget -- and since --popart-init-episodes and
        # ALPHAGRAD_HEALTH_EPISODES BOTH default to 3, the three health lines a
        # run printed were ALWAYS the three warm-up rows, reading
        # "ppo=nan value=nan ent=nan ratio/max_log=nan". The first REAL
        # training episode never printed one, so ratio/max_log -- the
        # ratio-1-at-epoch-0 tripwire this line exists to show -- was never
        # displayed at all, and the expected warm-up NaNs camouflaged that.
        # Warm-up rows are now labelled `[health warmup]`, print `n/a` for the
        # undefined fields, and do NOT consume the budget.
        if not warmup:
            _HEALTH_N[0] += 1
        _hlabel = "warmup" if warmup else "ep%d" % (_HEALTH_N[0] - 1)
        if warmup or _HEALTH_N[0] <= int(
                os.environ.get("ALPHAGRAD_HEALTH_EPISODES", "3")):
'''

OLD_2 = '''            tqdm.write("[health ep%d] ppo=%.4g value=%.4g ent=%.4g "
                       "ratio/max_log=%.3g kl/approx=%.3g mu_quality=%.4g "
                       "sec/ep=%.1f" % (
                           _HEALTH_N[0] - 1, ppo_loss, value_loss,
                           policy_entropy,
                           log_dict.get("ratio/max_log", float("nan")),
                           log_dict.get("kl/approx", float("nan")),
                           log_dict.get("popart/mu_quality", float("nan")),
                           log_dict.get("time/sec_per_episode", float("nan"))))
'''

NEW_2 = '''            # `n/a`, never NaN, for a metric that is UNDEFINED here: a warm-up
            # row has no gradient step, and an absent key is a missing
            # measurement rather than a bad number. Anything that still prints
            # `nan` on a `[health ep..]` line is therefore a REAL non-finite
            # value -- which is exactly what tools/smoke.sh exits non-zero on.
            def _hk(_k, _f="%.4g", _warm_undef=True):
                if (_warm_undef and warmup) or _k not in log_dict:
                    return "n/a"
                return _f % float(log_dict[_k])
            _hv = (lambda _v, _f="%.4g": "n/a" if warmup else _f % float(_v))
            tqdm.write("[health %s] ppo=%s value=%s ent=%s "
                       "ratio/max_log=%s kl/approx=%s mu_quality=%s "
                       "sec/ep=%s" % (
                           _hlabel, _hv(ppo_loss), _hv(value_loss),
                           _hv(policy_entropy),
                           _hk("ratio/max_log", "%.3g"),
                           _hk("kl/approx", "%.3g"),
                           _hk("popart/mu_quality", "%.4g", False),
                           _hk("time/sec_per_episode", "%.1f", False)))
'''

OLD_3 = '''                tqdm.write("[health ep%d] live-faces %s" % (
                    _HEALTH_N[0] - 1, _LIVE_FACES.consume_stats()))
'''

NEW_3 = '''                tqdm.write("[health %s] live-faces %s" % (
                    _hlabel, _LIVE_FACES.consume_stats()))
'''

OLD_4 = '''        pbar.set_description(
            f"ent:{policy_entropy:.3f} best:{b_ret_desc} means:{means_str}"
        )
'''

NEW_4 = '''        # `ent:nan` on the bar was the SAME warm-up artefact as the health
        # line (all-NaN `_wmets`), and it was read as a real failure more than
        # once. Undefined during a warm-up row -> say so.
        pbar.set_description(
            f"ent:{'n/a' if warmup else format(policy_entropy, '.3f')} "
            f"best:{b_ret_desc} means:{means_str}"
        )
'''

PAIRS = [(OLD_1, NEW_1), (OLD_2, NEW_2), (OLD_3, NEW_3), (OLD_4, NEW_4)]


def main(path):
    src = open(path).read()
    if "_hlabel" in src:
        print(f"ALREADY PATCHED: {path}")
        return 0
    for i, (old, new) in enumerate(PAIRS, 1):
        n = src.count(old)
        if n != 1:
            raise SystemExit(
                f"ANCHOR {i} matched {n} times (expected 1) in {path} -- "
                "refusing to patch")
        src = src.replace(old, new)
    open(path, "w").write(src)
    print(f"PATCHED (4 hunks): {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
