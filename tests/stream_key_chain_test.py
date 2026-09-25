# dsnn-dfw.188: the carried stream key is the rebuilt key at every step.
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import alphagrad.approx.env as E  # noqa: E402

COMMON = [
    "--seed", "250197", "--measure-latency", "--latency-inner-reps", "50",
    "--num-data-points", "5", "--reps-per-point", "4",
    "--ref-num-data-points", "5", "--ref-reps-per-point", "32",
    "--measure-budget-secs", "1.0", "--measure-window-secs", "0.05",
    "--incremental-encode", "--cmp-type", "latency",
    "--mem-type", "peak_memory", "--terminal-rewards-only",
    "--rewards", "cmp", "mem", "acc", "--reward-mode", "lagrangian",
    "--quality-metric", "grad_cosine", "--approx-add", "lossless",
    "--fixed-order", "free", "--cost-form", "paired-log",
    "--mem-channel", "watermark", "--reduce-axis-space", "physical",
    "--face-read", "last-row", "--set-pointer", "--face-actions",
    "--per-face-masks", "--unified-face-head", "--live-faces",
    "--dynamic-substeps", "--hidden-dim", "256", "--vocab-size", "256",
    "--num-layers", "3", "--tokenize-where", "local",
    "--approx-profile", "all", "--wandb", "disabled",
]
ARGV = {
    "nn256": ["--example", "NeuralNetwork", "--dataset", "mnist"],
    "tlm": ["--example", "TransformerLM", "--dataset", "wikitext2"],
}
ENVS = 4


def _clear():
    E._INCR_STREAM_CACHE.clear()
    E._STREAM_STEPS.clear()
    E._LIVE_CHAINS.clear()
    for k in E._STREAM_STEPS_STATS:
        E._STREAM_STEPS_STATS[k] = 0


@pytest.fixture
def env_of(monkeypatch):
    from alphagrad.approx.common import examples
    keep = E.MAX_FACES
    monkeypatch.delenv("ALPHAGRAD_MAX_FACES", raising=False)
    _clear()

    def build(target):
        if target == "tlm":
            monkeypatch.setenv("ALPHAGRAD_TLM_SEQ", "32")
            monkeypatch.setenv("ALPHAGRAD_TLM_DMODEL", "128")
            monkeypatch.setenv("ALPHAGRAD_TLM_VOCAB", "1024")
        else:
            monkeypatch.setattr(examples, "_EQ_NN_HIDDEN", 256)
        from alphagrad.approx import ppo
        from alphagrad.approx.cpu_approx_worker import _build_env_from_args
        ns, unknown = ppo.make_argparser().parse_known_args(
            ARGV[target] + COMMON)
        assert unknown == [], unknown
        d = dict(vars(ns))
        d["exec_on_gpu"] = False
        return _build_env_from_args(d, None, seed=int(d["seed"]))

    yield build
    E.MAX_FACES = keep
    _clear()


def _episodes(env, seed):
    from alphagrad.approx.env import FACE_SLOTS, MAX_RULES_PER_VERTEX
    rng = np.random.default_rng(seed)
    vv = [int(v) for v in env.valid_vertices]
    F = int(E.MAX_FACES)
    out = []
    for i in range(ENVS):
        rest = [v for v in rng.permutation(vv) if int(v) != vv[i]]
        order = np.asarray([vv[i]] + [int(v) for v in rest], np.int32)
        T = len(order)
        specs = np.full((T, MAX_RULES_PER_VERTEX, 3), -1, np.int32)
        specs[..., 2] = 0
        faces = np.full((T, F, FACE_SLOTS, 3), -1, np.int32)
        skips = np.zeros((T, F), np.int32)
        for k in range(T):
            for f in range(4):
                u = rng.random()
                if u < 0.2:
                    skips[k, f] = 1
                elif u < 0.5:
                    faces[k, f, 0] = (0, 0, -1)
        out.append((order, specs, faces, skips))
    return out


def _walk(env, eps, check):
    cfg = env.config
    consts, args = list(env.consts), list(env.args)
    base = (id(cfg.jaxpr), tuple(cfg.argnums))
    T = len(eps[0][0])
    toks = []
    for t in range(1, T):
        for order, specs, faces, skips in eps:
            out = E._callback(cfg, args, consts, order, specs, faces, skips, t)
            toks.append(np.array(out[0], copy=True))
            if check:
                want = E._stream_steps_rebuilt(
                    [int(x) for x in order[:t]], specs[:t].tolist())
                assert any(k[:2] == base and k[2] == want
                           for k in E._INCR_STREAM_CACHE), (
                    f"step {t}: the stream cache key is not the rebuilt key")
    return toks


@pytest.mark.parametrize("target", ["nn256", "tlm"])
def test_carried_key_is_the_rebuilt_key_over_whole_episodes(
        env_of, monkeypatch, target):
    env = env_of(target)
    eps = _episodes(env, seed=188)
    T = len(eps[0][0])
    carried = _walk(env, eps, check=True)
    stats = dict(E._STREAM_STEPS_STATS)
    assert stats == {"hit": 0, "ext": ENVS * (T - 2), "cold": ENVS}, stats
    _clear()
    monkeypatch.setattr(
        E, "_stream_steps",
        lambda base_key, o_list, specs_list:
            E._stream_steps_rebuilt(o_list, specs_list))
    rebuilt = _walk(env, eps, check=False)
    assert len(carried) == len(rebuilt) == ENVS * (T - 1)
    for i, (a, b) in enumerate(zip(carried, rebuilt)):
        assert a.dtype == b.dtype and np.array_equal(a, b), (
            f"call {i}: the tokens differ between the carried and the "
            f"rebuilt key")
