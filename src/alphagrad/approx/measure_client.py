"""Parent-side client for measure_server (subprocess CUDA-poison isolation).

MeasureClient.measure_seq(seq) -> np.array([lat, xla_peak, flops, cos]) or None.
The child owning the CUDA context dies on a poisoned kernel; the client detects
it (EOF/timeout), restarts the child, and returns None so the caller redraws.
"""
import os, sys, json, select, subprocess
import numpy as np


class MeasureClient:
    def __init__(self, timeout_s=None):
        self.timeout_s = float(timeout_s or os.environ.get(
            "ALPHAGRAD_MS_TIMEOUT", "900"))
        self._p = None
        self.restarts = 0

    def _start(self):
        env = os.environ.copy()
        # the child runs BARE sys.executable (no `uv run`): make alphagrad
        # importable explicitly, and NEVER swallow its stderr (the DEVNULL
        # trap already hid one silent-failure bug in this codebase).
        _src = os.path.expanduser("~/dsnn/alphagrad/src")
        env["PYTHONPATH"] = _src + os.pathsep + env.get("PYTHONPATH", "")
        # verify_pareto_solution.py sets JAX_PLATFORMS=cpu as an IMPORT side
        # effect; the parent's backend is already initialized (GPU) so only
        # CHILDREN get poisoned -> strip it so the measure child sees the GPU.
        # ALPHAGRAD_MS_PLATFORM overrides explicitly if ever needed.
        _plat = os.environ.get("ALPHAGRAD_MS_PLATFORM", "").strip()
        if _plat:
            env["JAX_PLATFORMS"] = _plat
        else:
            env.pop("JAX_PLATFORMS", None)
        # put the measure child on its OWN GPU when one is given: heavy tasks
        # (ViT COMPRESS densify) otherwise co-reside with the training encoder
        # on one card and OOM it. ALPHAGRAD_MS_GPU="1" -> child sees only cuda:1.
        _msgpu = os.environ.get("ALPHAGRAD_MS_GPU", "").strip()
        if _msgpu:
            env["CUDA_VISIBLE_DEVICES"] = _msgpu
        self._errlog = open(os.path.expanduser("~/dsnn/ms_child_stderr.log"), "a")
        self._p = subprocess.Popen(
            [sys.executable, "-m", "alphagrad.approx.measure_server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._errlog, text=True, bufsize=1, env=env)
        line = self._readline(self.timeout_s)          # wait for {"ready": true}
        if line is None or not json.loads(line).get("ready"):
            raise RuntimeError("measure_server failed to become ready")

    def _readline(self, timeout):
        if self._p is None or self._p.stdout is None:
            return None
        r, _, _ = select.select([self._p.stdout], [], [], timeout)
        if not r:
            return None
        line = self._p.stdout.readline()
        return line if line else None

    def _kill(self):
        if self._p is not None:
            try:
                self._p.kill(); self._p.wait(timeout=10)
            except Exception:
                pass
        self._p = None

    def measure_seq(self, seq):
        """seq = [(action_idx, [op strings])]. None on failure (caller redraws)."""
        if self._p is None or self._p.poll() is not None:
            self._kill(); self.restarts += 1
            try:
                self._start()
            except Exception:
                return None
        try:
            self._p.stdin.write(json.dumps(
                {"seq": [[int(a), list(ops)] for a, ops in seq]}) + "\n")
            self._p.stdin.flush()
            line = self._readline(self.timeout_s)
            if line is None:                            # died or hung -> restart next call
                self._kill()
                return None
            out = json.loads(line)
            if "raw" in out:
                return np.asarray(out["raw"], dtype=np.float64)
            # LOG the server-side error (the third silence layer bit us once):
            try:
                self._errlog.write(f"[server-error] {out.get('error', '?')}\n")
                self._errlog.flush()
            except Exception:
                pass
            return None
        except BaseException:
            self._kill()
            return None

    def close(self):
        self._kill()
