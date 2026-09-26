# dsnn-dfw.264: during a write, a reader sees the old file or none.
import gzip
import io
import os
import threading
import types
import urllib.request
import zipfile

import numpy as np

from alphagrad.approx.common import datasets as D

CHUNK = 1 << 16
WAIT = 60.0


def _seen(path, full):
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    return "full" if data == full else "partial"


def _bytes(n, seed):
    return np.random.default_rng(seed).bytes(n)


def _fake_urlretrieve(payloads, cache, probes):
    def fetch(url, filename, *args, **kwargs):
        name = url.rsplit("/", 1)[1]
        payload = payloads[name]
        with open(filename, "wb") as fh:
            for i in range(0, len(payload), CHUNK):
                fh.write(payload[i:i + CHUNK])
                fh.flush()
                probes.append((name, _seen(cache / name, payload)))
        return str(filename), None
    return fetch


def _wait(event, errors, what):
    if not event.wait(WAIT):
        errors.append(("timeout", what))


def _race(writer, target_seen, errors, probes, ev):
    def run_a():
        try:
            writer()
        except BaseException as exc:
            errors.append(("A", repr(exc)))
        probes.append(target_seen())
        ev["a_done"].set()

    def run_b():
        try:
            writer()
        except BaseException as exc:
            errors.append(("B", repr(exc)))

    ta = threading.Thread(target=run_a)
    ta.start()
    _wait(ev["a_paused"], errors, "A never paused")
    tb = threading.Thread(target=run_b)
    tb.start()
    ta.join(3 * WAIT)
    tb.join(3 * WAIT)
    assert not ta.is_alive() and not tb.is_alive()


def test_the_mnist_download_never_shows_a_partial_file(tmp_path, monkeypatch):
    payloads = {f: _bytes(3 * CHUNK + 17, i)
                for i, f in enumerate(D._MNIST_FILES.values())}
    probes = []
    monkeypatch.setattr(urllib.request, "urlretrieve",
                        _fake_urlretrieve(payloads, tmp_path, probes))
    D._download_mnist(tmp_path)
    assert len(probes) == 4 * len(payloads)
    assert [p for p in probes if p[1] == "partial"] == []
    for f, payload in payloads.items():
        assert (tmp_path / f).read_bytes() == payload
    assert sorted(os.listdir(tmp_path)) == sorted(payloads)


def test_the_wikitext_download_never_shows_a_partial_zip(tmp_path,
                                                         monkeypatch):
    reps = 10000
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("wikitext-2/wiki.train.tokens",
                    " the cat the <unk> dog the" * reps)
    payload = buf.getvalue()
    name = "wikitext-2-v1.zip"
    probes = []
    monkeypatch.setenv("DSNN_WIKITEXT_DIR", str(tmp_path))
    monkeypatch.setattr(D, "_DATASET_CACHE", {})
    monkeypatch.setattr(urllib.request, "urlretrieve",
                        _fake_urlretrieve({name: payload}, tmp_path, probes))
    ids = D.load_wikitext2(4)
    assert len(probes) == -(-len(payload) // CHUNK) > 2
    assert [p for p in probes if p[1] == "partial"] == []
    assert (tmp_path / name).read_bytes() == payload
    assert sorted(os.listdir(tmp_path)) == [name]
    assert ids.shape == (6 * reps,)


def test_the_shd_download_never_shows_a_partial_gz(tmp_path, monkeypatch):
    raw = _bytes(3 * CHUNK + 5, 250197)
    payload = gzip.compress(raw)
    name = "shd_train.h5.gz"
    probes = []
    monkeypatch.setattr(urllib.request, "urlretrieve",
                        _fake_urlretrieve({name: payload}, tmp_path, probes))
    out = D._download_shd(tmp_path, "train")
    assert out == tmp_path / "shd_train.h5"
    assert len(probes) == -(-len(payload) // CHUNK) > 2
    assert [p for p in probes if p[1] == "partial"] == []
    assert out.read_bytes() == raw
    assert (tmp_path / name).read_bytes() == payload
    assert sorted(os.listdir(tmp_path)) == ["shd_train.h5", name]


def test_two_shd_unpacks_never_show_a_partial_h5(tmp_path, monkeypatch):
    raw = _bytes(6 * CHUNK, 250198)
    (tmp_path / "shd_train.h5.gz").write_bytes(gzip.compress(raw))
    target = tmp_path / "shd_train.h5"
    ev = {k: threading.Event() for k in ("a_paused", "b_opened", "a_done")}
    errors, probes = [], []

    class _Source:
        def __init__(self, on_read):
            self._pos, self._n, self._on_read = 0, 0, on_read

        def read(self, n=-1):
            self._on_read(self._n)
            self._n += 1
            chunk = raw[self._pos:self._pos + CHUNK]
            self._pos += len(chunk)
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def a_reads(i):
        if i == 2:
            probes.append(_seen(target, raw))
            ev["a_paused"].set()
            _wait(ev["b_opened"], errors, "B never opened its file")

    def b_reads(i):
        if i == 0:
            ev["b_opened"].set()
            _wait(ev["a_done"], errors, "A never finished")

    roles = [a_reads, b_reads]
    monkeypatch.setattr(D, "gzip", types.SimpleNamespace(
        open=lambda path, mode="rb": _Source(roles.pop(0))))
    _race(lambda: D._download_shd(tmp_path, "train"),
          lambda: _seen(target, raw), errors, probes, ev)
    assert probes == [None, "full"], (probes, errors)
    assert errors == []
    assert target.read_bytes() == raw
    assert sorted(os.listdir(tmp_path)) == ["shd_train.h5", "shd_train.h5.gz"]


def test_two_binned_cache_writes_never_show_a_partial_npz(tmp_path,
                                                          monkeypatch):
    rng = np.random.default_rng(250199)
    x = rng.integers(0, 7, size=(24, D.SHD_TIME_BINS, D.SHD_CHANNELS),
                     dtype=np.uint8)
    y = rng.integers(0, D.SHD_CLASSES, size=(24,), dtype=np.uint8)
    npz = tmp_path / (f"shd_train_binned_{D.SHD_TIME_BINS}x{D.SHD_CHANNELS}_"
                      f"{np.dtype(D._SHD_COUNT_DTYPE).name}.npz")
    ev = {k: threading.Event() for k in ("a_paused", "b_opened", "a_done")}
    errors, probes = [], []

    def npz_seen():
        if not npz.exists():
            return None
        try:
            with np.load(npz) as z:
                ok = (np.array_equal(z["x"], x)
                      and np.array_equal(z["y"], y))
        except Exception as exc:
            return f"partial ({exc!r})"
        return "full" if ok else "partial"

    class _Handle:
        def __init__(self, fh, role):
            self._fh, self._role = fh, role

        def write(self, data):
            view = memoryview(data)
            if self._role == "B":
                self._role = None
                ev["b_opened"].set()
                _wait(ev["a_done"], errors, "A never finished")
            elif self._role == "A" and view.nbytes >= CHUNK:
                self._role = None
                half = view.nbytes // 2
                n = self._fh.write(view[:half])
                self._fh.flush()
                probes.append(npz_seen())
                ev["a_paused"].set()
                _wait(ev["b_opened"], errors, "B never opened its file")
                return n + self._fh.write(view[half:])
            return self._fh.write(data)

        def __getattr__(self, name):
            return getattr(self._fh, name)

    real_savez = np.savez
    roles = ["A", "B"]

    def savez(file, *args, **kwargs):
        return real_savez(_Handle(file, roles.pop(0)), *args, **kwargs)

    monkeypatch.setenv("DSNN_SHD_DIR", str(tmp_path))
    monkeypatch.setattr(D, "_SHD_BINNED", {})
    monkeypatch.setattr(D, "_download_shd",
                        lambda cache, subset: cache / "shd_train.h5")
    monkeypatch.setattr(D, "_bin_shd", lambda path: (x, y))
    monkeypatch.setattr(np, "savez", savez)
    _race(lambda: D._shd_binned("train"), npz_seen, errors, probes, ev)
    assert probes == [None, "full"], (probes, errors)
    assert errors == []
    assert npz_seen() == "full"
    assert sorted(os.listdir(tmp_path)) == [npz.name]
