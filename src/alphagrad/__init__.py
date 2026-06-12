import sys

# Lazy attribute access for the vertexgame shortcuts. Eagerly importing
# .vertexgame here would pull JAX into every consumer of the `alphagrad`
# package, including JAX-free Ray drivers (see alphagrad.approx.mu0_ray).
# Tests / interactive users that do `from alphagrad import make_graph`
# still work — the import is deferred to first attribute access.
_LAZY_VERTEXGAME_EXPORTS = frozenset({
    "clean", "compress", "cross_country", "embed", "forward",
    "get_graph_shape", "make_graph", "minimal_markowitz", "reverse",
    "safe_preeliminations", "step",
})


def __getattr__(name):
    if name in _LAZY_VERTEXGAME_EXPORTS:
        from . import vertexgame as _vg
        return getattr(_vg, name)
    raise AttributeError(f"module 'alphagrad' has no attribute {name!r}")


from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
