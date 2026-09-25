import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import face_stage2_redraw_test as _redraw                          # noqa: E402

import jax                                                         # noqa: E402

from alphagrad.approx.unified_face_policy import UnifiedFacePolicy  # noqa: E402

try:
    from jax.interpreters.batching import BatchTracer              # noqa: E402
except ImportError:                                                # pragma: no cover
    from jax._src.interpreters.batching import BatchTracer         # noqa: E402


# dsnn-dfw.11: XLA:GPU saturated the face head when the redraw ran it under jax.vmap.
def test_the_stage2_redraw_draws_no_face_under_a_vmap():
    draws = []
    sample = UnifiedFacePolicy.__dict__["sample_face"]

    def spy(self, *args, **kwargs):
        leaves = jax.tree_util.tree_leaves((args, kwargs))
        draws.append(any(isinstance(x, BatchTracer) for x in leaves))
        return sample.__get__(self, type(self))(*args, **kwargs)

    type.__setattr__(UnifiedFacePolicy, "sample_face", spy)
    try:
        _redraw._rows(steps=2)
    finally:
        type.__setattr__(UnifiedFacePolicy, "sample_face", sample)
    assert draws, "no face was drawn"
    assert not any(draws), (
        f"{sum(draws)} of {len(draws)} face draws ran under jax.vmap")
