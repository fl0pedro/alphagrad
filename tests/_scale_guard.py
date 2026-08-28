# -*- coding: utf-8 -*-
"""Import-order guard for the test-scale ``ALPHAGRAD_MAX_*`` knobs.

``alphagrad.approx.env`` freezes ``MAX_DELTA_TOKENS`` (and the ``MAX_FACES``
default) into module constants at ITS FIRST import.  A test module that asks
for a small test scale with ``os.environ.setdefault`` therefore only GETS it
when it happens to be the first module in the process to import alphagrad.

Run in a shared pytest process behind any module that imports alphagrad
without setting the knob (``tests/per_face_masks_test.py`` is one), the env
var reads 128 while ``env.MAX_DELTA_TOKENS`` is still 32768.  The stand-in
face-chunk callbacks size their token arrays from the env var, ``_face_loop``
sizes its ONE per-step chunk-stream buffer from the constant, and the two
disagree deep inside the while body::

    ValueError: Incompatible shapes for broadcasting:
                shapes=[(32768,), (128,), ()]

That is SCAFFOLDING, not the sampling/replay mirror.  In production the face
chunk callback is a ``LiveFaceStream(..., window=MAX_DELTA_TOKENS)`` built
from the very constant ``_face_loop`` sizes the stream with, so the two
cannot disagree -- and a shape error is a TRACE-TIME crash anyway, never a
silently wrong importance ratio.

``tests/face_read_point_test.py`` already reads the live constants for this
exact reason.  ``request_scale`` generalises that: it makes the request, then
returns what ``env.py`` ACTUALLY froze, and turns a request that did not take
into a NAMED module-level skip instead of a broadcast error 200 frames down.
``tools/ratio_gates.sh`` runs each affected module in its own process (where
the request always takes) and treats a skip as a failure, so the skip cannot
rot into silent green.
"""
import os


def request_scale(max_delta_tokens=None, max_faces=None):
    """Ask for a test scale.

    Returns ``(MAX_DELTA_TOKENS, MAX_FACES, pytestmark)`` -- the values
    ``env.py`` ACTUALLY holds, plus a ``pytestmark`` list that is empty when
    the request took and a single named ``skip`` when it did not.  Assign it::

        W, MAXF, pytestmark = request_scale(max_delta_tokens=128,
                                            max_faces=64)

    A mark rather than ``pytest.skip(allow_module_level=True)`` on purpose:
    the module still IMPORTS and its node ids still resolve, so a batched
    ``pytest a::t b::t ...`` run reports "skipped, <reason>" for this file
    and still RUNS every other gate on the command line.
    """
    if max_delta_tokens is not None:
        os.environ.setdefault(
            "ALPHAGRAD_MAX_DELTA_TOKENS", str(int(max_delta_tokens)))
    if max_faces is not None:
        os.environ.setdefault("ALPHAGRAD_MAX_FACES", str(int(max_faces)))

    import pytest
    from alphagrad.approx import env as _env

    bad = []
    if (max_delta_tokens is not None
            and int(_env.MAX_DELTA_TOKENS) != int(max_delta_tokens)):
        bad.append("MAX_DELTA_TOKENS: asked %d, env.py froze %d"
                   % (int(max_delta_tokens), int(_env.MAX_DELTA_TOKENS)))
    if max_faces is not None and int(_env.MAX_FACES) != int(max_faces):
        bad.append("MAX_FACES: asked %d, env.py froze %d"
                   % (int(max_faces), int(_env.MAX_FACES)))

    marks = []
    if bad:
        marks = [pytest.mark.skip(reason=(
            "alphagrad.approx.env was imported before this module, so its "
            "test-scale request did not take (" + "; ".join(bad) + "). This "
            "module builds stand-in face chunks at the requested width and "
            "would hit a shape mismatch against the frozen constants. Run it "
            "in its OWN process: tools/ratio_gates.sh, or "
            "`python -m pytest <this file>` with nothing else on the "
            "command line."))]
    return int(_env.MAX_DELTA_TOKENS), int(_env.MAX_FACES), marks
