"""Ticket `dsnn-dfw.68` -- a launcher must exit with the trainer's code.

Every generated launcher ran `$PY src/alphagrad/approx/ppo.py "${ARGS[@]}"`,
printed `echo "TRAINER exited with $?"`, and then fell off the end of the
script, so the script's own exit status was the echo's (0) rather than the
trainer's.  A crashed trainer therefore left sacct reading `COMPLETED 0:0`
(jobs 66737, 66212, and the four C rows 66768, 66770, 66771, 66772) and the
per-node singleton queue moved on as if the run had finished.

The fix is at the source, in `render()`: capture `$?` into a variable before
the echo consumes it, keep the echo, and `exit` with that captured code.
"""
from __future__ import annotations

import importlib.util
import os
import re

_ALPHAGRAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEN = os.path.join(_ALPHAGRAD, "tools", "gen_fq_launchers.py")

_TRAINER_LINE = re.compile(
    r'^\s*src/alphagrad/approx/ppo\.py "\$\{ARGS\[@\]\}"\s*$', re.M)


def _gen():
    spec = importlib.util.spec_from_file_location("gen_fq_launchers", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _one_row(gen):
    """One representative row, rendered end to end."""
    rows = gen.thesis_core_arms()
    assert rows, "the generator emits no thesis core row"
    return rows[0]


def test_the_tail_exits_with_the_trainers_captured_code():
    """The tail of a rendered launcher must capture the trainer's exit code
    into a variable, echo it (kept, for the log), and `exit` with it -- not
    fall through to the echo's own exit status of 0."""
    gen = _gen()
    a = _one_row(gen)
    text = gen.render(a)

    m = _TRAINER_LINE.search(text)
    assert m, "the trainer invocation line is missing from the rendered tail"
    tail = text[m.end():]

    # The echo must survive: it is the human-readable record in the log.
    assert 'echo "TRAINER exited with' in tail

    # A bare `echo "TRAINER exited with $?"` with nothing after it means the
    # script's own exit status is the echo's (always 0): this is the bug.
    # The fix must capture $? into a variable BEFORE the echo, then exit
    # with that variable -- never a bare, un-captured trailing echo.
    capture = re.search(r'^\s*([A-Za-z_][A-Za-z0-9_]*)=\$\?\s*$', tail, re.M)
    assert capture, (
        "the tail never captures $? into a variable right after the "
        "trainer runs; a trailing echo alone always exits 0 -- fix in "
        "render(), not in this test")
    var = capture.group(1)

    assert f"${var}" in tail or f"${{{var}}}" in tail, (
        f"captured variable {var!r} is never referenced again")

    exit_line = re.search(
        r'^\s*exit\s+"?\$\{?' + re.escape(var) + r'\}?"?\s*$', tail, re.M)
    assert exit_line, (
        f"the tail never exits with the captured code {var!r}; a launcher "
        f"that does not `exit \"${var}\"` reports COMPLETED 0:0 for a "
        f"crashed trainer regardless of the echo (jobs 66737, 66212, "
        f"66768, 66770, 66771, 66772)")

    # The capture must come strictly before the exit, and the echo must sit
    # between them so the log always shows the code before the job ends.
    echo_pos = tail.index('echo "TRAINER exited with')
    assert capture.start() < echo_pos < exit_line.start(), (
        "capture, echo and exit are out of order in the tail")


def test_the_tail_never_ends_on_a_bare_echo():
    """Regression guard for the exact defect: the script's last statement
    must not be the echo itself (that is the un-captured-$? bug verbatim)."""
    gen = _gen()
    for a in (gen.thesis_core_arms()[0], gen.orderonly_final_arms()[0]):
        text = gen.render(a).rstrip("\n")
        last_line = text.splitlines()[-1]
        assert not last_line.strip().startswith('echo "TRAINER exited'), (
            f"{a['name']}: the rendered script's last line is the trainer "
            f"echo -- its own exit status (0) is what sacct will see")
        assert last_line.strip().startswith("exit "), (
            f"{a['name']}: the rendered script must end with an explicit "
            f"`exit` of the trainer's captured code, not fall through")
