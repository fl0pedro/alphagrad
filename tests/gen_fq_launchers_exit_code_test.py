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
import subprocess

_ALPHAGRAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEN = os.path.join(_ALPHAGRAD, "tools", "gen_fq_launchers.py")

_TRAINER_LINE = re.compile(
    r'^\s*src/alphagrad/approx/ppo\.py "\$\{ARGS\[@\]\}"\s*$', re.M)


def _gen(pairs: bool = False):
    old = os.environ.pop("THESIS_PAIRS", None)
    if pairs:
        os.environ["THESIS_PAIRS"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("gen_fq_launchers", _GEN)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.environ.pop("THESIS_PAIRS", None)
        if old is not None:
            os.environ["THESIS_PAIRS"] = old
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


def _shell_options(text):
    return "".join(line + "\n" for line in text.splitlines()
                   if line.startswith("set "))


def _bash(script):
    return subprocess.run(["bash", "-c", script], capture_output=True,
                          text=True, timeout=60)


def test_a_crashed_trainer_is_the_exit_code_of_the_rendered_launcher():
    gen = _gen()
    for a in (gen.thesis_core_arms()[0], gen.orderonly_final_arms()[0]):
        text = gen.render(a)
        m = _TRAINER_LINE.search(text)
        assert m, a["name"]
        r = _bash(_shell_options(text) + "(exit 3)" + text[m.end():])
        assert r.returncode == 3, (
            a["name"], r.returncode, r.stdout[-500:], r.stderr[-500:])


def test_a_paired_launcher_exits_with_the_worse_of_its_two_trainers():
    gen = _gen(pairs=True)
    pairs = gen.thesis_pair_arms()
    assert pairs, "the generator emits no paired launcher"
    text = gen.render(pairs[0])
    tail = text[text.index('wait "$PID_A"'):]
    for code_a, code_b in ((0, 3), (5, 3), (0, 0)):
        stub = (f"(exit {code_a}) &\nPID_A=$!\n"
                f"(exit {code_b}) &\nPID_B=$!\n")
        r = _bash(_shell_options(text) + stub + tail)
        assert r.returncode == max(code_a, code_b), (
            pairs[0]["name"], code_a, code_b, r.returncode,
            r.stdout[-500:], r.stderr[-500:])
