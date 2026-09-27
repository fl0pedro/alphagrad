"""The stack is an input and the run tree is fixed (owner ruling 2026-09-27),
pinned on `tools/gen_fq_launchers.py`:

  1. RENDERING REFUSES WITHOUT A STACK.  /Scratch/assmuth/campaign/stack was
     two symlinks that agents repointed at a different staged directory for
     each batch, so a launcher's code was whatever the links said when its
     job started.  `render` raises StackError without a stack, `main` exits
     without writing a launcher, and a relative path, a path the launcher
     cannot carry unquoted and the retired symlinks are refused too.
  2. EVERY LAUNCHER NAMES ITS OWN CODE: -D, cd, PYTHONPATH, the pre-flight
     (66), the two-op test and the provenance lines all name the stack, the
     launcher echoes it before anything runs, and no rendered launcher
     contains /Scratch/assmuth/campaign/stack.
  3. THE RUN TREE: every launcher that runs a trainer exports
     WANDB_DIR=/Scratch/assmuth/campaign/runs before the trainer starts, so
     the run directories land in /Scratch/assmuth/campaign/runs/wandb and not
     in the stack's checkout.  No launcher reads <stack>/alphagrad/wandb,
     ppo.py passes wandb.init no directory of its own, and the nightly copy
     reads the run tree.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess

import pytest

_ALPHAGRAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEN = os.path.join(_ALPHAGRAD, "tools", "gen_fq_launchers.py")
_PPO = os.path.join(_ALPHAGRAD, "src", "alphagrad", "approx", "ppo.py")
_COPY = os.path.join(_ALPHAGRAD, "tools", "thesis_nightly_copy.sh")

# THE OWNER'S PATHS, TYPED HERE ON PURPOSE.
STACK = "/Scratch/assmuth/mrg/test-stack"
OTHER_STACK = "/Scratch/assmuth/mrg/stack-b8"
RETIRED = "/Scratch/assmuth/campaign/stack"
RUNS = "/Scratch/assmuth/campaign/runs"
WANDB_RUNS = RUNS + "/wandb"
_TRAINER = ('src/alphagrad/approx/ppo.py "${ARGS', "tools/smoke.sh",
            "tools/pool_liveness_gate.sh")


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_fq_launchers", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def texts(gen):
    return {a["name"]: gen.render(a, STACK) for a in gen.ARMS}


def _trainer_start(text: str) -> int:
    """Where the launcher first starts a trainer, or -1 when it runs none."""
    hits = [text.index(t) for t in _TRAINER if t in text]
    return min(hits) if hits else -1


# ------------------------------------------------- 1. no stack, no launcher

def test_render_refuses_without_the_stack(gen):
    assert issubclass(gen.StackError, ValueError)
    for a in (gen.ARMS[0], gen.campaign_arms()[0], gen.thesis_arms()[0]):
        for missing in (None, ""):
            with pytest.raises(gen.StackError) as e:
                gen.render(a, missing)
            assert "--stack" in str(e.value), a["name"]
            assert "2026-09-27" in str(e.value), a["name"]
        with pytest.raises(gen.StackError):
            gen.render(a)


@pytest.mark.parametrize("bad", [
    "relative/stack", "stack", "~/stack", "/Scratch/assmuth/mrg/a b",
    "/Scratch/assmuth/mrg/$STACK", "/Scratch/assmuth/mrg/x;rm",
    RETIRED, RETIRED + "/", RETIRED + "/alphagrad/..", RETIRED + "-b8",
])
def test_a_bad_or_retired_stack_is_refused(gen, bad):
    with pytest.raises(gen.StackError):
        gen.check_stack(bad)
    with pytest.raises(gen.StackError):
        gen.render(gen.ARMS[0], bad)


def test_a_staged_stack_is_taken_as_named(gen):
    assert gen.RETIRED_STACK == RETIRED
    assert gen.check_stack(STACK) == STACK
    assert gen.check_stack(OTHER_STACK + "/") == OTHER_STACK
    assert gen.check_stack("/Scratch/assmuth/mrg/blockprep/stack") \
        == "/Scratch/assmuth/mrg/blockprep/stack"


def test_main_refuses_without_the_stack_and_writes_nothing(gen, tmp_path,
                                                           capsys):
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SystemExit):
        gen.main(["--out", str(out)])
    assert "--stack" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        gen.main(["--dry-run", "--out", str(out), "--against",
                  str(tmp_path / "tree")])
    for bad in ("", "relative/stack", RETIRED):
        with pytest.raises(SystemExit) as e:
            gen.main(["--stack", bad, "--out", str(out)])
        assert "stack" in str(e.value), bad
    assert list(out.iterdir()) == []


def test_main_writes_the_launchers_of_its_stack(gen, tmp_path):
    """The write mode keeps its semantics and --out is where the launchers
    go, so a playground holds its own; --check then finds them current."""
    out = tmp_path / "playground"
    out.mkdir()
    assert gen.main(["--stack", STACK, "--out", str(out)]) == 0
    names = sorted(p.name for p in out.glob("fq_*.sbatch"))
    assert names == sorted(f"fq_{a['name']}.sbatch" for a in gen.ARMS)
    one = (out / f"fq_{gen.ARMS[0]['name']}.sbatch").read_text()
    assert one == gen.render(gen.ARMS[0], STACK)
    assert gen.main(["--stack", STACK, "--out", str(out), "--check"]) == 0
    # the same tree checked for ANOTHER stack is drift, launcher by launcher
    assert gen.main(["--stack", OTHER_STACK, "--out", str(out),
                     "--check"]) == 1


# ------------------------------------------- 2. every launcher names its code

def test_no_rendered_launcher_contains_the_retired_stack(gen, texts):
    assert len(texts) == len(gen.ARMS)
    for name, text in texts.items():
        assert RETIRED not in text, name
        assert "@AG_REPO@" not in text and "@GX_REPO@" not in text, name
        assert "@WANDB_RUNS@" not in text, name


def test_every_launcher_names_and_echoes_its_stack(gen, texts):
    ag, gx = f"{STACK}/alphagrad", f"{STACK}/graphax"
    for a in gen.ARMS:
        text = texts[a["name"]]
        assert f"#SBATCH -D {ag}\n" in text, a["name"]
        assert f"\ncd {ag}\n" in text, a["name"]
        assert (f"export PYTHONPATH={gx}/src:{ag}/src\n" in text
                or f'export PYTHONPATH="{gx}/src:{ag}/src"\n' in text), \
            a["name"]
        # the pre-flight's existence check (66) looks at this stack
        assert f'for P in "$PY" {ag}/src/alphagrad/approx/ppo.py \\\n' \
               f"         {gx}/src/graphax " in text, a["name"]
        assert "ABORT(66)" in text, a["name"]
        # the provenance line names the same two checkouts
        assert (f'echo "ag=$(git -C {ag} rev-parse --short HEAD)'
                f' gx=$(git -C {gx} rev-parse --short HEAD)"') in text, \
            a["name"]
        if a["kind"] != "cpu":
            assert f"{gx}/tests/misc/test_face_two_op_form.py" in text, \
                a["name"]
        # THE ECHO AT START: once, before the stack is checked and before
        # the interpreter every python line runs through is even bound.
        echo = f'echo "[cfg] stack {STACK}"\n'
        assert text.count(echo) == 1, a["name"]
        assert text.index(echo) < text.index(f"PY={gen.CAMPAIGN_PY}\n") \
            < text.index('for P in "$PY"'), a["name"]
        assert text.count(f"PY={gen.CAMPAIGN_PY}\n") == 1, a["name"]


def test_the_stack_is_the_only_input_that_moves(gen):
    """Rendered for two stacks, a launcher differs in the stack alone."""
    for a in gen.ARMS:
        one = gen.render(a, STACK)
        assert one.replace(STACK, OTHER_STACK) == gen.render(a, OTHER_STACK), \
            a["name"]


def test_the_tool_arm_greps_and_runs_the_tool_of_its_stack(gen, texts):
    tool = f"{STACK}/alphagrad/src/alphagrad/approx/tools/landscape_map.py"
    tool_arms = [a for a in gen.ARMS if a.get("needs_tool")]
    assert tool_arms
    for a in tool_arms:
        text = texts[a["name"]]
        assert f'if [ ! -f "{tool}" ]; then\n' in text, a["name"]
        assert f'FLAGSRC="{tool}"\n' in text, a["name"]
    face = texts["face_attrib"]
    assert f"TOOL={tool}\n" in face
    assert (f'ok = gx.startswith("{STACK}/graphax/") and '
            f'ag.startswith("{STACK}/alphagrad/")') in face


# ----------------------------------------------------------- 3. the run tree

def test_the_run_tree_is_fixed_and_admitted(gen):
    assert gen.CAMPAIGN_RUNS == RUNS
    assert gen.CAMPAIGN_WANDB_DIR == RUNS
    assert gen.CAMPAIGN_WANDB_RUNS == WANDB_RUNS
    assert "WANDB_DIR" in gen.STACK_ENV_NAMES
    assert "WANDB_DIR" in gen.CAMPAIGN_ENV_ALLOWED
    assert "WANDB_DIR" in gen.THESIS_ENV_ALLOWED
    assert "WANDB_DIR" not in gen.THESIS_TARGET_ENV_ALLOWED


def test_every_trainer_launcher_exports_wandb_dir(gen, texts):
    export = f"export WANDB_DIR={RUNS}\n"
    trainers = 0
    for a in gen.ARMS:
        text = texts[a["name"]]
        start = _trainer_start(text)
        if start < 0:
            continue
        trainers += 1
        assert text.count(export) == 1, a["name"]
        assert text.count("export WANDB_DIR=") == 1, a["name"]
        assert text.index(export) < start, a["name"]
    # the whole fleet but the two landscape tools runs a trainer
    assert trainers == len(gen.ARMS) - 2
    assert {a["name"] for a in gen.ARMS
            if _trainer_start(texts[a["name"]]) < 0} == {"w0_probe",
                                                         "face_attrib"}


def test_no_launcher_reads_a_stacks_wandb_tree(gen, texts):
    for name, text in texts.items():
        assert "/alphagrad/wandb" not in text, name
    assert f"\nW={WANDB_RUNS}\n" in texts["face_attrib"]


def test_ppo_passes_wandb_init_no_directory_of_its_own():
    """WANDB_DIR decides where the run directory goes only while nothing
    overrides it: wandb.init(dir=...) or settings=... would, and so would a
    WANDB_DIR written from the code."""
    tree = ast.parse(open(_PPO).read())
    inits = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "init" and isinstance(n.func.value, ast.Name)
             and n.func.value.id == "wandb"]
    assert len(inits) == 1
    kws = {k.arg for k in inits[0].keywords}
    assert "dir" not in kws and "settings" not in kws, kws
    approx = os.path.join(_ALPHAGRAD, "src", "alphagrad", "approx")
    for root, _dirs, files in os.walk(approx):
        for f in files:
            if f.endswith(".py"):
                src = open(os.path.join(root, f)).read()
                assert not re.search(r"environ\[.WANDB_DIR.\]\s*=(?!=)",
                                     src), f
                assert "environ.setdefault(\"WANDB_DIR\"" not in src, f


def test_the_nightly_copy_reads_the_run_tree(gen):
    src = open(_COPY).read()
    defaults = dict(re.findall(r"^(SRC_[A-Z]+)=\$\{\1:-([^}]*)\}$", src, re.M))
    assert defaults == {"SRC_WANDB": WANDB_RUNS, "SRC_LOGS": RUNS}
    assert defaults["SRC_WANDB"] == gen.CAMPAIGN_WANDB_RUNS
    assert defaults["SRC_LOGS"] == gen.CAMPAIGN_RUNS
    assert RETIRED not in src
    chk = subprocess.run(["bash", "-n", _COPY], capture_output=True, text=True)
    assert chk.returncode == 0, chk.stderr
