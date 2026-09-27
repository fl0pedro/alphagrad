"""Ticket dsnn-dfw.69, owner ruling 2026-09-20 -- ONE PROFILE FOR NN256, AND
TWO SEEDS PER 8-GPU NODE INSIDE ONE JOB.

Two facts are pinned here.

1. THE PROFILE IS THE ROW'S, NOT THE NODE'S.  A NN256 row used to take the
   size of whatever node the round robin gave it: on pgi15-gpu20 it rendered
   8 GPUs, 128 CPUs and --ray-measure 7 while its four sibling seeds rendered
   4, 64 and 3.  Latency measured under two fan-outs is not one distribution,
   so every NN256 row now renders the 4-GPU profile on every Blackwell node.

2. TWO ROWS ON ONE NODE ARE ONE SBATCH.  `/etc/slurm/epilog_reset_node.sh`
   kills every process of this user on a node when ANY job of theirs on it
   ends, so the second half of an 8-GPU node cannot be a second job.  The
   paired launcher runs both seeds concurrently inside one job, on disjoint
   GPUs and disjoint cores, and exits with the WORSE of the two trainer
   codes -- a half that crashed may not be hidden by a half that did not.
"""
from __future__ import annotations

import difflib
import importlib.util
import os
import re
import subprocess
import sys

import pytest

_ALPHAGRAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEN = os.path.join(_ALPHAGRAD, "tools", "gen_fq_launchers.py")
#: The stack every launcher here is rendered for: a generation-time input
#: with no default (owner ruling 2026-09-27).
STACK = "/Scratch/assmuth/mrg/test-stack"

#: The owner's numbers, typed here on purpose.
NODES = ("pgi15-gpu15", "pgi15-gpu16", "pgi15-gpu18", "pgi15-gpu19",
         "pgi15-gpu20")
EIGHT_GPU = ("pgi15-gpu19", "pgi15-gpu20")
NN256_GPUS = 4
NN256_CPUS = 64
#: A frozen round's memory; a row `thesis_arm` emits asks for the node's
#: (owner rulings 2026-09-25, sinfo RealMemory 770000 and 1540000 MB).
NN256_MEM = "400G"
NN256_ROW_MEM = "740G"
PAIR_MEM = "1480G"
NN256_ACTORS = "3"


def _load(pairs: bool):
    old = os.environ.pop("THESIS_PAIRS", None)
    os.environ["THESIS_PAIRS"] = "1" if pairs else "0"
    try:
        spec = importlib.util.spec_from_file_location("gen_fq_launchers", _GEN)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.environ.pop("THESIS_PAIRS", None)
        if old is not None:
            os.environ["THESIS_PAIRS"] = old
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load(pairs=True)


@pytest.fixture(scope="module")
def gen_off():
    return _load(pairs=False)


def _args(text: str) -> str:
    i = text.index("\nARGS=(\n")
    return text[i:text.index("\n)\n", i)]


def _nn256_rows(gen):
    """Every generated NN256 row on a Blackwell node: the matrix's own rows,
    the smoke's NN256 row and the five-seed order-only baseline."""
    return [a for a in gen.ARMS
            if a.get("thesis_target") == "nn256"
            and a.get("node") in gen.THESIS_NODE_GPUS
            and not a.get("paired") and not a.get("sweepl")
              and not a.get("sweepl2")]


# --------------------------------------------------- 1. the uniform profile

def test_the_node_list_is_the_five_cleared_blackwell_nodes(gen):
    assert gen.THESIS_NODES == gen.THESIS_NODES_ALL == NODES
    assert "pgi15-gpu17" not in gen.THESIS_NODES
    assert gen.ORDERONLY_FINAL_NODES == NODES
    for n in EIGHT_GPU:
        assert gen.THESIS_NODE_GPUS[n] == 8


def test_no_row_of_the_generator_targets_gpu17(gen):
    """gpu17 has no matched CUDA 12.9 ptxas or nvlink (job 66740 aborted 72).
    The whole generator, not one table: the arm record AND the rendered
    `#SBATCH -w` line."""
    offenders = [a["name"] for a in gen.ARMS if a.get("node") == "pgi15-gpu17"]
    assert not offenders, offenders
    for a in gen.ARMS:
        assert "#SBATCH -w pgi15-gpu17\n" not in gen.render(a, STACK), a["name"]


def test_every_nn256_row_renders_the_four_gpu_profile_on_every_node(gen):
    rows = _nn256_rows(gen)
    assert len(rows) >= 30, len(rows)
    seen_nodes = set()
    for a in rows:
        node = a["node"]
        seen_nodes.add(node)
        assert gen.thesis_row_gpus("nn256", node) == NN256_GPUS, a["name"]
        assert a["gpus"] == NN256_GPUS, a["name"]
        cli = dict(gen._merge_cli(a.get("cli", {})))
        assert cli["--ray-measure"] == NN256_ACTORS, a["name"]
        # Owner ruling 2026-09-23: 8 on every row `thesis_arm` emits; the
        # frozen rounds (the order-only baseline, the sweeps) keep their own
        # budget.
        frozen = any(a.get(k) for k in (
            "orderonly", "orderonly_final", "orderonly_tlm_final",
            "orderonly_rsnn", "sweepl", "sweepl2", "sweepl3"))
        b = (gen.FROZEN_CORE_BUDGET if frozen
             else gen.THESIS_CORE_BUDGET[NN256_GPUS])
        assert cli["--reserved-driver-cores"] == str(b["trainer"]), a["name"]
        want = (str(b["per_actor"]) if frozen
                else gen.THESIS_CORES_PER_ACTOR)
        assert cli["--cpu-cores-per-actor"] == want, a["name"]
        text = gen.render(a, STACK)
        assert f"#SBATCH --gres={gen.blackwell_gres(NN256_GPUS)}\n" in text, \
            a["name"]
        assert f"#SBATCH -c {NN256_CPUS}\n" in text, a["name"]
        want_mem = NN256_MEM if frozen else NN256_ROW_MEM
        assert f"#SBATCH --mem={want_mem}\n" in text, a["name"]
    # the profile is pinned ON EVERY NODE, so every node must carry one
    assert seen_nodes == set(NODES), sorted(seen_nodes)


def test_a_tlm_row_still_takes_the_node(gen):
    """The uniform profile is per TARGET, not a blanket rule: TLM keeps the
    node's own size, so an 8-GPU node still measures a TLM row with seven."""
    assert gen.THESIS_UNIFORM_GPUS == {"nn256": NN256_GPUS}
    for a in gen.ARMS:
        if a.get("thesis_target") != "tlm" or a.get("node") not in NODES:
            continue
        gpus = gen.THESIS_NODE_GPUS[a["node"]]
        assert a["gpus"] == gpus, a["name"]
        cli = dict(gen._merge_cli(a.get("cli", {})))
        assert cli["--ray-measure"] == gen.THESIS_RAY_MEASURE[gpus], a["name"]


# ------------------------------------------------------------- 2. the slots

def test_the_slot_ring_is_three_whole_nodes_and_two_halves_each(gen):
    assert gen.THESIS_SLOTS == (
        ("pgi15-gpu15", None), ("pgi15-gpu16", None), ("pgi15-gpu18", None),
        ("pgi15-gpu19", 0), ("pgi15-gpu19", 1),
        ("pgi15-gpu20", 0), ("pgi15-gpu20", 1))
    assert len(gen.THESIS_SLOTS) == 7


def test_every_half_row_names_the_pair_it_runs_inside(gen):
    halves = [a for a in gen.ARMS if a.get("paired_into")]
    assert halves
    pairs = {p["name"]: p for p in gen.thesis_pair_arms()}
    for a in halves:
        assert a["paired_into"] in pairs, a["name"]
        assert a["node"] in EIGHT_GPU, a["name"]
        # the file is still the readable record of the row, and it refuses
        text = gen.render(a, STACK)
        assert f"ABORT(74): {a['name']} runs as one half" in text, a["name"]
        assert 'if [ "${FQ_RELEASE_HALF:-0}" != "1" ]; then' in text, a["name"]
        assert "  exit 74" in text, a["name"]
    # every half of a pair points at it and no row points at two
    for p in pairs.values():
        pointing = {a["name"] for a in halves if a["paired_into"] == p["name"]}
        assert pointing == {h["name"] for h in p["halves"]}, p["name"]


# ---------------------------------------------------- 3. the paired launcher

def test_a_pair_is_two_seeds_of_one_arm_on_one_eight_gpu_node(gen):
    pairs = gen.thesis_pair_arms()
    assert pairs, "the generator emits no paired launcher"
    for p in pairs:
        assert p["node"] in EIGHT_GPU, p["name"]
        assert p["gpus"] == gen.THESIS_NODE_GPUS[p["node"]] == 8, p["name"]
        assert p["thesis_target"] == "nn256", p["name"]
        assert len(p["halves"]) == 2, p["name"]
        seeds = [h["seed"] for h in p["halves"]]
        assert len(set(seeds)) == 2, p["name"]
        assert p["name"] == gen.thesis_pair_name(
            p["thesis_arm"], p["thesis_target"], tuple(seeds)), p["name"]
        assert p["job"] == f"node-{p['node']}", p["name"]
        assert p["singleton"], p["name"]
    # a pair is NOT a matrix coordinate; its two halves are (core rows, or
    # the defense rows of dsnn-dfw.231, which are NN256 rows too)
    core = {a["name"] for a in gen.thesis_core_arms()
            + gen.thesis_defense_arms()}
    for p in pairs:
        assert p["name"] not in core, p["name"]
        for h in p["halves"]:
            assert h["name"] in core, h["name"]


def test_the_paired_launcher_asks_for_the_whole_node(gen):
    for p in gen.thesis_pair_arms():
        text = gen.render(p, STACK)
        assert f"#SBATCH -w {p['node']}\n" in text, p["name"]
        assert f"#SBATCH --gres={gen.blackwell_gres(8)}\n" in text, p["name"]
        assert f"#SBATCH -c {gen.BLACKWELL_CPUS[8]}\n" in text, p["name"]
        assert f"#SBATCH --mem={PAIR_MEM}\n" in text, p["name"]
        assert f"#SBATCH -J node-{p['node']}\n" in text, p["name"]
        assert "#SBATCH --dependency=singleton\n" in text, p["name"]


def test_the_two_halves_hold_disjoint_gpus_and_disjoint_cores(gen):
    for p in gen.thesis_pair_arms():
        text = gen.render(p, STACK)
        devices, cores = [], []
        for h in p["halves"]:
            d = [int(x) for x in h["devices"].split(",")]
            lo, hi = (int(x) for x in h["cores"].split("-"))
            assert len(d) == NN256_GPUS, p["name"]
            assert hi - lo + 1 == NN256_CPUS, p["name"]
            devices.append(set(d))
            cores.append(set(range(lo, hi + 1)))
            assert f"  export CUDA_VISIBLE_DEVICES={h['devices']}" in text, \
                p["name"]
            assert f"  taskset -c {h['cores']} " in text, p["name"]
        assert not devices[0] & devices[1], p["name"]
        assert not cores[0] & cores[1], p["name"]
        assert devices[0] | devices[1] == set(range(8)), p["name"]
        assert len(cores[0] | cores[1]) == gen.BLACKWELL_CPUS[8], p["name"]
        assert sorted(devices[0]) == [0, 1, 2, 3], p["name"]
        assert sorted(devices[1]) == [4, 5, 6, 7], p["name"]


def test_each_half_gets_its_own_ray_dir_log_file_and_wandb_run(gen):
    for p in gen.thesis_pair_arms():
        text = gen.render(p, STACK)
        rays, logs, names = set(), set(), set()
        for tag, h in zip(("A", "B"), p["halves"]):
            ray = "/tmp/ray_${SLURM_JOB_ID}_s%s" % h["seed"]
            assert f"  export RAY_TMPDIR={ray}" in text, p["name"]
            rays.add(ray)
            log = (f"> {gen.CAMPAIGN_RUNS}/{h['name']}"
                   f"_${{SLURM_JOB_ID}}_s{h['seed']}.log 2>&1 &")
            assert log in text, p["name"]
            logs.add(log)
            # the wandb run is the half's own --name, the matrix coordinate
            cli = dict(gen._merge_cli(h["cli"]))
            assert cli["--name"] == h["name"], p["name"]
            assert cli["--seed"] == h["seed"], p["name"]
            names.add(cli["--name"])
            assert f"ARGS_{tag}=(" in text, p["name"]
        assert len(rays) == len(logs) == len(names) == 2, p["name"]
        # both command lines are dry-parsed before either trainer starts
        assert text.count('"${ARGS_A[@]}" ||') == 1, p["name"]
        assert text.count('"${ARGS_B[@]}" ||') == 1, p["name"]


def test_the_pair_exits_with_the_worse_of_the_two_trainer_codes(gen):
    """A half that crashed may not be hidden by a half that did not: the job
    exits with the larger of the two captured codes (ticket dsnn-dfw.68 made
    a single launcher exit with its trainer's code; this is the same rule for
    two)."""
    for p in gen.thesis_pair_arms():
        text = gen.render(p, STACK)
        assert 'wait "$PID_A"' in text and "STATUS_A=$?" in text, p["name"]
        assert 'wait "$PID_B"' in text and "STATUS_B=$?" in text, p["name"]
        assert "TRAINER_STATUS=$STATUS_A" in text, p["name"]
        assert ('if [ "$STATUS_B" -gt "$TRAINER_STATUS" ]; then\n'
                "  TRAINER_STATUS=$STATUS_B\nfi\n") in text, p["name"]
        tail = text.rstrip("\n").splitlines()[-1]
        assert tail == 'exit "$TRAINER_STATUS"', (p["name"], tail)
        # and each half's own code reaches the log by name
        for tag, h in zip(("A", "B"), p["halves"]):
            assert re.search(
                r'echo "TRAINER half %s \(seed %s, %s\) exited with \$STATUS_%s"'
                % (tag, h["seed"], re.escape(h["name"]), tag), text), p["name"]


def test_the_paired_launcher_dumps_jax_devices_inside_each_half(gen):
    """The evidence that a half really sees four GPUs and not eight is in the
    log of the job itself, per half, after CUDA_VISIBLE_DEVICES is set.

    It must be JAX's device list and NOT `nvidia-smi`.  nvidia-smi asks the
    driver and enumerates the whole node whatever the mask says, so both
    halves printed the same eight lines and the mask was never evidenced;
    jax.devices() reads CUDA_VISIBLE_DEVICES, which is the thing under test.
    """
    for p in gen.thesis_pair_arms():
        text = gen.render(p, STACK)
        # the old call is gone from the halves
        assert "  nvidia-smi" not in text, p["name"]
        for tag, h in zip(("A", "B"), p["halves"]):
            # the echo still names the half and its mask
            assert (f'  echo "[half {tag}] seed {h["seed"]}'
                    f' CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES'
                    f' cores {h["cores"]} RAY_TMPDIR=$RAY_TMPDIR"') in text, \
                p["name"]
            dump = (f"import jax; print('[half {tag}] jax devices: ' + "
                    "', '.join(str(d.id) + ':' + d.platform + ':' + "
                    "d.device_kind for d in jax.devices()))")
            assert text.count(f'-c "{dump}"') == 1, p["name"]
            # under the mask: inside the half's subshell, after the export
            i_exp = text.index(f"  export CUDA_VISIBLE_DEVICES={h['devices']}")
            i_dump = text.index(dump)
            i_run = text.index(f"  taskset -c {h['cores']}")
            assert i_exp < i_dump < i_run, p["name"]


def test_the_half_device_dump_runs_on_a_node_with_no_gpu(gen):
    """CPU-SAFE.  The dump is the line a CPU test job runs too, so it must
    print a device list rather than raise when no GPU is visible.  Run the
    real snippet here, in a child with CUDA_VISIBLE_DEVICES empty."""
    p = gen.thesis_pair_arms()[0]
    text = gen.render(p, STACK)
    snippet = [ln for ln in text.splitlines() if "jax devices: " in ln][0]
    code = snippet.split(' -c "', 1)[1].rstrip('"')
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["JAX_PLATFORMS"] = "cpu"
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "[half A] jax devices: " in r.stdout, r.stdout
    assert ":cpu:" in r.stdout, r.stdout


def test_a_pair_holds_or_releases_both_halves_together(gen):
    """One job cannot be half held: a pair whose halves disagreed would
    either start a held run or hold a released one."""
    for p in gen.thesis_pair_arms():
        halves = {a["name"]: a for a in gen.ARMS if a.get("paired_into")}
        held = {bool(halves[h["name"]].get("held")) for h in p["halves"]}
        assert len(held) == 1, p["name"]
        assert bool(p.get("held")) == held.pop(), p["name"]


def test_thesis_pair_arm_refuses_a_mismatched_pair(gen):
    n0 = len(gen.ARMS)
    p = gen.thesis_pair_arms()[0]
    halves = {a["name"]: a for a in gen.ARMS if a.get("paired_into")}
    rows = [halves[h["name"]] for h in p["halves"]]
    try:
        with pytest.raises(gen.CampaignRowError):
            gen.thesis_pair_arm(rows[:1])
        other = dict(rows[1], thesis_arm="__not_the_same_arm__")
        with pytest.raises(gen.CampaignRowError):
            gen.thesis_pair_arm([rows[0], other])
        other = dict(rows[1], thesis_seed=rows[0]["thesis_seed"])
        with pytest.raises(gen.CampaignRowError):
            gen.thesis_pair_arm([rows[0], other])
    finally:
        del gen.ARMS[n0:]
    assert len(gen.ARMS) == n0


# ------------------------------------------- 4. the switch (dsnn-dfw.245)

def test_with_the_pairs_off_every_nn256_seed_is_a_single_row(gen, gen_off):
    assert gen_off.THESIS_PAIRS is False and gen.THESIS_PAIRS is True
    assert gen_off.thesis_pair_arms() == []
    assert not [a["name"] for a in gen_off.ARMS if a.get("paired_into")]
    on = {a["name"]: a for a in gen.ARMS}
    off = {a["name"]: a for a in gen_off.ARMS}
    assert set(off) == set(on) - {p["name"] for p in gen.thesis_pair_arms()}
    n_halves = 0
    for name, a in off.items():
        text = gen_off.render(a, STACK)
        assert "ABORT(74)" not in text and "exit 74" not in text, name
        ref = gen.render(on[name], STACK)
        if on[name].get("paired_into"):
            n_halves += 1
            # the off file is the half's file without its stub, line for line
            a_ln, b_ln = ref.splitlines(), text.splitlines()
            gone = []
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(
                    a=a_ln, b=b_ln, autojunk=False).get_opcodes():
                if op != "equal":
                    assert op == "delete", (name, op, b_ln[j1:j2])
                    gone += a_ln[i1:i2]
            assert any("ABORT(74)" in ln for ln in gone), name
            assert all(ln.startswith("#") or ln in ("", "fi", "  exit 74")
                       or "FQ_RELEASE_HALF" in ln or "ABORT(74)" in ln
                       for ln in gone), (name, gone)
            assert _args(text) == _args(ref), name
            assert '\n  src/alphagrad/approx/ppo.py "${ARGS[@]}"\n' in text, \
                name
            assert f"#SBATCH --gres={gen_off.blackwell_gres(NN256_GPUS)}\n" \
                in text, name
            assert f"#SBATCH -c {NN256_CPUS}\n" in text, name
            assert f"#SBATCH --mem={NN256_ROW_MEM}\n" in text, name
        else:
            assert text == ref, name
    assert n_halves == 2 * len(gen.thesis_pair_arms()) > 0
    for s in ("250197", "250198"):
        a = off[f"C_popart_nn256_s{s}"]
        assert a["node"] in EIGHT_GPU and not a.get("paired_into"), a["name"]


# -------------------------------------- 5. the devices a row names (dsnn-dfw.245)

_FROZEN = ("orderonly", "orderonly_rsnn", "orderonly_final",
           "orderonly_tlm_final", "sweepl", "sweepl2", "sweepl3")


def _array(text: str, tag: str) -> str:
    i = text.index(f"\nARGS_{tag}=(\n")
    return text[i:text.index("\n)\n", i)]


def test_each_half_names_its_own_trainer_gpu_and_measure_gpus(gen):
    from alphagrad.approx.common.device_guard import measure_devices

    want = {"A": ("0", "1,2,3"), "B": ("4", "5,6,7")}
    pairs = gen.thesis_pair_arms()
    assert pairs
    for p in pairs:
        text = gen.render(p, STACK)
        for tag, h in zip(("A", "B"), p["halves"]):
            cli = dict(gen._merge_cli(h["cli"]))
            assert (cli["--gpus"], cli["--measure-gpus"]) == want[tag], \
                (p["name"], tag)
            dev = h["devices"].split(",")
            assert [cli["--gpus"]] + cli["--measure-gpus"].split(",") == dev, \
                (p["name"], tag)
            arr = _array(text, tag)
            assert f"\n  --gpus {want[tag][0]}\n" in arr, (p["name"], tag)
            assert f"\n  --measure-gpus {want[tag][1]}\n" in arr, \
                (p["name"], tag)
            # ppo.py's own check takes the half's line under the half's mask
            assert measure_devices(
                int(cli["--ray-measure"]), cli["--measure-gpus"],
                trainer_gpus=cli["--gpus"], visible=h["devices"]) == \
                [int(d) for d in dev[1:]], (p["name"], tag)
        for flag in ("--gpus", "--measure-gpus"):
            assert flag in p["required_flags"], (p["name"], flag)


def test_every_single_row_names_gpu_0_and_the_measure_gpus_after_it(gen):
    rows = [a for a in gen.thesis_arms() if not a.get("paired")]
    frozen = [a for a in rows if any(a.get(k) for k in _FROZEN)]
    single = [a for a in rows if not any(a.get(k) for k in _FROZEN)]
    assert single and frozen
    for a in single:
        cli = dict(gen._merge_cli(a["cli"]))
        n = int(cli["--ray-measure"])
        assert n == a["gpus"] - 1, a["name"]
        assert cli["--gpus"] == "0", a["name"]
        assert cli["--measure-gpus"] == ",".join(
            str(k) for k in range(1, n + 1)), a["name"]
        for flag in ("--gpus", "--measure-gpus"):
            assert flag in a["required_flags"], (a["name"], flag)
        text = gen.render(a, STACK)
        assert f"\n  --measure-gpus {cli['--measure-gpus']}\n" in _args(text), \
            a["name"]
    for a in frozen:
        cli = dict(gen._merge_cli(a["cli"]))
        assert "--measure-gpus" not in cli and "--gpus" not in cli, a["name"]
        assert "--measure-gpus" not in a["required_flags"], a["name"]
        assert "--measure-gpus" not in gen.render(a, STACK), a["name"]
