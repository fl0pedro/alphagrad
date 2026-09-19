from __future__ import annotations

from typing import NamedTuple


class CoreLayout(NamedTuple):
    n_logical: int
    trainer: tuple[int, int]
    timing_actors: tuple[tuple[int, int], ...]
    oracle: tuple[int, int]
    spare: tuple[int, ...]


def node_core_layout(
    n_logical: int,
    n_timing_actors: int,
    *,
    trainer_cores: int,
    cores_per_actor: int,
    oracle_cores: int,
) -> CoreLayout:
    # Positions index the process's own sorted affinity mask, not raw CPU ids:
    # the SLURM mask on the 8-GPU nodes is 0-31,128-159.
    for name, value in (("n_logical", n_logical),
                        ("n_timing_actors", n_timing_actors),
                        ("trainer_cores", trainer_cores),
                        ("cores_per_actor", cores_per_actor),
                        ("oracle_cores", oracle_cores)):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive; got {value!r}")

    trainer = (0, int(trainer_cores))
    cursor = int(trainer_cores)
    actors = []
    for _ in range(int(n_timing_actors)):
        actors.append((cursor, int(cores_per_actor)))
        cursor += int(cores_per_actor)
    # The oracle actor pins itself to the LAST `oracle_cores` of the inherited
    # mask (common/grad_oracle_async.py), so the budget puts it there.
    oracle = (int(n_logical) - int(oracle_cores), int(oracle_cores))
    if cursor > oracle[0]:
        raise ValueError(
            f"the core budget does not fit {n_logical} logical CPUs: trainer "
            f"{trainer_cores} + {n_timing_actors}x{cores_per_actor} timing "
            f"actors = {cursor}, and the oracle needs {oracle_cores} at the "
            f"top")
    layout = CoreLayout(
        n_logical=int(n_logical), trainer=trainer,
        timing_actors=tuple(actors), oracle=oracle,
        spare=tuple(range(cursor, oracle[0])))
    check_disjoint(layout)
    return layout


def layout_slices(layout: CoreLayout) -> tuple[tuple[str, int, int], ...]:
    out = [("trainer", layout.trainer[0], layout.trainer[1])]
    for i, (base, width) in enumerate(layout.timing_actors):
        out.append((f"timing-actor-{i}", base, width))
    out.append(("oracle", layout.oracle[0], layout.oracle[1]))
    return tuple(out)


def check_disjoint(layout: CoreLayout) -> None:
    seen: dict[int, str] = {}
    for name, base, width in layout_slices(layout):
        if base < 0 or base + width > layout.n_logical:
            raise ValueError(
                f"{name} slice {base}..{base + width - 1} leaves the "
                f"{layout.n_logical} logical CPUs of the node")
        for pos in range(base, base + width):
            if pos in seen:
                raise ValueError(
                    f"core budget overlap at position {pos}: {seen[pos]} and "
                    f"{name} both hold it")
            seen[pos] = name


def core_ids(cpus, base: int, width: int) -> tuple[int, ...]:
    # The budget is positions; a Ray actor needs absolute cpu ids because the
    # mask it inherits from the narrowed trainer is not the node's.
    cpus = tuple(cpus)
    if base < 0 or base + width > len(cpus):
        raise ValueError(
            f"slice {base}..{base + width - 1} leaves the {len(cpus)} cpus "
            f"of this job")
    return cpus[base:base + width]


def describe(layout: CoreLayout) -> str:
    parts = [f"{name} {base}-{base + width - 1}"
             for name, base, width in layout_slices(layout)]
    if layout.spare:
        parts.append(f"spare {layout.spare[0]}-{layout.spare[-1]}")
    return f"{layout.n_logical} logical CPUs: " + ", ".join(parts)
