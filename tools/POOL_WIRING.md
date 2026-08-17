# Measurement pool: final wiring plan (2026-08-17)

Every design decision below is MEASURED, not assumed. Jobs cited inline.

## Settled facts

| fact | evidence |
|---|---|
| measure_wait is ~98% of the host callback; env compute is 2-6% | 61350/61368 prof counters |
| compile is 70-95% of a measure actor's host time, 20-90s/plan | wandb actor logs, `ALPHAGRAD_ACTOR_PROF_EVERY` |
| graphax trace+lower is 3.5% of compile (0.53s median) | 61445 — CPU-lowering pool design is DEAD |
| parallel-LLVM flag: -12% median, -35% heaviest, 0 regressions | 61445; SHIPPED in env.py `a9d499f` |
| compile time is structure-driven (3.3-45.6s across random orders) | 61445 — caches cannot hit; overlap is the lever |
| jit vs AOT: identical blocking, identical caching | `compiler.py:293/387` — the JIT fork is dead |
| in-process cross-id executable rebind DOES NOT work | 61455, 18/18 "replica assigned to cuda:0" (CPU and CUDA) |
| cross-PROCESS ship with CVD same-id pinning: bitwise identical | 61478, 3.4MB blob, 0.08s load, no recompile |
| process pool: 3.32x vs seq; latency ratio 0.995 med / 1.014 max; peak 1.000/1.000 | 61481, corrected `dev.clear_memory_stats` |
| 200-step walk after windows: median 7.6s/plan, +23% pool wall, windows stay <=1.022 | 61481 |
| walk-as-scan: rejected — a second per-plan compile of unique HLO doubles the bottleneck | analysis against 61445 numbers |

## Architecture (validated end to end by tools/bench_pool_proc.py, job 61481)

    K compile procs (CUDA_VISIBLE_DEVICES=0 — the trainer GPU; loads are
      transient there and it is never a measurement device)
      -> serialize_executable blobs, atomic tmp->rename publish
      -> measure procs, one per remaining GPU (CVD=i -> local id 0),
         atomic-rename claim (one winner, FIFO-ish, work-stealing)
      -> deserialize (0.08s) OUTSIDE the window
      -> clear -> barrier -> timed reps -> block -> peak   (the contract)
      -> optional 200-step walk AFTER the windows (quality slot 6,
         0.922 Pearson vs downstream accuracy)

## Wiring steps into ppo.py's measure path

1. Replace CpuApproximationActor's in-actor compile+measure with the
   two-role split above. Ray MAY remain as the process supervisor only
   (spawn / respawn / CVD pinning); all data-plane transport is blob files
   or plasma — the pure-JAX core is what 61481 validated.
2. Compile work by TICKET CLAIM (atomic rename), not static shard: a #104
   ptxas segfault then costs exactly one plan. Route the loss into the
   existing sentinel/drop path (#119: dropped as missing data, NEVER
   scored worst — scoring worst biases the reward and was already fixed
   once).
3. Supervisor: respawn dead workers; per-plan wall timeout (port the Ray
   pool's proven timeout/respawn policy); RSS cap per worker; orphaned
   .claim files reclaimed by lease timeout.
4. OPTIONAL, measured-ready: time the walk's 200 steps as 200 latency
   samples and DROP the separate 100-exec latency loop on walk arms (same
   executable, value-independent cost — per-step readings are 200 samples
   of ONE fact). Peak stays a single bare pre-walk window (the walk
   carries ~3x weight-size Adam state). Absolute latency shifts by the
   adam-step constant: PopArt absorbs it, but historical comparability
   breaks — one line in the campaign log when adopted.
5. Known-trap catalog, earned the hard way:
   - `XLA_PYTHON_CLIENT_PREALLOCATE=false` in every launcher (else
     nvidia-smi reads the 73.7GB preallocation ghost);
   - `dev.clear_memory_stats()` is a DEVICE method —
     `client.clear_memory_stats` does not exist, and a silent try/except
     around it voided an entire peak table;
   - TLM pins `SEQ=32 / DMODEL=128 / VOCAB=1024` or numbers are
     incomparable with every prior campaign.
