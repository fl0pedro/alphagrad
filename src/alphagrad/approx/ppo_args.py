"""JAX-free CLI argparser for `ppo_ray.py`.

Subset of `ppo.make_argparser`'s arguments — only the knobs the
first-cut Ray PPO trainer (`ppo_ray_worker.PPORayWorker`) actually
reads. The full single-process trainer in `ppo.py` carries ~80 args;
duplicating all of them here would just be drift bait. Add to this
file as the Ray trainer grows.

Kept JAX-free so the driver in `ppo_ray.py` can construct the parser
without triggering JAX init.
"""

from __future__ import annotations

import argparse


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ray-actor PPO trainer (first-cut) for vertex elimination.",
    )

    # Run / logging
    p.add_argument("--name", type=str, default="approx-ppo-ray")
    p.add_argument("--seed", type=int, default=250197)
    p.add_argument(
        "--wandb", type=str, default="offline",
        choices=["disabled", "offline", "online"],
    )
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--no-jit", action="store_true")
    p.add_argument(
        "--exec-on-gpu", action="store_true",
        help="Pin training to GPU 0 and the env eval callback to GPU 1.",
    )
    p.add_argument(
        "--cpu-actor-num-gpus", type=float, default=0.0,
        help="GPUs reserved per measurement (CpuApproximation) actor. >0 "
        "with --exec-on-gpu runs the Jacobian exec on GPU; use a fraction "
        "(e.g. 0.5) to co-locate the actor with the trainer on one GPU.",
    )
    p.add_argument(
        "--best-sequences-json", type=str, default="",
        help="Path to write the running per-channel + overall best "
        "sequences as JSON. Empty (default) → derive from the wandb "
        "run dir as ``<wandb_dir>/best_sequences.json``. The file holds "
        "one entry per reward channel + an ``overall`` entry, each "
        "with the full (a_i, b_i, c_i, r_i) reward tuple and the "
        "action sequence that produced it. Updated atomically every "
        "``--best-sequences-every`` episodes and at run end.",
    )
    p.add_argument(
        "--best-sequences-every", type=int, default=10,
        help="Cadence (in episodes) for writing the best-sequences "
        "JSON + logging the wandb table snapshot. Set to 0 to only "
        "write at run end.",
    )

    # Environment / reward
    p.add_argument("--example", type=str, default="Helmholtz")
    p.add_argument(
        "--cmp-type", type=str, default="flops",
        choices=["graphax", "flops", "latency"],
    )
    p.add_argument(
        "--mem-type", type=str, default="peak_memory",
        choices=["graphax", "bytes_accessed", "peak_memory", "xla_peak_memory"],
        help="Which measured channel drives the memory reward. "
        "'peak_memory' = ResourceMonitor high-water mark (idx 5; exact on GPU, "
        "sampled on CPU). 'xla_peak_memory' = deterministic XLA memory_analysis "
        "(idx 8; preferred on CPU). All channels are measured/logged regardless; "
        "this only selects the reward driver.",
    )
    p.add_argument(
        "--rewards", nargs="+", type=str,
        default=["cmp", "mem", "acc"], choices=["cmp", "mem", "acc", "all"],
        help="Reward channels. cmp/mem/acc = the 3 weighted slots (driver "
        "channel picked by --cmp-type/--mem-type/ALPHAGRAD_ACC_PROXY). ``all`` "
        "(or ALPHAGRAD_REWARD_ALL_CHANNELS=1) puts a uniform positive weight on "
        "EVERY applicable measured channel (flops, muls_adds, latency_ns, "
        "max_io, bytes, peak_memory, xla_peak_memory, cosine_sim, frob_residual "
        "+ bkstep_acc when ALPHAGRAD_BKSTEP=1); PopArt normalises each and the "
        "env's stored signs (costs negated, quality positive) make the single "
        "positive weight reward low-cost + high-fidelity.",
    )
    p.add_argument("--lambda-cmp", type=float, default=1.0)
    p.add_argument("--lambda-mem", type=float, default=1.0)
    p.add_argument(
        "--lambda-acc", type=float, default=1.0,
        help="Static weight on the cosine_sim (accuracy) reward channel. The "
        "scalar reward symlog's every channel, so cosine (symlog(1)~=0.69) is "
        "dwarfed by latency/peak-memory (symlog~=17-20) at the old hard-coded "
        "weight of 1.0 — set ~25 to bring cosine to a COMPARABLE magnitude so "
        "PPO actually trades accuracy against cost.",
    )
    # Both quality channels (cosine_sim direction + frob magnitude) are
    # now active rewards by default. cossim is gated by ``acc`` in
    # ``--rewards`` (weight = 1.0 when present). frob has its own
    # lambda; bumped from the prior 0.0 default so the Jacobian-error
    # magnitude contributes to the gradient alongside cossim.
    p.add_argument("--lambda-frob", type=float, default=1.0)
    p.add_argument(
        "--lambda-cossim-guide", type=float, default=0.0,
        help="Weight on the CAPPED-cossim GUIDE term added to the reward: "
        "reward += lambda_cossim_guide * min(cossim, C), where C is set via "
        "ALPHAGRAD_COSSIM_GUIDE_CAP (the trainability edge). Anti flat-zero-"
        "basin: when the acc channel is B_kstep(fracred) (ALPHAGRAD_ACC_PROXY="
        "bkstep) and the untrained policy sits at cos<=0, fracred has no "
        "gradient; this monotonic term (min, NOT clip-at-0) pulls the policy "
        "up to the edge, then the capped term goes constant and B_kstep "
        "dominates. 0 = off. Routes onto the cosine_sim channel weight.",
    )
    p.add_argument("--measure-latency", action="store_true")
    p.add_argument(
        "--latency-samples", type=int, default=1,
        help="LEGACY — superseded by --num-data-points × --reps-per-point. "
        "Retained for backward compat with non-PPO entry points.",
    )
    p.add_argument(
        "--num-data-points", type=int, default=5,
        help="Distinct args sets (data points) used for the noisy-channel "
        "measurement loop per env step. Default 5. Resampled once per "
        "rollout in the PPO worker (see generate_eval_samples call in "
        "run_rollout_and_train).",
    )
    p.add_argument(
        "--reps-per-point", type=int, default=4,
        help="Reruns of the compiled approx-fn against the same data point "
        "to expose timing/memory noise. Default 4. Total pool size per "
        "step is num_data_points × reps_per_point (default 5 × 4 = 20).",
    )
    p.add_argument(
        "--percentile-keep", type=float, default=0.60,
        help="Percentile (in [0, 1]) used to aggregate the noisy-channel "
        "pool. 0.60 = P60: 'slowest 60% latency / highest 60% memory / "
        "worst 60% frob'. Higher = more conservative / worse-case "
        "reward signal. Applies to latency_ns, peak_memory, "
        "frob_residual, cosine_sim.",
    )
    # --- Latency-measurement noise control (see env.py EnvConfig) ----------
    p.add_argument(
        "--latency-inner-reps", type=int, default=1,
        help="Time each latency reading over a tight inner loop of N "
        "back-to-back executions (one barrier, divided by N), via "
        "perf_counter. Amortizes per-call dispatch overhead — the dominant "
        "noise for sub-ms kernels. 1 = single call. Try 10-20.",
    )
    p.add_argument(
        "--latency-warmup", type=int, default=0,
        help="Discard the first K executions per data point before timing "
        "(first-touch / cache warm-up). Try 3.",
    )
    p.add_argument(
        "--latency-winsor", type=float, default=0.0,
        help="If >0, aggregate the latency pool with a symmetric winsorized "
        "mean at this trim fraction (e.g. 0.2) instead of --percentile-keep. "
        "Empirically the most reproducible/discriminative latency estimator.",
    )
    p.add_argument(
        "--measure-grad", action="store_true",
        help="Measure value_and_grad of the SCALAR training loss (the gradient "
        "that hits the optimizer) via graphax.value_and_grad instead of the "
        "full Jacobian via jacve. Quality channels (cosine_sim/frob_residual) "
        "then compare the approx gradient vs the exact gradient — a more "
        "faithful gradient-accuracy signal. Requires a loss-like example.",
    )
    p.add_argument(
        "--quant-once", action="store_true",
        help="Simplify the search space: only the FIRST Quant emitted per episode "
        "takes effect (later Quant ops dropped) — one global quantization choice "
        "(a single dtype, or none) instead of per-vertex repeated quant.",
    )
    p.add_argument(
        "--seed-vertices", action="store_true",
        help="Grad-mode only: treat the tangent + adjoint SEEDS as their own graph "
        "vertices (graphax.seed_vertices). The scalar loss becomes "
        "<ones/N, fn(p + t*dir)> with the tangent seed t appended as the LAST arg "
        "and the adjoint contraction explicit, so the elimination order chooses "
        "forward / reverse / cross-country seed timing. Same value/gradient as the "
        "plain scalar loss; only the action space (and graph) grows.",
    )
    p.add_argument(
        "--latency-timer", type=str, default="perf_counter",
        choices=["perf_counter", "rm"],
        help="Latency timer. 'perf_counter' (default): time an inner loop + one "
        "block_until_ready. 'rm': time via the fixed ResourceMonitor (barrier "
        "drains before stop; block inside the context) — yields latency AND "
        "peak memory from a single execution pass.",
    )
    p.add_argument(
        "--max-wall-seconds", type=float, default=0.0,
        help="Stop training cleanly once this many seconds of wall-clock have "
        "elapsed (final Pareto + all-time-candidate archives are dumped). 0 = "
        "unlimited (run to --episodes). Use with a larger SLURM --time so the "
        "run ends on the budget, not a hard kill.",
    )
    p.add_argument(
        "--spread-cpu-actors", action="store_true",
        help="Ray SPREAD scheduling for the CpuApproximationActors so "
        "they distribute across ALL cluster nodes (use every core on "
        "both the GPU node and the CPU node) instead of bin-packing onto "
        "one node. Pair with --cpu-cores-shared in the 2-node setup.",
    )
    p.add_argument(
        "--cpu-cores-shared", action="store_true",
        help="Multiple PPO jobs share one Ray cluster (the 2-node pooled "
        "setup). CpuApproximationActors then claim disjoint CPU-core "
        "slices from a per-node allocator (a cluster named actor) instead "
        "of the single-job static slice, so the concurrent jobs' actors "
        "don't collide on the same cores.",
    )
    # --- Off-policy replay (C-MORL V-trace actor-critic) -------------------
    # When >0, store each episode's per-env trajectories in a circular replay
    # buffer and train on minibatches SAMPLED from it (fresh + past episodes),
    # with V-trace value targets + clipped-IS (the PPO ratio) — i.e. genuine
    # off-policy actor-critic instead of on-policy PPO on the fresh rollout.
    p.add_argument(
        "--replay-buffer-size", type=int, default=0,
        help="Replay-buffer capacity in TRAJECTORIES (per-env episodes). "
        "0 = on-policy PPO on the fresh rollout (legacy). >0 = off-policy "
        "V-trace actor-critic sampling from the buffer.",
    )
    p.add_argument(
        "--replay-sample-trajs", type=int, default=0,
        help="Trajectories sampled from the buffer per update (0 = num_envs).",
    )
    p.add_argument(
        "--vtrace-rho-bar", type=float, default=1.0,
        help="V-trace ρ̄ clip on the policy IS weight (rewards correction).",
    )
    p.add_argument(
        "--vtrace-c-bar", type=float, default=1.0,
        help="V-trace c̄ clip on the trace IS weight (value propagation).",
    )
    # --- ASYNC PIPELINE (Stage 1, IMPALA-style decoupling) -----------------
    # Runs a SAMPLER actor (rollout + measurement -> push trajectories to the
    # shared replay buffer) CONCURRENTLY with the LEARNER actor (P3O updates
    # from the buffer + weight broadcast). measure(ep N+1) overlaps learn(ep N)
    # so the GPUs stay busy measuring while the learner updates. Requires --p3o
    # + --replay-buffer-size>0 (the off-policy IS/KL make the actor/learner lag
    # valid). OFF (default) = the synchronous run_rollout_and_train path.
    p.add_argument(
        "--async-pipeline", action="store_true",
        help="Enable the Stage-1 async pipeline: a sampler collects rollouts "
        "+ measures into the replay buffer while the learner runs P3O updates "
        "concurrently, syncing weights every --async-weight-sync-every "
        "updates. Requires --p3o and --replay-buffer-size>0. Off = sync.",
    )
    p.add_argument(
        "--async-updates-per-step", type=int, default=1,
        help="Learner P3O updates per pipeline step (per sampler traj). "
        "Higher = the learner drains the buffer faster relative to sampling.",
    )
    p.add_argument(
        "--async-weight-sync-every", type=int, default=1,
        help="Broadcast the learner's weights to the sampler every N pipeline "
        "steps. 1 = every step (tightest actor/learner lag). Larger = more "
        "off-policy staleness (bounded + corrected by P3O's IS ratio + KL).",
    )
    p.add_argument(
        "--async-warmup-trajs", type=int, default=2,
        help="Fill the replay buffer with this many sampler trajectories "
        "BEFORE the learner starts, so it never trains on an empty/tiny "
        "buffer. Default 2.",
    )
    # --- STAGE 2: compile/execute split via a dedicated compile-actor -------
    # Pulls the ~1.9s inline jacve compile OFF the measure critical path. A
    # compile-actor (co-resident on a measure GPU, CPU-bound compile, 0% GPU
    # compute) PRE-compiles the sampler's terminal orders into the shared
    # cluster CompileCacheCoordinator; the measure actors then LOAD the
    # executable (~10ms) instead of compiling inline. On a not-yet-compiled
    # order the measure falls back to inline compile (correctness) + counts the
    # miss. Requires --async-pipeline. OFF = Stage-1 (measure actors compile
    # inline).
    p.add_argument(
        "--async-compile-split", action="store_true",
        help="STAGE 2: dedicated compile-actor pre-compiles orders into the "
        "shared cache so measure actors exec-only (load+run, no inline "
        "compile). Requires --async-pipeline. Off = Stage 1.",
    )
    p.add_argument(
        "--cpu-cores-per-actor", type=int, default=0,
        help="Force each CpuApproximationActor to pin to exactly N CPU cores. "
        "1 = single-core-per-actor (cleanest per-reading latency CV, ~25-30× "
        "tighter, but single-threaded exec — set --num-cpu-workers ≈ #cores so "
        "every core hosts one actor and throughput is recovered by cross-actor "
        "parallelism). 0 = legacy auto slice (#cores // num_cpu_workers).",
    )
    p.add_argument(
        "--measure-queue", action="store_true",
        help="Terminal-step global measurement queue: decompose the "
        "N-env × n-point measurement into N×n_points per-(env,point) "
        "tasks dispatched via ray.util.ActorPool, so an env's points run "
        "in parallel and gated/cheap tasks free workers for stragglers "
        "(full core utilisation). Without it, each env measures its "
        "points sequentially on one actor's fixed cores, leaving most "
        "cores idle when envs are imbalanced (the common case once the "
        "FLOP-gate fires). Driver P60-aggregates per env across points.",
    )
    p.add_argument(
        "--reserved-driver-cores", type=int, default=8,
        help="CPU cores reserved for the main GPU/driver process (Ray "
        "driver, GCS, tokenization staging, GAE). The CPU-approx actors "
        "partition the REMAINING cores. Prevents the actors from "
        "blanketing all cores and starving the driver during "
        "rollout/update. 0 = no reservation (actors use all cores).",
    )
    p.add_argument(
        "--flop-gate-threshold", type=float, default=0.0,
        help="FLOP-gate: if an order's XLA cost_analysis FLOP count "
        "exceeds this, skip the expensive latency/memory/quality "
        "measurement entirely and assign FLOP/bytes-derived cost "
        "surrogates + worst-case quality. Catches the 500x-FLOP "
        "pathological orders (common early in training) whose ~50s "
        "uninterruptible exec the slow-exec cutoff can't avoid. 0 "
        "disables. Tune from the [DBG-env] flops= readings.",
    )
    p.add_argument(
        "--slow-exec-cutoff-seconds", type=float, default=8.0,
        help="Per-exec slow-order cutoff. If a single approx-Jacobian "
        "execution exceeds this, the policy's elimination order is "
        "pathologically expensive (the 500x-FLOP blowup common early in "
        "training); keep the samples gathered so far and stop instead of "
        "running all num_data_points*reps measurements. A good order "
        "(~4.5s exec) gets the full pool; a bad order (~44s) is capped at "
        "one sample. 0 disables (measure everything — can cost ~15 "
        "min/terminal-step on bad orders).",
    )

    # ------- 2-phase cost pipeline (cheap_first → full) -------
    # The full cost-channel path (XLA cost_analysis + ResourceMonitor +
    # compiled_exact comparison) is ~10× slower than the graphax-symbolic-
    # only path. ``--cost-pipeline-schedule cheap_first`` starts training
    # with only the cheap symbolic counts (muls_adds_fmas, max_io_sum)
    # plus the graphax-side jaxpr, runs until the policy stabilises
    # (KL window mean below threshold OR episode cutover, whichever
    # fires first), then swaps every CPU actor's env into the full path
    # for the rest of training. Wandb event ``phase/cutover_ep`` records
    # when the switch fires.
    p.add_argument(
        "--cost-pipeline-schedule", type=str, default="always_full",
        choices=["always_full", "cheap_first"],
        help="`always_full` (default): every step measures all 6 cost "
             "channels + cossim/frob. `cheap_first`: phase 1 = graphax "
             "symbolic only (target_fun=None — flops/latency/bytes/peak "
             "are 0, cossim/frob are 0); phase 2 enables the full path "
             "after the cutover trigger.",
    )
    p.add_argument(
        "--phase-cutover-ep", type=int, default=200,
        help="(cheap_first) Hard episode cutover — at episode N, force "
             "the swap regardless of KL. 0 disables this trigger.",
    )
    p.add_argument(
        "--phase-cutover-kl-threshold", type=float, default=0.01,
        help="(cheap_first) Cut over when the windowed-mean policy KL "
             "drops below this. 0 disables this trigger.",
    )
    p.add_argument(
        "--phase-cutover-kl-window", type=int, default=20,
        help="(cheap_first) Episode window over which KL is averaged "
             "before checking the threshold. Larger = less spiky trigger.",
    )

    # ------- RQ8 (Pitch A) — reward pipeline -------
    # All flags default to legacy behavior; --reward-pipeline pca2 opts
    # in. Full design: docs/experiments/reward_pipeline_pca.md.
    p.add_argument(
        "--reward-pipeline", type=str, default="legacy",
        choices=["legacy", "pca2"],
        help="Reward scalarization pipeline. ``legacy`` (default): "
             "current weighted-sum of 8 channels via --lambda-cmp/mem/frob. "
             "``pca2``: PCA-2 compresses the 6 cost channels into 2 "
             "decorrelated unit-variance latents, sums them; cossim/frob "
             "still enter via --lambda-frob.",
    )
    p.add_argument(
        "--pca-refit-every", type=int, default=100,
        help="(--reward-pipeline pca2) Episodes between eigendecomp refits "
             "of the EMA correlation matrix. Slow timescale so the latent "
             "reward stays continuous; Procrustes-aligned across refits.",
    )
    p.add_argument(
        "--pca-warmup-episodes", type=int, default=50,
        help="(--reward-pipeline pca2) Episodes of EMA-stats warmup before "
             "the first PCA refit. Until then, projection falls back to "
             "first-two-z-scored-channels.",
    )
    p.add_argument(
        "--running-max-channels", type=str, default="",
        help="Comma-separated channel names reduced via running-max instead "
             "of sum along the rollout, before GAE/aggregation. Typical: "
             "``peak_memory,max_io_sum`` since these are max-over-vertices "
             "quantities — summing them over the rollout overstates the "
             "true downstream cost. Implemented via "
             "common/telescoping.telescope_increments.",
    )

    # ------- RQ9 (Pitch B scaffold) — front coverage -------
    # Both flags default to legacy behavior (linear sum + Dirichlet);
    # the actual trainer-side hookup for Tchebycheff + Kronecker is
    # noted as outstanding in docs/experiments/pareto_front_tchebycheff.md
    # — these flags exist so RQ9 can flip them once the impl lands.
    p.add_argument(
        "--preference-sampler", type=str, default="dirichlet",
        choices=["dirichlet", "kronecker"],
        help="(--preference-conditioned) Preference-vector sampler. "
             "``dirichlet`` (default): current RQ7 behavior, O(N^-0.5) "
             "discrepancy. ``kronecker``: golden-ratio (K=2) / R_d "
             "low-discrepancy sequence, O(log N / N) — far more uniform "
             "coverage of the simplex per fixed env count. See "
             "common/preferences.kronecker_preferences.",
    )
    p.add_argument(
        "--scalarization", type=str, default="linear",
        choices=["linear", "tchebycheff"],
        help="Per-channel-reward → scalar reduction. ``linear`` (default): "
             "``sum_k w_k * r_k`` — current behavior, reaches only the "
             "convex hull of the Pareto front. ``tchebycheff``: augmented "
             "Tchebycheff ``-(max_k w_k * |r_k - z*|) - rho * sum|r_k - z*|`` "
             "— reaches concave regions per Miettinen 1999 §3.4.3. See "
             "common/scalarization.apply_scalarization.",
    )
    p.add_argument(
        "--tchebycheff-rho", type=float, default=0.05,
        help="(--scalarization tchebycheff) Augmentation weight on the "
             "``sum_k |r_k - z*|`` term. Rules out weakly-Pareto-optimal "
             "points; 0.05 per Steuer 1986.",
    )
    p.add_argument(
        "--intermediate-rewards",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit a reward vector at every rollout step (default: off — "
             "terminal-only). When off, non-terminal steps return a zero "
             "reward vector and skip the expensive per-step "
             "vertex_elimination_jaxpr; GAE back-propagates the terminal "
             "reward. Turn on for per-step cost rewards (more compute, more "
             "value-target variance).",
    )
    p.add_argument("--num-eval-samples", type=int, default=10)
    p.add_argument("--dataset", type=str, default="none")
    p.add_argument("--dataset-size", type=int, default=-1)

    # Rollout
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--ppo-epochs", type=int, default=4,
                   help="Number of passes over the rollout buffer per episode.")

    # Advantage normalisation strategy. `scalar` is the only path the Ray
    # trainer supports: PopArt value normalisation is unconditional there
    # and carries its own per-channel (mu, sigma), so it rejects `gdpo`
    # (which brings its own per-minibatch z-score — layering the two would
    # double-normalise). `gdpo` implements the per-channel z-score →
    # priority-weighted sum → batch-norm pipeline from the NVIDIA GDPO
    # paper (arXiv:2601.05242); it is retained for the single-process
    # ppo.py trainer only. Calibration is auto-skipped under `gdpo`
    # because per-minibatch z-scoring subsumes magnitude rescaling.
    p.add_argument(
        "--advantage-norm", type=str, default="scalar",
        choices=["gdpo", "scalar"],
        help="Advantage normalisation pipeline. `gdpo` (default): "
             "per-channel GAE + per-minibatch z-score per channel + "
             "priority-weighted sum + batch-norm. `scalar` (legacy): "
             "scalarise rewards via weights, symlog inside GAE, global "
             "advantage z-score over the whole rollout. Under `gdpo` "
             "the calibration phase is skipped (z-scoring is "
             "self-calibrating).",
    )
    p.add_argument(
        "--calibration-statistic", type=str, default="iqr",
        choices=["iqr", "mean_abs", "std"],
        help="Per-channel dispersion statistic used by the pre-training "
             "calibration phase to rescale reward weights. `iqr` "
             "(default, recommended): symlog-space IQR / 1.349 — robust "
             "to heavy-tail outliers. `mean_abs` (legacy): "
             "``|symlog(mean)|`` — normalises by the channel BIAS rather "
             "than its spread; kept for A/B regression. `std`: "
             "symlog-space std via the IQR proxy. Only relevant when "
             "calibration runs (advantage-norm=scalar AND "
             "calibrate-steps>0).",
    )

    # PPO hyperparameters
    p.add_argument("--ppo-eps", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument(
        "--entropy-coef",
        type=float,
        default=0.05,
        help="Entropy bonus coefficient in the PPO loss. Default bumped "
             "from 0.01 to 0.05 because the dynamic-substeps action "
             "space has 6 categorical heads (vertex / op_type / i / j / "
             "factor / quant) with joint entropy averaged across them — "
             "0.01 was insufficient (observed entropy collapsing 1.4 → "
             "0.4 by ep 13 in run-9s554e3x, where the per-head bonus "
             "drowns in value loss ≈ 4 and PPO clip loss ≈ 0.17).",
    )
    p.add_argument("--entropy-coef-final", type=float, default=0.001,
                   help="Final entropy coefficient after linear annealing.")
    p.add_argument("--gae-lambda", type=float, default=1.0)
    p.add_argument("--discount", type=float, default=1.0)

    # Conditioned-reward gating (the GDPO paper's "easy reward on hard
    # reward" trick). When the gate fails the easier channel is zeroed
    # for that transition, so the policy can't hill-climb the easy
    # signal without first satisfying the harder one.
    p.add_argument(
        "--reward-condition", nargs="*", type=str, default=[],
        help="Conditioned-reward gates of the form "
             "``<easier>:<harder>>=<threshold>`` or "
             "``<easier>:<harder><=<threshold>``. Channel names are "
             "those in the env's reward vector (muls_adds_fmas / flops "
             "/ latency_ns / max_io_sum / bytes_accessed / peak_memory "
             "/ cosine_sim / frob_residual). At every transition where "
             "``op(harder, threshold)`` is False, the easier channel's "
             "reward is set to zero before GAE. Used to force the "
             "policy to maximise the harder objective first — e.g. "
             "``--reward-condition 'flops:cosine_sim>=0.8'`` zeros the "
             "FLOPs reward unless the resulting Jacobian agrees with "
             "the exact one. Multiple gates compose AND-wise.",
    )

    # Optimizer
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-decay-min-mult", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--adam-eps", type=float, default=1e-8)

    # Dynamic substeps (Phase C). When --dynamic-substeps is on the agent
    # emits typed (op_type, i, j, factor) micro-actions per vertex which
    # env.micro_actions_to_rule_specs_jax translates into the legacy
    # rule_specs the env consumes. The first cut here is restricted to
    # ``max_substeps == 1`` — a single DIAG-or-END per vertex. Multi-
    # substep (sequence) support is a follow-up.
    p.add_argument(
        "--dynamic-substeps", action="store_true",
        help="Enable the typed micro-action policy (full multi-substep "
        "DIAG/COMPRESS/QUANT sub-episode per vertex).",
    )
    p.add_argument(
        "--max-substeps", type=int, default=16,
        help="Maximum number of typed micro-actions (DIAG/COMPRESS/QUANT) "
        "the policy may emit per vertex in the full dynamic-substeps scheme. "
        "Must be <= env.MAX_RULES_PER_VERTEX (16) to avoid silent truncation.",
    )
    p.add_argument(
        "--substep-curriculum", action="store_true",
        help="Ramp the per-vertex micro-action budget 0->CEIL in equal "
        "levels of STEP episodes: budget(ep)=min(CEIL, ep//STEP). Level 0 "
        "(budget=0) = pure exact elimination (order only). Runtime cap (no "
        "recompile). STEP=ALPHAGRAD_SUBSTEP_CURRICULUM_STEP (default 40), "
        "CEIL=ALPHAGRAD_SUBSTEP_CURRICULUM_CEIL (default --max-substeps). "
        "Also ALPHAGRAD_SUBSTEP_CURRICULUM=1. OFF = current behaviour.",
    )
    p.add_argument(
        "--factors", type=str, default="-1,2,3,4",
        help="Comma-separated DIAG factor choices the policy picks from. "
        "``-1`` resolves to ``gcd(n_i, n_j)`` per env edge; everything "
        "else is a literal divisor. When --variant is set, this is "
        "OVERRIDDEN by the variant's factor table at init (the agent is "
        "always built with the union factor table so curriculum stage "
        "transitions can mask via the policy logits rather than "
        "rebuilding the agent).",
    )

    # Variant / curriculum (mirrors mu0_args.py). PPO builds the agent
    # with the FULL action footprint (op_type ∈ {DIAG, COMPRESS, QUANT,
    # END}, full factor table) and masks per-stage in the act_step +
    # loss_fn — same trick MuZero uses for its prior-gating curriculum
    # so the policy weights survive stage transitions. ``full_curriculum``
    # is the recommended default: it auto-expands an empty --curriculum
    # to ``diag_gcd:N/3, diag_factor:N/3, full:N/3``.
    p.add_argument(
        "--variant",
        type=str,
        default="custom",
        choices=[
            "custom", "ve_only",
            "diag_gcd", "diag_factor",
            "compress", "compress_scalar",
            "quantize", "quant_smallest_float",
            "diag_compress", "diag_quant", "compress_quant",
            "full", "full_curriculum",
        ],
        help="Pre-canned action-space restriction. ``custom`` honours "
             "--factors / --dynamic-substeps directly. ``ve_only`` "
             "freezes op_type=END (pure vertex elimination). "
             "``diag_gcd`` allows DIAG with factor=-1 only. "
             "``diag_factor`` adds factors 2,3,4,8,16. ``compress`` / "
             "``quantize`` enable those op_types alongside DIAG. "
             "``full`` opens everything. ``full_curriculum`` is sugar "
             "for ``full`` plus the default 3-stage curriculum.",
    )
    p.add_argument(
        "--curriculum",
        type=str,
        default="",
        help="Curriculum of variants in sequence. Format: "
             "``stage1:N1,stage2:N2,...``. Empty = single training run "
             "on --variant (or the default 3-stage curriculum when "
             "--variant=full_curriculum and --curriculum is empty). "
             "Total episodes across stages must equal --episodes.",
    )

    # Model
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--embd-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument(
        "--policy-dims", type=str, default="128",
        help="Comma-separated MLP hidden dims for the vertex policy head.",
    )
    p.add_argument(
        "--value-dims", type=str, default="128",
        help="Comma-separated MLP hidden dims for the value head.",
    )

    return p
