#!/bin/bash
# THE NIGHTLY COPY of the thesis run data.
# Ticket dsnn-dfw.4, owner ruling 2026-09-16: "/Scratch is not persistent.
# Nightly checksum-verified copy to /Users/assmuth/thesis-runs/."
#
# WHAT IT COPIES.  Every wandb run directory of the campaign stack whose run
# name is a thesis run name: the run directory itself, which is where the
# trainer puts the plan log (plan_log_<name>.jsonl), the front dumps
# (pareto_front.json, best_sequences.json), the checkpoints
# (ppo_ckpt_ep000000NNN/) and auto_stop.json -- common/checkpoint.run_directory
# is the one rule all four follow -- plus the slurm log of the job.
#
# THE RULES IT KEEPS.
#   1. A STALLED EXPORT MUST NOT BREAK IT.  The pgi15 home export refused
#      every write from 2026-09-13 to 2026-09-15.  So each step that touches
#      /Users/assmuth runs under `timeout`, a timeout is logged and the script
#      exits 0 with NO completion marker, and the next night tries again.
#   2. A COPY IS COMPLETE ONLY AFTER A CHECKSUM LIST VERIFIES.  The list is
#      taken on the SOURCE and checked against the DESTINATION with
#      `sha256sum -c`. Only then is COPY_COMPLETE written.
#   3. IT NEVER DELETES ANYTHING on the destination.
set -uo pipefail

SRC_WANDB=${SRC_WANDB:-/Scratch/assmuth/campaign/stack/alphagrad/wandb}
SRC_LOGS=${SRC_LOGS:-/Scratch/assmuth/campaign/runs}
DEST=${DEST:-/Users/assmuth/thesis-runs}
STATE=${STATE:-/Scratch/assmuth/thesis_copy}
LOG=$STATE/copy.log
# The run names of the matrix and of the smoke (tools/gen_fq_launchers.py:
# thesis_run_name).  Anything else in the wandb tree is another experiment.
NAME_RE=${NAME_RE:-^(A|B|C|C_popart|condC)_(nn256|tlm)_s2501[0-9][0-9]$|^smoke_(C_tlm|condC_nn256)$}
T=${T:-120}          # seconds any single home-export operation may take

mkdir -p "$STATE"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

say "=== nightly copy starting on $(hostname) ==="

# --- rule 1: prove the export accepts writes BEFORE anything else.
if ! timeout $T mkdir -p "$DEST" 2>/dev/null; then
  say "the home export did not accept mkdir $DEST within ${T}s -- nothing"
  say "copied, nothing marked; the next run tries again."
  exit 0
fi
PROBE=$DEST/.write_probe.$$
if ! timeout $T touch "$PROBE" 2>/dev/null; then
  say "the home export did not accept a write within ${T}s -- nothing copied,"
  say "nothing marked; the next run tries again."
  exit 0
fi
timeout $T rm -f "$PROBE" 2>/dev/null
say "the home export accepts writes"

COPIED=0; SKIPPED=0; FAILED=0

for RD in "$SRC_WANDB"/run-*; do
  [ -d "$RD/files" ] || continue
  F=$RD/files
  # THE RUN NAME, from the artefacts themselves rather than from a guess.
  NAME=""
  PL=$(ls "$F"/plan_log_*.jsonl 2>/dev/null | head -1)
  if [ -n "$PL" ]; then
    NAME=$(basename "$PL"); NAME=${NAME#plan_log_}; NAME=${NAME%.jsonl}
  elif [ -f "$F/config.yaml" ] && grep -q "^name:" "$F/config.yaml"; then
    # wandb writes the whole argument namespace into config.yaml at
    # init, so the run name is there from episode 0 -- before any plan
    # log exists.  A run that died early is still copied.
    NAME=$(sed -n "/^name:/{n;s/^ *value: *//p;q}" "$F/config.yaml")
  elif [ -f "$F/pareto_front.json" ]; then
    NAME=$(sed -n "s/.*\"run_name\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" "$F/pareto_front.json" | head -1)
  fi
  [ -n "$NAME" ] || continue
  echo "$NAME" | grep -Eq "$NAME_RE" || continue

  D=$DEST/$NAME
  # A finished run whose copy is already verified is not copied again.
  if timeout $T test -f "$D/COPY_COMPLETE" 2>/dev/null; then
    NEW=$(find "$F" -newer "$D/COPY_COMPLETE" -type f 2>/dev/null | head -1)
    if [ -z "$NEW" ]; then SKIPPED=$((SKIPPED+1)); continue; fi
  fi

  say "copying $NAME  (from $RD)"
  if ! timeout $T mkdir -p "$D/files" 2>/dev/null; then
    say "  the export stalled on mkdir $D -- left incomplete, will retry"
    FAILED=$((FAILED+1)); continue
  fi
  # An incomplete copy must never look finished.
  timeout $T rm -f "$D/COPY_COMPLETE" 2>/dev/null

  if ! timeout $((T*20)) rsync -a --no-perms --no-owner --no-group \
         "$F"/ "$D/files"/ >>"$LOG" 2>&1; then
    say "  rsync of $NAME did not finish -- left incomplete, will retry"
    FAILED=$((FAILED+1)); continue
  fi
  # the slurm log of every job of this run, beside the run directory
  for SL in "$SRC_LOGS"/"$NAME"_*.log; do
    [ -f "$SL" ] || continue
    timeout $T cp -p "$SL" "$D/" 2>/dev/null
  done

  # --- rule 2: the checksum list is taken on the SOURCE and checked on the
  #     DESTINATION.  Nothing is marked complete until sha256sum -c passes.
  MAN=$STATE/manifest_$NAME.sha256
  ( cd "$F" && find . -type f -print0 | sort -z \
      | xargs -0 sha256sum ) > "$MAN" 2>/dev/null
  if ! timeout $((T*20)) bash -c "cd \"$D/files\" && sha256sum -c --quiet \"$MAN\"" >>"$LOG" 2>&1; then
    say "  CHECKSUM MISMATCH for $NAME -- not marked complete, will retry"
    FAILED=$((FAILED+1)); continue
  fi
  N=$(wc -l < "$MAN")
  if ! timeout $T cp -p "$MAN" "$D/MANIFEST.sha256" 2>/dev/null; then
    say "  the export stalled writing the manifest of $NAME -- will retry"
    FAILED=$((FAILED+1)); continue
  fi
  if ! timeout $T bash -c "printf \"verified %s files on %s from %s\n\" \"$N\" \"$(date -u +%FT%TZ)\" \"$RD\" > \"$D/COPY_COMPLETE\"" 2>/dev/null; then
    say "  the export stalled writing COPY_COMPLETE for $NAME -- will retry"
    FAILED=$((FAILED+1)); continue
  fi
  say "  $NAME complete: $N files verified"
  COPIED=$((COPIED+1))
done

say "=== done: $COPIED copied, $SKIPPED already current, $FAILED to retry ==="
exit 0
