#!/bin/bash
# ab_pomo_report.sh <out-dir>   e.g. ab_prof_out/nocache
D=$1
echo "=== $D : per-update phases (s) ==="
jq -rc '[.update, (.s_rollout//0), (.s_measure//0), (.s_update//0), (.s_confirm//0), (.unique//0), (.hits//0), (.n_feasible//0), (.skipped//"-")] | @tsv' $D/updates.jsonl \
  | awk 'BEGIN{printf "u\troll\tmeas\tupd\tconf\tuniq\thits\tfeas\tskip\n"}{printf "%s\t%.2f\t%.2f\t%.2f\t%.2f\t%s\t%s\t%s\t%s\n",$1,$2,$3,$4,$5,$6,$7,$8,$9}'

echo "=== $D : measurement rows by tag (wall_s / compile_s), fresh only ==="
jq -s 'map(select(.wall_s != null)) | group_by(.tag) | map({tag:.[0].tag, n:length,
        wall_total:(map(.wall_s)|add), wall_mean:((map(.wall_s)|add)/length),
        compile_total:(map(.compile_s//0)|add), compile_mean:((map(.compile_s//0)|add)/length)})' $D/measurements.jsonl

echo "=== $D : cached (no wall) rows by tag ==="
jq -s 'map(select(.wall_s == null)) | group_by(.tag) | map({tag:.[0].tag, n:length})' $D/measurements.jsonl

echo "=== $D : summary.json ==="
cat $D/summary.json 2>/dev/null | jq '{updates_done, stopped, unique_measurements, cache_hits, cache_hit_rate, measure_wall_s, wall_hours, worker_respawns}'
