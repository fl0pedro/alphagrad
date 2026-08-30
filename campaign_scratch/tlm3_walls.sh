#!/bin/bash
# tlm3_: per-episode wall from the epoch-prefixed smoke logs.
# Each line is "<epoch> <text>"; episode boundaries are the [prof ep=N] lines
# that ppo.py emits once per episode under ALPHAGRAD_PROFILE=1.
R=/Users/assmuth/dsnn/alphagrad/tlm3_out
for tag in tlm2 tlm3; do
  f=$R/ppo_$tag.log
  [ -f "$f" ] || continue
  echo "===== $tag ($f) ====="
  echo "-- first/last timestamps --"
  head -1 $f | cut -d' ' -f1
  tail -1 $f | cut -d' ' -f1
  echo "-- episode markers --"
  grep -E "\[prof ep=|\[mem ep=" $f | awk '{print $1, $2, $3, $4, $5, $6}'
  echo "-- per-episode wall (delta between consecutive [prof ep=] lines) --"
  grep -E "\[prof ep=" $f | awk '{t=$1; if (prev) printf "  ep_gap %6d s\n", t-prev; prev=t}'
  echo "-- peak gpu mem (MiB) --"
  sort -n $R/gpumem_$tag.txt 2>/dev/null | tail -1
  echo "-- chunk warning (must be empty) --"
  grep -i "EXTEND_CHUNK" $f | head -2
done
f=$R/gaz_gaz3.log
if [ -f "$f" ]; then
  echo "===== gaz3 ($f) ====="
  head -1 $f | cut -d' ' -f1
  tail -1 $f | cut -d' ' -f1
  grep -E "episode|measurement|ep=" $f | tail -10
  sort -n $R/gpumem_gaz3.txt 2>/dev/null | tail -1
fi
