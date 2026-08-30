#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu19
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -c 16
#SBATCH -t 00:40:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/decode3_trace_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ALPHAGRAD_SKIP_COUNT_OPS=1
S=/Users/assmuth/dsnn
D=$S/decode3_tiny.npz
[ -f $D ] || JAX_PLATFORMS=cpu uv run --no-sync python decode3_data.py $D 6 0,2,4,6 || exit 1
for ARM in new new_sizes sizes_only old_today; do
  JAX_PLATFORMS=cpu uv run --no-sync python decode3_face_train.py --data $D \
    --out /tmp/t.json --arm $ARM --batch 2 --n-test 2 --chunk-cap 4096 --trace-only \
    2>&1 | grep -E "TRACE OK|Error|Traceback|line [0-9]+," | tail -6
done
for ARM in new new_slots anchor no_ident no_dyn; do
  JAX_PLATFORMS=cpu uv run --no-sync python decode3_vertex_train.py --data $D \
    --out /tmp/t.json --arm $ARM --batch 2 --n-test 2 --trace-only \
    2>&1 | grep -E "TRACE OK|Error|Traceback|line [0-9]+," | tail -6
done
echo "########## TRACE ALL DONE ##########"
