set -x
cd ~/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/.jaxcache
D=/Users/assmuth/dsnn/decode2_ds_192.npz
for V in fold boundary split_part split_auth mix_nobias mix_adj; do
  uv run --no-sync python decode2_vertex_train.py --data $D --out /tmp/d2v_$V.json \
    --variant $V --steps 1 --eval-every 1 --batch 4 2>&1 | grep -Ev "Triton|^W0|^step" | tail -6
done
for R in today_bnd endpoints endpoints_sizes sizes_only; do
  uv run --no-sync python decode2_face_train.py --data $D --out /tmp/d2f_$R.json \
    --arm $R --steps 1 --eval-every 1 --batch 4 --chunk-cap 2048 2>&1 | grep -Ev "Triton|^W0|^step" | tail -4
done
