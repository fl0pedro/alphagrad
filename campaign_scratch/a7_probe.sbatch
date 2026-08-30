#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 0:40:00
#SBATCH -J a7probe
#SBATCH -o a7_probe.out
export PATH=$HOME/.local/bin:$PATH
export JAX_COMPILATION_CACHE_DIR=$HOME/dsnn/.jc_a7
cd $HOME/dsnn/alphagrad
uv run --no-sync python a7_probe_loss.py
