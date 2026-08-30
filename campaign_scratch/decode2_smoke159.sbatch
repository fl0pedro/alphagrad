#!/bin/bash
#SBATCH -J pal159
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/decode2_smoke159_%j.log
cd /Users/assmuth/dsnn/alphagrad
JAX_PLATFORMS=cpu uv run --no-sync python decode2_smoke159.py
