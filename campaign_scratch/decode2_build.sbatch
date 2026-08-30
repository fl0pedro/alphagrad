#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu18
#SBATCH --mem=200G
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/decode2_build_%j.log
cd ~/dsnn/alphagrad
JAX_PLATFORMS=cpu uv run --no-sync python decode2_face_data.py /Users/assmuth/dsnn/decode2_ds_192.npz 192
