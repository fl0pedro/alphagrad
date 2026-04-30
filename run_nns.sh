#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=2-00:00:00
##SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

cd ~/dsnn
rm -rf t3.out
alias uv="~/.local/bin/uv"
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0,3 nohup uv run alphagrad/src/alphagrad/approx/ppo.py --top-n 20 --episodes 500 --example VmappedNeuralNetwork --exec-on-gpu --lambda-cmp 0.2696296296296296 --lambda-mem 0.26843657817109146 > t0.out 2>&1 &
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=1,3 nohup uv run alphagrad/src/alphagrad/approx/ppo.py --top-n 20 --episodes 500 --example VmappedNeuralNetwork --reward-type estimated --exec-on-gpu --lambda-cmp 1.1087533156498675 --lambda-mem 1.0145631067961165 > t1.out 2>&1 &
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=2,3 nohup uv run alphagrad/src/alphagrad/approx/ppo.py --top-n 20 --episodes 500 --example VmappedNeuralNetwork --reward-type empirical --exec-on-gpu --lambda-cmp 0.19523637675929267 --lambda-mem 2.3728070175438596 > t2.out 2>&1 &
wait
