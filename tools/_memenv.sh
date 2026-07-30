cd $HOME/dsnn/alphagrad
export PYTHONPATH=$HOME/dsnn/gxf/src:$HOME/dsnn/alphagrad/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=0
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_NN_HIDDEN=256
