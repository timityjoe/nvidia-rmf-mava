#!/bin/bash

echo "Setting up Mava Environment..."
export MAVA_DIR="/media/timityjoe/DataSSD/workspace/marl/nvidia-rmf-mava"
conda activate conda39-mava
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/lib:/usr/lib:/usr/local/lib"
echo "$MAVA_DIR"
