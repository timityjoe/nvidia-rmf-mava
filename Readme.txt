
# See
# https://github.com/instadeepai/Mava
# Detailed installation Guide:
# https://github.com/instadeepai/Mava/blob/develop/docs/DETAILED_INSTALL.md
# Advanced Mava usage:
# https://github.com/timityjoe/nvidia-rmf-mava/tree/main/mava/advanced_usage

# Conda Setup
conda init bash
conda create --name conda39-mava python=3.9
conda activate conda39-mava
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/


# Pip setup (from pyproject.toml and setup.py)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113^C
python -m pip install .
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-dev.txt


# Training commands. See also: 
# /mava/configs/default_ff_ippo2.yaml
# /mava/configs/arch/anakin2.yaml
# /mava/configs/logger/base_logger2.yaml
# /mava/configs/logger/ff_ippo2.yaml
# /mava/configs/system/ff_ippo2.yaml
# /mava/configs/env/rware2.yaml
#
python3 -m mava.systems.ff_ippo env=rware
python3 -m mava.systems.ff_ippo_2 env=rware2 env/scenario=tiny-4ag



# Start Tensorboard
cd ./results
tensorboard --logdir=./ --port=8080


