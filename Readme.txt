
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

## Pull additional repos
vcs import < og_marl_repo.txt
pip install https://github.com/instadeepai/Mava/archive/refs/tags/0.1.2.zip


# Pip setup (from pyproject.toml and setup.py)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113^C
python -m pip install .
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-dev.txt
pip install chex


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
python3 -m mava.systems.ff_ippo_2 env=rware2 env/scenario=small-4ag
python3 -m mava.systems.ff_ippo_2 env=rware2 env/scenario=large-6ag


# Experience Recorder (Offline Data Generation)
python3 -m mava.advanced_usage.ff_ippo_store_experience_2 env=rware env/scenario=small-4ag
python3 -m mava.advanced_usage.ff_ippo_store_experience_2 env=rware2 env/scenario=small-4ag
python3 -m mava.advanced_usage.ff_ippo_store_experience_2 env=rware2 env/scenario=large-6ag


# Experience Playback (Offline Pretraining)
python3 -m mava.advanced_usage.train_offline_algo


# Start Tensorboard
cd ./results
tensorboard --logdir=./ --port=8080

# Some results:
large-6ag  num_envs:16  num_evaluation:1000  timetaken=2hrs20mins (no convergence)
large-6ag  num_envs:160  num_evaluation:1000  timetaken=12hrs00mins (no convergence)



