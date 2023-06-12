
Build and Installation steps 

###############################################################
    Jax - From Source (Install Cuda12, nvidia-driver-530 1st)
###############################################################

# https://jax.readthedocs.io/en/latest/developer.html#building-from-source
# 1) Build from source
git clone https://github.com/google/jax
git checkout -b v0.4.10 tags/jaxlib-v0.4.10
cd jax
sudo apt install g++ python python3-dev
pip install numpy wheel
pip install absl-py
python build/build.py --enable_cuda

# 2) Installs jax
pip install -e .
pip install -e /media/timityjoe/Data/workspace/marl/jax  

# 3a) Check: Running the Bazel tests
python build/build.py --configure_only
bazel test //tests:cpu_tests //tests:backend_independent_tests
bazel test //tests:gpu_tests --local_test_jobs=4 --test_tag_filters=multiaccelerator --//jax:build_jaxlib=false --test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform

NB_GPUS=2
JOBS_PER_ACC=4
J=$((NB_GPUS * JOBS_PER_ACC))
MULTI_GPU="--run_under $PWD/build/parallel_accelerator_execute.sh --test_env=JAX_ACCELERATOR_COUNT=${NB_GPUS} --test_env=JAX_TESTS_PER_ACCELERATOR=${JOBS_PER_ACC} --local_test_jobs=$J"
bazel test //tests:gpu_tests //tests:backend_independent_tests --test_env=XLA_PYTHON_CLIENT_PREALLOCATE=false --test_tag_filters=-multiaccelerator $MULTI_GPU

# 3b) Check: Running the Bazel tests
pip install -r build/test-requirements.txt

############################################
    Jax - Pip Install Precompiled (Cuda12)
############################################
# https://github.com/google/jax
# CUDA 12 installation
# pip installation: GPU (CUDA, installed via pip, easier)
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# pip installation: GPU (CUDA, installed locally, harder)
# Installs the wheel compatible with CUDA 12 and cuDNN 8.8 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# You can find your CUDA version with the command:
nvcc --version



###################################
   Mava
   - Don't use Docker Desktop
   - Use Mava tag 0.1.3,    docker build --target tf-core -t mava/tf:2.8.4-py38-ubuntu20.04 .
   - If development branch, docker build --target jax-core -t mava/tf:2.12.0-py39-ubuntu20.04 .
###################################
Install from source via
https://github.com/instadeepai/Mava/blob/develop/examples/quickstart.ipynb
But without Jyupyter Notebook

Check dependency:
------------------------------
ldd [OPTION] ...

Commands:
------------------------------
# docker build . -t mava:tf-core
# docker build . -t mava:jax-core
# docker pull instadeepct/mava:jax-core-latest
docker build . --target tf-core -t mava/tf:2.8.4-py38-ubuntu20.04

sudo docker run --gpus all -it --rm  -v $(pwd):/home/app/mava -w /home/app/mava instadeepct/mava:jax-core-latest python examples/debugging/simple_spread/feedforward/decentralised/run_ippo.py --base_dir /home/app/mava/logs/

sudo docker run --gpus all -it --rm  -v $(pwd):/home/app/mava -w /home/app/mava /var/lib/docker/mava:jax-core python examples/flatland/feedforward/decentralised/run_ippo.py --base_dir /home/app/mava/logs/

# Setup
docker context use default
whereis libnvidia-ml.so.1
sudo apt install nvidia-cuda-toolkit
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/compat
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/:/usr/local/cuda/lib64

# Cuda Image Test
docker context use default
# docker run -it nvidia/cuda:11.2.0-base-ubuntu20.04 bash
# docker run --gpus all -it nvidia/cuda:11.2.0-base-ubuntu20.04 bash
# docker run --rm --gpus all -it nvidia/cuda:11.2.0-base-ubuntu20.04 bash
docker run -it --rm --gpus all nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04
* Check that nvidia-smi, have Nvidia-Driver & CUDA
             nvcc -V, has nvidia cuda toolkit 

# Tensorflow Test
https://github.com/usr-ein/test-tensorflow
https://hub.docker.com/r/tensorflow/tensorflow/
https://github.com/nvidia/cuda-samples
docker run --gpus all -it --rm sam1902/test-tensorflow:latest
docker run -it --rm tensorflow/tensorflow bash
- Start a GPU container, using the Python interpreter.
docker run -it --rm -v $(realpath ~/notebooks):/tf/notebooks -p 8888:8888 tensorflow/tensorflow:latest-jupyter


# Conda Commands
# Setup
# https://utho.com/docs/tutorial/how-to-install-anaconda-on-ubuntu-20-04-lts/
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
bash Anaconda3-2023.03-1-Linux-x86_64.sh
	* Specify conda directory, ie:
	${DATA}/anaconda3
	*Do you wish the installer to initialize Anaconda3 by running conda init? [Set 'no'] 
	*Change via "conda config --set auto_activate_base false"
	* Add this to ~/.bashrc:
	export PATH=$DATA/anaconda3/bin:$PATH
	# If encountering "Your shell has not been properly configured to use 'conda activate'.", then "conda dactivate" from (base)
	source activate base	
conda init bash
conda create --name conda39-mava python=3.9
conda activate conda39-mava
conda deactivate
conda clean --all	# Purge cache and unused apps
condo info


# Tensorflow with Cuda GPU install (Conda)
# See
# https://utho.com/docs/tutorial/how-to-install-anaconda-on-ubuntu-20-04-lts/
# https://www.tensorflow.org/install/pip
conda install cuda -c nvidia
conda install -c nvidia cudnn
	* Set these env variables after CUDNN is installed
	CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib


# Tensorflow Python Test (Terminal)
# https://www.tensorflow.org/install/pip
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"


# Docker commands
# Copy files out Container - Desktop
* Clear out ~/.docker
rm -rf ./
docker context ls
docker image ls
sudo chmod 666 /var/run/docker.sock
docker cp bf77b897f29d:/home/app/mava/'/home/app/mava'/mava ~/mava_docker/mava


# Mava-Jax
(in conda virtual env) pip install -r requirements.txt
# docker build . -t mava:jax-core
# docker pull instadeepct/mava:jax-core-latest
docker build . --target tf-core -t mava/tf:2.8.4-py38-ubuntu20.04

docker context use default
# sudo docker run --gpus all -it instadeepct/mava:jax-core-latest bash
# sudo docker run --gpus all -it docker.io/library/mava:jax-core bash
sudo docker run --gpus all -it docker.io/mava/tf:2.8.4-py38-ubuntu20.04 bash
sudo docker run --rm --runtime=nvidia --gpus all -it docker.io/library/mava:jax-core bash


# Mava Tag 0.1.3
python3 -m examples.tf.debugging.simple_spread.recurrent.decentralised.run_madqn
python3 -m examples.tf.flatland.recurrent.decentralised.run_madqn


# Mava Development Branch
python3 -m examples.petting_zoo.simple_spread.feedforward.decentralised.run_ippo
python3 -m examples.petting_zoo.butterfly.run_ippo_with_monitoring
python3 -m examples.flatland.feedforward.decentralised.run_ippo
------------------------------

To run from python3 (example)
make run run_mappo=examples/tf/petting_zoo/sisl/multiwalker/feedforward/decentralised/run_mappo.py
make run run_ippo=examples/flatland/feedforward/decentralised/run_ippo.py

make run-tensorboard run_mappo=examples/tf/petting_zoo/sisl/multiwalker/feedforward/decentralised/run_mappo.py
make run-tensorboard run_ippo=examples/flatland/feedforward/decentralised/run_ippo.py

docker run --gpus all -p 6006:6006 -it --rm  -v /home/app/mava -w /home/app/mava docker.io/library/mava:jax-core /bin/bash -c "  tensorboard --bind_all --logdir  /home/app/mava/logs/ & python  --base_dir /home/app/mava/logs/; "














