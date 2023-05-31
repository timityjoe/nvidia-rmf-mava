
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
###################################
Install from source via
https://github.com/instadeepai/Mava/blob/develop/examples/quickstart.ipynb
But without Jyupyter Notebook

Commands:
------------------------------
docker build . -t mava:tf-core

docker pull instadeepct/mava:jax-core-latest

docker run --gpus all -it --rm  -v $(pwd):${MAVA_DIR} -w ${MAVA_DIR} instadeepct/mava:jax-core-latest python examples/debugging/simple_spread/feedforward/decentralised/run_ippo.py --base_dir ${MAVA_DIR}/logs/

docker run --gpus all -it --rm  -v $(pwd):${MAVA_DIR} -w ${MAVA_DIR} instadeepct/mava:jax-core-latest python examples/flatland/feedforward/decentralised/run_ippo.py --base_dir ${MAVA_DIR}/logs/

python3 -m examples.petting_zoo.simple_spread.feedforward.decentralised.run_ippo
python3 -m examples.petting_zoo.butterfly.run_ippo_with_monitoring
python3 -m examples.flatland.feedforward.decentralised.run_ippo
------------------------------

To run from python3 (example)
make run run_mappo=examples/tf/petting_zoo/sisl/multiwalker/feedforward/decentralised/run_mappo.py
make run-tensorboard run_mappo=examples/tf/petting_zoo/sisl/multiwalker/feedforward/decentralised/run_mappo.py














