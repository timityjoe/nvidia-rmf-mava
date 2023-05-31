##########################################################
# Core Mava image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as mava-core
# FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04 as mava-core
# Prevent situation where we get stuck
ENV TZ=Asia/Kuala_Lumpur \
    DEBIAN_FRONTEND=noninteractive
# Flag to record agents
ARG record
# Ensure no installs try launch interactive screen
ARG DEBIAN_FRONTEND=noninteractive
# Update packages
RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python3.9 && \
    apt install -y python3.9-dev && \
    apt-get install -y python3-pip && \
    apt-get install -y python3.9-venv

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 10

# Check python v
RUN python -V

# Setup virtual env
RUN python -m venv mava
ENV VIRTUAL_ENV /mava
ENV PATH /mava/bin:$PATH
RUN pip install --upgrade pip setuptools wheel
# Location of mava folder
ARG folder=/home/app/mava
## working directory
WORKDIR ${folder}
## Copy code from current path
COPY . /home/app/mava
# For box2d
RUN apt-get install swig -y
## Install core dependencies + reverb.
# Mod by Tim:
RUN pip install -e .[reverb]
# RUN pip install -e .[dm-reverb==0.7.2]
## Optional install for screen recording.
ENV DISPLAY=:0
RUN if [ "$record" = "true" ]; then \
    ./bash_scripts/install_record.sh; \
    fi
EXPOSE 6006
##########################################################

# Jax Images
##########################################################
# Core Mava image
FROM mava-core as jax-core
# Jax gpu config.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
## Install core jax dependencies.
# Install jax gpu
RUN pip install -e .[jax]
# TODO (Ruan): This version is pinned now to avoid a breaking change in jaxlib.
# Unpin when the issue is resolved.
# Please see https://github.com/deepmind/dm-haiku/issues/565 for details.
RUN pip install --upgrade "jax[cuda]==0.3.24" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
##########################################################

##########################################################
# PZ image
FROM jax-core AS pz
RUN pip install -e .[pz]
# PettingZoo Atari envs
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y unrar-free
RUN pip install autorom
RUN AutoROM -v
##########################################################

##########################################################
# Flatland Image
FROM jax-core AS flatland
RUN pip install -e .[flatland]
# To fix module 'jaxlib.xla_extension' has no attribute '__path__'
RUN pip install cloudpickle -U
##########################################################
