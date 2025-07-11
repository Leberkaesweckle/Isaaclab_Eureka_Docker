# Isaaclab_Eureka_Docker
Below you’ll find the basic code and workflow to create a Docker image containing IsaacSim 4.5, IsaacLab, and Eureka, and then run it on a cluster using Apptainer (formerly Singularity). If you need further assistance or have specific requirements, please feel free to ask!

# Creating Dockerfile:
The Isaaclab Dockerfile from Nvidia (https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-lab) did't work for me so I had to build my owm.
```bash
docker build -t isaaclab-eureka:latest .
```
```bash
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

##Conda installieren
##apt-get update && apt-get install -y wget
RUN apt-get update && apt-get install -y wget git
## mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/miniconda && \
    rm ~/miniconda.sh

###conda init all
ENV PATH="/opt/miniconda/bin:${PATH}"
##create conda environment 
##conda create -n env_isaaclab python=3.10
RUN conda create -n env_isaaclab python=3.10 -y && \
    conda init bash && \
    echo "conda activate env_isaaclab" >> ~/.bashrc

##Install CUDA 12
##pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

RUN conda run -n env_isaaclab pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
RUN conda run -n env_isaaclab pip install --upgrade pip

## Install additional dependencies, because of Errors
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1 \
      libglu1-mesa \
      libsm6 \
      libice6 \
      libxt6 \
      libxrender1 \
      libxrandr2 \
      libxi6 \
      libxfixes3 \
      libxcursor1 \
      libxinerama1 \
      libxxf86vm1 \
      libxss1 \
      libgtk2.0-0 \
      libnss3 \
      libasound2 \
      libx11-6 \
      libxext6 && \
    rm -rf /var/lib/apt/lists/*

##pip install isaacsim
RUN conda run -n env_isaaclab pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
##

##isaaclab
##git clone https://github.com/isaac-sim/IsaacLab.git

RUN git clone https://github.com/isaac-sim/IsaacLab.git

WORKDIR /IsaacLab

## sudo apt install cmake build-essential
RUN apt-get update && apt-get install -y cmake build-essential

## Install IsaacLab
RUN conda run -n env_isaaclab pip install -e source/isaaclab

## Install IsaacLab Eureka
## python -m pip install -e source/isaaclab_eureka
## https://github.com/isaac-sim/IsaacLabEureka

# Klone das Repo (im Arbeitsverzeichnis, z.B. /workspace)
RUN git clone https://github.com/isaac-sim/IsaacLabEureka

# Installiere das Package in der Conda-Umgebung
RUN conda run -n env_isaaclab pip install -e IsaacLabEureka/source/isaaclab_eureka

##Eula akzeptierencd

ENV ACCEPT_EULA=Y

ENV OMNIVERSE_ACCEPT_EULA=Y

ENV OMNI_KIT_ACCEPT_EULA=Y

### Installation of additional dependencies
# Installiere Python-Pakete mit exakten Versionen und downgrade pydantic
RUN conda run -n env_isaaclab pip install \
    "tensorboard==2.19.0" \
    "h5py==3.14.0" \
    "wandb==0.12.21" \
    "numba==0.61.2"     

# Stelle sicher, dass Pydantic 1.x benutzt wird!
RUN conda run -n env_isaaclab pip uninstall -y pydantic && \
    conda run -n env_isaaclab pip install "pydantic<2.0.0"

# Installiere Vulkan-Tools und numba-Dependencies
RUN apt-get update && apt-get install -y vulkan-tools

RUN conda run -n env_isaaclab pip install \
    "scikit-learn==1.7.0" \
    "rl_games==1.6.1" 

ENTRYPOINT ["bash"]
```
### Converting it into Apptainer
```bash
apptainer build isaaclab-eureka.sif docker-daemon://isaaclab-eureka:latest
```
### Running it on the Cluster
You have to create the folder for the bindings. To run it, you can use the following code. Please set the proxies; IsaacSim/IsaacLab needs a constant internet connection to run.
```bash
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --job-name=Isaac-Cartpole-Direct-v0
#SBATCH --export=NONE
#SBATCH --gres=gpu:a40:1
#SBATCH --output=log_sbatch/%x_%j.out
#SBATCH --error=log_sbatch/%x_%j.err

unset SLURM_EXPORT_ENV


# set number of threads to requested cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 

#set the proxys
export http_proxy=§§§§§§§§§§§§§§§§§§§§§§§§§§
export https_proxy=§§§§§§§§§§§§§§§§§§§§§§§§§§

echo "=== STARTE NETWORK TEST ==="

echo
echo "Ping-Test"
ping -c 3 www.google.com
ping -c 3 omniverse-content-production.s3-us-west-2.amazonaws.com

echo "=== END NETWORK TEST ==="

echo "Setting OpenAI API Key"

export OPENAI_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

echo "=== END OpenAI API Key ==="

echo "Starting IsaacLab training script"

ARGS="--task=Isaac-Cartpole-Direct-v0 --max_training_iterations=2000 --rl_library=rl_games --gpt_model=gpt-4.1"

apptainer exec --nv \
  --env http_proxy=§§§§§§§§§§§§§§§§§§§§§§§ \
  --env https_proxy=§§§§§§§§§§§§§§§§§§§§§§§ \
  --env PYTHONNOUSERSITE=1 \
  --bind $PWD/eureka_mounts/fake_deriveddatacache:/opt/miniconda/envs/env_isaaclab/lib/python3.10/site-packages/omni/cache/DerivedDataCache \
  --bind $PWD/eureka_mounts/fake_documents:/opt/miniconda/envs/env_isaaclab/lib/python3.10/site-packages/omni/data/documents \
  --bind $PWD/eureka_mounts/fake_kit_data:/opt/miniconda/envs/env_isaaclab/lib/python3.10/site-packages/omni/data/Kit \
  --bind $PWD/eureka_mounts/fake_deriveddatacache/Kit/106.5/d02c707b:/opt/miniconda/envs/env_isaaclab/lib/python3.10/site-packages/omni/cache/Kit/106.5/d02c707b \
  --bind $PWD/eureka_mounts/fake_ogn_generated:/opt/miniconda/envs/env_isaaclab/lib/python3.10/site-packages/omni/cache/ogn_generated \
  --bind $PWD/eureka_mounts/cache:/workspace/isaaclab/cache \
  --bind $PWD/eureka_mounts/cache/ov:/root/.cache/ov \
  --bind $PWD/eureka_mounts/cache/pip:/root/.cache/pip \
  --bind $PWD/eureka_mounts/cache/glcache:/root/.cache/nvidia/GLCache \
  --bind $PWD/eureka_mounts/cache/computecache:/root/.nv/ComputeCache \
  --bind $PWD/eureka_mounts/logs:/root/.nvidia-omniverse/logs \
  --bind $PWD/eureka_mounts/data:/root/.local/share/ov/data \
  --bind $PWD/eureka_mounts/outputs:/workspace/isaaclab/outputs \
  --bind $PWD/logs_eureka:/Isaaclab/logs \
  --bind $PWD/logs_eureka:/IsaacLab/IsaacLabEureka/logs \
  --bind $PWD/logs_eureka:/workspace/logs \
  --bind $PWD/eureka_programming/source/:/IsaacLab/IsaacLabEureka/source/ \
  --bind $PWD/eureka_programming/scripts/:/IsaacLab/IsaacLabEureka/scripts/ \
  --bind $PWD/eureka_mounts/sources/isaaclab_tasks/direct/direct/:/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/ \
  isaaclab-eureka.sif \
  bash -c "
    source /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate env_isaaclab && \
    cd /workspace && \
    python /IsaacLab/IsaacLabEureka/scripts/train.py $ARGS
  "
```
