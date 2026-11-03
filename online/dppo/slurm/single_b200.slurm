#!/bin/bash
#SBATCH --job-name=larbrid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=resident
#SBATCH --gres=gpu:b200:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/yuanchu/projects/LARBRID/logs/%x_%j.out
#SBATCH --error=/home/yuanchu/projects/LARBRID/logs/%x_%j.err

mkdir -p /home/yuanchu/projects/LARBRID/logs

SEED=1234

cd /home/yuanchu/projects/LARBRID/dppo
source /home/yuanchu/projects/LARBRID/.venv/bin/activate

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/yuanchu/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python script/run.py --config-name=ft_ppo_diffusion_mlp_img --config-dir=cfg/robomimic/finetune/square seed=${SEED}
