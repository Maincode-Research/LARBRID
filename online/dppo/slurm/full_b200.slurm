#!/bin/bash
#SBATCH --job-name=larbrid_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=resident
#SBATCH --gres=gpu:b200:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/yuanchu/projects/LARBRID/logs/%x_%A_%a.out
#SBATCH --error=/home/yuanchu/projects/LARBRID/logs/%x_%A_%a.err
#SBATCH --array=0-2

SEEDS=(1234 2314 3214)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== SLURM job started on $(hostname) at $(date) (array task ${SLURM_ARRAY_TASK_ID}, seed ${SEED}) ==="

mkdir -p /home/yuanchu/projects/LARBRID/logs

cd /home/yuanchu/projects/LARBRID/dppo
source /home/yuanchu/projects/LARBRID/.venv/bin/activate

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/yuanchu/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python script/run.py --config-name=ft_ppo_diffusion_mlp_img --config-dir=cfg/robomimic/finetune/square seed=${SEED}

echo "=== Job finished at $(date) with exit code $? ==="
