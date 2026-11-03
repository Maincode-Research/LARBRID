#!/bin/bash
#SBATCH --job-name=larbrid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-1
#SBATCH --reservation=gpu-4_gpu-13_gpu-16_gpu-18_gpu-21_gpu-22_gpu-23_gpu-28_reservation
#SBATCH --partition=256C8G1H_MI325X_Ubuntu22
#SBATCH --account=ck1
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --output=/shared/amdgpu/home/lyc1197023878_ck1/projects/LARBRID/logs/%x_%A_%a.out
#SBATCH --error=/shared/amdgpu/home/lyc1197023878_ck1/projects/LARBRID/logs/%x_%A_%a.err

mkdir -p /shared/amdgpu/home/lyc1197023878_ck1/projects/LARBRID/logs

SEEDS=(12 123)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

singularity exec \
    --bind /shared/amdgpu/home/lyc1197023878_ck1/projects/LARBRID:/workspace \
    --bind /shared/amdgpu/home/lyc1197023878_ck1/.mujoco:/root/.mujoco \
    --env LD_LIBRARY_PATH=/shared/amdgpu/home/lyc1197023878_ck1/.mujoco/mujoco210/bin:/shared/amdgpu/home/lyc1197023878_ck1/miniconda3/lib \
    --env MUJOCO_GL=osmesa \
    --env PYOPENGL_PLATFORM=osmesa \
    /shared/amdgpu/home/lyc1197023878_ck1/projects/LARBRID/larbrid-env.sif \
    bash -c "cd /workspace/dppo && source /workspace/.venv/bin/activate && python script/run.py --config-name=ft_dql_diffusion_mlp --config-dir=cfg/robomimic/finetune/transport seed=${SEED}"
