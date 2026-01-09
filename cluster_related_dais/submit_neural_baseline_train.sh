#!/bin/bash
#SBATCH -J neural_baseline_train
#SBATCH -o /dais/u/hevrapetek/thesis_outer/thesis/logs/current.out
#SBATCH -e /dais/u/hevrapetek/thesis_outer/thesis/logs/current.err
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=100000M

# --- Environment setup ---
module purge
source ~/.bashrc
source /dais/u/hevrapetek/miniforge3/etc/profile.d/conda.sh
conda activate thesis_gpu_wheels
source /dais/u/hevrapetek/thesis_outer/thesis/.wandb_secrets.sh
cd /dais/u/hevrapetek/thesis_outer/thesis/

# Helpful runtime env for performance/stability
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NCCL_DEBUG=warn
# New name (old NCCL_ASYNC_ERROR_HANDLING is deprecated)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "[RUN] SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
echo "[RUN] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "[RUN] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[RUN] nvidia-smi -L:" && nvidia-smi -L || true

NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}

# --- Arguments for Hydra / Python module ---
# Start with the experiment choice
ARGS=( experiment=token_estimator_classification_neural_baseline_training
	   experiment.dataset.root="/dais/fs/scratch/hevrapetek/"
     experiment.dataset.split=train
	   experiment.project_name=neural_baselines
 )


# (Optional) print final args for debugging
echo "[RUN] Hydra args:"
printf '  %q\n' "${ARGS[@]}"

# --- Run ---
HYDRA_FULL_ERROR=1 torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  -m neural_baseline.training \
  "${ARGS[@]}"
