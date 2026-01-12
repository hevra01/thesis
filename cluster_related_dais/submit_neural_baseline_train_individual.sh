#!/bin/bash
#SBATCH -J neural_baseline_train
#SBATCH -o /dais/u/hevrapetek/thesis_outer/thesis/logs/current.out
#SBATCH -e /dais/u/hevrapetek/thesis_outer/thesis/logs/current.err
#SBATCH --time=0-3:00:00
#SBATCH --nodes=1
#SBATCH --mem=100000

#SBATCH --gres=gpu:h200:2

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

# --- LPIPS bin ranges (min/max per job index) ---
# Source edges (11 values). We'll create 10 bins: [edges[i], edges[i+1]] for i=0..9.
EDGES=(
  0.02079272 0.12662399 0.23245525 0.33828652 0.44411778 0.54994905 \
  0.65578032 0.76161158 0.86744285 0.97327411 1.07910538
)

MIN_ERRORS=()
MAX_ERRORS=()
for i in {0..9}; do
  MIN_ERRORS+=("${EDGES[$i]}")
  next=$((i+1))
  MAX_ERRORS+=("${EDGES[$next]}")
done

# Resolve job index: prefer SLURM_ARRAY_TASK_ID, else first CLI arg, else 0.
JOB_INDEX=${SLURM_ARRAY_TASK_ID:-${1:-0}}

# Clamp and validate JOB_INDEX in [0, 9]
if [[ "$JOB_INDEX" -lt 0 || "$JOB_INDEX" -gt 9 ]]; then
  echo "[ERROR] JOB_INDEX=$JOB_INDEX out of range [0..9]." >&2
  exit 1
fi

MIN_ERR=${MIN_ERRORS[$JOB_INDEX]}
MAX_ERR=${MAX_ERRORS[$JOB_INDEX]}

echo "[RUN] Using LPIPS bin index $JOB_INDEX: min_error=$MIN_ERR, max_error=$MAX_ERR"
SIGMA=0.6

# --- Arguments for Hydra / Python module ---
# Start with the experiment choice
ARGS=( experiment=token_estimator_classification_neural_baseline_training_resnet
	   experiment.dataset.root="/dais/fs/scratch/hevrapetek/"
     experiment.dataset.split=train
     experiment.reconstruction_dataset.batch_size=1024
	   experiment.project_name=neural_baselines_new_lr
     experiment.reconstruction_dataset.min_error=${MIN_ERR}
     experiment.reconstruction_dataset.max_error=${MAX_ERR}
     experiment.experiment_name="classification_${SIGMA}"
     experiment.group_name="LPIPS_all_finetune_resnet_sigma${SIGMA}"
     experiment.training.loss_training.sigma=${SIGMA}
     experiment.checkpoint_path="neural_baseline/checkpoint/lpips_${MIN_ERR}"
 )


# (Optional) print final args for debugging
echo "[RUN] Hydra args:"
printf '  %q\n' "${ARGS[@]}"

# --- Run ---
HYDRA_FULL_ERROR=1 torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  -m neural_baseline.training \
  "${ARGS[@]}"
