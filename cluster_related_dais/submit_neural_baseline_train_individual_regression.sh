#!/bin/bash
#SBATCH -J neural_baseline_train
#SBATCH -o /dais/u/hevrapetek/thesis_outer/thesis/logs/current.out
#SBATCH -e /dais/u/hevrapetek/thesis_outer/thesis/logs/current.err
#SBATCH --time=0-5:00:00
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
K_VALUES=(1 2 4 8 16 32 64 128 256)

MIN_Ks=()
MAX_Ks=()
for k in "${K_VALUES[@]}"; do
  MIN_Ks+=("$k")
  MAX_Ks+=("$k")
done
# Resolve job index: prefer SLURM_ARRAY_TASK_ID, else first CLI arg, else 0.
JOB_INDEX=${SLURM_ARRAY_TASK_ID:-${1:-0}}

# Clamp and validate JOB_INDEX in [0, 9]
if [[ "$JOB_INDEX" -lt 0 || "$JOB_INDEX" -gt 9 ]]; then
  echo "[ERROR] JOB_INDEX=$JOB_INDEX out of range [0..9]." >&2
  exit 1
fi

MIN_K=${MIN_Ks[$JOB_INDEX]}
MAX_K=${MAX_Ks[$JOB_INDEX]}

echo "[RUN] Using LPIPS bin index $JOB_INDEX: min_error=$MIN_K, max_error=$MAX_K"

# --- Arguments for Hydra / Python module ---
# Start with the experiment choice
ARGS=( 
     experiment=token_estimator_classification_neural_baseline_training_resnet

	   experiment.dataset_root="/dais/fs/scratch/hevrapetek/"

     experiment.reconstruction_dataset.batch_size=1024
     experiment.reconstruction_dataset.min_error=${MIN_K}
     experiment.reconstruction_dataset.max_error=${MAX_K}

	   experiment.project_name=neural_baselines_regression_recon_loss_prediction
     #experiment.group_name="regression__${MIN_K}"
     experiment.experiment_name="min_${MIN_K}"
     experiment.reconstruction_dataset.filter_key="k_value"
     experiment.task_type=regression
     experiment.model.use_condition=false
     experiment.checkpoint_path="neural_baseline/checkpoint/regression/min_${MIN_K}.pt"
 )


# (Optional) print final args for debugging
echo "[RUN] Hydra args:"
printf '  %q\n' "${ARGS[@]}"

# --- Run ---
HYDRA_FULL_ERROR=1 torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  -m neural_baseline.training \
  "${ARGS[@]}"
