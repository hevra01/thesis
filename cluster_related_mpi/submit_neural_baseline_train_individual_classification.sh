#!/bin/bash
# This script is a Slurm job submission script for running a Python script on a cluster.


# ---------------- Slurm job configuration ---------------- #
# #SBATCH is a directive for Slurm to configure the job.
# The lines below are Slurm directives that specify job parameters.

#SBATCH --job-name lid_estimations       # Name of the job (shows up in squeue)
#SBATCH --output logs/output_bpp_100_epoch.txt       # Stdout log file (in logs/ directory) (anything your script prints, like print()) will go to logs/output_<jobid>.txt
#SBATCH --error logs/error_bpp_100_epoch.txt         # Stderr log file (errors go here) (any errors or exceptions) go to logs/error_<jobid>.txt

#SBATCH --partition gpu22             # Partition to submit to (e.g., gpu24 is H100, 80GB VRAM)
#SBATCH --gres gpu:a40:2                   # Request 4 GPUs on this node (edit as needed)
#SBATCH --time=0-5:00:00              # Max wall time for the job
#SBATCH --nodes 1                      # Single-node multi-GPU

# ---------------- Setup runtime environment ---------------- #

# Ensure shell is properly initialized (for mamba)
source ~/.bashrc

# Activate mamba (replace 'myenv' with your actual environment name)
source /BS/data_mani_compress/work/miniforge3/etc/profile.d/conda.sh
conda activate dgm_geometry

# Load W&B secrets (make sure this file is secure and not shared)
source /BS/data_mani_compress/work/thesis/thesis/.wandb_secrets.sh

# ---------------- Run your code ---------------- #

# Move to your working directory (adjust this to where your code lives)
cd /BS/data_mani_compress/work/thesis/thesis

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
ARGS=( 
     experiment=token_estimator_classification_neural_baseline_training_resnet

	   experiment.dataset_root="/dais/fs/scratch/hevrapetek/"

     experiment.reconstruction_dataset.batch_size=1024
     experiment.reconstruction_dataset.num_workers=16
     experiment.reconstruction_dataset.prefetch_factor=6

     experiment.reconstruction_dataset.filter_key="LPIPS"
     experiment.reconstruction_dataset.min_error=${MIN_ERR}
     experiment.reconstruction_dataset.max_error=${MAX_ERR}

	   experiment.project_name=neural_baselines_new_lr
     experiment.group_name="dais_LPIPS_range_finetune_resnet_train_val_sigma_${SIGMA}"
     experiment.experiment_name="min_${MIN_ERR}"

     experiment.checkpoint_path="neural_baseline/checkpoint/LPIPS/min_${MIN_ERR}.pt"
     experiment.training.loss_training.sigma=${SIGMA}
 )


# (Optional) print final args for debugging
echo "[RUN] Hydra args:"
printf '  %q\n' "${ARGS[@]}"

# --- Run ---
HYDRA_FULL_ERROR=1 torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  -m neural_baseline.training \
  "${ARGS[@]}"
