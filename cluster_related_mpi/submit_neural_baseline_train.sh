#!/bin/bash
# This script is a Slurm job submission script for running a Python script on a cluster.


# ---------------- Slurm job configuration ---------------- #
# #SBATCH is a directive for Slurm to configure the job.
# The lines below are Slurm directives that specify job parameters.

#SBATCH --job-name lid_estimations       # Name of the job (shows up in squeue)
#SBATCH --output logs/output_bpp_100_epoch.txt       # Stdout log file (in logs/ directory) (anything your script prints, like print()) will go to logs/output_<jobid>.txt
#SBATCH --error logs/error_bpp_100_epoch.txt         # Stderr log file (errors go here) (any errors or exceptions) go to logs/error_<jobid>.txt

#SBATCH --partition gpu22             # Partition to submit to (e.g., gpu24 is H100, 80GB VRAM)
#SBATCH --gres gpu:4                   # Request 4 GPUs on this node (edit as needed)
#SBATCH --time=0-8:00:00              # Max wall time for the job
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

# --- Arguments for Hydra / Python module ---
# Start with the experiment choice
ARGS=( experiment=token_estimator_classification_neural_baseline_training_resnet
     experiment.dataset.split=train
	   experiment.project_name=neural_baselines
	   experiment.reconstruction_dataset.batch_size=220
	   
 )


# (Optional) print final args for debugging
echo "[RUN] Hydra args:"
printf '  %q\n' "${ARGS[@]}"

# --- Run ---
HYDRA_FULL_ERROR=1 torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  -m neural_baseline.training \
  "${ARGS[@]}"
