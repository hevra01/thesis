#!/bin/bash
# =============================================================================
# Slurm job submission script for HeuristicTokenCountPredictor training
# =============================================================================
# This script trains a lightweight MLP that predicts token counts from:
#   - Reconstruction loss (e.g., LPIPS)
#   - Optional features: LID, local_density, etc.
# No images are loaded - only precomputed scalar features.

# ---------------- Slurm job configuration ---------------- #
#SBATCH --job-name heuristic_baseline
#SBATCH --output logs/heuristic_baseline_output.txt
#SBATCH --error logs/heuristic_baseline_error.txt

#SBATCH --partition gpu22
#SBATCH --gres gpu:a40:1                   # Only 1 GPU needed for lightweight MLP
#SBATCH --time=0-2:00:00                   # Shorter time - MLP trains fast
#SBATCH --nodes 1

# ---------------- Setup runtime environment ---------------- #

# Ensure shell is properly initialized (for mamba)
source ~/.bashrc

# Activate mamba environment
source /BS/data_mani_compress/work/miniforge3/etc/profile.d/conda.sh
conda activate dgm_geometry

# Load W&B secrets
source /BS/data_mani_compress/work/thesis/thesis/.wandb_secrets.sh

# ---------------- Run your code ---------------- #

# Move to your working directory
cd /BS/data_mani_compress/work/thesis/thesis

# Helpful runtime env for performance/stability
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NCCL_DEBUG=warn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "[RUN] SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
echo "[RUN] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "[RUN] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[RUN] nvidia-smi -L:" && nvidia-smi -L || true

NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}

# =============================================================================
# Training Configuration
# =============================================================================
SIGMA=0.2
RECON_LOSS_KEY="DINOv2FeatureLoss"

# --- Arguments for Hydra / Python module ---
ARGS=(
    # Use the heuristic baseline experiment config
    experiment=token_estimator_heuristic_baseline_training

    # -----------------------------------------------------------------
    # Optimizer settings
    # -----------------------------------------------------------------
    experiment.optimizer.lr=0.001

    # -----------------------------------------------------------------
    # Training settings
    # -----------------------------------------------------------------
    experiment.training.num_epochs=100
    experiment.training.loss_training_classification.sigma=${SIGMA}

    # -----------------------------------------------------------------
    # Dataset settings
    # -----------------------------------------------------------------
    experiment.reconstruction_dataset.reconstruction_loss="${RECON_LOSS_KEY}"
    experiment.reconstruction_dataset.batch_size=1024
    experiment.reconstruction_dataset.num_workers=4
    experiment.reconstruction_dataset.additional_feature_keys='["dino_dist"]' #'["lid", "local_density"]' or '["dino_dist"]'


    # -----------------------------------------------------------------
    # Model settings
    # -----------------------------------------------------------------
    experiment.model.num_classes=256
    experiment.model.num_additional_features=1
    experiment.model.hidden_dim=64

    # -----------------------------------------------------------------
    # Experiment metadata
    # -----------------------------------------------------------------
    experiment.project_name="heuristic_baselines_classification"
    experiment.experiment_name="heuristic_${RECON_LOSS_KEY}_sigma${SIGMA}_with_normalization_dino_dist"
    experiment.group_name="heuristic_token_prediction"

    # -----------------------------------------------------------------
    # Checkpoint paths
    # -----------------------------------------------------------------
    experiment.checkpoint_path_best="/BS/data_mani_compress/work/thesis/thesis/heuristic_baseline/checkpoint/${RECON_LOSS_KEY}_with_normalization_64best_recon_loss_dino_dist.pt"
    experiment.checkpoint_path_latest="/BS/data_mani_compress/work/thesis/thesis/heuristic_baseline/checkpoint/${RECON_LOSS_KEY}_with_normalization_64latest_recon_loss_dino_dist.pt"
)

# (Optional) print final args for debugging
echo "[RUN] Hydra args:"
printf '  %q\n' "${ARGS[@]}"

# --- Run ---
# Note: Using single GPU since this is a lightweight MLP (no need for DDP)
# But torchrun still works for consistency with other scripts
HYDRA_FULL_ERROR=1 torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    -m heuristic_baseline.training \
    "${ARGS[@]}"
