#!/bin/bash
# filepath: /BS/data_mani_compress/work/thesis/thesis/cluster_related_mpi/submit_eval_heuristic_baseline.sh
# =============================================================================
# Slurm job submission script for HeuristicTokenCountPredictor evaluation
# =============================================================================
# This script evaluates a trained lightweight MLP that predicts token counts from:
#   - Reconstruction loss (e.g., LPIPS)
#   - Optional features: LID, local_density, etc.
# No images are loaded - only precomputed scalar features.

# ---------------- Slurm job configuration ---------------- #
#SBATCH --job-name eval_heuristic_baseline
#SBATCH --output logs/eval_heuristic_baseline_output.txt
#SBATCH --error logs/eval_heuristic_baseline_error.txt

#SBATCH --partition gpu22
#SBATCH --gres gpu:a40:1                   # Only 1 GPU needed for lightweight MLP
#SBATCH --time=0-0:30:00                   # Short time - evaluation is fast
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

echo "[RUN] SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
echo "[RUN] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "[RUN] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[RUN] nvidia-smi -L:" && nvidia-smi -L || true

# =============================================================================
# Evaluation Configuration
# =============================================================================
RECON_LOSS_KEY="DINOv2FeatureLoss"

# --- Arguments for Hydra / Python module ---
ARGS=(
    # -----------------------------------------------------------------
    # Device settings
    # -----------------------------------------------------------------
    experiment.device="cuda:0"

    # -----------------------------------------------------------------
    # Checkpoint path (trained model to evaluate)
    # -----------------------------------------------------------------
    # either lpips_with_heuristic_best or LPIPS_only_recon_loss_best (which is the one trained without heuristic features, only recon loss)
    experiment.checkpoint_path="/BS/data_mani_compress/work/thesis/thesis/heuristic_baseline/checkpoint/DINOv2FeatureLoss_with_normalization_best_recon_loss_dino_dist.pt"

    # -----------------------------------------------------------------
    # Dataset settings (must match training config)
    # -----------------------------------------------------------------
    experiment.reconstruction_dataset.reconstruction_loss="${RECON_LOSS_KEY}"
    experiment.reconstruction_dataset.batch_size=512
    experiment.reconstruction_dataset.num_workers=4
    experiment.reconstruction_dataset.additional_feature_keys='["dino_dist"]' # '["lid", "local_density"]' or [] or '["dino_dist"]' depending on whether evaluating the model trained with or without heuristic features

    # -----------------------------------------------------------------
    # Model settings (must match training config)
    # -----------------------------------------------------------------
    experiment.model.num_classes=256
    experiment.model.num_additional_features=1 # either 0 or 2 (for lid and local_density)
    experiment.model.hidden_dim=32

    # -----------------------------------------------------------------
    # Evaluation settings
    # -----------------------------------------------------------------
    experiment.reconstruction_dataset.eval_per_class=true
    experiment.reconstruction_dataset.eval_by_recon_loss_ranges=true

    # -----------------------------------------------------------------
    # Experiment metadata
    # -----------------------------------------------------------------
    experiment.project_name="heuristic_baselines_evaluation"
    experiment.experiment_name="eval_heuristic_${RECON_LOSS_KEY}_normalization_dino_dist"
    experiment.group_name="heuristic_evaluation"
)

# (Optional) print final args for debugging
echo "[RUN] Hydra args:"
printf '  %q\n' "${ARGS[@]}"

# --- Run ---
# Note: Using single process since this is evaluation (no DDP needed)
HYDRA_FULL_ERROR=1 python -m heuristic_baseline.eval "${ARGS[@]}"
