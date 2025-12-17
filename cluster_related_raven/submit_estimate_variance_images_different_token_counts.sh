#!/bin/bash
#SBATCH -J reconstruct_tokens
#SBATCH -o /ptmp/hevrapetek/thesis/logs/current.out
#SBATCH -e /ptmp/hevrapetek/thesis/logs/current.err
#SBATCH --time=0-15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

# --- Environment setup ---
module purge
module load anaconda/3/2023.03
source ~/.bashrc
source activate /u/hevrapetek/conda-envs/thesis
source /ptmp/hevrapetek/thesis/.wandb_secrets.sh

cd /ptmp/hevrapetek/thesis

# --- Arguments for Python script ---
ARGS=(
  experiment=${EXPERIMENT_NAME:-estimate_variance_images_different_token_counts}
  experiment.data_path="/ptmp/hevrapetek/reconstruction_imagenet_stochastic/val"
  experiment.out_file="/ptmp/hevrapetek/thesis/variance_lpips_results.json"
)

python find_variance_images_different_token_counts.py "${ARGS[@]}"
