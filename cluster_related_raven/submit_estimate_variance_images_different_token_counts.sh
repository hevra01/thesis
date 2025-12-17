#!/bin/bash
#SBATCH -J lpips_variance
#SBATCH -o /ptmp/hevrapetek/thesis/logs/lpips_%A_%a.out
#SBATCH -e /ptmp/hevrapetek/thesis/logs/lpips_%A_%a.err
#SBATCH --time=0-05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --array=0-8

# --------------------------------------------------
# Environment setup
# --------------------------------------------------
module purge
module load anaconda/3/2023.03
source ~/.bashrc
source activate /u/hevrapetek/conda-envs/thesis
source /ptmp/hevrapetek/thesis/.wandb_secrets.sh

cd /ptmp/hevrapetek/thesis

# --------------------------------------------------
# Map SLURM array index -> token count
# --------------------------------------------------
K_LIST=(1 2 4 8 16 32 64 128 256)
RECONST_K=${K_LIST[$SLURM_ARRAY_TASK_ID]}

echo "Running LPIPS variance for reconst_k=${RECONST_K}"

# --------------------------------------------------
# Run experiment
# --------------------------------------------------
python find_variance_images_different_token_counts.py \
  experiment.reconst_k=${RECONST_K} \
  experiment.data_path="/ptmp/hevrapetek/reconstruction_imagenet_stochastic/val" \
  experiment.out_file="/ptmp/hevrapetek/thesis/variance_lpips_results/reconst_${RECONST_K}.json"
