#!/bin/bash
#SBATCH -J reconstruct_tokens
#SBATCH -o /ptmp/hevrapetek/thesis/logs/current.out
#SBATCH -e /ptmp/hevrapetek/thesis/logs/current.err
#SBATCH --time=0-8:00:00
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

# --- Pick k based on Slurm job-array index ---
# Submit with: sbatch --array=0-8 this_script.sh
k_keep_list=(1 2 4 8 16 32 64 128 256)

JOB_INDEX=${SLURM_ARRAY_TASK_ID:-0}
K=${k_keep_list[$JOB_INDEX]}

echo "SLURM_ARRAY_TASK_ID=${JOB_INDEX} -> K=${K}"

# --- Arguments for Python script ---
ARGS=(
  experiment=${EXPERIMENT_NAME:-estimate_reconstruction_loss}
  experiment.reconstructed_data_path=/ptmp/hevrapetek/reconstruction_imagenet_APC_true/train
  experiment.reconstruction_loss_output_path=/ptmp/hevrapetek/thesis/data/datasets/imagenet_reconstruction_losses/train/${K}.json
  experiment.register_token_path="/ptmp/hevrapetek/thesis/data/datasets/imagnet_register_tokens/imagnet_train_register_tokens.npz"
  experiment.dataset.root="/ptmp/hevrapetek/ILSVR2012"
  experiment.dataset.split="train"
  experiment.k_keep_list=[${K}]
)

python find_reconstruction_loss.py "${ARGS[@]}"
