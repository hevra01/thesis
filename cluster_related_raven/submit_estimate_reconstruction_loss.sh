#!/bin/bash
#SBATCH -J reconstruct_tokens
#SBATCH -o /ptmp/hevrapetek/thesis/logs/current.out
#SBATCH -e /ptmp/hevrapetek/thesis/logs/current.err
#SBATCH --time=1-00:00:00
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
  experiment.reconstructed_data_path=/ptmp/hevrapetek/reconstruction_imagenet/train
  experiment.reconstruction_loss_output_path=/ptmp/hevrapetek/thesis/data/datasets/imagenet_reconstruction_losses_new/all.json
  experiment.register_token_path="/ptmp/hevrapetek/thesis/data/datasets/imagnet_register_tokens/imagnet_train_register_tokens.npz"
  experiment.dataset.root="/ptmp/hevrapetek/ILSVR2012"
)

python find_reconstruction_loss.py "${ARGS[@]}"
