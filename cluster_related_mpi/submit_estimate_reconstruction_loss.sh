#!/bin/bash
# This script is a Slurm job submission script for running a Python script on a cluster.


# ---------------- Slurm job configuration ---------------- #
# #SBATCH is a directive for Slurm to configure the job.
# The lines below are Slurm directives that specify job parameters.



#SBATCH --job-name lid_estimations       # Name of the job (shows up in squeue)
#SBATCH --output logs/1out.txt       # Stdout log file (in logs/ directory) (anything your script prints, like print()) will go to logs/output_<jobid>.txt
#SBATCH --error logs/1err.txt         # Stderr log file (errors go here) (any errors or exceptions) go to logs/error_<jobid>.txt

#SBATCH --partition gpu22              # Partition to submit to (e.g., gpu24 is H100, 80GB VRAM)
#SBATCH --gres gpu:1                   # Request 1 GPU
#SBATCH --mem 64G                      # Amount of RAM to allocate
#SBATCH --cpus-per-task 4              # Number of CPU cores to allocate
#SBATCH --time 0-10:00:00              # Max wall time for the job (2 days)
#SBATCH --nodes 1                      # Number of nodes (machines)
#SBATCH --ntasks 1                     # Number of tasks (normally 1 for single GPU jobs)

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

ARGS=(
  experiment=${EXPERIMENT_NAME:-estimate_reconstruction_loss}
  experiment.reconstructed_data_path=/BS/data_mani_compress/work/thesis/imagenet_reconstructed_APC_true/val_categorized
  experiment.reconstruction_loss_output_path=/BS/data_mani_compress/work/thesis/thesis/data/datasets/imagenet_reconstruction_losses/val_categorized/all_APG_on.json
  experiment.register_token_path="/BS/data_mani_compress/work/thesis/thesis/data/datasets/imagnet_register_tokens/imagnet_val_register_tokens.npz"
  experiment.dataset.root="/scratch/inf0/user/mparcham/ILSVRC2012"
  experiment.dataset.split="val_categorized"
  "experiment.k_keep_list=[1, 2, 4, 8, 16, 32, 64, 128, 256]"
)

python find_reconstruction_loss.py "${ARGS[@]}"
