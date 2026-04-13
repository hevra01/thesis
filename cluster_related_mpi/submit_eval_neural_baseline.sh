#!/bin/bash


#SBATCH --job-name lid_estimations       # Name of the job (shows up in squeue)
#SBATCH --output logs/1out.txt       # Stdout log file (in logs/ directory) (anything your script prints, like print()) will go to logs/output_<jobid>.txt
#SBATCH --error logs/1err.txt         # Stderr log file (errors go here) (any errors or exceptions) go to logs/error_<jobid>.txt

#SBATCH --partition gpu22              # Partition to submit to (e.g., gpu24 is H100, 80GB VRAM)
#SBATCH --gres gpu:a40:1                   # Request 1 GPU
#SBATCH --mem 64G                      # Amount of RAM to allocate
#SBATCH --cpus-per-task 4              # Number of CPU cores to allocate
#SBATCH --time 0-03:00:00              # Max wall time for the job (2 days)
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

EXPERIMENT_NAME=${EXPERIMENT_NAME:-eval_neural_baseline}

ARGS=(
    experiment=$EXPERIMENT_NAME
    experiment.experiment_name=normal_distro
    experiment.project_name=neural_baseline_evaluation
    experiment.group_name=conditional_Film_classification_lpips_based_on_token_count_extra_layer

    experiment.dataset.root=/scratch/inf0/user/mparcham/ILSVRC2012/
    experiment.dataset.split=val_categorized
    experiment.dataset.batch_size=1024

    experiment.filter_based_on=k_value  # recon_loss or k_value. determines the filter_key.
    experiment.model.task_type=classification
    experiment.model.num_classes=9
    experiment.model.checkpoint_path=neural_baseline/checkpoint/predict_token_count_filM/lpips_best_extra_layer.pt # predict_token_count for classification, predict_recon_loss for regression
    
    experiment.reconstruction_dataset.batch_size=64
    experiment.reconstruction_dataset.reconstruction_loss_key="LPIPS"
)
echo "ARGS: ${ARGS[@]}"

python -m neural_baseline.eval "${ARGS[@]}"