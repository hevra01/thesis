#!/bin/bash
# This script is a Slurm job submission script for running a Python script on a cluster.


# ---------------- Slurm job configuration ---------------- #
# #SBATCH is a directive for Slurm to configure the job.
# The lines below are Slurm directives that specify job parameters.



#SBATCH --job-name dataset_prep_single     # Name of the job (shows up in squeue)
#SBATCH --output logs/output_prep_%j.txt   # Stdout log file (unique per job)
#SBATCH --error logs/error_prep_%j.txt     # Stderr log file (unique per job)

#SBATCH --partition a40              # Partition to submit to (e.g., gpu24 is H100, 80GB VRAM)
#SBATCH --gres gpu:1                   # Request 1 GPU
#SBATCH --cpus-per-task 4              # Number of CPU cores to allocate
#SBATCH --time=1-0:0:0              # Max valid value less than 7-00:00:00
#SBATCH --nodes 1                      # Number of nodes (machines)
#SBATCH --ntasks 1                     # Number of tasks (normally 1 for single GPU jobs)

# ---------------- Setup runtime environment ---------------- #


# Activate your environment
# Load conda correctly
source /home/hpc/v114be/v114be16/.conda/etc/profile.d/conda.sh
conda activate thesis


# Optionally load secrets or other setup files
source /anvme/workspace/v114be16-hevra/thesis/.wandb_secrets.sh

# ---------------- Execution ---------------- #
cd /anvme/workspace/v114be16-hevra/thesis


# ---------------- Run your code ---------------- #

# ----- Configurable via env -----
EXPERIMENT_NAME=${EXPERIMENT_NAME:-reconstruct_with_tokens}

# Class range chunking
BASE_START_CLASS=${385:-0}
CLASSES_PER_JOB=${CLASSES_PER_JOB:-20}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
START_CLASS=$(( BASE_START_CLASS + TASK_ID * CLASSES_PER_JOB ))
END_CLASS=$(( START_CLASS + CLASSES_PER_JOB ))

echo "Reconstructing classes [$START_CLASS, $END_CLASS) for experiment=$EXPERIMENT_NAME"

ARGS=(
  experiment=$EXPERIMENT_NAME
  experiment.start_class_idx=$START_CLASS
  experiment.end_class_idx=$END_CLASS
  experiment.output_path=anvme/workspace/v114be16-hevra/reconstruction_imagenet/train
  experiment.register_tokens_path=anvme/workspace/v114be16-hevra/imagenet_register_tokens/imagnet_train_register_tokens.npz
  experiment.data_root=anvme/workspace/v114be16-hevra/thesis/ILSVRC2012/train
)

python -u reconstruct_with_tokens.py "${ARGS[@]}"
