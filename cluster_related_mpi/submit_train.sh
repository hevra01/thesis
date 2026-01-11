#!/bin/bash
# This script is a Slurm job submission script for running a Python script on a cluster.


# ---------------- Slurm job configuration ---------------- #
# #SBATCH is a directive for Slurm to configure the job.
# The lines below are Slurm directives that specify job parameters.



#SBATCH --job-name run_diffusion       # Name of the job (shows up in squeue)
#SBATCH --output logs/output.txt       # Stdout log file (in logs/ directory) (anything your script prints, like print()) will go to logs/output_<jobid>.txt
#SBATCH --error logs/error.txt         # Stderr log file (errors go here) (any errors or exceptions) go to logs/error_<jobid>.txt
#SBATCH --partition gpu20              # Partition to submit to (e.g., gpu20 from sinfo)
#SBATCH --gres gpu:1                   # Request 1 GPU
#SBATCH --mem 16G                      # Amount of RAM to allocate
#SBATCH --cpus-per-task 4              # Number of CPU cores to allocate
#SBATCH --time 01:00:00                # Max wall time for the job (1 hour)
#SBATCH --nodes 1                      # Number of nodes (machines)
#SBATCH --ntasks 1                     # Number of tasks (normally 1 for single GPU jobs)

# ---------------- Setup runtime environment ---------------- #

# Ensure shell is properly initialized (for mamba)
source ~/.bashrc

# Activate mamba (replace 'myenv' with your actual environment name)
source /BS/data_mani_compress/work/miniforge3/etc/profile.d/conda.sh
conda activate dgm_geometry


# ---------------- Run your code ---------------- #

# Move to your working directory (adjust this to where your code lives)
cd /BS/data_mani_compress/work/thesis/thesis

# Run your Python script
python train.py experiment=train_mnist
