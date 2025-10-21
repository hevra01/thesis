#!/bin/bash
#SBATCH --account=mpi_cpu
#SBATCH --qos=debug

#SBATCH -J reconstruct_tokens
#SBATCH -o logs/currenttest.log
#SBATCH -e logs/currenttest.error
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

was passiert heier
# --- Environment setup ---
module purge
module load anaconda/3/2023.03
source activate /u/hevrapetek/conda-envs/thesis

# Debug: Print GPU and node information
echo "Job running on nodes: $SLURM_JOB_NODELIST"
echo "Total nodes: $SLURM_NNODES" 
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
#
srun ./test.py
