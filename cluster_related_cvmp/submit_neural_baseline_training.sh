#!/bin/bash
# This script submits a job to the NHR@FAU Slurm cluster for running a Python training script.

#SBATCH --job-name=lid_estimations
#SBATCH --output=logs/output_bpp_100_epoch_%j.txt
#SBATCH --error=logs/error_bpp_100_epoch_%j.txt

# Partition and GPU configuration (see sinfo for options like gpu-a40, gpu-a100)
#SBATCH --partition=gpu-a100              # Use an A100 GPU node
#SBATCH --gres=gpu:4                      # Number of GPUs (adjust if needed)
#SBATCH --nodes=1                         # Single node
#SBATCH --ntasks=1                        # One launcher process (torchrun spawns workers)
#SBATCH --cpus-per-task=16                # CPU cores
#SBATCH --mem=64G                         # Memory
#SBATCH --time=1-00:00:00                 # 1 day walltime

# ---------------- Environment Setup ---------------- #
module load slurm_setup
module load cuda/12.1                     # Load CUDA module
module load anaconda/3                    # Load Anaconda (adjust if needed)

# Activate your environment
source ~/.bashrc
conda activate thesis

# Optionally load secrets or other setup files
source /home/hevra/thesis/.wandb_secrets.sh

# ---------------- Execution ---------------- #
cd /home/hevra/thesis

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NCCL_DEBUG=warn
export NCCL_ASYNC_ERROR_HANDLING=1

echo "[RUN] SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
echo "[RUN] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "[RUN] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[RUN] nvidia-smi -L:" && nvidia-smi -L || true

NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}

torchrun --standalone --nproc_per_node=${NUM_GPUS} \
    -m neural_baseline.training \
    experiment=token_estimator_regression_neural_baseline_training
#!/bin/bash
# This script submits a job to the NHR@FAU Slurm cluster for running a Python training script.

#SBATCH --job-name=lid_estimations
#SBATCH --output=logs/output_bpp_100_epoch_%j.txt
#SBATCH --error=logs/error_bpp_100_epoch_%j.txt

# Partition and GPU configuration (see sinfo for options like gpu-a40, gpu-a100)
#SBATCH --partition=gpu-a100              # Use an A100 GPU node
#SBATCH --gres=gpu:4                      # Number of GPUs (adjust if needed)
#SBATCH --nodes=1                         # Single node
#SBATCH --ntasks=1                        # One launcher process (torchrun spawns workers)
#SBATCH --cpus-per-task=16                # CPU cores
#SBATCH --mem=64G                         # Memory
#SBATCH --time=1-00:00:00                 # 1 day walltime

# ---------------- Environment Setup ---------------- #
module load slurm_setup
module load cuda/12.1                     # Load CUDA module
module load anaconda/3                    # Load Anaconda (adjust if needed)

# Activate your environment
source ~/.bashrc
conda activate dgm_geometry

# Optionally load secrets or other setup files
source /home/hevra/thesis/.wandb_secrets.sh

# ---------------- Execution ---------------- #
cd /home/hevra/thesis

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NCCL_DEBUG=warn
export NCCL_ASYNC_ERROR_HANDLING=1

echo "[RUN] SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
echo "[RUN] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "[RUN] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[RUN] nvidia-smi -L:" && nvidia-smi -L || true

NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}

torchrun --standalone --nproc_per_node=${NUM_GPUS} \
    -m neural_baseline.training \
    experiment=token_estimator_regression_neural_baseline_training
#!/bin/bash
# This script submits a job to the NHR@FAU Slurm cluster for running a Python training script.

#SBATCH --job-name=lid_estimations
#SBATCH --output=logs/output_bpp_100_epoch_%j.txt
#SBATCH --error=logs/error_bpp_100_epoch_%j.txt

# Partition and GPU configuration (see sinfo for options like gpu-a40, gpu-a100)
#SBATCH --partition=gpu-a100              # Use an A100 GPU node
#SBATCH --gres=gpu:4                      # Number of GPUs (adjust if needed)
#SBATCH --nodes=1                         # Single node
#SBATCH --ntasks=1                        # One launcher process (torchrun spawns workers)
#SBATCH --cpus-per-task=16                # CPU cores
#SBATCH --mem=64G                         # Memory
#SBATCH --time=1-00:00:00                 # 1 day walltime

# ---------------- Environment Setup ---------------- #
module load slurm_setup
module load cuda/12.1                     # Load CUDA module
module load anaconda/3                    # Load Anaconda (adjust if needed)

# Activate your environment
source ~/.bashrc
conda activate thesis

# Optionally load secrets or other setup files
source /home/hevra/thesis/.wandb_secrets.sh

# ---------------- Execution ---------------- #
cd /anvme/workspace/v114be16-hevra/thesis

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NCCL_DEBUG=warn
export NCCL_ASYNC_ERROR_HANDLING=1

echo "[RUN] SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
echo "[RUN] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "[RUN] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[RUN] nvidia-smi -L:" && nvidia-smi -L || true

NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}

torchrun --standalone --nproc_per_node=${NUM_GPUS} \
    -m neural_baseline.training \
    experiment=token_estimator_regression_neural_baseline_training
