#!/bin/bash
#SBATCH -J reconstruct_tokens
#SBATCH -o /ptmp/hevrapetek/thesis/logs/current.out
#SBATCH -e /ptmp/hevrapetek/thesis/logs/current.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=10
#SBATCH --gres=gpu:4

# --- Environment setup ---
module purge
module load anaconda/3/2023.03
source ~/.bashrc
source activate /u/hevrapetek/conda-envs/thesis
source /ptmp/hevrapetek/thesis/.wandb_secrets.sh
cd /ptmp/hevrapetek/thesis

# uncomment below for multi-GPU runs with torchrun
# Helpful runtime env for performance/stability
# - OMP_NUM_THREADS: max CPU threads OpenMP-backed libs (BLAS, MKL, etc.) will use per process.
#   Set it to your Slurm CPU allocation so you don't oversubscribe cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# - NCCL_DEBUG: logging level for NCCL (GPU comms). Use 'warn' for light logs, 'info' for debugging.
export NCCL_DEBUG=warn
# - NCCL_ASYNC_ERROR_HANDLING: surface NCCL errors promptly instead of hanging.
export NCCL_ASYNC_ERROR_HANDLING=1
# - NCCL_SOCKET_IFNAME: (optional) select network interface NCCL should use (e.g., ib0 or eth0).
#   Only needed if your cluster has multiple NICs or NCCL picks the wrong one.
# export NCCL_SOCKET_IFNAME=eth0

# Single-node multi-GPU with torchrun.
# - --nproc_per_node: number of worker processes to spawn on this node (usually = #GPUs).
# - --standalone: use a local rendezvous for process coordination (single-node only).
#                 For multi-node you would drop --standalone and pass --rdzv_backend/--rdzv_endpoint.

# Log a few environment values to stdout for sanity checks.
echo "[RUN] SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
echo "[RUN] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "[RUN] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[RUN] nvidia-smi -L:" && nvidia-smi -L || true

NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
torchrun --standalone --nproc_per_node=${NUM_GPUS} \
	-m neural_baseline.training \
	experiment=token_estimator_regression_neural_baseline_training

# for single GPU runs, just use python directly:
#python -m neural_baseline.training experiment=token_estimator_regression_neural_baseline_training