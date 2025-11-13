#!/bin/bash
#SBATCH -J reconstruct_tokens
#SBATCH -o /ptmp/hevrapetek/thesis/logs/current.out
#SBATCH -e /ptmp/hevrapetek/thesis/logs/current.err
#SBATCH --time=0-01:00:00
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

EXPERIMENT_NAME=${EXPERIMENT_NAME:-estimate_density_RF_0}
BASE_BATCH=${BASE_START_BATCH:-0}
BATCHES_PER_JOB=${BATCHES_PER_JOB:-200} # check how many batches can fit into 1D per GPU

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
START_BATCH=$(( BASE_BATCH + TASK_ID * BATCHES_PER_JOB ))
END_BATCH=$(( START_BATCH + BATCHES_PER_JOB ))


# ---------------- Run your code ---------------- #


# --- Arguments for Python script ---
ARGS=(
  experiment=$EXPERIMENT_NAME
  experiment.start_batch_idx=$START_BATCH
  experiment.end_batch_idx=$END_BATCH
  experiment.output_path=/ptmp/hevrapetek/thesis/data/datasets/density_imagenet_val/corrected_hutchinson_4/token_count_128/
  experiment.keep_k=128
  experiment.register_path=/ptmp/hevrapetek/thesis/data/datasets/imagnet_register_tokens/imagnet_val_register_tokens.npz
  experiment.dataset.root=/ptmp/hevrapetek/ILSVR2012/
  experiment.dataset.batch_size=12
  experiment.dataset.split="val_categorized"
)

# --- Run ---
cd /ptmp/hevrapetek/thesis
python -u estimate_density_RF.py "${ARGS[@]}"
