#!/bin/bash
#SBATCH -J reconstruct_tokens
#SBATCH -o /ptmp/hevrapetek/thesis/logs/current.out
#SBATCH -e /ptmp/hevrapetek/thesis/logs/current.err
#SBATCH --time=0-02:00:00
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

EXPERIMENT_NAME=${EXPERIMENT_NAME:-estimate_density_RF_0_2}
BASE_BATCH=${BASE_START_BATCH:-0}
BATCHES_PER_JOB=${BATCHES_PER_JOB:-100} # check how many batches can fit into 1D per GPU

# Allow sweeping over keep_k values via the same job array index.
# Provide a space-separated list in KEEP_K_VALUES or use the default FlexTok sweep.
KEEP_K_VALUES=${KEEP_K_VALUES:-"1 2 4 8 16 32 64 128 256"}
read -r -a KEEP_K_ARRAY <<< "$KEEP_K_VALUES"
NUM_K=${#KEEP_K_ARRAY[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Map the 1D array index into (k_index, batch_block_index):
#   k_index varies fastest, batch block varies slowest.
k_index=$(( TASK_ID % NUM_K ))
block_index=$(( TASK_ID / NUM_K ))

START_BATCH=$(( BASE_BATCH + block_index * BATCHES_PER_JOB ))
END_BATCH=$(( START_BATCH + BATCHES_PER_JOB ))
KEEP_K=${KEEP_K_ARRAY[$k_index]}

echo "[submit] TASK_ID=$TASK_ID -> k_index=$k_index keep_k=$KEEP_K, block_index=$block_index, start=$START_BATCH end=$END_BATCH"


# ---------------- Run your code ---------------- #


# --- Arguments for Python script ---
ARGS=(
  experiment=$EXPERIMENT_NAME
  experiment.start_batch_idx=$START_BATCH
  experiment.end_batch_idx=$END_BATCH
  experiment.output_path=/ptmp/hevrapetek/thesis/data/datasets/density_imagenet_val/guidance_7.5/token_count_${KEEP_K}/
  experiment.keep_k=${KEEP_K}
  experiment.guidance_scale=7.5
  experiment.register_path=/ptmp/hevrapetek/thesis/data/datasets/imagnet_register_tokens/imagnet_val_register_tokens.npz
  experiment.dataset.root=/ptmp/hevrapetek/ILSVR2012/
  experiment.dataset.split="val_categorized"
  experiment.dataset.batch_size=12
)

# --- Run ---
cd /ptmp/hevrapetek/thesis
python -u estimate_density_RF.py "${ARGS[@]}"
