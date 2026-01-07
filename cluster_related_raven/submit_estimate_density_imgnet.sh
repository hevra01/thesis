#!/bin/bash
#SBATCH -J reconstruct_tokens
#SBATCH -o /ptmp/hevrapetek/thesis/logs/current.out
#SBATCH -e /ptmp/hevrapetek/thesis/logs/current.err
#SBATCH --time=0-04:30:00
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

EXPERIMENT_NAME=${EXPERIMENT_NAME:-estimate_density_RF}

# Toggle sweeping behavior
USE_KEEP_K_SWEEP=${USE_KEEP_K_SWEEP:-1}   # 1 to sweep keep_k list; 0 to disable sweeping
USE_BATCH_WINDOW=${USE_BATCH_WINDOW:-1}   # 1 to override batch indices; 0 to use config values

# Batch parameters (only used when USE_BATCH_WINDOW=1)
BASE_START_BATCH=${BASE_START_BATCH:-0}
BATCHES_PER_JOB=${BATCHES_PER_JOB:-480}   # Adjust based on GPU time/memory

# keep_k list (only used when USE_KEEP_K_SWEEP=1)
if [ "$USE_KEEP_K_SWEEP" -eq 1 ]; then
# [1,2,4,8,16,32,64,128,256] for conditional but 0 for unconditional
  KEEP_K_LIST_RAW=${KEEP_K_LIST:-[256]}
else
  # Optional single override
  if [ -n "${KEEP_K_LIST:-}" ]; then
    echo "Note: USE_KEEP_K_SWEEP=0, ignoring KEEP_K_LIST='${KEEP_K_LIST}'" >&2
  fi
  KEEP_K_LIST_RAW=""
fi

# Normalize bracketed comma-separated form
KEEP_K_LIST_NORM="$KEEP_K_LIST_RAW"
if [[ "$KEEP_K_LIST_NORM" =~ ^\[.*\]$ ]]; then
  KEEP_K_LIST_NORM="${KEEP_K_LIST_NORM#[}"
  KEEP_K_LIST_NORM="${KEEP_K_LIST_NORM%]}"
  KEEP_K_LIST_NORM="${KEEP_K_LIST_NORM//,/ }"
fi
read -r -a KEEP_K_ARRAY <<< "$KEEP_K_LIST_NORM"
NUM_KEEP_K=${#KEEP_K_ARRAY[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ "$NUM_KEEP_K" -gt 0 ]; then
  keep_k_index=$(( TASK_ID % NUM_KEEP_K ))
  SELECTED_KEEP_K=${KEEP_K_ARRAY[$keep_k_index]}
  if [ "$USE_BATCH_WINDOW" -eq 1 ]; then
    block_index=$(( TASK_ID / NUM_KEEP_K ))
  else
    block_index=0
  fi
else
  # No keep_k sweep
  SELECTED_KEEP_K=""
  if [ "$USE_BATCH_WINDOW" -eq 1 ]; then
    block_index=$TASK_ID
  else
    block_index=0
  fi
fi

if [ "$USE_BATCH_WINDOW" -eq 1 ]; then
  START_BATCH=$(( BASE_START_BATCH + block_index * BATCHES_PER_JOB ))
  END_BATCH=$(( START_BATCH + BATCHES_PER_JOB ))
fi

# If sweeping disabled but user set KEEP_K_OVERRIDE, use it
if [ "$USE_KEEP_K_SWEEP" -eq 0 ] && [ -n "${KEEP_K_OVERRIDE:-}" ]; then
  SELECTED_KEEP_K=$KEEP_K_OVERRIDE
fi

# Informative summary
if [ -n "$SELECTED_KEEP_K" ] && [ "$USE_BATCH_WINDOW" -eq 1 ]; then
  echo "Running experiment=$EXPERIMENT_NAME keep_k=$SELECTED_KEEP_K batches [$START_BATCH,$END_BATCH) TASK_ID=$TASK_ID"
elif [ -n "$SELECTED_KEEP_K" ]; then
  echo "Running experiment=$EXPERIMENT_NAME keep_k=$SELECTED_KEEP_K (batch indices from config) TASK_ID=$TASK_ID"
elif [ "$USE_BATCH_WINDOW" -eq 1 ]; then
  echo "Running experiment=$EXPERIMENT_NAME batches [$START_BATCH,$END_BATCH) (keep_k from config) TASK_ID=$TASK_ID"
else
  echo "Running experiment=$EXPERIMENT_NAME (keep_k & batches from config) TASK_ID=$TASK_ID"
fi

# Construct output base path. estimate_density_RF.py appends _start_end.json.
OUTPUT_ROOT=/ptmp/hevrapetek/thesis/data/datasets/density_imagenet/val/reconst_256/
if [ -n "$SELECTED_KEEP_K" ]; then
  OUTPUT_BASE="$OUTPUT_ROOT/token_count${SELECTED_KEEP_K}/"
else
  # Fallback base name when keep_k not explicitly selected here
  OUTPUT_BASE="$OUTPUT_ROOT/token_count_config"
fi


# ---------------- Run your code ---------------- #


# --- Arguments for Python script ---
ARGS=( experiment=$EXPERIMENT_NAME )

if [ "$USE_BATCH_WINDOW" -eq 1 ]; then
  ARGS+=(
    experiment.start_batch_idx=$START_BATCH
    experiment.end_batch_idx=$END_BATCH
  )
fi

if [ -n "$SELECTED_KEEP_K" ]; then
  ARGS+=( experiment.keep_k=$SELECTED_KEEP_K )
fi

# Static overrides (adjust as needed)
ARGS+=(
  experiment.output_path=$OUTPUT_BASE
  experiment.register_path=/ptmp/hevrapetek/thesis/data/datasets/imagnet_register_tokens/imagnet_val_register_tokens.npz
  experiment.dataset.root=/ptmp/hevrapetek/reconstruction_imagenet_APC_true/val/reconst_256
  experiment.dataset.batch_size=20
  experiment.guidance_scale=7.5
  experiment.dataset.split=""
  experiment.hutchinson_samples=4
)

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_BASE")"

# --- Run ---
python -u estimate_density_RF.py "${ARGS[@]}"
