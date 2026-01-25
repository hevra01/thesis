#!/bin/bash
#SBATCH -J neural_baseline_train
#SBATCH -o /dais/u/hevrapetek/thesis_outer/thesis/logs/current.out
#SBATCH -e /dais/u/hevrapetek/thesis_outer/thesis/logs/current.err
#SBATCH --time=0-1:00:00
#SBATCH --nodes=1
#SBATCH --mem=500GB
#SBATCH --gres=gpu:h200:1

# --- Environment setup ---
module purge
source ~/.bashrc
source /dais/u/hevrapetek/miniforge3/etc/profile.d/conda.sh
conda activate thesis_gpu_wheels
source /dais/u/hevrapetek/thesis_outer/thesis/.wandb_secrets.sh
cd /dais/u/hevrapetek/thesis_outer/thesis/

# Helpful runtime env for performance/stability
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NCCL_DEBUG=warn
# New name (old NCCL_ASYNC_ERROR_HANDLING is deprecated)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "[RUN] SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
echo "[RUN] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "[RUN] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[RUN] nvidia-smi -L:" && nvidia-smi -L || true

NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}


# ---------------- Sweep configuration (array-friendly) ---------------- #
# Name your Hydra experiment (maps to conf/estimate_lid.yaml entries)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-estimate_lid_imageNet_flextok}

# Control whether to sweep t and/or override batch start/end indices via job id mapping.
# USE_T_SWEEP: 1 to sweep t (use T_VALUES or default list), 0 to disable t sweep entirely
USE_T_SWEEP=${USE_T_SWEEP:-1}
# USE_BATCH_WINDOW: 1 to compute and pass experiment.start_batch_idx and experiment.end_batch_idx, 0 to use config values
# 1 (default): compute and pass experiment.start_batch_idx and experiment.end_batch_idx
# 0: do not override; use values from the Hydra config
USE_BATCH_WINDOW=${USE_BATCH_WINDOW:-1}

# Batch window parameters (used only when USE_BATCH_WINDOW=1)
BASE_START_BATCH=${BASE_START_BATCH:-0}   # Starting batch index for block_index=0
BATCHES_PER_JOB=${BATCHES_PER_JOB:-329}  # Number of batches covered by each job

# Optional: sweep over t values. Provide as space-separated list ("5 10 15"),
# or bracketed comma-separated list ("[5,10,15]"). Defaults to commonly used values when USE_T_SWEEP=1.
if [ "$USE_T_SWEEP" -eq 1 ]; then
    T_VALUES_RAW=${T_VALUES:-[0.04, 0.08, 0.12, 0.2]}
else
    # Disable t sweep regardless of T_VALUES; if you want to sweep, set USE_T_SWEEP=1
    if [ -n "${T_VALUES:-}" ]; then
        echo "Note: USE_T_SWEEP=0, ignoring T_VALUES='${T_VALUES}'" >&2
    fi
    T_VALUES_RAW=""
fi
T_VALUES_NORM="$T_VALUES_RAW"

# Normalize bracketed/comma-separated form into space-separated tokens
if [[ "$T_VALUES_NORM" =~ ^\[.*\]$ ]]; then
    T_VALUES_NORM="${T_VALUES_NORM#[}"
    T_VALUES_NORM="${T_VALUES_NORM%]}"
    T_VALUES_NORM="${T_VALUES_NORM//,/ }"
fi

# Split into an array; empty input yields zero-length array
read -r -a T_ARRAY <<< "$T_VALUES_NORM"
NUM_T=${#T_ARRAY[@]}

# Current array task id (0 if not submitted as an array job)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Map TASK_ID to (t_index, block_index) depending on the configured sweeps
if [ "$NUM_T" -gt 0 ]; then
    # There is a t sweep; always map a t_index
    t_index=$(( TASK_ID % NUM_T ))
    T_VALUE_SELECTED=${T_ARRAY[$t_index]}
    if [ "$USE_BATCH_WINDOW" -eq 1 ]; then
        # Also sweep batch windows by the quotient
        block_index=$(( TASK_ID / NUM_T ))
    else
        # t-only sweep; keep batch window fixed (from config)
        block_index=0
    fi
else
    # No t sweep; optionally sweep batches by TASK_ID
    T_VALUE_SELECTED=""
    if [ "$USE_BATCH_WINDOW" -eq 1 ]; then
        block_index=$TASK_ID
    else
        block_index=0
    fi
fi

# Compute batch window only if enabled
if [ "$USE_BATCH_WINDOW" -eq 1 ]; then
    START_BATCH=$(( BASE_START_BATCH + block_index * BATCHES_PER_JOB ))
    END_BATCH=$(( START_BATCH + BATCHES_PER_JOB ))
fi

# Base directory for LID output files; estimate_LID.py will append _{start}_{end}.json
OUT_ROOT="data/datasets/imageNet_LID_values/flextok_based/original_images/val"
if [ -n "$T_VALUE_SELECTED" ]; then
    # Include t in the path to distinguish sweeps, e.g., .../val/t_18/lid_0000_3123.json
    OUT_BASE="$OUT_ROOT/t_${T_VALUE_SELECTED}/lid"
else
    # No t override; keep a generic base
    OUT_BASE="$OUT_ROOT/lid"
fi

# Ensure the output directory exists (estimate_LID.py doesn't create dirs)
mkdir -p "$(dirname "$OUT_BASE")"

# Informative log of what will run
if [ -n "$T_VALUE_SELECTED" ] && [ "$USE_BATCH_WINDOW" -eq 1 ]; then
    echo "Running experiment=$EXPERIMENT_NAME with t_value=$T_VALUE_SELECTED, batches [$START_BATCH, $END_BATCH) (TASK_ID=$TASK_ID)"
elif [ -n "$T_VALUE_SELECTED" ]; then
    echo "Running experiment=$EXPERIMENT_NAME with t_value=$T_VALUE_SELECTED, using batch indices from config (TASK_ID=$TASK_ID)"
elif [ "$USE_BATCH_WINDOW" -eq 1 ]; then
    echo "Running experiment=$EXPERIMENT_NAME with batches [$START_BATCH, $END_BATCH) (TASK_ID=$TASK_ID)"
else
    echo "Running experiment=$EXPERIMENT_NAME using t_value and batch indices from config (TASK_ID=$TASK_ID)"
fi

# Build Hydra overrides. Putting them in an array preserves argument boundaries.
ARGS=(
    experiment=$EXPERIMENT_NAME
)

if [ "$USE_BATCH_WINDOW" -eq 1 ]; then
    ARGS+=(
        experiment.start_batch_idx=$START_BATCH
        experiment.end_batch_idx=$END_BATCH
        experiment.dataset.root="/dais/fs/scratch/hevrapetek/"
        experiment.dataset.split="val_categorized"
        experiment.dataset.batch_size=152
        experiment.register_path="data/datasets/imagnet_register_tokens/imagnet_val_register_tokens.npz"
    )
fi

# Optionally override t_value if a sweep list was provided
if [ -n "$T_VALUE_SELECTED" ]; then
    ARGS+=( experiment.t_value=$T_VALUE_SELECTED )
fi

# Always set the base output path for LID values; estimate_LID.py appends _start_end.json
ARGS+=( experiment.output_lid_file_path=$OUT_BASE )

# Run your Python script; -u for unbuffered stdout
python -u estimate_LID_flextok.py "${ARGS[@]}"
