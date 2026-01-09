#!/bin/bash
#SBATCH -J reconstruct_tokens
#SBATCH -o /ptmp/hevrapetek/thesis/logs/current.out
#SBATCH -e /ptmp/hevrapetek/thesis/logs/current.err
#SBATCH --time=0-03:00:00
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

# --- Configurable variables ---
EXPERIMENT_NAME=${EXPERIMENT_NAME:-reconstruct_with_tokens}
BASE_START_CLASS=${BASE_START_CLASS:-0}
CLASSES_PER_JOB=${CLASSES_PER_JOB:-10}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
START_CLASS=$(( BASE_START_CLASS + TASK_ID * CLASSES_PER_JOB ))
END_CLASS=$(( START_CLASS + CLASSES_PER_JOB ))

echo "[$(date)] Running task $TASK_ID: classes [$START_CLASS, $END_CLASS) for experiment=$EXPERIMENT_NAME"
echo "SLURM_JOB_ID=$SLURM_JOB_ID on node $HOSTNAME"

# --- Arguments for Python script ---
ARGS=(
  experiment=$EXPERIMENT_NAME
  experiment.start_class_idx=$START_CLASS
  experiment.end_class_idx=$END_CLASS
  experiment.output_path=/ptmp/hevrapetek/reconstruction_imagenet_stochastic/val
  experiment.register_tokens_path=/ptmp/hevrapetek/thesis/data/datasets/imagnet_register_tokens/imagnet_val_register_tokens.npz
  experiment.data_root=/ptmp/hevrapetek/ILSVR2012/val
  experiment.perform_apc=true
)

# --- Run ---
cd /ptmp/hevrapetek/thesis
python -u reconstruct_with_tokens.py "${ARGS[@]}"
