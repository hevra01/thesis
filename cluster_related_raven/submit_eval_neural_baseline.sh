#!/bin/bash
#SBATCH -J lpips_variance
#SBATCH -o /ptmp/hevrapetek/thesis/logs/lpips.out
#SBATCH -e /ptmp/hevrapetek/thesis/logs/lpips.err
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

# --------------------------------------------------
# Environment setup
# --------------------------------------------------
module purge
module load anaconda/3/2023.03
source ~/.bashrc
source activate /u/hevrapetek/conda-envs/thesis
source /ptmp/hevrapetek/thesis/.wandb_secrets.sh

cd /ptmp/hevrapetek/thesis

EXPERIMENT_NAME=${EXPERIMENT_NAME:-eval_neural_baseline}

ARGS=(
    experiment=$EXPERIMENT_NAME
    experiment.experiment_name=neural_baseline_eval
    experiment.project_name=neural_baseline_evaluation

    experiment.dataset.root=/scratch/inf0/user/mparcham/ILSVRC2012/
    experiment.dataset.split=val_categorized
    experiment.dataset.batch_size=4

    experiment.task=classification
    experiment.checkpoint_path=neural_baseline/checkpoint/predict_token_count/all.pt

    experiment.reconstruction_dataset.batch_size=4
)

python -m neural_baseline.eval "${ARGS[@]}"