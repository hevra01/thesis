"""
This script will be used to evaluate the neural baseline models.
We have 2 neural baseline models:
1. for predicting the reconstruction loss, which is a regression task.
2. for predicting the token count, which is a classification task.

To evaluate the performance of these models, we will:
1. evaluate the performance for a given range of reconstruction loss values.
   because the model may perform differently for different ranges of reconstruction loss values.

2. evaluate the performance for each token count class separately.
   because the model may perform differently for different token count classes.
"""

import json
import os
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from data.utils.dataloaders import ReconstructionDataset_Neural
import wandb


def compute_hard_nll_mean(logits: torch.Tensor, k_int: torch.Tensor) -> torch.Tensor:
    C = logits.size(1)
    log_p = F.log_softmax(logits, dim=1)
    idx = (k_int - 1).clamp(0, C - 1).view(-1, 1)
    hard_nll = -log_p.gather(1, idx).squeeze(1)
    return hard_nll.mean()

def main(cfg: DictConfig):
    device = cfg.experiment.device

    run = wandb.init(
            name=cfg.experiment.experiment_name,
            project=cfg.experiment.project_name,
            group=cfg.experiment.group_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_slurm")

    wandb.run.summary["slurm_job_id"] = slurm_job_id



    # in the config for the dataset, we have already filtered the dataset according to the desired evaluation ranges.
    # we either filter by reconstruction loss range or by token count class based on the task.
    base_dataset = instantiate(cfg.experiment.dataset)
    model = instantiate(cfg.experiment.model).to(device).eval()

    base_dataloader = torch.utils.data.DataLoader(base_dataset, batch_size=cfg.experiment.batch_size, shuffle=False)

    task = cfg.experiment.task  # either "classification" or "regression"

    if task == "classification":
        filter_key = cfg.experiment.reconstruction_dataset.filter_key_classification
        filter_range = cfg.experiment.reconstruction_dataset.recon_loss_ranges_classification
        skip_range = cfg.experiment.reconstruction_dataset.skip_range_classification
    elif task == "regression":
        filter_key = cfg.experiment.reconstruction_dataset.filter_key_regression
        filter_range = cfg.experiment.reconstruction_dataset.recon_loss_ranges_regression
        skip_range = cfg.experiment.reconstruction_dataset.skip_range_regression
    else:
        raise ValueError(f"Unknown task: {task}")

    # this will either be used to fetch the recon loss that is to be predicted
    # or it will be used as a condition to predict the token count class.
    recon_loss_key = cfg.experiment.reconstruction_dataset.reconstruction_loss_key

    # reconstruction_data holds per-image errors for multiple K values + token counts.
    with open(cfg.experiment.reconstruction_dataset.reconstruction_train_data_path, "r") as f:
        reconstruction_data = json.load(f)


    for current_range in range(0, len(filter_range) - 1, skip_range):
        min_error, max_error = filter_range[current_range], filter_range[current_range + 1]
        print(f"Evaluating for reconstruction loss range: [{min_error}, {max_error}]")

        # since we are evaluating the model performance for specific reconstruction loss ranges,
        # we create a new dataset that filters the images based on the current reconstruction loss range.
        recon_dataset = ReconstructionDataset_Neural(
            reconstruction_data=reconstruction_data,
            dataloader=base_dataloader,
            filter_key=filter_key,
            min_error=min_error,
            max_error=max_error,
            error_key=recon_loss_key,
        )

        recon_dataloader = torch.utils.data.DataLoader(recon_dataset, batch_size=cfg.experiment.reconstruction_dataset.batch_size, shuffle=False)

        if task == "classification":
            # Evaluate classification performance (predicting token count)

            hard_nll_mean = 0.0

            # Iterate over the dataloader
            for batch in recon_dataloader:
                images = batch["images"].to(device)
                k_value = batch["k_value"].to(device)  # true token count class
                recon_loss = batch[recon_loss].to(device) # condition

                logits = model(images, recon_loss)  # [B,C]
                hard_nll_mean += compute_hard_nll_mean(logits, k_value)

            # find average hard NLL over all batches
            hard_nll_mean /= len(recon_dataset)
            print(f"Hard NLL Mean (Classification Task) for {min_error}-{max_error}: {hard_nll_mean.item():.4f}")

        elif task == "regression":
            # Evaluate regression performance
            mae_loss_fn = torch.nn.L1Loss(reduction='mean')
            mae_mean = 0.0

            max_logk = 8.0  # since log2(256) = 8

            # Iterate over the dataloader
            for batch in recon_dataset:
                images = batch["images"].to(device)
                k_value = batch["k_value"].to(device)  # condition
                cond = torch.log2(k_value) / max_logk

                true_recon_loss = batch[recon_loss].to(device) # condition
                pred_recon_loss = model(images, cond)  # predicted reconstruction loss
                mae_mean += mae_loss_fn(pred_recon_loss, true_recon_loss).item()

            # find average MAE over all batches
            mae_mean /= len(recon_dataset)
            print(f"MAE Mean (Regression Task) for {min_error}-{max_error}: {mae_mean:.4f}")
        else:
            raise ValueError(f"Unknown task: {task}")