import os
import sys

import numpy as np

from utils import compute_knee, plot_lid_curve_with_knee

# Automatically add the project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    print(project_root)
    sys.path.insert(0, project_root)

import json
import torch
from data.utils.dataloaders import get_mnist_dataloader
from fokker_planck_estimator import FlipdEstimator
import argparse
from models.mlp import MLPUnet
from sde.sdes import VpSDE

def estimate_LID_over_t_range(
    x,  # [data_dim] or [batch_size, data_dim]
    lid_estimator,
    t_values,
    ambient_dim,
    device,
    return_info=False,
    **knee_kwargs
):
    """
    Estimates LID over a range of t values for a single instance or a batch.
    Returns the best LID using the knee algorithm.
    """
    x = x.to(device)
    lid_curve = estimate_LID_over_t_range_batch(
            x=x,
            t_values=t_values,
            lid_estimator=lid_estimator
        )
    lid_curve = np.array(lid_curve)
    t_values = np.array(t_values)
    return compute_knee(
        t_values, lid_curve, ambient_dim, return_info=return_info, **knee_kwargs
    )


def estimate_LID_over_t_range_batch(x, t_values, lid_estimator):
    # this list will get filled with LID values for each t in t_values
    lid_curve = []

    # loop over each t value to estimate LID
    for t in t_values:
        lid_vals = lid_estimator._estimate_lid(x, t=t)
        print(f"Estimated LID for t={t}: {lid_vals}")
        exit(0)

        # For a batch, we take the mean of LID values across the batch.
        # However, if the input is a single instance, we just take the value directly.
        lid_curve.append(lid_vals.mean().item())
        
    return lid_curve



def estimate_LID_over_t_range_dataloader(
    dataloader,
    lid_estimator,
    t_values,
    ambient_dim,
    device,
    return_info=False,
    **knee_kwargs
):
    """
    Estimates LID over a range of t values for an entire dataset (dataloader).
    Processes the dataset batch-by-batch to avoid memory issues.
    Aggregates mean LID for each t across all batches, then finds the best LID using the knee algorithm.
    """
    # Initialize an array to accumulate the sum of mean LID values for each t
    lid_curve_sum = np.zeros(len(t_values), dtype=np.float64)
    # Counter for the number of batches processed
    lid_curve_count = 0

    # Outer loop: iterate over each batch in the dataloader
    # For each batch, we compute the mean LID for every t value.
    for batch in dataloader:
        images = batch["image"].to(device)

        # You accumulate these means across all batches.
        batch_lid_curve = []

        batch_lid_curve = estimate_LID_over_t_range_batch(
            x=images,
            t_values=t_values,
            lid_estimator=lid_estimator
        )

        # Add the batch's mean LID curve to the running sum
        # This operation adds the values element-wise (for each t).
        lid_curve_sum += np.array(batch_lid_curve)

        # Increment the batch counter
        lid_curve_count += 1

    # After all batches, compute the average LID curve over the dataset
    lid_curve = lid_curve_sum / lid_curve_count
    t_values = np.array(t_values)

    # Use the knee algorithm to find the best LID estimate from the averaged curve
    return compute_knee(
        t_values, lid_curve, ambient_dim, return_info=return_info
    )

def main():
    parser = argparse.ArgumentParser(description="Estimate LID using Fokker-Planck Estimator")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (JSON)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = json.load(f)

    # Set device
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Time at which to estimate LID. Theoretically, this should be a float value very close to zero.
    t = cfg["t"] 

    # Prepare the model for score estimation, which will be used for the LID estimation.
    model_cfg = cfg["model"]
    ckpt_path = cfg["checkpoint_path"]
    score_model = MLPUnet(
        data_dim=model_cfg["data_dim"],
        hidden_sizes=model_cfg["hidden_sizes"],
        time_embedding_dim=model_cfg["time_embedding_dim"]
    ).to(device)
    score_model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])

    model = VpSDE(score_net=score_model)

    lid_estimator = FlipdEstimator(ambient_dim=model.score_net.data_dim, model=model, device=device)

    # NOTE: this is a bit hard coded for MNIST, it needs to be more 
    # flexible to work with other datasets.
    # Initialize the dataset
    dataloader = get_mnist_dataloader(flatten=True)

    # the range of t values over which to estimate LID
    t_values = torch.linspace(0.0, 1, 100)

    # Estimate LID over the range of t values for the entire dataset.
    # Then use the knee algorithm to find the best LID estimate.
    knee_info = estimate_LID_over_t_range_dataloader(lid_estimator, dataloader,
                                                   t_values, 
                                                   ambient_dim=model_cfg["data_dim"],
                                                   device=device, return_info=True)
    
    # visualize the LID curve and knee point
    plot_lid_curve_with_knee(
        knee_info["convex_hull"],
        knee_info["knee_timestep"],
        t_values,
        knee_info["lid"],
        ambient_dim=model_cfg["data_dim"],
        output_path=cfg.get("output_plot_path", "lid_curve_with_knee.png")
    )

   

if __name__ == "__main__":

    main()
