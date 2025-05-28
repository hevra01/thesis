import os
import sys

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


def estimate_LID(dataloader, model, t, device):
    """
    Estimates the Local Intrinsic Dimensionality (LID) using the Fokker-Planck Estimator.
    
    Args:
        dataloader: DataLoader containing the data points.
        model: The trained model used for score estimation.
        t: Time at which to estimate LID.
        device: Device to run the model on (e.g., 'cuda' or 'cpu').
    
    Returns:
        lid_values: Estimated LID values for the data points.
    """

    lid_estimator = FlipdEstimator(ambient_dim=model.score_net.data_dim, model=model, device=device)

    lid_values = []
    for batch in dataloader:
        images = batch["image"].to(device)
        lid_value = lid_estimator._estimate_lid(images, t=t)
        lid_values.append(lid_value.cpu())
    
    return torch.cat(lid_values)


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

    # NOTE: this is a bit hard coded for MNIST, it needs to be more 
    # flexible to work with other datasets.
    # Initialize the dataset
    dataloader = get_mnist_dataloader(flatten=True)

    LID_values = estimate_LID(dataloader, model, t, device)

    print(LID_values.mean().item())

    # Save the LID values to a file
    output_path = cfg.get("output_path", "lid_values.pt")
    torch.save(LID_values, output_path)
    print(f"LID values saved to {output_path}")

if __name__ == "__main__":

    main()
