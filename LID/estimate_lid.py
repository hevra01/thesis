import json
import torch
from data.utils.dataloaders import get_mnist_dataloader
from fokker_planck_estimator import FlipdEstimator
import argparse
from sde.sdes import VpSDE

from models.mlp import MLPUnet

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

    lid_estimator = FlipdEstimator(ambient_dim=model.data_dim, model=model, device=device)

    lid_values = []
    print(dataloader)
    for x in dataloader:
        print(x, x.shape)
        exit()
        x = x.to(device)
        with torch.no_grad():
            lid_value = lid_estimator._estimate_lid(x, t=t)
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

    # NOTE: this is a bit hard coded for MNIST, it needs to be more 
    # flexible to work with other datasets.
    # Initialize the dataset
    dataloader = get_mnist_dataloader()

    estimate_LID(dataloader, score_model, t, device)



if __name__ == "__main__":

    main()
