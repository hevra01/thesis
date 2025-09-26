import json
import hydra
from hydra.utils import instantiate
import sys

import numpy as np
from omegaconf import OmegaConf
import wandb
from utils import compute_edge_ratio

sys.path.append('/BS/data_mani_compress/work/thesis/thesis')

@hydra.main(version_base=None, config_path="../conf", config_name="estimate_edge_ratio")
def main(cfg):

    # Initialize W&B and dump Hydra config
    wandb.init(
        project="dataset_prep", 
        name=f"edge_ratio_estimation_train", 
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # Output file path
    file_path = cfg.experiment.output_file

    # instantiate the data loader
    dataloader = instantiate(cfg.experiment.dataset)

    # variables for storing edges
    edges = []

    for images, labels in dataloader:
        for image in images:
            image = image.cpu().numpy()
            image = np.transpose(image, (1, 2, 0))  # Convert to HWC format
            edge = compute_edge_ratio(image)
            edges.append(edge)

        with open(file_path, "w") as f:
            json.dump(edges, f, indent=2)

if __name__ == "__main__":
    main()
