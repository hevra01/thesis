import json
import hydra
from hydra.utils import instantiate
import sys

import numpy as np
from utils import compute_edge_ratio

sys.path.append('/BS/data_mani_compress/work/thesis/thesis')

@hydra.main(version_base=None, config_path="../conf", config_name="estimate_edge_ratio")
def main(cfg):
    
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

    # # Save edges to file
    # np.save(file_path, edges)

if __name__ == "__main__":
    main()
