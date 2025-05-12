from pathlib import Path
import hydra
import lightning as L
import torch
import torchvision.transforms.functional as TVF
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, errors as omegaconf_errors
from pprint import pprint


@hydra.main(version_base=None, config_path="../conf/", config_name="train")
def main(cfg: DictConfig):

    # Configure dataset
    train_dataset = instantiate(cfg.dataset.train)
    train_loader = torch.utils.data.DataLoader(train_dataset, **cfg.train.loader)

    # Configure model
    dgm = instantiate(cfg.train.dgm)  # instantiate the DGM model
    

if __name__ == "__main__":

    main()
