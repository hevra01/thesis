"""
Given the original dataset and reconstructed images from tokens, compute reconstruction losses.
Saves per-image, per-k records.

Directory layout expected under `reconstructed_data_path`:
  reconstructed_data_path/
    reconst_1/
      <wnid>/<filename>
    reconst_2/
      <wnid>/<filename>
    ...
    reconst_256/
      <wnid>/<filename>
"""

import csv
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image
import torch
from torchvision import transforms as T
from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from lpips import LPIPS

# Reuse the aggregation helper (handles LPIPS specially)
from reconstruction_loss import reconstruction_error
from data.utils.dataloaders import get_imagenet_dataloader


@hydra.main(version_base=None, config_path="conf", config_name="estimate_reconstruction_loss")
def main(cfg: DictConfig):
    # Initialize W&B and dump Hydra config
    wandb.init(
        project="reconstruction_loss_estimation", 
        name=f"reconstruction_loss_estimation", 
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    device = cfg.experiment.device

    # Instantiate loss functions from config (e.g., L1 and LPIPS)
    loss_fns = [instantiate(loss_cfg).to(device) for loss_cfg in cfg.experiment.loss_functions]

    # instantiate original dataset
    original_images_dataloader = instantiate(cfg.experiment.dataset)

    # Where reconstructed images live
    reconstructed_data_path = cfg.experiment.reconstructed_data_path

    # Output directory for metrics
    output_path = cfg.experiment.reconstruction_loss_output_path

    # Load normalization values from Hydra config
    mean = torch.tensor(cfg.experiment.image_normalization.mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(cfg.experiment.image_normalization.std).view(1, 3, 1, 1).to(device)

    reconstruction_errors = []
    batch_size = original_images_dataloader.batch_size

    # outer loop for images reconstructed with different k values
    for k_keep in cfg.experiment.k_keep_list:
        # initialize the dataloader for reconstructed images with k_keep
        reconstructed_k_root = os.path.join(reconstructed_data_path, f"reconst_{k_keep}")
        reconstructed_dataloader = get_imagenet_dataloader(
            root=reconstructed_k_root,
            batch_size=original_images_dataloader.batch_size,
            shuffle=False,
            split="",
        )

        # inner loop over images.
        # For original images, it will stay the same across k_keep.
        # For reconstructed images, we get from reconstructed_dataloader. 
        # use the dataloader to get img. ignore the labels
        for batch_idx, (original_batch, reconstructed_batch) in enumerate(zip(original_images_dataloader, reconstructed_dataloader)):
            
            original_imgs = original_batch[0].to(device)  # get the images without labels
            reconst_imgs = reconstructed_batch[0].to(device)

            # unnormalize images 
            original_imgs = original_imgs * std + mean
            reconst_imgs = reconst_imgs * std + mean

            # Compute reconstruction errors
            with torch.no_grad():
                total_loss, loss_dict = reconstruction_error(
                    reconst_imgs, original_imgs, loss_fns=loss_fns, device=device
                )
            # Store errors for each image in the batch
            for img_idx in range(original_imgs.size(0)):
                reconstruction_errors.append({
                    "image_id": batch_idx * batch_size + img_idx,
                    "k_value": k_keep,
                    "L1Loss": loss_dict["L1Loss"][img_idx].item(),
                    "LPIPS": loss_dict["LPIPS"][0][img_idx].item(),
                    "LPIPS_layers": [t[img_idx].item() for t in loss_dict["LPIPS"][1]],
                    "DINOv2FeatureLoss": loss_dict["DINOv2FeatureLoss"][img_idx].item()
                })
            print(f"Processed batch {batch_idx} for k={k_keep}")
            
        # Save after each batch (overwrite)
        with open(output_path, "w") as f:
            json.dump(reconstruction_errors, f)

        wandb.log(
            {
                "progress/k_value": k_keep,
            }
        )
        
    # Save reconstruction errors to a JSON file
    with open(output_path, "w") as f:
        json.dump(reconstruction_errors, f)

    print("Reconstruction errors saved in:", output_path)

if __name__ == "__main__":
    main()
