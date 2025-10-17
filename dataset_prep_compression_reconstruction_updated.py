import time
import torch
from flextok.flextok_wrapper import FlexTokFromHub
from data.utils.dataloaders import get_imagenet_dataloader
from flextok.utils.misc import detect_bf16_support, get_bf16_context
from reconstruction_loss import MAELoss, VGGPerceptualLoss, reconstruction_error
import json
from itertools import islice
import hydra
import torch
import os
import wandb
from omegaconf import OmegaConf
from hydra.utils import instantiate
import numpy as np

"""
This file does the same thing as dataset_prep_compression_reconstruction.py,
however, it doesn't perform tokenization since we already have the tokens saved in a .npz file.
This is useful to save time when the tokenization step is slow.
"""

@hydra.main(version_base=None, config_path="conf", config_name="prep_dataset_compression_reconstruction")
def main(cfg):

    # Check start_batch_idx and end_batch_idx 
    start_batch_idx = cfg.experiment.start_batch_idx
    end_batch_idx = cfg.experiment.end_batch_idx

    # Initialize W&B and dump Hydra config
    wandb.init(
        project="dataset_prep_updated", 
        name=f"prep_dataset_{start_batch_idx:04d}_{end_batch_idx:04d}", 
        config=OmegaConf.to_container(cfg, resolve=True)
    )
 
    # Set device
    device = torch.device(cfg.device)

    # Instantiate the FlexTok model (used only for detokenization)
    model = instantiate(cfg.experiment.model).eval().to(device)

    # Configure the dataloader
    dataloader = instantiate(cfg.experiment.dataset)

    # Loss functions
    loss_fns = [instantiate(loss_cfg) for loss_cfg in cfg.experiment.loss_functions]
    for i, loss_fn in enumerate(loss_fns):
        if hasattr(loss_fn, "to"):
            loss_fns[i] = loss_fn.to(device)

    # Compression rates to test
    k_keep_list = cfg.experiment.k_keep_list

    # Format output filename to include batch range
    base_output_path = cfg.experiment.output_path  
    output_path = f"{base_output_path}_{start_batch_idx:04d}_{end_batch_idx:04d}.json"

    # Load precomputed register token ids (shape [N, 256])
    tokens_path = cfg.experiment.register_tokens_path
    assert os.path.isfile(tokens_path), f"register_tokens_path not found: {tokens_path}"
    npz = np.load(tokens_path, mmap_mode="r")
    if "token_ids" not in npz.files:
        raise KeyError(f"NPZ at {tokens_path} must contain 'token_ids'")
    token_ids = npz["token_ids"]  # memmap array [N, 256]

    reconstruction_errors = []

    # Slice the dataloader range for parallelization
    sliced_loader = islice(dataloader, start_batch_idx, end_batch_idx)

    enable_bf16 = detect_bf16_support()
    print('BF16 enabled:', enable_bf16)

    # Image normalization for unnormalizing to [0,1]
    mean = torch.tensor(cfg.experiment.image_normalization.mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(cfg.experiment.image_normalization.std).view(1, 3, 1, 1).to(device)

    # Process each batch
    for batch_idx, (images, labels) in enumerate(sliced_loader, start=start_batch_idx):
        images = images.to(device)
        B = images.size(0)

        images_unnorm = images * std + mean

        # Reconstruct with different k values
        for k_keep in k_keep_list:
            # Build token id sequences for this batch using global image ids
            tokens_list_filtered = []
            for img_idx in range(B):
                global_id = batch_idx * dataloader.batch_size + img_idx
                ids_np = token_ids[global_id, :k_keep]  # (k_keep,)
                ids_t = torch.from_numpy(np.asarray(ids_np, dtype=np.int64)).unsqueeze(0).to(device)
                tokens_list_filtered.append(ids_t)

            # Detokenize to reconstruct images
            with get_bf16_context(enable_bf16):
                reconstructed_imgs = model.detokenize(
                    tokens_list_filtered,
                    timesteps=20,
                    guidance_scale=7.5,
                )
            # Scale reconstructed images to [0, 1]
            reconstructed_imgs = (reconstructed_imgs.clamp(-1, 1) + 1) / 2

            # Compute reconstruction errors
            total_loss, loss_dict = reconstruction_error(
                reconstructed_imgs, images_unnorm, loss_fns=loss_fns
            )
            # Store errors for each image in the batch
            for img_idx in range(B):
                reconstruction_errors.append({
                    "image_id": batch_idx * dataloader.batch_size + img_idx,
                    "k_value": k_keep,
                    # Keep both losses if present
                    **{name: (val[img_idx].item() if hasattr(val, "__getitem__") else float(val)) for name, val in loss_dict.items()},
                })

        # Save after each batch (overwrite)
        with open(output_path, "w") as f:
            json.dump(reconstruction_errors, f)

        wandb.log({"progress/batch_idx": batch_idx + 1}, step=batch_idx)
        
    # Final save
    with open(output_path, "w") as f:
        json.dump(reconstruction_errors, f)

    print("Reconstruction errors saved in:", output_path)

    
if __name__ == "__main__":
    main()