"""
Estimate the density of images using a pre-trained FlexTok rectified flow based decoder model with register tokens.
"""


from itertools import islice
import json
import time
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from hydra.utils import instantiate
import wandb
from flextok.utils.misc import detect_bf16_support, get_bf16_context


@hydra.main(version_base=None, config_path="conf", config_name="estimate_density")
def main(cfg):
    device = cfg.device

    # Initialize W&B and dump Hydra config
    wandb.init(
        project="estimate_density_RF", 
        name=f"density_estimation_RF", 
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # Check start_batch_idx and end_batch_idx 
    start_batch_idx = cfg.experiment.start_batch_idx
    end_batch_idx = cfg.experiment.end_batch_idx

    # Format output filename to include batch range
    base_output_path = cfg.experiment.output_path  
    output_path = f"{base_output_path}_{start_batch_idx:04d}_{end_batch_idx:04d}.json"

    # instantiate the data loader
    dataloader = instantiate(cfg.experiment.dataset)

    # go over the imagenet dataset and estimate the density of the first num_images
    densities = []
    #divergences = []

    # read the register tokens from the .npz file
    register_tokens_npz = cfg.experiment.register_path
    data = np.load(register_tokens_npz)
    register_tokens = data['token_ids']

    batch_size = dataloader.batch_size

    # 👈 choose how many registers to keep (user-defined)
    keep_k = cfg.experiment.keep_k  

    hutchinson_samples = cfg.experiment.hutchinson_samples 

    # since this task is embarassingly parallel, we will use
    # multiple GPUs to speed up the process, where each GPU will handle 
    # a specific range of batches.
    sliced_loader = islice(dataloader, start_batch_idx, end_batch_idx)

    # instantiate the density estimator model
    flextok = instantiate(cfg.experiment.model).to(device)

    # we only need the gradients w.r.t the inputs for density estimation
    # not for the weights as well.
    for p in flextok.parameters():
        p.requires_grad_(False)
    flextok.eval()

    enable_bf16 = detect_bf16_support()
    print('BF16 enabled:', enable_bf16)

    flextok.pipeline.count_decoder_params()  # Print decoder parameters

    for batch_idx, (images, labels) in enumerate(sliced_loader, start=start_batch_idx):

        # Convert JSON lists to torch.Tensors and batch them
        token_ids_list = [
            torch.tensor(register_ids[:keep_k]).unsqueeze(0).to(device)
            for register_ids in register_tokens[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        ]
        start = time.time()
        
        # call the density estimation function
        with get_bf16_context(enable_bf16):
            current_densities = flextok.estimate_log_density(images.to(device), token_ids_list=token_ids_list, hutchinson_samples=hutchinson_samples)
        current_densities = [density.item() for density in current_densities]
        
        #current_divergences = [div.item() for div in no]
        densities.extend(current_densities)
        #divergences.extend(current_divergences)

        # Save the estimated densities to a file
        with open(output_path, "w") as f:
            json.dump(densities, f)

        wandb.log(
            {
                "progress/batch_idx": batch_idx + 1,   # current batch
            },
            step=batch_idx
        )
        print(f"Processed batch in {time.time() - start:.2f} seconds.")  

if __name__ == "__main__":
    main()