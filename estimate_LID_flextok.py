import json
import os

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import wandb
from hydra.utils import instantiate
from itertools import islice
from flextok.utils.misc import detect_bf16_support, get_bf16_context
from LID.fokker_planck_estimator import RectifiedFlowLIDEstimator
from sde.sdes import VpSDE

@hydra.main(version_base=None, config_path="conf", config_name="estimate_lid")
def main(cfg):

    # Check start_batch_idx and end_batch_idx 
    start_batch_idx = cfg.experiment.start_batch_idx
    end_batch_idx = cfg.experiment.end_batch_idx

    # Initialize W&B and dump Hydra config
    wandb.init(
        project="LID_estimate_flextok",  
        name=f"LID_estimate_flextok_{start_batch_idx:04d}_{end_batch_idx:04d}",
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # Set device
    device = torch.device(cfg.device)

    # Data dimension
    data_dim = cfg.experiment.data_dim

    # Configure model
    flextok = instantiate(cfg.experiment.model).to(device)  # instantiate the model
    # Ensure inference-only setup to save memory
    for p in flextok.parameters():
        p.requires_grad_(False)
    flextok.eval()

    # Format output filename to include batch range
    base_output_path = cfg.experiment.output_lid_file_path
    output_path = f"{base_output_path}_{start_batch_idx:04d}_{end_batch_idx:04d}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  

    # some models support fp16, so we convert the model to fp16 if specified
    enable_bf16 = detect_bf16_support()
    print('BF16 enabled:', enable_bf16)

    # Configure the dataloader
    dataloader = instantiate(cfg.experiment.dataset)

    batch_size = dataloader.batch_size

    # read the register tokens from the .npz file
    register_tokens_npz = cfg.experiment.register_path
    data = np.load(register_tokens_npz)
    register_tokens = data['token_ids']

    lid_estimator = RectifiedFlowLIDEstimator(ambient_dim=data_dim, model=flextok, device=device)

    # the range of t values over which to estimate LID
    t_value = cfg.experiment.t_value
    
    hutchinson_sample_count = cfg.hutchinson_sample_count

    lid_list = []

    # since this task is embarassingly parallel, we will use
    # multiple GPUs to speed up the process, where each GPU will handle 
    # a specific range of batches.
    sliced_loader = islice(dataloader, start_batch_idx, end_batch_idx)

    for batch_idx, (images, labels) in enumerate(sliced_loader, start=start_batch_idx):
        images = images.to(device)

        n = images.size(0) 

        # compute the global sample range for this batch
        start_sample = batch_idx * batch_size
        end_sample = start_sample + n  # NOT (batch_idx+1)*batch_size

        # slice exactly n token lists
        slice_tokens = register_tokens[start_sample:end_sample]
        assert len(slice_tokens) == n, f"token/image mismatch at batch {batch_idx}: tokens={len(slice_tokens)} images={n}"

        token_ids_list = [
            torch.as_tensor(t[:1], dtype=torch.long, device=device).unsqueeze(0)
            for t in slice_tokens
        ]

        # estimating for a single t value
        # the lid estimation will be unconditional, but we need to provide the token_ids_list, maybe in the future it can be used.
        # Use bf16 autocast where available for lower memory footprint
        with get_bf16_context(enable_bf16):
            div, norm = lid_estimator.estimate(images, t_hyper=t_value, hutchinson_samples=hutchinson_sample_count, token_ids_list=token_ids_list)

        per_instance = list(zip(div.tolist(), norm.tolist()))
        lid_list.extend(per_instance)
        print(lid_list)

    lid_values = [item for sublist in lid_list for item in sublist]

    # Convert to pure Python floats
    lid_list_serializable = [float(k) for k in lid_values]


    with open(output_path, "w") as f:
        json.dump(lid_list_serializable, f)

    print(f"Saved lid list to {output_path}")


if __name__ == "__main__":

    main()
