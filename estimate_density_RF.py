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

    densities = []

    # read the register tokens from the .npz file
    register_tokens_npz = cfg.experiment.register_path
    data = np.load(register_tokens_npz)
    register_tokens = data['token_ids']

    batch_size = dataloader.batch_size

    # choose how many registers to keep (user-defined)
    keep_k = cfg.experiment.keep_k
    hutchinson_samples = cfg.experiment.hutchinson_samples

    # slice the loader by batch indices (end is exclusive)
    sliced_loader = islice(dataloader, start_batch_idx, end_batch_idx)

    # instantiate the density estimator model
    flextok = instantiate(cfg.experiment.model).to(device)

    for p in flextok.parameters():
        p.requires_grad_(False)
    flextok.eval()

    enable_bf16 = detect_bf16_support()
    print('BF16 enabled:', enable_bf16)

    flextok.pipeline.count_decoder_params()

    for batch_idx, (images, labels) in enumerate(sliced_loader, start=start_batch_idx):
        # ---- FIX: make token slicing match the actual batch length ----
        # number of samples in THIS batch (handles short last batch)
        n = images.size(0) 

        # compute the global sample range for this batch
        start_sample = batch_idx * batch_size
        end_sample = start_sample + n  # NOT (batch_idx+1)*batch_size

        # slice exactly n token lists
        slice_tokens = register_tokens[start_sample:end_sample]
        assert len(slice_tokens) == n, f"token/image mismatch at batch {batch_idx}: tokens={len(slice_tokens)} images={n}"

        token_ids_list = [
            torch.as_tensor(t[:keep_k], dtype=torch.long, device=device).unsqueeze(0)
            for t in slice_tokens
        ]
        # ---------------------------------------------------------------

        start_t = time.time()
        with get_bf16_context(enable_bf16):
            current_densities = flextok.estimate_log_density(
                images.to(device),
                token_ids_list=token_ids_list,
                hutchinson_samples=hutchinson_samples
            )
        current_densities = [d.item() for d in current_densities]
        densities.extend(current_densities)

        # Save incrementally
        with open(output_path, "w") as f:
            json.dump(densities, f)

        wandb.log({"progress/batch_idx": batch_idx + 1}, step=batch_idx)
        print(f"Processed batch in {time.time() - start_t:.2f} seconds.")

if __name__ == "__main__":
    main()
