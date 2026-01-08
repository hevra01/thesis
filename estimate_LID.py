import json
import os

import hydra
from omegaconf import OmegaConf
import torch
import wandb
from hydra.utils import instantiate
from itertools import islice

from LID.estimate_lid import compute_knees_for_all_data_points_in_batch, estimate_LID_over_t_range, estimate_LID_over_t_range_dataloader
from LID.fokker_planck_estimator import FlipdEstimator
from LID.utils import compute_knee, plot_lid_curve_with_knee
from sde.sdes import VpSDE

@hydra.main(version_base=None, config_path="conf", config_name="estimate_lid")
def main(cfg):

    # Check start_batch_idx and end_batch_idx 
    start_batch_idx = cfg.experiment.start_batch_idx
    end_batch_idx = cfg.experiment.end_batch_idx

    # Initialize W&B and dump Hydra config
    wandb.init(
        project="LID_estimate",  
        name=f"LID_estimate_reconst_1{start_batch_idx:04d}_{end_batch_idx:04d}",
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # Set device
    device = torch.device(cfg.device)

    # Data dimension
    data_dim = cfg.experiment.data_dim

    # Configure model
    score_net, _ = instantiate(cfg.experiment.model)  # instantiate the model
    
    # Move the model to the specified device
    score_net = score_net.to(device)  

    checkpoint_path = cfg.experiment.checkpoint_path

    # Format output filename to include batch range
    base_output_path = cfg.experiment.output_lid_file_path
    #os.makedirs(os.path.dirname(base_output_path), exist_ok=True)  
    output_path = f"{base_output_path}_{start_batch_idx:04d}_{end_batch_idx:04d}.json"

    # Load the pretrained checkpoint
    ckpt = torch.load(checkpoint_path)
    score_net.load_state_dict(ckpt, strict=True)

    # some models support fp16, so we convert the model to fp16 if specified
    enable_fp16 = cfg.experiment.enable_fp16
    if enable_fp16:
        score_net.convert_to_fp16()

    # Configure the dataloader
    dataloader = instantiate(cfg.experiment.dataset)

    # variance-preserving SDE
    model = VpSDE(score_net=score_net)

    lid_estimator = FlipdEstimator(ambient_dim=data_dim, model=model, device=device)

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

        # estimating for a single t value
        lid_vals = lid_estimator._estimate_lid(images, t=t_value, hutchinson_sample_count=hutchinson_sample_count)
        
        lid_list.append(lid_vals.tolist())
    

    lid_values = [item for sublist in lid_list for item in sublist]

    # Convert to pure Python floats
    lid_list_serializable = [float(k) for k in lid_values]


    with open(output_path, "w") as f:
        json.dump(lid_list_serializable, f)

    print(f"Saved lid list to {output_path}")


if __name__ == "__main__":

    main()
