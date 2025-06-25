import os

import hydra
import torch
import wandb
from hydra.utils import instantiate

from LID.estimate_lid import estimate_LID_over_t_range, estimate_LID_over_t_range_dataloader
from LID.fokker_planck_estimator import FlipdEstimator
from LID.utils import compute_knee, plot_lid_curve_with_knee
from sde.sdes import VpSDE

@hydra.main(version_base=None, config_path="conf", config_name="estimate_lid")
def main(cfg):
    # # Initialize wandb if enabled
    # if cfg["wand_enabled"]:
    #     os.environ["WANDB_API_KEY"] = "4fb9acb164293e106d3b4e0787abfb22bcfe9afa"
    #     wandb.init(
    #         project=cfg["wandb"]["project"],
    #         name=cfg["wandb"]["run_name"],
    #         mode=cfg["wandb"]["mode"],
    #     )

    # Set device
    device = torch.device(cfg.device)

    # Data dimension
    data_dim = cfg.experiment.data_dim

    # Configure model
    score_net, _ = instantiate(cfg.experiment.model)  # instantiate the model
    # Move the model to the specified device
    score_net = score_net.to(device)  

    checkpoint_path = cfg.experiment.checkpoint_path

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
    t_values = torch.linspace(0.0, model.t_max, 100)
    hutchinson_sample_count = cfg.hutchinson_sample_count

    # Estimate LID over the range of t values for the entire dataset.

    lid_curve = estimate_LID_over_t_range(next(iter(dataloader))[0], lid_estimator,t_values, ambient_dim=data_dim, hutchinson_sample_count=hutchinson_sample_count, device=device, return_info=True)
    # lid_curve = estimate_LID_over_t_range_dataloader(dataloader, lid_estimator, 
    #                                                t_values, hutchinson_sample_count=hutchinson_sample_count,
    #                                                ambient_dim=data_dim,
    #                                                device=device, return_info=True)

     # Use the knee algorithm to find the best LID estimate from the averaged curve
    knee_info = compute_knee(t_values, lid_curve, ambient_dim=data_dim, return_info=True)

    # visualize the LID curve and knee point
    plot_lid_curve_with_knee(
        lid_curve,
        t_values,
        knee_info["knee_timestep"],
        knee_info["lid"],
        save_path=cfg.get("save_path")
    )

   

if __name__ == "__main__":

    main()
