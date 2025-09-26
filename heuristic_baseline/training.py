import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from data.utils.dataloaders import ReconstructionDataset_Heuristic
import json
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="../conf", config_name="heuristic_baseline_training")
def main(cfg: DictConfig):

    # Device configuration
    device = torch.device(cfg.device)

    # Initialize W&B and dump Hydra config
    wandb.init(
        project="dataset_prep", 
        name=f"trainset_heuristic_baseline_training", 
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # Load JSONs
    # this data holds images, mse_errors, vgg_errors, token counts for different k_values
    with open(cfg.experiment.reconstruction_dataset.reconstruction_data_path, "r") as f:
        reconstruction_data = json.load(f)


    # for the heuristic baseline we need the edge ratio information as a proxy for complexity
    with open(cfg.experiment.reconstruction_dataset.edge_ratio_path, "r") as f:
        edge_ratio_information = json.load(f)

    # this is just to get the images from the dataloader to be used by ReconstructionDataset
    # we will need them for the compression_rate_predictor which will get the latents of the images to make bpp predictions.
    dataloader = instantiate(cfg.experiment.dataset)

    # This dataset holds the mse_errors, vgg_errors for all the images for different 
    # values of k_values and the  bpp.
    recon_dataset = ReconstructionDataset_Heuristic(
        reconstruction_data=reconstruction_data,
        edge_ratio_information=edge_ratio_information
    )

    # Convert recon_dataset into a DataLoader
    recon_dataloader = DataLoader(recon_dataset, batch_size=cfg.experiment.reconstruction_dataset.batch_size, shuffle=False)

    # ---------------------------
    # Model, optimizer, loss
    # ---------------------------
    token_count_predictor = instantiate(cfg.experiment.model).to(device)
    optimizer = instantiate(cfg.experiment.optimizer, params=token_count_predictor.parameters())
    loss = instantiate(cfg.experiment.training.loss)

    # ---------------------------
    # SIMPLE CHECKPOINT RESUME
    #   - We use ONE file only (e.g., ".../last.pt").
    #   - If it exists, load it and continue from its saved epoch.
    #   - If not, start from epoch 0.
    # ---------------------------
    # Put a single path in your config, e.g.:
    # cfg.experiment.checkpoint_path = "checkpoints/last.pt"
    checkpoint_path = cfg.experiment.checkpoint_path
    start_epoch = 0  # default: start from scratch

    if os.path.isfile(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Loading and resuming...")
        ckpt = torch.load(checkpoint_path, map_location=device)

        # Restore model/optimizer and figure out where to resume
        token_count_predictor.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))  # next epoch will be start_epoch
        last_loss = ckpt.get("loss", None)
        if last_loss is not None:
            print(f"Resuming from epoch {start_epoch} with last avg loss = {last_loss:.6f}")
        else:
            print(f"Resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting from epoch 0.")

    # ---------------------------
    # Training loop (resume-aware)
    # ---------------------------
    num_epochs = cfg.experiment.training.num_epochs
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        token_count_predictor.train()

        for batch in recon_dataloader:
            vgg_error = batch["vgg_error"].to(device).float().unsqueeze(1)
            true_token_count = batch["k_value"].to(device).float()  # keep as-is if your loss handles mapping

            optimizer.zero_grad()
            edge_ratio = batch["edge_ratio"].to(device).float().unsqueeze(1)

            # Forward: model returns logits [B, 256] (classes 0..255 ↔ counts 1..256)
            token_count_prediction = token_count_predictor(edge_ratio, vgg_error)

            # Loss with Gaussian soft targets (your custom CE-style loss)
            current_loss = loss(token_count_prediction, true_token_count).mean()

            current_loss.backward()
            optimizer.step()

            epoch_loss += float(current_loss.item())

        avg_loss = epoch_loss / max(1, len(recon_dataloader))
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.6f}")

        # ---------------------------
        # ✅ Log to Weights & Biases
        # ---------------------------
        wandb.log({
            "epoch": epoch + 1,
            "avg_loss": avg_loss
        })

        # ---------------------------
        # Save/overwrite the single checkpoint file
        #   - We store 'epoch' as the index of the *next* epoch to run.
        #     That way, if we load later, training continues at the correct loop index.
        # ---------------------------
        torch.save(
            {
                "epoch": epoch + 1,  # next epoch to run
                "model_state_dict": token_count_predictor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()