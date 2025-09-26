import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from data.utils.dataloaders import ReconstructionDataset_Neural
import json
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="../conf", config_name="neural_baseline_training")
def main(cfg: DictConfig):

    # Device configuration
    device = torch.device(cfg.device)

    # Initialize W&B and dump Hydra config
    wandb.init(
        project="dataset_prep", 
        name=f"CNN_neural_baseline_training_loss_regression_VAE", 
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # Load JSONs
    # this data holds images, mse_errors, vgg_errors, token counts for different k_values
    with open(cfg.experiment.reconstruction_dataset.reconstruction_data_path, "r") as f:
        reconstruction_data = json.load(f)

    # this is just to get the images from the dataloader to be used by ReconstructionDataset
    # we will need them for the compression_rate_predictor which will get the latents of the images to make bpp predictions. 
    dataloader = instantiate(cfg.experiment.dataset)

    shuffle = cfg.experiment.reconstruction_dataset.shuffle

    # This dataset holds the mse_errors, vgg_errors for all the images for different 
    # values of k_values and the  bpp.
    recon_dataset = ReconstructionDataset_Neural(
        reconstruction_data=reconstruction_data,
        dataloader=dataloader
    )

    batch_size = cfg.experiment.reconstruction_dataset.batch_size
    

    # Convert recon_dataset into a DataLoader
    recon_dataloader = DataLoader(recon_dataset, batch_size=batch_size, shuffle=shuffle)

    # ---------------------------
    # Model, optimizer, loss
    # ---------------------------
    token_count_predictor = instantiate(cfg.experiment.model).to(device)

    # Count only trainable parameters
    num_params = sum(p.numel() for p in token_count_predictor.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    optimizer = instantiate(cfg.experiment.optimizer, params=token_count_predictor.parameters())
    training_loss = instantiate(cfg.experiment.training.loss_training)
    mae_loss = instantiate(cfg.experiment.training.loss_analysis)

    # check if regression or classification
    is_reg = getattr(cfg.experiment.model, "head", "classification") == "regression"
    num_classes = getattr(cfg.experiment.model, "num_classes", 256)


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

    # set the model in training mode
    token_count_predictor.train()

    print("len(recon_dataloader):", len(recon_dataloader))
    dataset_size = len(recon_dataloader.dataset)
    print("dataset_size:", dataset_size)
    
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        epoch_loss_analysis = 0.0

        # Accumulators for epoch-wide MAE metrics
        total_mae_expected_sum = 0.0  # classification (expected-count)
        total_mae_argmax_sum = 0.0    # classification (argmax)

        for batch in recon_dataloader:
            images = batch["image"].to(device).float()
            vgg_error = batch["vgg_error"].to(device).float().unsqueeze(1)
            k_value = batch["k_value"].to(device).float().unsqueeze(1)  

            optimizer.zero_grad()

            # Forward:
            #  - classification: logits [B, C] where classes 1..C map to counts 1..C
            #  - regression: output [B, 1] with scaled target
            logits = token_count_predictor(images, vgg_error)

            if is_reg:
                # Scale target to [0,1]: y = (K-1)/(C-1)
                y = (k_value - 1.0) / float(num_classes - 1)


                # loss is MAE between model output and scaled target
                # logits here are actually a scalar output [B,1]
                loss = training_loss(logits, y)

                # Use the model prediction (not the target!) to compute token-count MAE
                pred_k = (logits * (num_classes - 1) + 1.0).round().clamp(1, num_classes)

                # Per-sample MAE and epoch accumulation
                loss_analysis = mae_loss(pred_k, k_value)  

            else:
                # Your Gaussian-CE expects float K; keep as-is
                loss = training_loss(logits, k_value)
                # For analysis, prefer expected-count MAE which aligns with soft targets
                prob = torch.softmax(logits, dim=1)
                classes = torch.arange(1, num_classes + 1, device=logits.device, dtype=prob.dtype).unsqueeze(0)
                expected_count = (prob * classes).sum(dim=1)

                # Also compute argmax-based MAE for reference
                predicted_token_count = token_count_predictor.logits_to_token_count(logits).float()
                per_mae_expected = mae_loss(expected_count, k_value)  # [B]
                per_mae_argmax = mae_loss(predicted_token_count, k_value)  # [B]

                # Use expected-count MAE as the main analysis metric (batch mean)
                loss_analysis = per_mae_expected

                total_mae_expected_sum += float(per_mae_expected.sum().item())
                total_mae_argmax_sum += float(per_mae_argmax.sum().item())
                

            # Backprop + optimization step
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_loss_analysis += float(loss_analysis.item())

            print("loss:", float(loss.item()))

        # to find the average loss over an epoch, divide the accumulated loss by the number of batches.
        avg_loss = epoch_loss / len(recon_dataloader)
        avg_loss_analysis = epoch_loss_analysis / len(recon_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.6f}, Avg Loss Analysis (MAE): {avg_loss_analysis:.6f}")

        # ---------------------------
        # ✅ Log to Weights & Biases
        # ---------------------------
        # Log to W&B, include epoch-averaged MAEs
        log_dict = {
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "avg_loss_analysis": avg_loss_analysis,
        }
        # if total_samples > 0:
        #     if is_reg:
        #         log_dict.update({
        #             "mae_epoch": total_mae_reg_sum / float(dataset_size),
        #         })
        #     else:
        #         log_dict.update({
        #             "mae_expected_epoch": total_mae_expected_sum / float(dataset_size),
        #             "mae_argmax_epoch": total_mae_argmax_sum / float(dataset_size),
        #         })
        wandb.log(log_dict)

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