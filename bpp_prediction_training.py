import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.utils.dataloaders import ReconstructionDataset, get_imagenet_dataloader  # Ensure this is defined elsewhere
import json
from hydra.utils import instantiate
from models.bpp_predictor import CompressionRatePredictor


@hydra.main(version_base=None, config_path="conf", config_name="bpp_estimator_training")
def main(cfg: DictConfig):

    # Device configuration
    device = torch.device(cfg.device)

    # Load JSONs
    # this data holds images, mse_errors, vgg_errors, bpp for different k_values
    with open(cfg.experiment.reconstruction_dataset.reconstruction_data_path, "r") as f:
        reconstruction_data = json.load(f)

    # this is used for entropy coding, it holds the registers for each image
    # e.g., [[registers], [registers], ...] found by flextok.
    with open(cfg.experiment.reconstruction_dataset.all_registers_path, "r") as f:
        all_registers = json.load(f)

    # Compute number of pixels (e.g., 256x256)
    # this will be used to compute bits per pixel (bpp) = num_bits / num_pixels
    num_pixels = cfg.experiment.reconstruction_dataset.num_pixels

    # this is just to get the images from the dataloader to be used by ReconstructionDataset
    # we will need them for the compression_rate_predictor which will get the latents of the images to make bpp predictions. 
    dataloader_for_img = instantiate(cfg.experiment.dataset)

    # This dataset holds the mse_errors, vgg_errors for all the images for different 
    # values of k_values and the  bpp.
    recon_dataset = ReconstructionDataset(
        reconstruction_data=reconstruction_data,
        all_registers=all_registers,
        num_pixels=num_pixels,
        dataloader=dataloader_for_img,
    )

    # Convert recon_dataset into a DataLoader
    recon_dataloader = DataLoader(recon_dataset, batch_size=cfg.experiment.reconstruction_dataset.batch_size, shuffle=False)

    # Instantiate the CompressionRatePredictor, which is the model to be trained
    compression_rate_predictor = instantiate(cfg.experiment.model)
    compression_rate_predictor = compression_rate_predictor.to(device)

    # Define optimizer
    optimizer = instantiate(cfg.experiment.optimizer, params=compression_rate_predictor.parameters())

    mse_loss = instantiate(cfg.experiment.training.loss)

    # Training loop
    num_epochs = cfg.experiment.training.num_epochs
    for epoch in range(num_epochs):
        epoch_loss = 0  # Accumulator for total epoch loss

        for batch in recon_dataloader:
            images = batch["image"].to(device).float()  # Ensure images are in float32
            mse_error = batch["mse_error"].to(device).float().unsqueeze(1)  # Ensure mse_error is in float32
            bpp = batch["bpp"].to(device).float()  # Ensure bpp is in float32

            # Set model to training mode
            compression_rate_predictor.train()

            # Forward pass
            optimizer.zero_grad()
            predicted_bpp = compression_rate_predictor(images, mse_error)  # [B]

            # Compute MSE loss between predicted and true bpp
            loss = mse_loss(predicted_bpp, bpp)

            # Backward pass and optimizer update
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Optionally compute average loss per batch
        avg_loss = epoch_loss / len(recon_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.6f}")

        # Save model checkpoint
        checkpoint_path = cfg.experiment.checkpoint_path + f"{epoch + 1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": compression_rate_predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, checkpoint_path)

        print(f"Saved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":

    main()