import torch
from flextok.flextok_wrapper import FlexTokFromHub
from data.utils.dataloaders import get_imagenet_dataloader
from flextok.utils.misc import detect_bf16_support, get_bf16_context
from reconstruction_loss import MAELoss, VGGPerceptualLoss, reconstruction_error
import json
from itertools import islice
import hydra
import torch
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path="conf", config_name="prep_dataset_compression_reconstruction")
def main(cfg):
 
    # Set device
    device = torch.device(cfg.device)

    # Instantiate the FlexTok model using Hydra. This model is performing
    # compression and reconstruction of images.
    model = instantiate(cfg.experiment.model).eval().to(device)

    # Configure the dataloader
    dataloader = instantiate(cfg.experiment.dataset)

    # Loss functions
    # Instantiate loss functions dynamically
    loss_fns = [instantiate(loss_cfg) for loss_cfg in cfg.experiment.loss_functions]

    # Move all loss functions to the specified device
    for i, loss_fn in enumerate(loss_fns):
        if hasattr(loss_fn, "to"):
            loss_fns[i] = loss_fn.to(device)

    # Compression rates to test
    k_keep_list = cfg.experiment.k_keep_list

    # Check start_batch_idx and end_batch_idx 
    start_batch_idx = cfg.experiment.start_batch_idx
    end_batch_idx = cfg.experiment.end_batch_idx

    # Format output filename to include batch range
    base_output_path = cfg.experiment.output_path  
    output_path = f"{base_output_path}_{start_batch_idx:04d}_{end_batch_idx:04d}.json"

    reconstruction_errors = []

    # since this task is embarassingly parallel, we will use
    # multiple GPUs to speed up the process, where each GPU will handle 
    # a specific range of batches.
    sliced_loader = islice(dataloader, start_batch_idx, end_batch_idx)

    enable_bf16 = detect_bf16_support()
    print('BF16 enabled:', enable_bf16)

    # Load normalization values from Hydra config
    mean = torch.tensor(cfg.experiment.image_normalization.mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(cfg.experiment.image_normalization.std).view(1, 3, 1, 1).to(device)

    # Process each batch in the dataloader
    for batch_idx, (images, labels) in enumerate(sliced_loader, start=start_batch_idx):
        images = images.to(device)

        # Tokenize the images
        with get_bf16_context(enable_bf16):
            tokens_list = model.tokenize(images)

        images_unnorm = images * std + mean

        # Compute reconstruction errors for different k values
        for k_keep in k_keep_list:
            # Keep only the first k tokens
            tokens_list_filtered = [t[:, :k_keep] for t in tokens_list]
            
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
            for img_idx in range(images.size(0)):
                reconstruction_errors.append({
                    "image_id": batch_idx * dataloader.batch_size + img_idx,
                    "k_value": k_keep,
                    "mse_error": loss_dict["MAELoss"][img_idx].item(),
                    "vgg_error": loss_dict["VGGPerceptualLoss"][img_idx].item()
                })

            # Save after each batch (overwrite)
            with open(output_path, "w") as f:
                json.dump(reconstruction_errors, f)

        print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")
        
    # Save reconstruction errors to a JSON file
    with open(output_path, "w") as f:
        json.dump(reconstruction_errors, f)

    print("Reconstruction errors saved in:", output_path)

    
if __name__ == "__main__":
    main()