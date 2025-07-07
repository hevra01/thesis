import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL


class CompressionRatePredictor(nn.Module):
    """
    A model to predict the compression rate based on VAE encoder features and tolerated maximum loss.

    Args:
        hf_hub_path (str): Hugging Face Hub path for the VAE model.
        input_dim (int): Dimension of the flattened VAE encoder features.
        loss_dim (int): Dimension of the tolerated maximum loss (usually 1).
        hidden_dims (list): List of integers specifying the hidden layer sizes for the MLP.
    """

    def __init__(self, input_dim, loss_dim=1, hidden_dims=[512, 256, 128, 64], hf_hub_path='stabilityai/sd-vae-ft-mse'):
        super().__init__()
        
        # Initialize the Stable Diffusion VAE encoder
        # this is pre-trained
        self.vae = AutoencoderKL.from_pretrained(hf_hub_path, low_cpu_mem_usage=False)

        # Define the MLP layers
        mlp_input_dim = input_dim + loss_dim  # Concatenate VAE features and tolerated loss
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(mlp_input_dim, hidden_dim))
            layers.append(nn.ReLU())
            mlp_input_dim = hidden_dim
        layers.append(nn.Linear(mlp_input_dim, 1))  # Output layer for compression rate
        layers.append(nn.Sigmoid())  # Ensure output is between 0 and 1
        self.mlp = nn.Sequential(*layers)

    def forward(self, images, tolerated_loss):
        """
        Forward pass to predict the compression rate.

        Args:
            images (torch.Tensor): Input images of shape [B, C, H, W].
            tolerated_loss (torch.Tensor): Tolerated maximum loss of shape [B, 1].

        Returns:
            torch.Tensor: Predicted compression rate of shape [B, 1].
        """
        # Encode images using the VAE encoder.
        # This outputs a distribution
        vae_output = self.vae.encode(images) 

        # Here, I could've used `.sample()` to get the latents more randomly.
        # So, in vae, each image is encoded to a distribution, and we can sample from it.
        # However, for consistency, I will use `.mode()` to get the most probable latent
        # for the purpose of relating images and reconstruction loss to compression rate.
        vae_latents = vae_output.latent_dist.mode() # shape is [batch_size, 4, 32, 32]

        # Flatten the VAE features
        flattened_features = vae_latents.view(vae_latents.size(0), -1) # torch.Size([batch_size, 4096])

        # Concatenate flattened features with tolerated loss
        input_features = torch.cat([flattened_features, tolerated_loss], dim=1)
        # Pass through the MLP to predict compression rate
        compression_rate = self.mlp(input_features)
        return compression_rate