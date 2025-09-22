import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL
from typing import List, Optional

class NeuralTokenCountPredictor(nn.Module):
    """
    Configurable predictor of token count given:
      - image (encoded by SD VAE to latents),
      - an auxiliary conditioning vector (e.g., tolerated loss).

    Key design choices are configurable:
      * num_classes:      number of discrete token-count classes (e.g., 256).
      * pooling:          'gap' (global average pool) or 'flatten'.
      * feature_dim:      required only for 'flatten' (e.g., 4*(H/8)*(W/8) = 4096 for 256x256).
      * loss_dim:         dimension of the auxiliary conditioning vector (e.g., 1 for a single scalar).
      * hidden_dims:      MLP hidden sizes (list). [] means a single Linear to num_classes.
      * activation:       'relu' | 'gelu' | 'silu' | 'tanh'.
      * dropout:          dropout probability applied between hidden layers.
      * hf_hub_path:      VAE checkpoint.
    """

    def __init__(self, hf_hub_path: str = "stabilityai/sd-vae-ft-mse", freeze_vae: bool = True, 
                 num_classes: int = 256, pooling: str = 'GAP', feature_dim: Optional[int] = None, 
                 dropout: float = 0.0, hidden_dims: Optional[List[int]] = None,
                 activation: str = "gelu", loss_dim: Optional[int] = 1):
        super().__init__()

        # map string to activation module
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU()
        }
        act_fn = activations[activation.lower()]
        if act_fn is None:
            raise ValueError(f"Unsupported activation: {activation}. Supported: {list(activations.keys())}")

         # ---- 1) Load SD VAE encoder (we only need the encoder path) ----
        self.vae = AutoencoderKL.from_pretrained(hf_hub_path, low_cpu_mem_usage=False)
        if freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad = False
            self.vae.eval()  # deterministic stats


        # ---- 2) Head configuration (all from config, nothing hardcoded) ----
        self.num_classes = int(num_classes)
        self.pooling = pooling.lower()
        self.loss_dim = int(loss_dim)
        self.hidden_dims = list(hidden_dims or [])
        self.dropout_p = float(dropout)
        self.activation = activation 

        # Determine latent feature dimension after pooling
        # - For 'gap' we can read latent channel count from the VAE config (usually 4 for SD VAEs).
        # - For 'flatten' we cannot know H',W' at init without assuming input size,
        #   so we require feature_dim from the config.
        if self.pooling == "gap":
            # diffusers AutoencoderKL exposes latent_channels in config for SD VAEs
            latent_ch = getattr(self.vae.config, "latent_channels", 4)  # fallback to 4 if missing
            pooled_feat_dim = latent_ch  # GAP: [B, C, H', W'] -> [B, C]
        elif self.pooling == "flatten":
            if feature_dim is None:
                raise ValueError(
                    "feature_dim is required when pooling='flatten'. "
                    "Tip: for SD VAE with 256x256 inputs: feature_dim = 4*(256/8)*(256/8) = 4096."
                )
            pooled_feat_dim = int(feature_dim)
        else:
            raise ValueError("pooling must be 'gap' or 'flatten'")

        # Final input dim to the MLP head = pooled features + conditioning vector
        in_dim = pooled_feat_dim + self.loss_dim

        # ---- 3) Build a small MLP classifier head to num_classes ----
        layers: List[nn.Module] = []
        prev = in_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_fn)
            if self.dropout_p > 0:
                layers.append(nn.Dropout(self.dropout_p))
            prev = h
        layers.append(nn.Linear(prev, self.num_classes))  # final logits
        self.classifier = nn.Sequential(*layers)

    @torch.no_grad()
    def _encode_latents(self, images: torch.Tensor) -> torch.Tensor:
        """
        Deterministic VAE encoding (mode of posterior).
        images: [B, 3, H, W] in the VAE's expected range (often [-1, 1]).
        returns: [B, C, H/8, W/8] for SD VAEs (C ~ 4).
        """
        posterior = self.vae.encode(images).latent_dist
        return posterior.mode()
    
    def _pool(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Apply the configured pooling to latents.
          - 'gap'     : mean over spatial dims -> [B, C]
          - 'flatten' : flatten spatial dims   -> [B, C*H'*W']
        """
        if self.pooling == "gap":
            return latents.mean(dim=(2, 3))
        else:  # 'flatten'
            return latents.flatten(start_dim=1)

    def forward(self, images: torch.Tensor, tolerated_loss: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W]  (normalize to [-1, 1] if using SD VAEs)
        tolerated_loss: [B, 1] (scalar per image)

        returns:
            logits: [B, 256] (class 0..255 -> token count 1..256)
        """

        # 1) Encode to latents
        latents = self._encode_latents(images)

        # 2) Pool per config
        pooled = self._pool(latents)  # [B, pooled_feat_dim]

        # 3) Concatenate conditioning vector, which is the loss
        if tolerated_loss.dim() == 1:
            tolerated_loss = tolerated_loss.unsqueeze(1)  # [B] -> [B,1] for loss_dim==1
        feats = torch.cat([pooled, tolerated_loss], dim=1)  # [B, pooled_feat_dim + loss_dim]

        # linear classifier -> [B, 256]
        logits = self.classifier(feats)
        return logits

    @staticmethod
    def logits_to_token_count(logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to predicted counts in [1..256].
        """
        preds = logits.argmax(dim=1) + 1
        return preds