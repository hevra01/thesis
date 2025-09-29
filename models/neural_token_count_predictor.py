import torch
import torch.nn as nn
import warnings
from diffusers import AutoencoderKL
from typing import List, Optional
from transformers import CLIPModel
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor

RESNET_FACTORIES = {
    "resnet18":  (resnet18,  ResNet18_Weights.DEFAULT),
    "resnet34":  (resnet34,  ResNet34_Weights.DEFAULT),
    "resnet50":  (resnet50,  ResNet50_Weights.DEFAULT),
}

# --- Helper functions for activations and normalizations ---
def make_act(name: str) -> nn.Module:
    """
    Map a string to an activation module.
    """
    name = name.lower()
    if name == "relu":  return nn.ReLU(inplace=True)
    if name == "gelu":  return nn.GELU()
    if name == "silu":  return nn.SiLU()   # same as Swish, default in Stable Diffusion
    if name == "tanh":  return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")

def make_norm(kind: str, num_ch: int, num_groups: int = 8) -> nn.Module:
    """
    Map a string to a normalization layer.
    - "group": good for small batch sizes, works well in vision.
    - "batch": standard BatchNorm2d, needs larger batches.
    - "none" : skip normalization.
    """
    kind = kind.lower()
    if kind == "group": return nn.GroupNorm(num_groups=min(num_groups, num_ch), num_channels=num_ch)
    if kind == "batch": return nn.BatchNorm2d(num_ch)
    if kind == "none":  return nn.Identity()
    raise ValueError(f"Unknown norm: {kind}")

class NeuralTokenCountPredictor(nn.Module):
    """
    Predicts the token count from an input image. Optionally conditions
    on a user-specified tolerated reconstruction loss (a scalar per item).

    If `use_loss_condition = True`, the forward pass expects `tolerated_loss`
    and concatenates a learned embedding of it to image features.
    Otherwise the model ignores `tolerated_loss`.
    """
    def __init__(
        self,
        backbone_type: str = "resnet",               # backbone for feature extraction: "sd-vae" | "resnet" | "none"
        
        hf_hub_path: str = "stabilityai/sd-vae-ft-mse",  # pretrained VAE
        freeze_vae: bool = True,                         # freeze encoder weights
        
        resnet_name="resnet18", # resnet model
        resnet_layer="layer2", # layer to extract features from
        resnet_feature_dim=128, # feature dimension of the extracted layer
        resnet_pretrained=True,

        clip_model_name: str = "openai/clip-vit-base-patch32", # CLIP model name (if used)
        clip_feature_dim: int = 512,  # CLIP feature dimension 
        clip_pretrained: bool = True,  # whether to use pretrained CLIP model

        head: str = "regression",            # "classification" | "regression"
        num_classes: int = 256,                 # classification target size

        # --- conditioning vector ---
        # --- optional loss conditioning ---
        use_loss_condition: bool = True,
        loss_dim: int = 1,       # e.g. tolerated loss (scalar)
        loss_proj_dim: int = 64, # project conditioning vector into this many dims

        # --- conv head ---
        conv_channels: List[int] = (64, 128, 192, 256),  # out_channels for each conv
        conv_strides:  List[int] = (1,   2,   2,   1),   # stride for each conv
        norm: str = "group",                             # group | batch | none
        num_groups: int = 8,                             # groups for GroupNorm
        activation: str = "silu",                   # activation after conv
        dropout2d: float = 0.10,                         # spatial dropout prob
        post_conv_pool: str = "none",                    # "none" (flatten) | "gap"

        # --- fc head ---
        fc_hidden: List[int] = (512, 256),               # MLP hidden sizes
        device: str = "cuda"                            # device to run on
    ):
        super().__init__()

        # --- config / switches ---
        self.device = device
        self.head_type = head.lower()
        self.backbone_type = backbone_type.lower()
        self.use_loss_condition = use_loss_condition

        # set the backbone accordingly
        if self.backbone_type == "sd-vae":
            print("Using VAE encoder for image feature extraction.")
            # ---- 1. VAE encoder (latent extractor) ----
            self.backbone = AutoencoderKL.from_pretrained(hf_hub_path, low_cpu_mem_usage=False)
            if freeze_vae:
                for p in self.backbone.parameters():
                    p.requires_grad = False
                self.backbone.eval()  # avoid updating stats
    
            latent_ch = getattr(self.backbone.config, "latent_channels", 4)  # default for SD
            c_in = latent_ch
        elif self.backbone_type == "resnet":
            # build a feature extractor that returns a chosen internal layer
            factory, weights = RESNET_FACTORIES[resnet_name]
            model = factory(weights=weights if resnet_pretrained else None)
            model.eval()
            self.backbone = create_feature_extractor(model, return_nodes={resnet_layer: "feat"})
            c_in = resnet_feature_dim
        elif self.backbone_type == "clip":
            print(f"Using CLIP ({clip_model_name}) for image feature extraction.")
    
            self.backbone = CLIPModel.from_pretrained(clip_model_name)
            if not clip_pretrained:
                print("Warning: CLIP without pretrained weights is unusual.")
            
            # CLIP encoders output a pooled embedding (batch_size, feature_dim)
            c_in = self.backbone.visual_projection.out_features  # e.g., 512 or 768

        else:
            print("Skipping VAE encoder & resnet; using raw images as input.")
            c_in = 3  # RGB images

        self.num_classes = num_classes
        self.post_conv_pool = post_conv_pool.lower()
        assert self.post_conv_pool in {"none", "gap"}, "pool must be 'none' or 'gap'"

        # ---- 2. Convolutional head ----
        blocks = []
        act_conv = make_act(activation)

        for c_out, s in zip(conv_channels, conv_strides):
            # Each block = Conv → Norm → Activation (+ optional dropout)
            blocks += [
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=s, padding=1, bias=False),
                make_norm(norm, c_out, num_groups),
                act_conv,
            ]
            if dropout2d and dropout2d > 0:
                blocks += [nn.Dropout2d(dropout2d)]
            c_in = c_out

        self.conv = nn.Sequential(*blocks)
        self.flatten = nn.Flatten(start_dim=1)  # used if pool="none"

        # ------------------------------------------------------------------
        # Optional loss projection
        # Only create the module if we plan to use it. This makes checkpointing
        # and JIT traces cleaner when conditioning is disabled.
        # ------------------------------------------------------------------
        if self.use_loss_condition:
            assert loss_dim > 0 and loss_proj_dim > 0, \
                "loss_dim and loss_proj_dim must be > 0 when use_loss_condition=True."
            self.loss_proj = nn.Sequential(
                nn.Linear(loss_dim, loss_proj_dim),
                nn.SiLU(),
                nn.Linear(loss_proj_dim, loss_proj_dim),
            )
            self.loss_proj_dim = loss_proj_dim
        else:
            self.loss_proj = None
            self.loss_proj_dim = 0  # contributes nothing to the head input

        # ---- 4. Fully-connected head (lazy init) ----
        # We can’t build it until we know the conv output size,
        # so we initialize as None and build it after the first forward pass.
        self.fc_hidden = list(fc_hidden)
        self.fc_activation = activation
        self.fc_dropout = dropout2d
        # head is built lazily on first forward once input dim is known
        self.head: Optional[nn.Sequential] = None

    def _build_head(self, feat_dim: int):
        """
        Build the FC head dynamically once we know conv output size.
        """
        layers = []
        act_fc = make_act(self.fc_activation)

        for h in self.fc_hidden:
            layers += [nn.Linear(feat_dim, h), act_fc]
            if self.fc_dropout > 0:
                layers += [nn.Dropout(self.fc_dropout)]
            feat_dim = h

        # --- Final linear depends on head type ---
        if self.head_type == "classification":
            # logits: no activation
            layers += [nn.Linear(feat_dim, self.num_classes)]
        else:
            # regression to [0,1]
            layers += [nn.Linear(feat_dim, 1)]
            layers += [nn.Sigmoid()]  # guarantees output in [0,1]

        self.head = nn.Sequential(*layers).to(self.device)

    @torch.no_grad()
    def _extract_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract feature map from the backbone.
        Input:  images [B, 3, H, W] in [-1,1]
        Output: features
          - For conv backbones (sd-vae, resnet, raw): [B, C, H', W']
          - For CLIP: [B, D] pooled embeddings
        """
        if self.backbone_type == "sd-vae":
            posterior = self.backbone.encode(images).latent_dist
            return posterior.mode()  # deterministic (mean of distribution)
        elif self.backbone_type == "resnet":
            return self.backbone(images)["feat"]       # [B, C, H', W']
        elif self.backbone_type == "clip":
            return self.backbone.vision_model(pixel_values=images).pooler_output
        else:
            return images                              # [B, 3, H, W]


    def forward(self, images: torch.Tensor, tolerated_loss: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass:
          1. Encode image into latents.
          2. Apply conv head.
          3. Pool or flatten.
          4. Concat with conditioning embedding.
          5. Pass through FC classifier → logits.

        Args:
            images: [B, 3, H, W]
            tolerated_loss: [B, 1] or [B] (scalar conditioning feature)
        Returns:
            output: [B, num_classes] for classification, or [B, 1] for regression
        """

        # --- Sanity around loss conditioning ---
        if self.use_loss_condition:
            if tolerated_loss is None:
                raise ValueError("tolerated_loss must be provided when use_loss_condition=True.")
            if tolerated_loss.dim() == 1:
                tolerated_loss = tolerated_loss.unsqueeze(1)

        # --- Backbone / conv pathway ---
        cnn_input = self._extract_feature_map(images)

        # If backbone outputs 1D features (e.g., CLIP), conv blocks (2D) are incompatible.
        if cnn_input.dim() == 2: # [B, D]
            if len(self.conv) > 0:
                raise RuntimeError(
                    "Backbone produced 2D features but a conv head is configured. "
                    "For CLIP, set conv_channels: [] in config so no Conv2d layers are added."
                )
            feats = cnn_input  # [B, D]
        else:
            feats = self.conv(cnn_input)  # [B, C, H', W'] after convs

        # --- Pooling / Flattening to [B, F] ---
        if feats.dim() == 2:
            if self.post_conv_pool == "gap":
                warnings.warn(
                    "post_conv_pool='gap' requested but features are 2D (e.g., CLIP). "
                    "Skipping GAP and using the features as-is. Set post_conv_pool='none' to silence this warning.",
                    RuntimeWarning,
                )
            # keep feats as [B, F]
        else:
            if self.post_conv_pool == "gap":
                feats = feats.mean(dim=(2, 3))  # [B, C]
            elif self.post_conv_pool == "none":
                feats = self.flatten(feats)     # [B, C*H'*W']
            else:
                raise ValueError(f"Unknown post_conv_pool: {self.post_conv_pool}")

        # --- Optional loss embedding and concatenation ---
        if self.use_loss_condition:
            loss_emb = self.loss_proj(tolerated_loss)   # [B, loss_proj_dim]
            x = torch.cat([feats, loss_emb], dim=1)     # [B, F + loss_proj_dim]
        else:
            x = feats                                   # [B, F]

        # --- Build FC head lazily the first time ---
        if self.head is None:
            self._build_head(x.size(1))

        # --- Head ---
        out = self.head(x)
        return out

    @staticmethod
    def logits_to_token_count(logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to predicted token count in [1..num_classes].
        """
        return logits.argmax(dim=1) + 1
