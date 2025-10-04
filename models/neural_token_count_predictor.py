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
        
        # --- optional transformer aggregator (after conv/backbone) ---
        # If enabled, we treat the spatial feature map [B,C,H,W] as a sequence of
        # tokens (length H*W, dim C), add 2D positional encodings, run a
        # Transformer encoder, and pool (cls/mean) to a vector for the FC head.
        use_transformer: bool = False,
        transformer_dim: int = 256,
        transformer_layers: int = 4,
        transformer_nhead: int = 8,
        transformer_mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.1,
        transformer_add_cls: bool = True,          # add a learnable [CLS] token
        transformer_pool: str = "cls",             # cls | mean
        pos_encoding: str = "sincos2d",            # sincos2d | none

        # --- expected input resolution (H, W) ---
        # Provide this to enable eager FC head build for 'flatten' pooling by
        # computing spatial dims analytically instead of waiting for a real batch.
        input_resolution: Optional[List[int]] = None,
        
        device: str = "cuda"                            # device to run on
    ):
        super().__init__()

        # --- config / switches ---
        self.device = device
        self.head_type = head.lower()
        self.backbone_type = backbone_type.lower()
        self.use_loss_condition = use_loss_condition
        self.use_transformer = use_transformer
        self.transformer_pool = transformer_pool.lower()
        assert self.transformer_pool in {"cls", "mean"}
        self.pos_encoding = pos_encoding.lower()
        assert self.pos_encoding in {"sincos2d", "none"}

        #  ---- 1. backbone ----
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
            if resnet_pretrained:
                for p in model.parameters():
                    p.requires_grad = False
            model.eval()
            self.backbone = create_feature_extractor(model, return_nodes={resnet_layer: "feat"})
            c_in = resnet_feature_dim
        elif self.backbone_type == "clip":
            print(f"Using CLIP ({clip_model_name}) for image feature extraction.")
    
            self.backbone = CLIPModel.from_pretrained(clip_model_name)
            if not clip_pretrained:
                print("Warning: CLIP without pretrained weights is unusual.")
            
            if self.use_transformer:
                c_in = self.backbone.vision_model.config.hidden_size   # tokens
            else:
                c_in = self.backbone.visual_projection.out_features     # pooled

        else:
            print("Skipping VAE encoder & resnet; using raw images as input.")
            c_in = 3  # RGB images

        self.num_classes = num_classes
        self.post_conv_pool = post_conv_pool.lower()
        # valid options incl. 'flatten' (explicit); message kept generic
        assert self.post_conv_pool in {"none", "gap", "flatten"}, "post_conv_pool must be one of: none | gap | flatten"

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

        # For transformer we need to know the channel dimension C after convs.
        # "c_in" at this point holds the last conv out_channels (or backbone dim if no convs).
        conv_out_channels = c_in

        # ---- 3. Optional Transformer encoder aggregator ----
        # Note: We build the transformer here because we know the feature dim (C).
        #       Spatial size (H*W) is dynamic; positional encodings will be generated on-the-fly.
        if self.use_transformer:
            # check if the input dim to the transformer is compatible
            # with its own expected dim, if not add a linear projection
            d_model = transformer_dim
            self._t_in_proj = nn.Identity() if conv_out_channels == d_model else nn.Linear(conv_out_channels, d_model)

            # Learnable [CLS] token if requested (pooled representation)
            self._t_add_cls = transformer_add_cls
            if self._t_add_cls:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
                nn.init.trunc_normal_(self.cls_token, std=0.02) # initializes parameter from a truncated normal distribution
            else:
                self.cls_token = None

            # Standard TransformerEncoder (Pre-LN for stability)
            # a single one layer looks like:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, # internal dim
                nhead=transformer_nhead,
                dim_feedforward=int(d_model * transformer_mlp_ratio), # after the attention block, the mlp expands to this many dims
                dropout=transformer_dropout,
                batch_first=True, # [L, B, D] (sequence length first), or [B, L, D] if batch_first=True
                norm_first=True, # x = x + Sublayer(LayerNorm(x)) or x = LayerNorm(x + Sublayer(x)).
                activation="gelu",
            )

            # grouping the transformer layers together
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
            self._t_d_model = d_model
        else:
            self.transformer = None

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

        # ---- 4. Fully-connected head (eager when deterministic) ----
        # We build the FC head immediately when we can determine the input
        # feature dimension purely from configuration:
        #   - Transformer enabled: head input dim = transformer_dim (+ loss proj)
        #   - No transformer and post_conv_pool == 'gap': dim = last conv channels
        #   - CLIP pooled with no convs: dim = CLIP visual_projection.out_features
        # Otherwise (e.g., post_conv_pool == 'flatten' where H' and W' depend on
        # input resolution), we keep the previous lazy build on first forward.
        self.fc_hidden = list(fc_hidden)
        self.fc_activation = activation
        self.fc_dropout = dropout2d
        self.head = None

        # Try to determine head input dim without running a dummy forward
        feat_core_dim: Optional[int] = None
        if self.use_transformer:
            # Transformer pools to transformer_dim by design
            feat_core_dim = self._t_d_model
        else:
            # Non-transformer flow
            if self.backbone_type == "clip" and len(self.conv) == 0:
                # CLIP pooled output (vector); conv head is not used with 2D features
                feat_core_dim = self.backbone.visual_projection.out_features
            elif self.post_conv_pool == "gap":
                # GAP collapses spatial dims; only channels matter
                feat_core_dim = conv_out_channels
            elif self.post_conv_pool == "flatten":
                # If input resolution is provided, we can compute H'*W' analytically.
                if input_resolution is not None:
                    H_in, W_in = int(input_resolution[0]), int(input_resolution[1])

                    # Backbone spatial stride
                    if self.backbone_type == "sd-vae":
                        # Stable Diffusion VAE encoder downsamples by 8 (assumes divisible by 8)
                        backbone_stride = 8
                        C0 = getattr(self.backbone.config, "latent_channels", 4)
                    elif self.backbone_type == "resnet":
                        # ResNet spatial stride at the tapped layer (conv1:2, maxpool:2)
                        # layer1: 4, layer2: 8, layer3: 16, layer4: 32
                        layer_stride_map = {"layer1": 4, "layer2": 8, "layer3": 16, "layer4": 32}
                        # Default to 8 if unknown layer name
                        backbone_stride = layer_stride_map.get(resnet_layer, 8)
                        C0 = None  # will be superseded by conv_out_channels below
                    else:
                        # Raw RGB pathway (no backbone downsample)
                        backbone_stride = 1
                        C0 = None

                    # Additional stride from our conv head
                    conv_stride_total = 1
                    for s in conv_strides:
                        conv_stride_total *= int(s)

                    total_stride = backbone_stride * conv_stride_total
                    if H_in % total_stride != 0 or W_in % total_stride != 0:
                        warnings.warn(
                            f"Input resolution {(H_in, W_in)} not divisible by total stride {total_stride}. "
                            "Using floor division for spatial dims.", RuntimeWarning,
                        )
                    H_out = H_in // total_stride
                    W_out = W_in // total_stride

                    # Channel dim after conv stack (or backbone if no convs)
                    C_out = conv_out_channels
                    feat_core_dim = int(C_out * H_out * W_out)
                else:
                    # No resolution → keep lazy
                    feat_core_dim = None

        if feat_core_dim is not None:
            feat_in = feat_core_dim + (self.loss_proj_dim if self.use_loss_condition else 0)
            self._build_head(feat_in)
        else:
            # We couldn't resolve the head input size statically; require input_resolution
            raise ValueError(
                "Cannot determine FC head input dimension. If post_conv_pool='flatten', "
                "please set input_resolution: [H, W] (and ensure strides are correct)."
            )

    # ------------------------------ Positional Encoding ------------------------------
    @staticmethod
    def _build_2d_sincos_pos_embed(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
        """
        Create a 2D sine-cosine positional encoding of shape [1, H*W, dim].
        Requires dim to be divisible by 4 (half for y, half for x; sin+cos per axis).
        """
        if dim % 4 != 0:
            raise ValueError(f"sincos2d pos_encoding requires dim % 4 == 0, got dim={dim}")
        # Based on common ViT/MAE formulations.
        y_pos = torch.arange(h, dtype=torch.float32, device=device)  # [H]
        x_pos = torch.arange(w, dtype=torch.float32, device=device)  # [W]
        yy, xx = torch.meshgrid(y_pos, x_pos, indexing="ij")        # [H,W]

        omega_y = torch.arange(dim // 4, dtype=torch.float32, device=device)
        omega_y = 1.0 / (10000 ** (omega_y / (dim // 4)))            # [dim/4]
        omega_x = omega_y.clone()

        # [H,W,dim/4]
        out_y = yy[..., None] * omega_y
        out_x = xx[..., None] * omega_x

        # sin/cos and concat along channel
        pos_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=-1)  # [H,W,dim/2]
        pos_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=-1)  # [H,W,dim/2]
        pos = torch.cat([pos_y, pos_x], dim=-1)                           # [H,W,dim]

        pos = pos.view(1, h * w, dim)  # [1, HW, dim]
        return pos

    # (No eager helper needed; head input dim is derived statically when possible.)

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
            outputs = self.backbone.vision_model(pixel_values=images)
            # If a transformer aggregator is used downstream, prefer token sequence.
            return outputs.last_hidden_state if self.use_transformer else outputs.pooler_output
        else:
            return images                              # [B, 3, H, W]

    def _apply_transformer(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Apply Transformer to a feature map or token sequence.
        Inputs:
          - feats: [B, C, H, W] or [B, T, C] or [B, C] (degenerate)
        Returns:
          - pooled: [B, D] where D = transformer_dim
        """
        assert self.transformer is not None, "Transformer path requested but module not built."

        if feats.dim() == 4:
            B, C, H, W = feats.shape
            x = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, HW, C]
        elif feats.dim() == 3:
            B, T, C = feats.shape
            x = feats
        elif feats.dim() == 2:
            B, C = feats.shape
            x = feats.unsqueeze(1)  # [B,1,C]
        else:
            raise ValueError(f"Unsupported feats shape for transformer: {feats.shape}")

        # Project to transformer dimension if needed
        x = self._t_in_proj(x)

        # Positional encodings (for 2D only)
        if self.pos_encoding == "sincos2d" and feats.dim() == 4:
            _, _, H, W = feats.shape
            pos = self._build_2d_sincos_pos_embed(H, W, self._t_d_model, feats.device)  # [1, HW, D]
            x = x + pos

        # Optional [CLS] token
        if self._t_add_cls:
            cls = self.cls_token.expand(x.size(0), -1, -1)  # [B,1,D]
            x = torch.cat([cls, x], dim=1)                  # [B,1+T,D]

        x = self.transformer(x)  # [B, 1+T, D] or [B, T, D]

        # Pooling
        if self._t_add_cls and self.transformer_pool == "cls":
            pooled = x[:, 0]                # [B, D]
        else:
            if self._t_add_cls:
                x = x[:, 1:]               # drop cls for mean pool
            pooled = x.mean(dim=1)          # [B, D]

        return pooled

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

        # Handle different feature shapes from backbone
        if cnn_input.dim() == 2:
            # 2D vector (e.g., CLIP pooled). Conv2d head cannot consume this.
            if len(self.conv) > 0:
                raise RuntimeError(
                    "Backbone produced 2D features but a conv head is configured. "
                    "For CLIP, set conv_channels: [] in config so no Conv2d layers are added."
                )
            feats = cnn_input  # [B, D]
        elif cnn_input.dim() == 3:
            # 3D tokens (e.g., CLIP last_hidden_state when use_transformer=True)
            if len(self.conv) > 0:
                raise RuntimeError(
                    "Backbone produced 3D token sequence but a Conv2d head is configured. "
                    "When using token sequences, set conv_channels: [] and rely on the Transformer aggregator."
                )
            feats = cnn_input  # [B, T, C]
        else:
            feats = self.conv(cnn_input)  # [B, C, H', W'] after convs (or identity if no convs)

        # --- Optional Transformer aggregator ---
        if self.use_transformer:
            feats_vec = self._apply_transformer(feats)  # [B, D]
        else:
            # --- Pooling / Flattening to [B, F] (classic CNN path) ---
            if feats.dim() == 2:
                if self.post_conv_pool == "gap":
                    warnings.warn(
                        "post_conv_pool='gap' requested but features are 2D (e.g., CLIP). "
                        "Skipping GAP and using the features as-is. Set post_conv_pool='none' to silence this warning.",
                        RuntimeWarning,
                    )
                feats_vec = feats  # [B, F]
            else:
                if self.post_conv_pool == "gap":
                    feats_vec = feats.mean(dim=(2, 3))  # [B, C]
                elif self.post_conv_pool == "flatten":
                    feats_vec = self.flatten(feats)     # [B, C*H'*W']
                else:
                    raise ValueError(f"Unknown post_conv_pool: {self.post_conv_pool}")

        # --- Optional loss embedding and concatenation ---
        if self.use_loss_condition:
            loss_emb = self.loss_proj(tolerated_loss)   # [B, loss_proj_dim]
            x = torch.cat([feats_vec, loss_emb], dim=1) # [B, F + loss_proj_dim]
        else:
            x = feats_vec                               # [B, F]

        # --- Head ---
        out = self.head(x)
        return out

    @staticmethod
    def logits_to_token_count(logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to predicted token count in [1..num_classes].
        """
        return logits.argmax(dim=1) + 1
