from typing import Optional
import os
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetCond(nn.Module):
    """
    ResNet backbone + (optional) scalar conditioning.

    This version replaces the "concat features + cond_mlp output" approach with
    FiLM-style conditioning:

        f = backbone(x)                             # [B, in_dim]
        (gamma, beta) = film(cond_scalar)           # [B, in_dim] each
        f_mod = f * (1 + gamma) + beta              # [B, in_dim]
        out = head(f_mod)

    FiLM encourages *feature modulation* based on the condition, which often
    prevents the network from ignoring the condition (a common failure mode
    with simple concatenation).
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        backbone: str = "resnet50",
        pretrained: bool = True,
        use_condition: bool = True,
        freeze_backbone: bool = False,
        keep_backbone_eval: bool = True,
        task_type: str = "classification",
        # --- optional: load weights from a checkpoint ---
        checkpoint_path: Optional[str] = None,
        checkpoint_strict: bool = True,
        map_location: Optional[str] = "cpu",
    ):
        super().__init__()
        weights = "IMAGENET1K_V2" if pretrained else None
        self.backbone = getattr(models, backbone)(weights=weights)

        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Freeze backbone params if requested (train only the head / FiLM)
        self.freeze_backbone = bool(freeze_backbone)
        self.keep_backbone_eval = bool(keep_backbone_eval)
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if self.keep_backbone_eval:
                # Avoid BatchNorm running stats updates
                self.backbone.eval()

        self.use_condition = bool(use_condition)

        # -------------------------
        # FiLM conditioning module
        # -------------------------
        # We keep the same "scalar -> small MLP" idea, but instead of producing a
        # 32-d vector to concatenate, we produce (gamma, beta) vectors that
        # modulate backbone features directly.
        #
        # NOTE:
        # - This code assumes the condition you pass is a *scalar float* (shape [B] or [B,1]).
        # - If you later want categorical conditions, replace this with an Embedding
        #   and produce gamma/beta from the embedding.
        self.film = (
            nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 2 * in_dim),  # outputs concatenated [gamma | beta]
            )
            if self.use_condition
            else None
        )

        # -------------------------
        # Task head (unchanged idea)
        # -------------------------
        if task_type == "regression":
            # With FiLM, the head only needs backbone feature dim
            self.head = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        elif task_type == "classification":
            if num_classes is None:
                raise ValueError("num_classes must be provided for classification task")
            self.head = nn.Linear(in_dim, num_classes)
        else:
            raise ValueError(f"Unknown task: {task_type}")

        # ------------------------------------------------------------
        # Optional checkpoint loading
        # - Accept either a raw state_dict or a dict with 'model_state_dict'.
        # - Strip an optional 'module.' prefix (from DDP checkpoints).
        # ------------------------------------------------------------
        if checkpoint_path:
            try:
                if not os.path.isfile(checkpoint_path):
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
                ckpt = torch.load(checkpoint_path, map_location=map_location or "cpu")

                state_dict = ckpt.get("model_state_dict", ckpt)

                # Handle DDP 'module.' prefix if present
                if any(k.startswith("module.") for k in state_dict.keys()):
                    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

                missing, unexpected = self.load_state_dict(state_dict, strict=checkpoint_strict)
                if missing or unexpected:
                    print(
                        f"[ResNetCond] Loaded checkpoint with missing keys: {len(missing)}, "
                        f"unexpected keys: {len(unexpected)}"
                    )
            except Exception as e:
                raise RuntimeError(f"Checkpoint loading failed: {e}")

    def forward(self, x, recon_loss_scalar=None):
        # Ensure backbone stays in eval during training if frozen
        if self.freeze_backbone and self.keep_backbone_eval:
            self.backbone.eval()

        # Backbone features
        f = self.backbone(x)  # [B, in_dim]

        if self.use_condition:
            if recon_loss_scalar is None:
                raise ValueError("recon_loss_scalar must be provided when use_condition=True")

            # Ensure condition is shaped [B,1]
            l = recon_loss_scalar.view(-1, 1).float()

            # Keep your original log transform (useful when the scalar spans a wide range).
            # If your scalar can be 0, this is safe due to +1e-8.
            l = torch.log(l + 1e-8)

            # Produce FiLM parameters and modulate features
            gb = self.film(l)  # [B, 2*in_dim]
            gamma, beta = gb.chunk(2, dim=1)  # each [B, in_dim]

            # FiLM modulation (feature-wise affine transform)
            f = f * (1.0 + gamma) + beta

        return self.head(f)

    def train(self, mode: bool = True):
        """Override to keep frozen backbone in eval when training.

        When `freeze_backbone=True` and `keep_backbone_eval=True`, the backbone
        is kept in eval mode even if the module is in train mode; this avoids
        BatchNorm running-stat updates while training the head/FiLM.
        """
        super().train(mode)
        if self.freeze_backbone and self.keep_backbone_eval:
            self.backbone.eval()
        return self

    def head_parameters(self):
        """Return an iterable of parameters for the trainable head only.

        Includes `head` and, if enabled, FiLM parameters (`film`).
        Useful for constructing an optimizer that excludes the frozen backbone.
        """
        params = list(self.head.parameters())
        if self.use_condition and self.film is not None:
            params += list(self.film.parameters())
        return params