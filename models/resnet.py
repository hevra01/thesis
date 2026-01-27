from typing import Optional
import os
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetCond(nn.Module):
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

        # Freeze backbone params if requested (train only the head)
        self.freeze_backbone = bool(freeze_backbone)
        self.keep_backbone_eval = bool(keep_backbone_eval)
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if self.keep_backbone_eval:
                # Avoid BatchNorm running stats updates
                self.backbone.eval()

        self.use_condition = use_condition
        cond_dim = 32 if use_condition else 0
        self.cond_mlp = (
            nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
            ) if use_condition else None
        )

        # the head is either for regression or classification
        if task_type == "regression":
            self.head = nn.Linear(in_dim + cond_dim, 1)
        elif task_type == "classification":  # classification
            self.head = nn.Linear(in_dim + cond_dim, num_classes)
        else:
            raise ValueError(f"Unknown task: {task_type}")

        # ------------------------------------------------------------
        # Optional checkpoint loading
        # - Accept either a raw state_dict or a dict with 'model_state_dict'.
        # - Strip an optional 'module.' prefix (from DDP checkpoints).
        # ------------------------------------------------------------
        if checkpoint_path:
            try:
                # check if the file exists, and if so, load it
                if not os.path.isfile(checkpoint_path):
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
                ckpt = torch.load(checkpoint_path, map_location=map_location or "cpu")

                # If ckpt has a key called "model_state_dict", return its value.
                # Otherwise, return ckpt itself.
                state_dict = ckpt.get("model_state_dict", ckpt)

                # Handle DDP 'module.' prefix if present
                if any(k.startswith("module.") for k in state_dict.keys()):
                    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

                # load the weights into the model and report missing/unexpected keys
                missing, unexpected = self.load_state_dict(state_dict, strict=checkpoint_strict)
                if missing or unexpected:
                    # Provide a concise log for debugging; users can set strict=False to ignore
                    print(
                        f"[ResNetCond] Loaded checkpoint with missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
                    )
            except Exception as e:
                print(f"[ResNetCond] Failed to load checkpoint: {e}")

    def forward(self, x, recon_loss_scalar=None):
        # Ensure backbone stays in eval during training if frozen
        if self.freeze_backbone and self.keep_backbone_eval:
            self.backbone.eval()
        f = self.backbone(x)

        if self.use_condition:
            if recon_loss_scalar is None:
                raise ValueError("recon_loss_scalar must be provided when use_condition=True")
            l = recon_loss_scalar.view(-1, 1)
            l = torch.log(l + 1e-8)
            c = self.cond_mlp(l)
            z = torch.cat([f, c], dim=1)
            return self.head(z)
        else:
            return self.head(f)

    def train(self, mode: bool = True):
        """Override to keep frozen backbone in eval when training.

        When `freeze_backbone=True` and `keep_backbone_eval=True`, the backbone
        is kept in eval mode even if the module is in train mode; this avoids
        BatchNorm running-stat updates while training the head.
        """
        super().train(mode)
        if self.freeze_backbone and self.keep_backbone_eval:
            self.backbone.eval()
        return self

    def head_parameters(self):
        """Return an iterable of parameters for the trainable head only.

        Includes `head` and, if enabled, `cond_mlp`.
        Useful for constructing an optimizer that excludes the frozen backbone.
        """
        params = list(self.head.parameters())
        if self.use_condition and self.cond_mlp is not None:
            params += list(self.cond_mlp.parameters())
        return params
