import torch
import torch.nn as nn
import torchvision.models as models

class ResNetCond(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        use_condition: bool = True,
        freeze_backbone: bool = False,
        keep_backbone_eval: bool = True,
        task: str = "classification",
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
        if task == "regression":
            self.head = nn.Linear(in_dim + cond_dim, 1)
        elif task == "classification":  # classification
            self.head = nn.Linear(in_dim + cond_dim, num_classes)
        else:
            raise ValueError(f"Unknown task: {task}")

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

        Includes `classifier` and, if enabled, `cond_mlp`.
        Useful for constructing an optimizer that excludes the frozen backbone.
        """
        params = list(self.head.parameters())
        if self.use_condition and self.cond_mlp is not None:
            params += list(self.cond_mlp.parameters())
        return params
