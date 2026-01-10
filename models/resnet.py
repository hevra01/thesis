import torch
import torch.nn as nn
import torchvision.models as models

class ResNetCondClassifier(nn.Module):
    def __init__(self, num_classes: int, backbone="resnet50", pretrained=True, use_condition: bool = False):
        super().__init__()
        weights = "IMAGENET1K_V2" if pretrained else None
        self.backbone = getattr(models, backbone)(weights=weights)

        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

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

        self.classifier = nn.Linear(in_dim + cond_dim, num_classes)

    def forward(self, x, recon_loss_scalar=None):
        f = self.backbone(x)

        if self.use_condition:
            if recon_loss_scalar is None:
                raise ValueError("recon_loss_scalar must be provided when use_condition=True")
            l = recon_loss_scalar.view(-1, 1)
            l = torch.log(l + 1e-8)
            c = self.cond_mlp(l)
            z = torch.cat([f, c], dim=1)
            return self.classifier(z)
        else:
            return self.classifier(f)
