"""
Simple, heavily-commented MLP that maps a scalar reconstruction loss → token count (classification).

Why classification?
- Token count is an ordinal/categorical target in {1..C}. You already have
  GaussianCrossEntropyLoss in `reconstruction_loss.py` that creates distance-aware
  soft targets across classes, making near-misses cheaper than far misses.

Input / Output contract:
- Input: a tensor of shape [B] or [B, 1] containing per-sample loss values (floats).
- Output: raw logits of shape [B, num_classes]. Use these logits with
  `GaussianCrossEntropyLoss(num_classes)` for training.

Implementation details:
- We first apply a light preprocessing to the scalar: y = log1p(loss), which compresses
  the scale for very large values (optional but often helpful for stabilizing training).
- A small learnable affine transform (scale + bias) provides flexibility for rescaling.
- A two- or three-layer MLP then produces class logits.

In practice you can tune hidden sizes/activation/dropout from the constructor.
"""

from typing import Iterable, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossToTokenCountModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 256,
        hidden_sizes: Iterable[int] = (64, 64),
        activation: str = "silu",  # relu|gelu|silu|tanh
        dropout: float = 0.0,
        use_log1p: bool = True,
        affine_preact: bool = True,
    ) -> None:
        """
        Args:
            num_classes: number of token count classes (C). Labels are expected in [1..C].
            hidden_sizes: MLP layer widths between input (1) and output (C).
            activation: nonlinearity to use between layers.
            dropout: dropout probability between hidden layers (0 to disable).
            use_log1p: if True, preprocess input as log(1 + loss).
            affine_preact: if True, apply learnable affine (scale, bias) before MLP.
        """
        super().__init__()

        self.num_classes = int(num_classes)
        self.use_log1p = bool(use_log1p)
        self.affine_preact = bool(affine_preact)

        # 1) Optional learnable affine transformation on the (preprocessed) scalar input
        # This is a learnable linear transformation on the scalar input — just like batch normalization 
        # or layer normalization would do, but applied manually and before the MLP.
        if self.affine_preact:
            # scale initialized to 1.0, bias to 0.0
            # When you wrap a tensor in nn.Parameter, 
            # PyTorch automatically treats it as a learnable parameter.
            self.input_scale = nn.Parameter(torch.tensor(1.0))
            self.input_bias = nn.Parameter(torch.tensor(0.0))
        else:
            # “I want to tell PyTorch that there could be a parameter 
            # named input_bias, but for now, I don’t have one.”
            # This ensures the module still has an attribute called input_bias,
            # but it’s set to None instead of a real parameter.
            self.register_parameter("input_scale", None)
            self.register_parameter("input_bias", None)

        # 2) Choose activation
        act = activation.lower()
        if act == "relu":
            act_layer = nn.ReLU
        elif act == "gelu":
            act_layer = nn.GELU
        elif act == "silu":
            act_layer = nn.SiLU
        elif act == "tanh":
            act_layer = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 3) Build the MLP: [1] -> hidden_sizes -> [num_classes]
        layers: List[nn.Module] = []
        in_dim = 1
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_layer())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, self.num_classes))  # output logits

        self.mlp = nn.Sequential(*layers)

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize shapes and apply optional preprocessing.

        Accepts x with shape [B] or [B, 1]. Returns tensor with shape [B, 1].
        """
        # Ensure a 2D tensor with a single feature per sample
        if x.dim() == 1:
            x = x.unsqueeze(1)  # [B] -> [B,1]
        elif x.dim() == 2 and x.size(1) == 1:
            pass  # already [B,1]
        else:
            raise ValueError(f"Expected shape [B] or [B,1], got {tuple(x.shape)}")

        # Optional log1p to reduce dynamic range for large losses (squishing large values down)
        # this is base e. if x=0, log(1+0)=log(1)=0.
        if self.use_log1p:
            x = torch.log1p(torch.clamp(x, min=0))  # clamp to avoid log1p of negative

        # Optional learnable affine rescaling before the MLP
        if self.affine_preact:
            x = self.input_scale * x + self.input_bias

        return x

    def forward(self, loss_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            loss_values: [B] or [B,1] float tensor with per-sample reconstruction losses.

        Returns:
            logits: [B, C] raw class scores. Use with GaussianCrossEntropyLoss for training.
        """
        x = self._preprocess_input(loss_values)
        logits = self.mlp(x)
        return logits

    @torch.no_grad()
    def predict_proba(self, loss_values: torch.Tensor) -> torch.Tensor:
        """Return class probabilities P(class | loss) with softmax over logits.

        Args:
            loss_values: [B] or [B,1]
        Returns:
            probs: [B, C]
        """
        logits = self.forward(loss_values)
        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def predict_class(self, loss_values: torch.Tensor) -> torch.Tensor:
        """Return the argmax class index in [0..C-1].

        Note on labels: if your labels are in [1..C] (token counts), you may want to add 1 to
        this prediction when comparing to ground-truth counts.
        """
        logits = self.forward(loss_values)
        return torch.argmax(logits, dim=1)
