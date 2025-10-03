from typing import List, Optional
import torch
import torch.nn as nn

_ACTS = {
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
}

class HeuristicTokenCountPredictor(nn.Module):
    """
    this will be a regression task, where the input is the edge ratio and the tolerated loss (vgg error)
    and the output is the predicted token count.
    """
    def __init__(self,
        hidden_dims: List[int],
        hidden_activation: str,
        dropout: float = 0.1,
        output_activation: str = "sigmoid",   # linear | sigmoid | relu | softplus
        output_min: Optional[float] = None,  # used for sigmoid-range mapping
        output_max: Optional[float] = None
        ):
        super().__init__()
        if hidden_activation.lower() not in _ACTS:
            raise ValueError(f"hidden_activation must be one of {list(_ACTS)}")
        self.hidden_act = _ACTS[hidden_activation.lower()]()

        self.output_activation = _ACTS[output_activation.lower()]()
        self.output_min = output_min
        self.output_max = output_max
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        layers = []
        in_dim = 2  # [edge_ratio, vgg_error]
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), self.hidden_act, nn.Dropout(dropout)]
            in_dim = h
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()

        # Output head
        self.head = nn.Linear(in_dim, 1)

        # sanity when using sigmoid-range
        if self.output_activation == "sigmoid":
            if self.output_min is None or self.output_max is None:
                raise ValueError("Set output_min/output_max when output_activation='sigmoid'")


    def forward(self, edge_ratio: torch.Tensor, vgg_error: torch.Tensor) -> torch.Tensor:
        """
        edge_ratio: [batch, 1]
        vgg_error:  [batch, 1]
        returns:    [batch, 1]
        """

        # Concatenate features across last dim
        x = torch.cat([edge_ratio, vgg_error], dim=-1)

        h = self.backbone(x)
        y = self.head(h)  # [B,1] (logits)

        y01 = self.output_activation(y)
        return self.output_min + (self.output_max - self.output_min) * y01
        