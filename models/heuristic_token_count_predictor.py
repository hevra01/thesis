from typing import List
import torch
import torch.nn as nn

class HeuristicTokenCountPredictor(nn.Module):
    """
    this will be a regression task, where the input is the edge ratio and the tolerated loss (vgg error)
    and the output is the predicted token count.
    """
    def __init__(self, edge_ratio: float, tolerated_loss: float, hidden_dim: List[int], activation: str, droupout: float = 0.1):
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

        self.edge_ratio = edge_ratio
        self.tolerated_loss = tolerated_loss
        self.hidden_dim = hidden_dim
        self.activation = activation    
        self.dropout = droupout

        # Build the model architecture
        self.build_model()

    def build_model(self):
        layers = []
        in_dim = 2  # input is the edge_ratio and the tolerated_loss

        for h_dim in self.hidden_dim:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(self.dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))  # output is the predicted token count
        self.model = nn.Sequential(*layers)


    def forward(self, edge_ratio: torch.Tensor, vgg_error: torch.Tensor) -> torch.Tensor:
        """
        edge_ratio: [batch, 1]
        vgg_error:  [batch, 1]
        returns:    [batch, 1]
        """

        # Concatenate features across last dim
        x = torch.cat([edge_ratio, vgg_error], dim=-1)
        return self.model(x)