from typing import Iterable, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F



class HeuristicTokenCountPredictor(nn.Module):
    """
    A simple MLP classifier that predicts token counts based on reconstruction loss
    and optional additional features.
    
    Architecture:
        - Input: reconstruction loss (scalar) + optional features (concatenated)
        - Input normalization (BatchNorm1d) to handle different feature scales
        - Single hidden layer with nonlinear activation (ReLU)
        - Output: class logits for token count prediction
    
    The model is trained using GaussianCrossEntropyLoss which handles
    classification with Gaussian assumptions on the output distribution.
    """
    
    def __init__(
        self,
        num_classes: int = 256,
        num_additional_features: int = 0,
        hidden_dim: int = 64,
        use_log1p_recon: bool = True,
        normalize_inputs: bool = True,
    ):
        """
        Initialize the MLP classifier.
        
        Args:
            num_classes: Number of token count classes to predict
            num_additional_features: Number of additional features beyond recon loss.
                                     Can be 0 if only using reconstruction loss.
            hidden_dim: Dimension of the hidden layer
            use_log1p_recon: If True, apply log(1 + recon_loss) to compress dynamic range
            normalize_inputs: If True, apply BatchNorm1d to normalize all input features
        """
        super().__init__()
        
        # Input dimension: 1 (recon_loss) + number of additional features
        self.input_dim = 1 + num_additional_features
        self.num_classes = num_classes
        self.num_additional_features = num_additional_features
        self.use_log1p_recon = use_log1p_recon
        self.normalize_inputs = normalize_inputs
        
        # -----------------------------------------------------------------
        # Input normalization: BatchNorm1d to handle different feature scales
        # This learns running mean/std and normalizes inputs to zero mean, unit variance
        # -----------------------------------------------------------------
        if self.normalize_inputs:
            self.input_norm = nn.BatchNorm1d(self.input_dim)
        else:
            self.input_norm = nn.Identity()
        
        # -----------------------------------------------------------------
        # Architecture: Single nonlinear layer for classification
        # Input -> Normalize -> Linear -> ReLU (nonlinearity) -> Linear -> Logits
        # -----------------------------------------------------------------
        
        # First linear layer: projects input to hidden dimension
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        
        # Nonlinear activation function
        self.activation = nn.ReLU()
        
        # Output layer: projects hidden representation to class logits
        self.fc_out = nn.Linear(hidden_dim, num_classes)
    
    def forward(
        self,
        recon_loss: torch.Tensor,
        additional_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the MLP.
        
        Args:
            recon_loss: Reconstruction loss tensor of shape (batch_size,) or (batch_size, 1)
            additional_features: Optional tensor of shape (batch_size, num_additional_features)
                                 Can be None if num_additional_features=0
        
        Returns:
            logits: Class logits of shape (batch_size, num_classes)
        """
        # -----------------------------------------------------------------
        # Step 1: Ensure recon_loss has shape (batch_size, 1)
        # -----------------------------------------------------------------
        if recon_loss.dim() == 1:
            recon_loss = recon_loss.unsqueeze(-1)  # (batch_size,) -> (batch_size, 1)
        
        # -----------------------------------------------------------------
        # Step 2: Apply log1p to recon_loss to compress dynamic range
        # This helps with very large reconstruction loss values
        # -----------------------------------------------------------------
        if self.use_log1p_recon:
            recon_loss = torch.log1p(torch.clamp(recon_loss, min=0))
        
        # -----------------------------------------------------------------
        # Step 3: Concatenate recon_loss with additional features (if any)
        # -----------------------------------------------------------------
        if additional_features is not None and self.num_additional_features > 0:
            # Concatenate along feature dimension: (batch_size, 1 + num_features)
            x = torch.cat([recon_loss, additional_features], dim=-1)
        else:
            # Only use reconstruction loss as input
            x = recon_loss
        
        # -----------------------------------------------------------------
        # Step 4: Normalize inputs to bring all features to same scale
        # BatchNorm1d learns running statistics and normalizes to ~N(0,1)
        # -----------------------------------------------------------------
        x = self.input_norm(x)
        
        # -----------------------------------------------------------------
        # Step 5: Apply the single nonlinear layer
        # Linear transformation -> Nonlinearity (ReLU)
        # -----------------------------------------------------------------
        x = self.fc1(x)          # (batch_size, hidden_dim)
        x = self.activation(x)   # Apply ReLU nonlinearity
        
        # -----------------------------------------------------------------
        # Step 6: Project to class logits
        # -----------------------------------------------------------------
        logits = self.fc_out(x)  # (batch_size, num_classes)
        
        return logits