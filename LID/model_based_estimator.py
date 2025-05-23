import torch


class ModelBasedLIDEstimator:
    def __init__(self, ambient_dim: int, model: torch.nn.Module, device: torch.device):
        """
        Initializes the ModelBasedLIDEstimator.
        Args:
            ambient_dim (int): The dimensionality of the ambient space.
            model (torch.nn.Module): A torch Module that is a likelihood-based 
            deep generative model and that one can compute LID from.
            device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').
        """
        self.ambient_dim = ambient_dim
        self.model = model
        self.device = device
