import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import torchvision.transforms as T
from data.datasets.huggingface_dataset import HuggingFaceDataset


class Unflatten:
    """
    Unflatten a tensor given a shape.
    Args:
        shape (tuple): The shape to unflatten to.
    """
    def __init__(self, shape):
        self.shape = shape
    def __call__(self, x):
         # x is [batch_size, flat_dim]
        batch_size = x.shape[0]
        return x.view(batch_size, *self.shape)
    

