from torch import T
from data.datasets.huggingface_dataset import HuggingFaceDataset
from datasets import load_dataset
from torch.utils.data import DataLoader

def get_mnist_dataloader(batch_size=64, split="test"):
    # Load raw MNIST from HuggingFace
    raw_dataset = load_dataset("mnist", split=split)
    # Compose transforms (match your config)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])
    # Wrap with your HuggingFaceDataset class
    dataset = HuggingFaceDataset(dataset=raw_dataset, transform=transform)
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader