import torchvision.transforms as T
from data.datasets.huggingface_dataset import HuggingFaceDataset
from datasets import load_dataset
from torch.utils.data import DataLoader

def get_mnist_dataloader(batch_size=64, split="test", flatten=False):
    # Load raw MNIST from HuggingFace
    raw_dataset = load_dataset("mnist", split=split)

    transform_list = [
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ]

    if flatten:
        transform_list.append(T.Lambda(lambda x: x.view(-1)))  # flatten to 1D vector


    # Compose transforms (match your config)
    transform = T.Compose(transform_list)
        
    # Wrap with your HuggingFaceDataset class
    dataset = HuggingFaceDataset(dataset=raw_dataset, transform=transform)
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader