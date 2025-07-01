import torchvision.transforms as T
from data.datasets.huggingface_dataset import HuggingFaceDataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def get_mnist_dataloader(batch_size=120, split="test", flatten=False, class_filter=None):
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
    dataset = HuggingFaceDataset(dataset=raw_dataset, class_filter=class_filter, transform=transform)
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def get_imagenet_dataloader(
    root="/BS/databases23/imagenet/original/",
    split="val",  # or "train"
    batch_size=64,
    image_size=256,
    class_filter=None,
    num_workers=64,
    shuffle=False
):
    # Define standard ImageNet transforms
    transform = T.Compose([
        # Rescales the shorter side of the image to image_size. Note that this could be down or upsampling, 
        # depending on whether the shorter side is smaller or larger than the img_size.
        # The longer side is then resized to maintain the aspect ratio.
        T.Resize(image_size), 
        # Crops out a square of size (image_size × image_size) from the center of the resized image.
        T.CenterCrop(image_size), 
        # Converts the image from a PIL/numpy array in range [0, 255] to a PyTorch tensor in range [0.0, 1.0]
        T.ToTensor(),
        # Applies per-channel normalization
        # Given that the normal (expected) distribution has this mean/std, 
        # re-center and rescale the image data accordingly so it behaves like zero-mean, unit-variance data.
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset from disk
    dataset = ImageFolder(root=f"{root}/{split}", transform=transform)

    # Optional class filter
    if class_filter is not None:
        class_to_idx = {v: k for k, v in dataset.class_to_idx.items()}
        allowed_idxs = [class_to_idx[c] for c in class_filter if c in class_to_idx]
        dataset.samples = [s for s in dataset.samples if s[1] in allowed_idxs]

    # Wrap in DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader