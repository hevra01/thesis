import itertools
from pathlib import Path
import torchvision.transforms as T
from data.datasets.huggingface_dataset import HuggingFaceDataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from dahuffman import HuffmanCodec
from torchvision.datasets.folder import default_loader

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}

# Simple transform profiles so sizing/normalization can be configured from config

# these are mean and std for dataset used in CLIP training
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# dataset specific
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_imagenet_transform(
    profile: str = "imagenet",
    train_or_eval = "eval"
):
    """
    Build a torchvision transform pipeline for different backbones with fixed,
    known sizes and normalizations to avoid config skew:
      - "imagenet": 256 center-crop, ImageNet mean/std.
      - "clip":     224 center-crop, CLIP mean/std.
      - "sd_vae":   256 center-crop, scale to [-1,1] via mean=std=0.5.
    For custom setups, pass a full `transform` to the dataloader instead.
    """
    profile = (profile or "imagenet").lower()
    if profile == "clip":
        size = 224
        crop_size = 224
        mean, std = CLIP_MEAN, CLIP_STD
    elif profile == "sd_vae":
        size = 256
        crop_size = 256
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    elif profile == "imagenet":
        size = 256
        crop_size = 224
        mean, std = IMAGENET_MEAN, IMAGENET_STD

        if train_or_eval == "train":
            return T.Compose([
                T.Resize(size),
                T.RandomCrop(crop_size),
                T.RandomHorizontalFlip(),      # data augmentation
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])

    return T.Compose([
        T.Resize(size),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


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

class FlatImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.paths = sorted(
            p for p in self.root.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {self.root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = default_loader(str(self.paths[idx]))  # loads using PIL and handles grayscale/RGB
        if self.transform:
            img = self.transform(img)
        return img, -1  # dummy label

def get_imagenet_dataloader(
    root="/scratch/inf0/user/mparcham/ILSVRC2012",
    split="val",  # or "train"
    batch_size=64,
    class_filter=None,
    num_workers=16,
    shuffle=False,
    transform_profile: str = "imagenet",
    train_or_eval="eval", # this will be used to determine the transformation
    transform=None,
):
    """Flexible ImageNet loader with fixed, profile-based transforms.
    - transform_profile in {"imagenet", "clip", "sd_vae"} selects size+normalization.
    - Or pass a ready-made `transform` to override everything.
    """
    # Build transform unless custom provided
    if transform is None:
        transform = build_imagenet_transform(profile=transform_profile, train_or_eval=train_or_eval)

    data_root = f"{root}/{split}"
    has_class_subdirs = any(Path(data_root).glob("*/"))

    if has_class_subdirs:
        # Original behavior
        dataset = ImageFolder(root=data_root, transform=transform)

        # Optional class filter (only meaningful with class subdirs)
        if class_filter is not None:
            class_to_idx = {k: v for k, v in dataset.class_to_idx.items()}
            allowed_idxs = {class_to_idx[c] for c in class_filter if c in class_to_idx}
            dataset.samples = [s for s in dataset.samples if s[1] in allowed_idxs]
    else:
        # New: flat folder (no labels needed)
        dataset = FlatImageDataset(root=data_root, transform=transform)
        # Note: class_filter is ignored in this mode.


    # Wrap in DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


class ReconstructionDataset_Neural(Dataset):
    def __init__(
        self,
        reconstruction_data,
        dataloader,
        filter_key: str | None = None,
        min_error: float | None = None,
        max_error: float | None = None,
        error_key: str = "vgg_error",
    ):
        """
        Args:
            reconstruction_data (list): List of dicts containing reconstruction metrics.
                (img, k_value, mse_error, vgg_error).
            dataloader (DataLoader): Dataloader from which images can be fetched.
            filter_key (str|None): Which error field to filter on (e.g., "vgg_error" or "mse_error"). If None, no filtering.
            min_error (float|None): Keep samples with error >= min_error (if provided).
            max_error (float|None): Keep samples with error <= max_error (if provided).
            error_key (str): Name of the field in each data_point to read the error from (default: "vgg_error").
        """

        self.dataloader = dataloader

        # Apply optional filtering on the provided error field
        self.num_original = len(reconstruction_data)
        if filter_key is not None and (min_error is not None or max_error is not None):
            lo = float(min_error) if min_error is not None else float("-inf")
            hi = float(max_error) if max_error is not None else float("inf")
            filtered = []
            missing_key = 0
            for d in reconstruction_data:
                if filter_key not in d:
                    missing_key += 1
                    continue
                v = d[filter_key]
                # tolerate nested structures but we expect scalars
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    continue
                if lo <= val <= hi:
                    filtered.append(d)
            self.reconstruction_data = filtered
            self.filter_key = filter_key
            self.filter_bounds = (lo, hi)
            self.missing_key_count = missing_key
        else:
            self.reconstruction_data = reconstruction_data
            self.filter_key = None
            self.filter_bounds = None
            self.missing_key_count = 0

        self.num_kept = len(self.reconstruction_data)
        self.error_key = error_key

    def __len__(self):
        # this will be the number of images * the number of k values for each image.
        # e.g. for imagenet val, it is 50000 * len([1, 2, 4, 8, 16, 32, 64, 128, 180, 256]) = 500000
        return len(self.reconstruction_data)

    def __getitem__(self, idx):
        """
        Returns:
            dict: Contains image ID, k value, MSE error, vgg error (or any other error) and compression rate.
        """
        data_point = self.reconstruction_data[idx]
        image_id = data_point["image_id"]
        k_value = data_point["k_value"]
        # Read the requested error field, defaulting to vgg if present
        err_val = data_point[self.error_key]

        return {
            "image": self.dataloader.dataset[image_id][0],  # Get the image from the dataloader
            self.error_key: err_val,
            "k_value": k_value
        }
    

class ReconstructionDataset_Heuristic(Dataset):
    """
    this is different from ReconstructionDataset_token_based in the sense that this also has the edge information.
    Optionally, it can include additional per-image scalar features such as LID and local density.
    """
    def __init__(self, reconstruction_data, edge_ratio_information=None,
                 lid_information=None, local_density_information=None,
                 lpips_variance_information=None,
                 dino_dist_information=None,
                 error_key: list[str] = ["vgg_error"]):
        """
        Args:
            reconstruction_data (list): List of dicts containing reconstruction metrics.
                (img_id, k_value, mse_error, vgg_error, ...).
            edge_ratio_information (list|dict|None): Per-image edge ratio values.
            lid_information (list|dict|None): Per-image LID values.
            local_density_information (list|dict|None): Per-image local density values.
            error_key (str): Name of the error field in reconstruction_data to expose in samples.
        """
        self.reconstruction_data = reconstruction_data
        self.edge_ratio_information = edge_ratio_information
        self.lid_information = lid_information
        self.local_density_information = local_density_information
        self.lpips_variance_information = lpips_variance_information
        self.dino_dist_information = dino_dist_information

        # Apply optional filtering on the provided error field
        self.num_original = len(reconstruction_data)
        self.reconstruction_data = reconstruction_data
        self.error_key = error_key

    def __len__(self):
        return len(self.reconstruction_data)

    def __getitem__(self, idx):
        """
        Returns a dict with:
          - error value under `self.error_key`
          - k_value (int)
          - optionally: edge_ratio, lid, local_density if the corresponding info was provided
          - image_id (int) for traceability
        """
        data_point = self.reconstruction_data[idx]
        k_value = data_point["k_value"]
        err_vals = {key: data_point[key] for key in self.error_key}
        image_id = data_point["image_id"]

        out = {
            **err_vals,
            "k_value": k_value,
            "image_id": image_id,
        }
        if self.edge_ratio_information is not None:
            out["edge_ratio"] = self.edge_ratio_information[image_id]
        if self.lid_information is not None:
            out["lid"] = self.lid_information[k_value][image_id]
        if self.local_density_information is not None:
            out["local_density"] = self.local_density_information[k_value][image_id]
        if self.lpips_variance_information is not None:
            out["lpips_variance"] = self.lpips_variance_information[k_value][image_id]
        if self.dino_dist_information is not None:
            out["dino_dist"] = self.dino_dist_information[image_id]
        return out