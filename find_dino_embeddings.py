import argparse
import json
import os
from typing import Tuple

import torch
from transformers import AutoModel, AutoImageProcessor

from data.utils.dataloaders import get_imagenet_dataloader

# Optional: try to use RAPIDS cuML for fast GPU KMeans.
# If it's not installed, we fall back to scikit-learn.
try:
    from cuml.cluster import KMeans as cuKMeans  # type: ignore
    _HAS_CUML = True
except ImportError:
    _HAS_CUML = False
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances

def get_dino_features(
    model: AutoModel,
    processor: AutoImageProcessor,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract pooled DINO features for all images in the dataloader.

    We assume the dataloader yields batches of images in a format that the
    HuggingFace AutoImageProcessor can handle (PIL images or tensors).

    Args:
        model:       The DINO model used for feature extraction.
        processor:   Corresponding image processor (handles resize, crop, norm).
        dataloader:  DataLoader providing the images (shuffle=False recommended).
        device:      Device to run the model on (cuda / cpu).

    Returns:
        features: Tensor of shape [N, D] containing pooled DINO features
                  for all N images in the dataset, in the same order as the
                  dataloader iteration.
    """

    # Load normalization values from Hydra config
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    all_features = []

    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            # unnormalized images from dataloader based on imagenet stats
            images_unnormalized = images.to(device) * std + mean

            # images might already be tensors; processor accepts both tensors and PILs
            # We let the processor handle batching & normalization.
            inputs = processor(
                images=[t for t in images_unnormalized],   # convert batch tensor -> list of images
                return_tensors="pt",       # output PyTorch tensors
                do_rescale=False           # skip scaling 0–255 → 0–1
            )
            pixel_values = inputs["pixel_values"].to(device)

            # Forward pass through the ViT backbone
            outputs = model(pixel_values=pixel_values)

            # For DINOv2 in HF, `pooler_output` is the global embedding (CLS pooled).
            # If that ever changes, `last_hidden_state[:, 0, :]` is a safe fallback.
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                batch_feats = outputs.pooler_output  # [B, D]
            else:
                batch_feats = outputs.last_hidden_state[:, 0, :]  # CLS token

            all_features.append(batch_feats.cpu())

    features = torch.cat(all_features, dim=0)
    return features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute DINO distances to nearest centroids for ImageNet val."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/datasets/dino_embeddings/val_categorized",
        help="Output directory to save computed DINO embeddings.",
    )
    parser.add_argument(
        "--dinov2_dir",
        type=str,
        default="models/weights/dinov2-base",
        help="Local directory containing DINOv2-base weights.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for feature extraction.",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # ---------------------------------------------------------------------
    # 1) Instantiate the ImageNet val dataloader
    # ---------------------------------------------------------------------
    imagenet_val = get_imagenet_dataloader(
        split="val_categorized",
        batch_size=args.batch_size,
        shuffle=False,
    )
    print(f"Loaded ImageNet validation set with {len(imagenet_val.dataset)} images.")

    # ---------------------------------------------------------------------
    # 2) Load DINOv2-base model + processor from local folder
    # ---------------------------------------------------------------------
    dinov2_dir = args.dinov2_dir
    processor = AutoImageProcessor.from_pretrained(
        dinov2_dir,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        dinov2_dir,
        local_files_only=True,
    )

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Freeze parameters (we only do inference)
    for p in model.parameters():
        p.requires_grad = False

    # ---------------------------------------------------------------------
    # 3) Extract DINO features for all images
    # ---------------------------------------------------------------------
    print("Extracting DINO features...")
    features = get_dino_features(model, processor, imagenet_val, device)
    print(f"Extracted features with shape: {features.shape}")  # [N, D]
    
    # Save features to output directory
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "dino_features.pt")
    torch.save(features, out_path)
    print(f"Saved DINO features to {out_path}")

if __name__ == "__main__":
    main()