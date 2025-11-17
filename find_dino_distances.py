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


def run_kmeans(
    features: torch.Tensor,
    num_clusters: int,
    use_cuml: bool = _HAS_CUML,
) -> torch.Tensor:
    """
    Run KMeans clustering on the DINO features to obtain centroids.

    Args:
        features:     Tensor [N, D] of feature vectors on CPU.
        num_clusters: Number of clusters (K in KMeans).
        use_cuml:     If True and cuML is available, use GPU-accelerated KMeans.

    Returns:
        centroids: Tensor [num_clusters, D] of cluster centers on CPU.
    """
    N, D = features.shape
    print(f"Running KMeans on {N} samples of dim {D} with K={num_clusters}...")

    if use_cuml:
        print("Using cuML KMeans (GPU).")
        import cupy as cp  # type: ignore

        # Move data to GPU as CuPy array
        feats_cu = cp.asarray(features.numpy())
        kmeans = cuKMeans(
            n_clusters=num_clusters,
            max_iter=300,
            init="k-means++",
            random_state=0,
            verbose=1,
        )
        kmeans.fit(feats_cu)
        centroids_cu = kmeans.cluster_centers_
        centroids = torch.from_numpy(cp.asnumpy(centroids_cu))
    else:
        print("Using scikit-learn KMeans (CPU).")
        feats_np = features.numpy()
        kmeans = KMeans(
            n_clusters=num_clusters,
            max_iter=300,
            init="k-means++",
            random_state=0,
            verbose=1,
        )
        kmeans.fit(feats_np)
        centroids = torch.from_numpy(kmeans.cluster_centers_)

    print(f"Computed centroids with shape: {centroids.shape}")
    return centroids


def compute_nearest_centroid_distances(
    features: torch.Tensor,
    centroids: torch.Tensor,
    metric: str = "euclidean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each feature vector, compute the distance to its nearest centroid.

    Args:
        features:  Tensor [N, D] of feature vectors (CPU).
        centroids: Tensor [K, D] of cluster centers (CPU).
        metric:    Distance metric ('euclidean' or 'cosine').

    Returns:
        min_distances: Tensor [N] of distances to closest centroid.
        min_indices:   Tensor [N] of indices of the closest centroid.
    """
    feats = features
    cents = centroids

    if metric == "cosine":
        # Normalize and use cosine similarity
        feats = torch.nn.functional.normalize(feats, dim=1)
        cents = torch.nn.functional.normalize(cents, dim=1)

        # Compute similarity matrix [N, K] = feats @ cents^T
        sim = feats @ cents.t()
        # Cosine distance = 1 - similarity
        dist = 1.0 - sim
    elif metric == "euclidean":
        # (a - b)^2 = |a|^2 + |b|^2 - 2 a·b
        feats_sq = (feats ** 2).sum(dim=1, keepdim=True)      # [N, 1]
        cents_sq = (cents ** 2).sum(dim=1, keepdim=True).t()  # [1, K]
        sim = feats @ cents.t()                               # [N, K]
        dist = torch.sqrt(torch.clamp(feats_sq + cents_sq - 2 * sim, min=0.0))
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    min_distances, min_indices = torch.min(dist, dim=1)
    return min_distances, min_indices


def save_distances_to_json(
    distances: torch.Tensor,
    indices: torch.Tensor,
    output_path: str,
    num_clusters: int,
    distance_metric: str,
):
    """
    Save per-image DINO distances and nearest-centroid indices to a JSON file.

    We assume images are presented in the same order as the dataloader (i.e.,
    index 0 corresponds to the first image, etc.).

    Args:
        distances:       Tensor [N] of nearest-centroid distances.
        indices:         Tensor [N] of nearest-centroid indices.
        output_path:     Path to the JSON file to write.
        num_clusters:    Number of clusters used in KMeans.
        distance_metric: Distance metric name ('euclidean' or 'cosine').
    """
    N = distances.shape[0]
    print(f"Saving distances for {N} images to {output_path}")

    data = {
        "num_clusters": int(num_clusters),
        "distance_metric": distance_metric,
        "num_images": int(N),
        "images": [],
    }

    for i in range(N):
        entry = {
            "index": int(i),
            "nearest_centroid": int(indices[i].item()),
            "dino_distance": float(distances[i].item()),
        }
        data["images"].append(entry)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print("JSON saved.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute DINO distances to nearest centroids for ImageNet val."
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=1000,
        help="Number of KMeans clusters for DINO features.",
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Distance metric for nearest-centroid computation.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/dino_nearest_centroid_distances.json",
        help="Where to save the JSON with distances.",
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
        num_workers=args.num_workers,
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

    # ---------------------------------------------------------------------
    # 4) Run KMeans to obtain centroids
    # ---------------------------------------------------------------------
    centroids = run_kmeans(features, num_clusters=args.num_clusters)

    # ---------------------------------------------------------------------
    # 5) Compute nearest-centroid distances for each image
    # ---------------------------------------------------------------------
    print(
        f"Computing nearest-centroid distances using metric='{args.distance_metric}'..."
    )
    min_distances, min_indices = compute_nearest_centroid_distances(
        features, centroids, metric=args.distance_metric
    )

    # ---------------------------------------------------------------------
    # 6) Save results to JSON
    # ---------------------------------------------------------------------
    save_distances_to_json(
        distances=min_distances,
        indices=min_indices,
        output_path=args.output_json,
        num_clusters=args.num_clusters,
        distance_metric=args.distance_metric,
    )


if __name__ == "__main__":
    main()
