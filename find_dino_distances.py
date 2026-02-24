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

    return dist


def save_nearest_centroid_distances(
    all_distances: torch.Tensor,   # [N, K]
    output_path: str,
    distance_metric: str = "cosine",
):
    """
    Save only nearest-centroid distance and index for each image.

    Args:
        all_distances:   Tensor [N, K], distances[i, j] = distance(image_i, centroid_j)
        output_path:     Path where output will be saved (.pt format).
        distance_metric: 'euclidean' or 'cosine'
    """
    all_distances = all_distances.cpu()
    N, K = all_distances.shape

    print(f"Computing nearest centroid for {N} images from {K} centroids...")

    # Only compute nearest-centroid info (minimal memory)
    min_distances, min_indices = torch.min(all_distances, dim=1)  # [N]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as compact .pt file
    torch.save({
        "num_images": N,
        "num_centroids": K,
        "distance_metric": distance_metric,
        "nearest_distances": min_distances,  # [N]
        "nearest_indices": min_indices,       # [N]
    }, output_path)

    print(f"Saved nearest centroid distances to {output_path}")


def save_all_centroid_distances_to_json(
    all_distances: torch.Tensor,   # [N, K]
    output_path: str,
    distance_metric: str = "cosine",
):
    """
    Save per-image distances to ALL centroids into a JSON file.

    Args:
        all_distances:   Tensor [N, K], distances[i, j] = distance(image_i, centroid_j)
        output_path:     Path where JSON will be saved.
        distance_metric: 'euclidean' or 'cosine'
    """
    all_distances = all_distances.cpu()
    N, K = all_distances.shape

    print(f"Saving full centroid distance matrix: {N} images × {K} centroids → {output_path}")

    # Compute nearest-centroid info
    min_distances, min_indices = torch.min(all_distances, dim=1)          # [N]
    sorted_dists, sorted_indices = torch.sort(all_distances, dim=1)       # [N, K]

    data = {
        "num_images": int(N),
        "num_centroids": int(K),
        "distance_metric": distance_metric,
        "images": []
    }

    for i in range(N):
        entry = {
            "index": int(i),
            "nearest_centroid": int(min_indices[i].item()),
            "nearest_distance": float(min_distances[i].item()),
            "distances": all_distances[i].tolist(),              # 🔥 full distance vector
            "sorted_distances": sorted_dists[i].tolist(),        # optional
            "sorted_indices": sorted_indices[i].tolist(),        # optional
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
        "--dinov2_feature_dir",
        type=str,
        default="data/datasets/dino_embeddings/train/dino_features.pt",
        help="dino features are pre-computed.",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=1000,
        help="Number of clusters for KMeans.",
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        default="cosine",
        choices=["euclidean", "cosine"],
        help="Distance metric for nearest-centroid computation.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="data/datasets/dino_distances/train/dino_all_centroid_distances_cosine.json",
        help="Where to save the JSON with distances.",
    )
    parser.add_argument(
        "--output_pt",
        type=str,
        default="data/datasets/dino_distances/train/dino_nearest_centroid_distances.pt",
        help="Where to save the .pt file with nearest distances only.",
    )
    parser.add_argument(
        "--save_all",
        action="store_true",
        help="If set, save all centroid distances (memory-intensive). Otherwise, save only nearest.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # load dino features as .pt
    features = torch.load(args.dinov2_feature_dir)
    print(f"Loaded DINO features from {args.dinov2_feature_dir} with shape: {features.shape}")

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
    dist = compute_nearest_centroid_distances(
        features, centroids, metric=args.distance_metric
    )

    # ---------------------------------------------------------------------
    # 6) Save results
    # ---------------------------------------------------------------------
    if args.save_all:
        save_all_centroid_distances_to_json(
            all_distances=dist,
            output_path=args.output_json,
            distance_metric=args.distance_metric,
        )
    else:
        save_nearest_centroid_distances(
            all_distances=dist,
            output_path=args.output_pt,
            distance_metric=args.distance_metric,
        )


if __name__ == "__main__":
    main()
