"""
Compute mean LPIPS variance across multiple stochastic reconstructions
for a SINGLE token count (reconst_k).

This script is intended to be run as ONE SLURM JOB PER reconst_k.

Expected folder structure:
root_path/
  reconst_{k}/
    class_id/
      image_id/
        image_r0.jpg
        image_r1.jpg
        ...
"""

import os
import json
from typing import Dict, List

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch
from torchvision.io import read_image
from torchvision.transforms.functional import resize

import wandb


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def list_dir(path: str) -> List[str]:
    """Safe sorted directory listing."""
    if not os.path.isdir(path):
        return []
    return sorted(os.listdir(path))


def to_lpips_tensor(img_path: str, device: torch.device) -> torch.Tensor:
    """
    Load image and convert to LPIPS-compatible tensor:
      - shape [1,3,256,256]
      - values in [-1, 1]
    """
    img = read_image(img_path)  # [C,H,W], uint8

    # Ensure RGB
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.shape[0] == 4:
        img = img[:3]
    elif img.shape[0] != 3:
        raise ValueError(f"Unsupported channel count {img.shape[0]} for {img_path}")

    img = resize(img, [256, 256])
    img = img.float() / 255.0
    img = img * 2.0 - 1.0
    return img.unsqueeze(0).to(device)


# ---------------------------------------------------------------------
# Efficient LPIPS computation
# ---------------------------------------------------------------------

def mean_pairwise_lpips(
    img_paths: List[str],
    distance_fns,
    device: torch.device,
) -> float:
    """
    Compute mean LPIPS over all unique pairs of images.

    This implementation:
      - loads each image once
      - builds all (i,j) pairs
      - computes LPIPS in ONE batched forward pass

    Returns:
        mean LPIPS distance (float)
    """
    imgs = []
    for p in img_paths:
        try:
            imgs.append(to_lpips_tensor(p, device))
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")

    if len(imgs) < 2:
        return 0.0

    X = torch.cat(imgs, dim=0)  # [N,3,H,W]
    n = X.shape[0]

    # Build all unique index pairs
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(i + 1, n):
            idx_i.append(i)
            idx_j.append(j)

    A = X[idx_i]
    B = X[idx_j]

    with torch.no_grad():
        total = 0.0
        for fn in distance_fns:
            d = fn(A, B)          # [num_pairs]
            total += d.mean().item()

    return total / len(distance_fns)


# ---------------------------------------------------------------------
# Main scan logic for ONE reconst_k
# ---------------------------------------------------------------------

def scan_single_reconst(
    root_path: str,
    reconst_k: int,
    distance_fns,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Scan root_path/reconst_{k}/ and compute LPIPS variance.

    Returns:
        results[class_id][image_id] = mean_lpips
    """
    reconst_dir = f"reconst_{reconst_k}"
    reconst_path = os.path.join(root_path, reconst_dir)

    if not os.path.isdir(reconst_path):
        raise FileNotFoundError(f"Missing directory: {reconst_path}")

    print(f"[START] Processing {reconst_dir}")

    results: Dict[str, Dict[str, float]] = {}

    # Loop over ImageNet classes
    for class_idx, class_dir in enumerate(list_dir(reconst_path)):
        class_path = os.path.join(reconst_path, class_dir)
        if not os.path.isdir(class_path):
            continue

        results.setdefault(class_dir, {})

        # Loop over images (each folder = one image's reconstructions)
        for image_id_dir in list_dir(class_path):
            image_folder = os.path.join(class_path, image_id_dir)
            if not os.path.isdir(image_folder):
                continue

            img_paths = [
                os.path.join(image_folder, f)
                for f in list_dir(image_folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]

            mean_lp = mean_pairwise_lpips(img_paths, distance_fns, device)
            results[class_dir][image_id_dir] = mean_lp

        # Log progress once per class
        wandb.log(
            {
                "progress/reconst_k": reconst_k,
                "progress/class_name": class_dir,
                "progress/class_index": class_idx,
            },
            step=class_idx,
        )

        print(f"[DONE] reconst_{reconst_k} | class {class_dir}")

    return results


# ---------------------------------------------------------------------
# JSON saving
# ---------------------------------------------------------------------

def save_results_json(results: Dict, out_path: str) -> None:
    """Save results dict to JSON."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="estimate_variance_images_different_token_counts")
def main(cfg):

    wandb.init(
        project="imagenet_reconstruct_tokens",
        name=f"lpips_variance_reconst_{cfg.experiment.reconst_k}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    device = torch.device(cfg.experiment.device)
    root_path = cfg.experiment.data_path
    reconst_k = int(cfg.experiment.reconst_k)
    out_file = cfg.experiment.out_file

    # Instantiate LPIPS models
    distance_fns = [
        instantiate(dcfg).to(device)
        for dcfg in cfg.experiment.distance_functions
    ]

    results = scan_single_reconst(
        root_path=root_path,
        reconst_k=reconst_k,
        distance_fns=distance_fns,
        device=device,
    )

    save_results_json(results, out_file)

    print(f"[FINISHED] reconst_{reconst_k} → {out_file}")
    wandb.finish()


if __name__ == "__main__":
    main()
