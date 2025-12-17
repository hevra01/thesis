"""
Compute mean LPIPS variance across multiple stochastic reconstructions per image, for each token count.

Expected folder structure (default root_path=/ptmp/hevrapetek/reconstruction_imagenet_stochastic/val):
- root_path/
  - reconst_1/
    - n0210847/
      - n0210847_12345/     # folder for one image's reconstructions
        - n0210847_12345_r0.JPEG
        - n0210847_12345_r1.JPEG
        - ...
      - ... other image folders ...
  - reconst_2/
  - ...
  - reconst_256/

For each image folder (contains multiple reconstructions), we compute the mean pairwise LPIPS distance
among all reconstructions. This measures stochastic variance at each token count. Results are saved to JSON.
"""

import os
import json
from typing import Dict, List
from hydra.utils import instantiate
import hydra
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import resize

def list_dir(path: str) -> List[str]:
    """Safe directory listing; returns sorted children or [] if path missing."""
    if not os.path.isdir(path):
        return []
    return sorted(os.listdir(path))


def to_lpips_tensor(img_path: str, device: torch.device) -> torch.Tensor:
    """
    Load image and convert to LPIPS-compatible tensor:
    - Shape [1,3,H,W]
    - Values in [-1,1]
    - Resized to 256x256 for consistency
    - Ensures 3 channels (drops alpha or repeats grayscale)
    """
    img = read_image(img_path)  # [C,H,W] in [0,255]

    # Ensure RGB
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.shape[0] == 4:
        img = img[:3]
    elif img.shape[0] != 3:
        raise ValueError(f"Unsupported channel count {img.shape[0]} for {img_path}")

    img = resize(img, [256, 256])
    img = img.float() / 255.0
    img = img * 2.0 - 1.0  # [-1,1]
    return img.unsqueeze(0).to(device)


def mean_pairwise_lpips(img_paths: List[str], distance_fns, device: torch.device) -> float:
    """
    Mean of LPIPS over all unique pairs of images in img_paths.
    If <2 images are loadable, returns 0.0.
    """
    tensors = []
    for p in img_paths:
        try:
            tensors.append(to_lpips_tensor(p, device))
        except Exception as ex:
            print(f"Warning: failed to load {p}: {ex}")
    n = len(tensors)
    if n < 2:
        return 0.0

    total = 0.0
    count = 0
    with torch.no_grad():
        for i in range(n):
            for j in range(i + 1, n):
                for fn in distance_fns:
                    d = fn(tensors[i], tensors[j])
                    total += float(d.item())
                    count += 1
    return total / count if count > 0 else 0.0


def scan_reconstruction_root(root_path: str, distance_fns, device) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Walk the tree and compute mean LPIPS per image-folder.
    Returns nested dict: {reconst_k: {class_id: {image_id_folder: mean_lpips}}}
    """
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    # Top-level: reconst_{k}
    for reconst_dir in list_dir(root_path):
        if not reconst_dir.startswith("reconst_"):
            continue
        reconst_path = os.path.join(root_path, reconst_dir)
        if not os.path.isdir(reconst_path):
            continue
        results.setdefault(reconst_dir, {})

        # Class folders (e.g., n0210847)
        for class_dir in list_dir(reconst_path):
            class_path = os.path.join(reconst_path, class_dir)
            if not os.path.isdir(class_path):
                continue
            results[reconst_dir].setdefault(class_dir, {})

            # Image-id folders: each holds multiple reconstructions
            for image_id_dir in list_dir(class_path):
                image_folder_path = os.path.join(class_path, image_id_dir)
                if not os.path.isdir(image_folder_path):
                    continue

                # Collect image files in folder
                img_files = [
                    f for f in list_dir(image_folder_path)
                    if os.path.isfile(os.path.join(image_folder_path, f))
                    and f.lower().split('.')[-1] in {"jpg", "jpeg", "png", "bmp"}
                ]
                img_paths = [os.path.join(image_folder_path, f) for f in img_files]

                mean_lp = mean_pairwise_lpips(img_paths, distance_fns, device)
                results[reconst_dir][class_dir][image_id_dir] = mean_lp

    return results


def save_results_json(results: Dict, out_path: str) -> None:
    """Write results dict to JSON at out_path."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)


@hydra.main(version_base=None, config_path="conf", config_name="estimate_variance_images_different_token_counts")
def main(cfg):
    """
    Entry point to compute LPIPS variance per image-folder for each token count.
    - root_path: base directory containing reconst_{k} folders
    - out_json: path to write JSON results
    """
    device = cfg.experiment.device


    root_path = cfg.experiment.data_path
    out_file = cfg.experiment.out_file

    distance_fns = [instantiate(distance_cfg).to(device) for distance_cfg in cfg.experiment.distance_functions]

    print(f"Scanning root: {root_path}")
    results = scan_reconstruction_root(root_path, distance_fns, device)
    print(f"Saving results to: {out_file}")
    save_results_json(results, out_file)
    print("Done.")


if __name__ == "__main__":
    main()
