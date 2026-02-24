"""
Evaluation script for HeuristicTokenCountPredictor.

This script evaluates the heuristic baseline model that predicts token counts from:
  - Reconstruction loss (primary input)
  - Additional features: LID (Local Intrinsic Dimensionality), density, etc.

Evaluation modes:
1. Evaluate overall performance on the entire validation set
2. Evaluate performance for specific reconstruction loss ranges
3. Evaluate performance per token count class

No images are used - only precomputed scalar features.
"""

import json
import os
from typing import Dict, List, Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from data.utils.dataloaders import ReconstructionDataset_Heuristic


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_hard_nll_mean(logits: torch.Tensor, k_int: torch.Tensor) -> torch.Tensor:
    """Compute mean negative log-likelihood for hard class labels."""
    C = logits.size(1)
    log_p = F.log_softmax(logits, dim=1)
    # k_int is in [1..C], convert to 0-indexed
    idx = (k_int - 1).clamp(0, C - 1).view(-1, 1)
    hard_nll = -log_p.gather(1, idx).squeeze(1)
    return hard_nll.mean()


def compute_mean_absolute_error_classes(logits: torch.Tensor, k_int: torch.Tensor) -> float:
    """
    Compute mean absolute error between predicted and true class indices.
    
    Args:
        logits: Model output logits [B, num_classes]
        k_int: Ground truth token counts [B], values in [1..C]
    
    Returns:
        MAE of class predictions
    """
    preds = logits.argmax(dim=1) + 1  # Convert 0-indexed to 1-indexed
    mae = (preds - k_int).abs().float().mean().item()
    return mae


# =============================================================================
# Dataset Building (without DDP for evaluation)
# =============================================================================

def build_eval_dataloader(
    reconstruction_data_path: str,
    recon_loss_key: str,
    batch_size: int,
    additional_feature_keys: Optional[List[str]] = None,
    edge_ratio_path: Optional[str] = None,
    lid_path: Optional[str] = None,
    local_density_path: Optional[str] = None,
    lpips_variance_path: Optional[str] = None,
    dino_dist_path: Optional[str] = None,
    num_workers: int = 4,
    filter_key: Optional[str] = None,
    min_error: Optional[float] = None,
    max_error: Optional[float] = None,
    k_values: Optional[List[int]] = [1, 2, 4, 8, 16, 32, 64, 128, 256]
) -> DataLoader:
    """
    Build a DataLoader for evaluation using ReconstructionDataset_Heuristic.
    
    Optionally filters samples by reconstruction loss range.
    """
    # -----------------------------------------------------------------
    # Load reconstruction data
    # -----------------------------------------------------------------
    with open(reconstruction_data_path, "r") as f:
        reconstruction_data = json.load(f)


    # -----------------------------------------------------------------
    # Load optional feature files
    # -----------------------------------------------------------------
    edge_ratio_info = None
    local_density_info = None
    lpips_variance_info = None
    dino_dist_info = None

    if edge_ratio_path is not None:
        with open(edge_ratio_path, "r") as f:
            edge_ratio_info = json.load(f)

    best_lids_per_k = {}
    if lid_path is not None:
        with open(lid_path, "r") as f:
            lid_values = json.load(f)
        for k in (k_values):
            paired = lid_values
            best_lids_per_k[k] = paired

    density_dict = {}
    if local_density_path is not None:
        with open(local_density_path, "r") as f:
            local_density_info = json.load(f)
        for k in (k_values):
            density_dict[k] = [sum(v[0][:]) for v in local_density_info]

    if lpips_variance_path is not None:
        with open(lpips_variance_path, "r") as f:
            lpips_variance_info = json.load(f)

    if dino_dist_path is not None:
        with open(dino_dist_path, "r") as f:
            dino_dist_info = json.load(f)

    # -----------------------------------------------------------------
    # Build dataset
    # -----------------------------------------------------------------
    dataset = ReconstructionDataset_Heuristic(
        reconstruction_data=reconstruction_data,
        edge_ratio_information=edge_ratio_info,
        lid_information=best_lids_per_k,
        local_density_information=density_dict,
        lpips_variance_information=lpips_variance_info,
        dino_dist_information=dino_dist_info,
        error_key=[recon_loss_key],
        filter_key=filter_key,
        min_error=min_error,
        max_error=max_error
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for evaluation
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


# =============================================================================
# Evaluation Functions
# =============================================================================

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    recon_loss_key: str,
    additional_feature_keys: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate the heuristic model on a dataset.
    
    Args:
        model: HeuristicTokenCountPredictor
        dataloader: DataLoader for evaluation
        device: Device to run on
        recon_loss_key: Key for reconstruction loss in batch
        additional_feature_keys: List of additional feature keys
    
    Returns:
        Dict with evaluation metrics (nll, mae, count)
    """
    model.eval()

    nll_sum = 0.0
    correct = 0
    mae_sum = 0.0
    count = 0

    for batch in dataloader:
        # -----------------------------------------------------------------
        # Extract reconstruction loss (primary input)
        # -----------------------------------------------------------------
        recon_loss = batch[recon_loss_key].to(device, non_blocking=True).float()
        k_value = batch["k_value"].to(device, non_blocking=True).float().unsqueeze(1)

        # -----------------------------------------------------------------
        # Extract additional features if specified
        # -----------------------------------------------------------------
        if additional_feature_keys and len(additional_feature_keys) > 0:
            feature_list = []
            for key in additional_feature_keys:
                if key in batch:
                    feature_list.append(batch[key].to(device, non_blocking=True).float())
            if feature_list:
                additional_features = torch.stack(feature_list, dim=-1)
            else:
                additional_features = None
        else:
            additional_features = None

        # -----------------------------------------------------------------
        # Forward pass
        # -----------------------------------------------------------------
        logits = model(recon_loss, additional_features)

        # -----------------------------------------------------------------
        # Compute metrics
        # -----------------------------------------------------------------
        bs = recon_loss.size(0)
        count += bs
        k_float = k_value.long().squeeze(1)  # [B, 1]
        # NLL
        hard_nll = compute_hard_nll_mean(logits, k_float)
        nll_sum += float(hard_nll.item()) * bs

        
    return {
        "nll": nll_sum / count if count > 0 else float("nan"),
        "count": count,
    }


@torch.no_grad()
def evaluate_per_class(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    recon_loss_key: str,
    additional_feature_keys: Optional[List[str]] = None,
    num_classes: int = 9,
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate the model performance per token count class.
    
    Returns:
        Dict mapping class index (1..num_classes) to metrics dict
    """
    model.eval()

    # Per-class accumulators
    class_metrics = {c: {"nll_sum": 0.0, "correct": 0, "count": 0} for c in range(1, num_classes + 1)}

    for batch in dataloader:
        recon_loss = batch[recon_loss_key].to(device, non_blocking=True).float()
        k_value = batch["k_value"].to(device, non_blocking=True)

        # Extract additional features
        if additional_feature_keys and len(additional_feature_keys) > 0:
            feature_list = []
            for key in additional_feature_keys:
                if key in batch:
                    feature_list.append(batch[key].to(device, non_blocking=True).float())
            if feature_list:
                additional_features = torch.stack(feature_list, dim=-1)
            else:
                additional_features = None
        else:
            additional_features = None

        # Forward pass
        logits = model(recon_loss, additional_features)
        preds = logits.argmax(dim=1) + 1
        log_p = F.log_softmax(logits, dim=1)

        # Accumulate per-class metrics
        for c in range(1, num_classes + 1):
            mask = (k_value == c)
            if mask.sum() == 0:
                continue

            class_metrics[c]["count"] += int(mask.sum().item())
            class_metrics[c]["correct"] += int((preds[mask] == c).sum().item())

            # NLL for this class
            idx = (c - 1)  # 0-indexed
            nll_c = -log_p[mask, idx].sum().item()
            class_metrics[c]["nll_sum"] += nll_c

    # Compute averages
    results = {}
    for c in range(1, num_classes + 1):
        cnt = class_metrics[c]["count"]
        if cnt > 0:
            results[c] = {
                "nll": class_metrics[c]["nll_sum"] / cnt,
                "count": cnt,
            }
        else:
            results[c] = {"nll": float("nan"), "count": 0}

    return results


# =============================================================================
# Main Entry Point
# =============================================================================

@hydra.main(version_base=None, config_path="../conf", config_name="eval_heuristic_baseline")
def main(cfg: DictConfig):
    """
    Main evaluation function for HeuristicTokenCountPredictor.
    """
    device = torch.device(cfg.experiment.device)

    # -----------------------------------------------------------------
    # Initialize wandb
    # -----------------------------------------------------------------
    run = wandb.init(
        name=cfg.experiment.experiment_name,
        group=cfg.experiment.group_name,
        project=cfg.experiment.project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_slurm")
    wandb.run.summary["slurm_job_id"] = slurm_job_id

    # -----------------------------------------------------------------
    # Load model from checkpoint
    # -----------------------------------------------------------------
    model = instantiate(cfg.experiment.model).to(device)
    
    checkpoint_path = cfg.experiment.checkpoint_path
    if os.path.isfile(checkpoint_path):
        print(f"[Eval] Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Eval] Loaded model from epoch {ckpt.get('epoch', 'unknown')}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.eval()

    # -----------------------------------------------------------------
    # Get config values
    # -----------------------------------------------------------------
    recon_cfg = cfg.experiment.reconstruction_dataset
    recon_loss_key = recon_cfg.reconstruction_loss
    additional_feature_keys = list(recon_cfg.get("additional_feature_keys", []))
    num_classes = cfg.experiment.model.num_classes

    # -----------------------------------------------------------------
    # Evaluation Mode 1: Overall evaluation on full validation set
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Evaluation Mode 1: Overall Performance")
    print("=" * 60)

    full_dataloader = build_eval_dataloader(
        reconstruction_data_path=recon_cfg.reconstruction_data_path,
        recon_loss_key=recon_loss_key,
        batch_size=recon_cfg.batch_size,
        edge_ratio_path=recon_cfg.get("edge_ratio_path"),
        lid_path=recon_cfg.get("lid_path"),
        local_density_path=recon_cfg.get("local_density_path"),
        lpips_variance_path=recon_cfg.get("lpips_variance_path"),
        dino_dist_path=recon_cfg.get("dino_dist_path"),
        num_workers=recon_cfg.get("num_workers", 4),
    )

    overall_metrics = evaluate_model(
        model=model,
        dataloader=full_dataloader,
        device=device,
        recon_loss_key=recon_loss_key,
        additional_feature_keys=additional_feature_keys,
    )

    print(f"Overall NLL: {overall_metrics['nll']:.4f}")
    print(f"Total samples: {overall_metrics['count']}")

    wandb.log({
        "eval/overall_nll": overall_metrics["nll"],
        "eval/total_samples": overall_metrics["count"],
    })

    # -----------------------------------------------------------------
    # Evaluation Mode 2: Per reconstruction loss range
    # -----------------------------------------------------------------
    if recon_cfg.get("eval_by_recon_loss_ranges", False):
        print("\n" + "=" * 60)
        print("Evaluation Mode 2: Per Reconstruction Loss Range")
        print("=" * 60)

        filter_key = recon_cfg.get("filter_key", recon_loss_key)
        loss_ranges = recon_cfg.get("recon_loss_ranges", [])

        for i in range(len(loss_ranges) - 1):
            min_error = loss_ranges[i]
            max_error = loss_ranges[i + 1]

            print(f"\nEvaluating range [{min_error:.4f}, {max_error:.4f}]")

            range_dataloader = build_eval_dataloader(
                reconstruction_data_path=recon_cfg.reconstruction_data_path,
                recon_loss_key=recon_loss_key,
                batch_size=recon_cfg.batch_size,
                edge_ratio_path=recon_cfg.get("edge_ratio_path"),
                lid_path=recon_cfg.get("lid_path"),
                local_density_path=recon_cfg.get("local_density_path"),
                lpips_variance_path=recon_cfg.get("lpips_variance_path"),
                dino_dist_path=recon_cfg.get("dino_dist_path"),
                num_workers=recon_cfg.get("num_workers", 4),
                filter_key=filter_key,
                min_error=min_error,
                max_error=max_error,
            )

            range_metrics = evaluate_model(
                model=model,
                dataloader=range_dataloader,
                device=device,
                recon_loss_key=recon_loss_key,
                additional_feature_keys=additional_feature_keys,
            )

            print(f"  NLL: {range_metrics['nll']:.4f}")
            print(f"  Samples: {range_metrics['count']}")

            wandb.log({
                f"eval/range_{i}_nll": range_metrics["nll"],
                f"eval/range_{i}_count": range_metrics["count"],
                f"eval/range_{i}_min": min_error,
                f"eval/range_{i}_max": max_error,
            })

    # -----------------------------------------------------------------
    # Evaluation Mode 3: Per token count class
    # -----------------------------------------------------------------
    if recon_cfg.get("eval_per_class", True):
        print("\n" + "=" * 60)
        print("Evaluation Mode 3: Per Token Count Class")
        print("=" * 60)

        per_class_metrics = evaluate_per_class(
            model=model,
            dataloader=full_dataloader,
            device=device,
            recon_loss_key=recon_loss_key,
            additional_feature_keys=additional_feature_keys,
            num_classes=num_classes,
        )

        for c, metrics in per_class_metrics.items():
            print(f"Class {c}: NLL={metrics['nll']:.4f}, Count={metrics['count']}")
            wandb.log({
                f"eval/class_{c}_nll": metrics["nll"],
                f"eval/class_{c}_count": metrics["count"],
            })

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------
    wandb.finish()
    print("\n[Eval] Evaluation complete!")


if __name__ == "__main__":
    main()