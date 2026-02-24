"""
Training script for HeuristicTokenCountPredictor.

This model predicts token counts from:
  - Reconstruction loss (primary input)
  - Additional features: LID (Local Intrinsic Dimensionality) and density

No images are used - this is a lightweight MLP classifier.
"""

import os
import json
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import wandb

from data.utils.dataloaders import ReconstructionDataset_Heuristic


# =============================================================================
# Distributed Utilities (simplified from neural_baseline)
# =============================================================================

def dist_is_initialized() -> bool:
    """Return True if torch.distributed is both available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the global rank if initialized, else 0 (single-process)."""
    return dist.get_rank() if dist_is_initialized() else 0


def get_world_size() -> int:
    """Return the world size if initialized, else 1."""
    return dist.get_world_size() if dist_is_initialized() else 1


def is_main_process() -> bool:
    """True only for rank 0 (used to gate logging/checkpointing)."""
    return get_rank() == 0


def setup_distributed_and_device():
    """Setup distributed process group (if launched with torchrun) and return device + local rank."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and not dist_is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return device, local_rank


def compute_hard_nll_mean(logits: torch.Tensor, k_int: torch.Tensor) -> torch.Tensor:
    """Compute mean negative log-likelihood for hard class labels."""
    C = logits.size(1)
    log_p = F.log_softmax(logits, dim=1)
    # k_int is in [1..C], convert to 0-indexed
    idx = (k_int - 1).clamp(0, C - 1).view(-1, 1)
    hard_nll = -log_p.gather(1, idx).squeeze(1)
    return hard_nll.mean()


# =============================================================================
# Dataloader Builder using ReconstructionDataset_Heuristic
# =============================================================================

def build_heuristic_dataloader(
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
    shuffle: bool = True,
    k_values: Optional[List[int]] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
) -> DataLoader:
    """
    Build a DataLoader using ReconstructionDataset_Heuristic.
    
    Args:
        reconstruction_data_path: Path to JSON file containing reconstruction data
        recon_loss_key: Key for reconstruction loss (e.g., "LPIPS", "vgg_error")
        batch_size: Batch size for the dataloader
        additional_feature_keys: List of additional feature keys to include
                                 (e.g., ["lid", "local_density", "edge_ratio"])
        edge_ratio_path: Path to edge ratio JSON file (optional)
        lid_path: Path to LID JSON file (optional)
        local_density_path: Path to local density JSON file (optional)
        lpips_variance_path: Path to LPIPS variance JSON file (optional)
        dino_dist_path: Path to DINO distance JSON file (optional)
        num_workers: Number of dataloader workers
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader for the heuristic dataset
    """
    # -----------------------------------------------------------------
    # Load reconstruction data
    # -----------------------------------------------------------------
    with open(reconstruction_data_path, "r") as f:
        reconstruction_data = json.load(f)

    if is_main_process():
        print(f"[Dataloader] Loaded {len(reconstruction_data)} samples from {reconstruction_data_path}")

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
        if is_main_process():
            print(f"  - Loaded edge_ratio from {edge_ratio_path}")

    best_lids_per_k = {}
    if lid_path is not None:
        with open(lid_path, "r") as f:
            lid_values = json.load(f)
        for k in (k_values):
            paired = lid_values
            best_lids_per_k[k] = paired
        if is_main_process():
            print(f"  - Loaded LID from {lid_path}")

    density_dict = {}
    if local_density_path is not None:
        with open(local_density_path, "r") as f:
            local_density_info = json.load(f)
        for k in (k_values):
            density_dict[k] = [sum(v[0][:]) for v in local_density_info]
        if is_main_process():
            print(f"  - Loaded local_density from {local_density_path}")

    if lpips_variance_path is not None:
        with open(lpips_variance_path, "r") as f:
            lpips_variance_info = json.load(f)
        if is_main_process():
            print(f"  - Loaded lpips_variance from {lpips_variance_path}")

    if dino_dist_path is not None:
        with open(dino_dist_path, "r") as f:
            dino_dist_info = json.load(f)
        if is_main_process():
            print(f"  - Loaded dino_dist from {dino_dist_path}")

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
        error_key=[recon_loss_key],  # Pass as list as expected by the dataset
    )

    # -----------------------------------------------------------------
    # DDP sampler for distributed training
    # -----------------------------------------------------------------
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            drop_last=False,
        )
        if dist_is_initialized()
        else None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler is not None else shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return dataloader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    global_step: int,
    recon_loss_key: str,
    additional_feature_keys: Optional[List[str]] = None,
) -> Dict:
    """
    Train the heuristic model for one epoch.

    Args:
        model: HeuristicTokenCountPredictor
        dataloader: DataLoader for training data
        optimizer: Optimizer
        loss_fn: GaussianCrossEntropyLoss
        device: Device to train on
        epoch: Current epoch number
        global_step: Global training step counter
        recon_loss_key: Key for reconstruction loss in batch (e.g., "LPIPS")
        additional_feature_keys: List of additional feature keys (e.g., ["lid", "local_density"])

    Returns:
        Dict with training metrics (sums and counts for DDP reduction)
    """
    model.train()

    loss_sum = 0.0
    nll_sum = 0.0
    count = 0

    # Set epoch for DistributedSampler to ensure proper shuffling
    if isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)

    for batch_idx, batch in enumerate(dataloader):
        # -----------------------------------------------------------------
        # Extract reconstruction loss (primary input)
        # -----------------------------------------------------------------
        recon_loss = batch[recon_loss_key].to(device, non_blocking=True).float()
        k_value = batch["k_value"].to(device, non_blocking=True).float().unsqueeze(1)

        # -----------------------------------------------------------------
        # Extract additional features (LID, local_density, etc.) if specified
        # Stack them into a single tensor [B, num_features]
        # -----------------------------------------------------------------
        if additional_feature_keys and len(additional_feature_keys) > 0:
            feature_list = []
            for key in additional_feature_keys:
                if key in batch:
                    feature_list.append(batch[key].to(device, non_blocking=True).float())
            if feature_list:
                additional_features = torch.stack(feature_list, dim=-1)  # [B, num_features]
            else:
                additional_features = None
        else:
            additional_features = None

        # -----------------------------------------------------------------
        # Forward pass
        # -----------------------------------------------------------------
        optimizer.zero_grad(set_to_none=True)

        logits = model(recon_loss, additional_features)  # [B, num_classes]

        # GaussianCrossEntropyLoss expects float targets for soft labels
        k_float = k_value.long().squeeze(1)  # [B, 1]
        loss = loss_fn(logits, k_value)

        # -----------------------------------------------------------------
        # Backward pass
        # -----------------------------------------------------------------
        loss.backward()
        optimizer.step()

        # -----------------------------------------------------------------
        # Accumulate metrics
        # -----------------------------------------------------------------
        bs = recon_loss.size(0)
        count += bs
        loss_sum += float(loss.item()) * bs

        # Compute hard NLL for monitoring
        hard_nll = compute_hard_nll_mean(logits, k_float)
        nll_sum += float(hard_nll.item()) * bs

        # -----------------------------------------------------------------
        # Logging
        # -----------------------------------------------------------------
        if is_main_process() and (batch_idx < 3 or batch_idx % 100 == 0):
            print(
                f"  [Epoch {epoch} | Batch {batch_idx}/{len(dataloader)}] "
                f"loss: {loss.item():.4f} | hard_nll: {hard_nll.item():.4f}"
            )

        if is_main_process() and global_step % 50 == 0:
            wandb.log(
                {
                    "train/batch_loss": float(loss.item()),
                    "train/batch_hard_nll": float(hard_nll.item()),
                    "train/global_step": global_step,
                },
                step=global_step,
            )

        global_step += 1

    return {
        "loss_sum": loss_sum,
        "nll_sum": nll_sum,
        "count": count,
        "global_step": global_step,
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    recon_loss_key: str,
    additional_feature_keys: Optional[List[str]] = None,
) -> Dict:
    """
    Validate the heuristic model for one epoch.

    Args:
        model: HeuristicTokenCountPredictor
        dataloader: DataLoader for validation data
        loss_fn: Loss function (GaussianCrossEntropyLoss)
        device: Device to run on
        recon_loss_key: Key for reconstruction loss in batch (e.g., "LPIPS")
        additional_feature_keys: List of additional feature keys (e.g., ["lid", "local_density"])

    Returns:
        Dict with validation metrics (sums and counts for DDP reduction)
    """
    model.eval()

    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            # -----------------------------------------------------------------
            # Extract reconstruction loss (primary input)
            # -----------------------------------------------------------------
            recon_loss = batch[recon_loss_key].to(device, non_blocking=True).float()
            k_value = batch["k_value"].to(device, non_blocking=True).float().unsqueeze(1)

            # -----------------------------------------------------------------
            # Extract additional features (LID, local_density, etc.) if specified
            # -----------------------------------------------------------------
            if additional_feature_keys and len(additional_feature_keys) > 0:
                feature_list = []
                for key in additional_feature_keys:
                    if key in batch:
                        feature_list.append(batch[key].to(device, non_blocking=True).float())
                if feature_list:
                    additional_features = torch.stack(feature_list, dim=-1)  # [B, num_features]
                else:
                    additional_features = None
            else:
                additional_features = None

            # -----------------------------------------------------------------
            # Forward pass
            # -----------------------------------------------------------------
            logits = model(recon_loss, additional_features)

            # Loss computation
            k_float = k_value.long().squeeze(1)
            loss = loss_fn(logits, k_value)

            # -----------------------------------------------------------------
            # Metrics
            # -----------------------------------------------------------------
            bs = recon_loss.size(0)
            count += bs
            loss_sum += float(loss.item()) * bs

            # Hard NLL
            hard_nll = compute_hard_nll_mean(logits, k_float)
            nll_sum += float(hard_nll.item()) * bs

            # Accuracy (predicted class vs true class)
            preds = logits.argmax(dim=1) + 1  # Convert 0-indexed to 1-indexed
            correct += (preds == k_value).sum().item()

    return {
        "loss_sum": loss_sum,
        "nll_sum": nll_sum,
        "correct": correct,
        "count": count,
    }


def ddp_reduce_metrics(metrics: Dict, device: torch.device) -> Dict:
    """Reduce metrics across all DDP processes."""
    tensors = {k: torch.tensor(v, device=device, dtype=torch.float32) for k, v in metrics.items()}

    if dist_is_initialized():
        for t in tensors.values():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    count = tensors["count"].item()
    results = {}

    for k, t in tensors.items():
        if k == "count":
            results[k] = count
        elif k == "correct":
            results["accuracy"] = t.item() / count if count > 0 else 0.0
        else:
            # Average the sum
            key_name = k.replace("_sum", "")
            results[f"avg_{key_name}"] = t.item() / count if count > 0 else float("nan")

    return results


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(path: str, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, loss: float):
    """Save model checkpoint (only on main process)."""
    if not is_main_process():
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    target = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": target.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    print(f"[Checkpoint] Saved to {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    """Load checkpoint if it exists."""
    if not os.path.isfile(path):
        if is_main_process():
            print(f"[Checkpoint] No checkpoint found at {path}. Starting fresh.")
        return 0, None

    if is_main_process():
        print(f"[Checkpoint] Loading from {path}")

    ckpt = torch.load(path, map_location=device)

    target = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    target.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_epoch = ckpt.get("epoch", 0)
    best_loss = ckpt.get("loss", None)

    if is_main_process():
        print(f"[Checkpoint] Resuming from epoch {start_epoch}, best_loss={best_loss}")

    return start_epoch, best_loss


# =============================================================================
# Main Training Entry Point
# =============================================================================

@hydra.main(version_base=None, config_path="../conf", config_name="heuristic_baseline_training")
def main(cfg: DictConfig):
    """
    Main training function for HeuristicTokenCountPredictor.
    """
    # -----------------------------------------------------------------
    # Setup distributed training and device
    # -----------------------------------------------------------------
    device, local_rank = setup_distributed_and_device()

    # -----------------------------------------------------------------
    # Initialize wandb (main process only)
    # -----------------------------------------------------------------
    if is_main_process():
        wandb.init(
            name=cfg.experiment.experiment_name,
            project=cfg.experiment.project_name,
            group=cfg.experiment.group_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # -----------------------------------------------------------------
    # Build dataloaders
    # -----------------------------------------------------------------
    recon_cfg = cfg.experiment.reconstruction_dataset
    recon_loss_key = recon_cfg.reconstruction_loss
    
    # Additional feature keys: these are the keys used in ReconstructionDataset_Heuristic
    # e.g., ["lid", "local_density", "edge_ratio", "lpips_variance", "dino_dist"]
    additional_feature_keys = list(recon_cfg.get("additional_feature_keys", []))

    train_loader = build_heuristic_dataloader(
        reconstruction_data_path=recon_cfg.reconstruction_train_data_path,
        recon_loss_key=recon_loss_key,
        batch_size=recon_cfg.batch_size,
        additional_feature_keys=additional_feature_keys,
        edge_ratio_path=recon_cfg.get("edge_ratio_path"),
        lid_path=recon_cfg.get("lid_path"),
        local_density_path=recon_cfg.get("local_density_path"),
        lpips_variance_path=recon_cfg.get("lpips_variance_path"),
        dino_dist_path=recon_cfg.get("dino_dist_path"),
        num_workers=recon_cfg.get("num_workers", 4),
        shuffle=True,
    )

    val_loader = build_heuristic_dataloader(
        reconstruction_data_path=recon_cfg.reconstruction_test_data_path,
        recon_loss_key=recon_loss_key,
        batch_size=recon_cfg.batch_size,
        additional_feature_keys=additional_feature_keys,
        edge_ratio_path=recon_cfg.get("edge_ratio_path_val"),
        lid_path=recon_cfg.get("lid_path_val"),
        local_density_path=recon_cfg.get("local_density_path_val"),
        lpips_variance_path=recon_cfg.get("lpips_variance_path_val"),
        dino_dist_path=recon_cfg.get("dino_dist_path_val"),
        num_workers=recon_cfg.get("num_workers", 4),
        shuffle=False,
    )

    # -----------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------
    model = instantiate(cfg.experiment.model).to(device)

    if is_main_process():
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Model] HeuristicTokenCountPredictor with {num_params:,} trainable parameters")

    # Wrap in DDP if distributed
    if dist_is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    # -----------------------------------------------------------------
    # Build optimizer and loss function
    # Simple single-LR optimizer (no backbone/head split needed)
    # -----------------------------------------------------------------
    optimizer = instantiate(
        cfg.experiment.optimizer,
        params=model.parameters(),
        _convert_="all",
    )

    loss_fn = instantiate(cfg.experiment.training.loss_training_classification)

    # -----------------------------------------------------------------
    # Load checkpoint if exists
    # -----------------------------------------------------------------
    start_epoch, best_loss = load_checkpoint(
        cfg.experiment.checkpoint_path_best,
        model,
        optimizer,
        device,
    )

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------
    num_epochs = cfg.experiment.training.num_epochs
    global_step = 0

    for epoch in range(start_epoch, num_epochs):
        if is_main_process():
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print(f"{'='*60}")

        # Train
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            global_step=global_step,
            recon_loss_key=recon_loss_key,
            additional_feature_keys=additional_feature_keys,
        )
        global_step = train_metrics["global_step"]

        # Reduce train metrics across DDP
        train_reduced = ddp_reduce_metrics(
            {"loss_sum": train_metrics["loss_sum"], "nll_sum": train_metrics["nll_sum"], "count": train_metrics["count"]},
            device,
        )

        # Validate
        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            recon_loss_key=recon_loss_key,
            additional_feature_keys=additional_feature_keys,
        )

        # Reduce val metrics across DDP
        val_reduced = ddp_reduce_metrics(val_metrics, device)

        # -----------------------------------------------------------------
        # Logging
        # -----------------------------------------------------------------
        if is_main_process():
            print(f"\n[Epoch {epoch}] Train: loss={train_reduced['avg_loss']:.4f}, nll={train_reduced['avg_nll']:.4f}")
            print(f"[Epoch {epoch}] Val:   loss={val_reduced['avg_loss']:.4f}, nll={val_reduced['avg_nll']:.4f}, acc={val_reduced['accuracy']:.4f}")

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_reduced["avg_loss"],
                    "train/nll": train_reduced["avg_nll"],
                    "val/loss": val_reduced["avg_loss"],
                    "val/nll": val_reduced["avg_nll"],
                    "val/accuracy": val_reduced["accuracy"],
                },
                step=global_step,
            )

        # -----------------------------------------------------------------
        # Checkpointing (save best model)
        # -----------------------------------------------------------------
        current_loss = val_reduced["avg_loss"]
        if best_loss is None or current_loss < best_loss:
            best_loss = current_loss
            save_checkpoint(
                cfg.experiment.checkpoint_path_best,
                epoch + 1,
                model,
                optimizer,
                best_loss,
            )

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------
    if is_main_process():
        wandb.finish()

    if dist_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()