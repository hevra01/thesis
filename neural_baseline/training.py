import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import wandb
from data.utils.dataloaders import ReconstructionDataset_Neural
import json
from hydra.utils import instantiate
from typing import Dict, List, Tuple


# ------------------------------
# Distributed utilities (DDP)
# ------------------------------
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

def build_optimizer_param_groups(
    model: torch.nn.Module,
    lr_backbone: float = None,
    lr_head: float = None,
    backbone_key: str = "backbone",
    head_key: str = "classifier",
) -> List[Dict]:
    """
    Build optimizer parameter groups for head/backbone with separate learning rates.

    Args:
        model: the PyTorch model
        lr_backbone: learning rate for backbone params (None → ignored)
        lr_head: learning rate for head params (None → ignored)
        backbone_key: substring identifying backbone params in name
        head_key: substring identifying head params in name

    Returns:
        A list of parameter groups for the optimizer.
    """

    backbone_params = []
    head_params     = []
    other_params    = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # pick group
        if backbone_key and backbone_key in name:
            backbone_params.append(p)
        elif head_key and head_key in name:
            head_params.append(p)
        else:
            other_params.append(p)

    param_groups = []

    # attach backbone group
    if lr_backbone is not None and backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})

    # attach head group
    if lr_head is not None and head_params:
        param_groups.append({"params": head_params, "lr": lr_head})

    # any leftover params (neither backbone nor head)
    if other_params:
        # if no groups defined yet, treat these as default group
        default_lr = None
        if lr_backbone is not None and not param_groups:
            default_lr = lr_backbone
        if lr_head is not None and not param_groups:
            default_lr = lr_head

        group = {"params": other_params}
        if default_lr is not None:
            group["lr"] = default_lr

        param_groups.append(group)

    # fallback
    if not param_groups:
        raise ValueError("No parameters found for optimizer!")

    return param_groups


@hydra.main(version_base=None, config_path="../conf", config_name="neural_baseline_training")
def main(cfg: DictConfig):
    # =====================================
    # 1) Device + Distributed Initialization
    # =====================================
    # If launched with torchrun, these env vars exist.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Initialize process group if torchrun env vars are present
    if ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and not dist_is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else cfg.device)
    torch.backends.cudnn.benchmark = True

    # =======================
    # 2) Initialize Weights&Biases
    # =======================
    if is_main_process():
        run = wandb.init(
            name=cfg.experiment.experiment_name,
            project=cfg.experiment.project_name,
            group=cfg.experiment.group_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.run.summary["run_id"] = run.id

    # Log runtime / DDP info (rank 0 only)
    if is_main_process():
        try:
            num_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
            gpu_names = [torch.cuda.get_device_name(i) for i in range(num_visible)] if num_visible > 0 else []
            ddp_info = {
                "ddp": {
                    "enabled": dist_is_initialized(),
                    "world_size": get_world_size(),
                    "rank": get_rank(),
                    "local_rank": local_rank,
                },
                "cuda": {
                    "available": torch.cuda.is_available(),
                    "num_visible_gpus": num_visible,
                    "device": str(device),
                    "visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "gpu_names": "; ".join(gpu_names) if gpu_names else "",
                },
                "nccl": {
                    "debug": os.environ.get("NCCL_DEBUG", ""),
                    "async_error_handling": os.environ.get("NCCL_ASYNC_ERROR_HANDLING", ""),
                },
                "omp": {
                    "num_threads": os.environ.get("OMP_NUM_THREADS", ""),
                },
                "cudnn": {
                    "benchmark": bool(torch.backends.cudnn.benchmark),
                },
                "torch": {
                    "version": torch.__version__,
                },
            }
            wandb.config.update(ddp_info)
        except Exception as e:
            print(f"[DDP] Failed to gather/log runtime info: {e}")

    # =======================
    # 3) Load data + DataLoader for train set
    # =======================
    # reconstruction_data holds per-image errors for multiple K values + token counts.
    with open(cfg.experiment.reconstruction_dataset.reconstruction_train_data_path, "r") as f:
        train_reconstruction_data = json.load(f)

    # This base dataloader supplies images to the ReconstructionDataset implementation.
    train_base_dataloader = instantiate(cfg.experiment.dataset_train)

    shuffle = bool(cfg.experiment.reconstruction_dataset.shuffle)

    # Optional filtering to constrain error range
    filter_key = getattr(cfg.experiment.reconstruction_dataset, "filter_key", None)
    if filter_key is not None:
        min_error  = getattr(cfg.experiment.reconstruction_dataset, "min_error", None)
        max_error  = getattr(cfg.experiment.reconstruction_dataset, "max_error", None)
    else:
        filter_key = min_error = max_error = None

    recon_loss_key = cfg.experiment.reconstruction_dataset.reconstruction_loss

    # NOTE: ReconstructionDataset_Neural is assumed to exist in your codebase
    train_recon_dataset = ReconstructionDataset_Neural(
        reconstruction_data=train_reconstruction_data,
        dataloader=train_base_dataloader,
        filter_key=filter_key,
        min_error=min_error,
        max_error=max_error,
        error_key=recon_loss_key,
    )

    batch_size = int(cfg.experiment.reconstruction_dataset.batch_size)
    num_workers = int(getattr(cfg.experiment.reconstruction_dataset, "num_workers", 4))

    # In DDP, DistributedSampler shards the dataset across processes.
    train_sampler = (
        DistributedSampler(
            train_recon_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            drop_last=False,
        )
        if dist_is_initialized()
        else None
    )

    train_recon_dataloader = DataLoader(
        train_recon_dataset,
        batch_size=batch_size,
        shuffle=(False if train_sampler is not None else shuffle),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # =======================
    # 3) Load data + DataLoader for test set
    # =======================
    # reconstruction_data holds per-image errors for multiple K values + token counts.
    with open(cfg.experiment.reconstruction_dataset.reconstruction_test_data_path, "r") as f:
        test_reconstruction_data = json.load(f)

    # This base dataloader supplies images to the ReconstructionDataset implementation.
    test_base_dataloader = instantiate(cfg.experiment.dataset_val)

    shuffle = bool(cfg.experiment.reconstruction_dataset.shuffle)

    # NOTE: ReconstructionDataset_Neural is assumed to exist in your codebase
    test_recon_dataset = ReconstructionDataset_Neural(
        reconstruction_data=test_reconstruction_data,
        dataloader=test_base_dataloader,
        filter_key=filter_key,
        min_error=min_error,
        max_error=max_error,
        error_key=recon_loss_key,
    )

    # In DDP, DistributedSampler shards the dataset across processes.
    test_sampler = (
        DistributedSampler(
            test_recon_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            drop_last=False,
        )
        if dist_is_initialized()
        else None
    )

    test_recon_dataloader = DataLoader(
        test_recon_dataset,
        batch_size=batch_size,
        shuffle=(False if test_sampler is not None else shuffle),
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # =======================
    # 4) Model, optimizer, losses
    # =======================
    # Classification head: model outputs logits [B, C] (counts 1..C)
    # Regression head: model outputs scalar [B, 1] (scaled target in [0, 1])
    token_count_predictor = instantiate(cfg.experiment.model).to(device)

    if dist_is_initialized():
        token_count_predictor = torch.nn.parallel.DistributedDataParallel(
            token_count_predictor,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    # Count trainable parameters
    if is_main_process():
        num_params = sum(p.numel() for p in token_count_predictor.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params:,}")
        print("len(recon_dataloader):", len(train_recon_dataloader))
        print("len(recon_dataloader.dataset):", len(train_recon_dataloader.dataset))
        print("model:\n", token_count_predictor)

    lr_backbone = cfg.experiment.optimizer_lr.get("lr_backbone", None)
    lr_head     = cfg.experiment.optimizer_lr.get("lr_head", None)

    param_groups = build_optimizer_param_groups(
        model=token_count_predictor,
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        backbone_key="backbone",  # adjust if your model uses a different naming
        head_key="classifier",
    )

    optimizer = instantiate(
        cfg.experiment.optimizer,
        params=param_groups,
        _convert_="all",
    )

    # training_loss:
    #  - classification: Gaussian-soft cross-entropy 
    training_loss = instantiate(cfg.experiment.training.loss_training)

    is_reg = (getattr(cfg.experiment.model, "head", "classification") == "regression")
    num_classes = int(getattr(cfg.experiment.model, "num_classes", 256))

    # =======================
    # 5) Resume checkpoint (single file)
    # =======================
    checkpoint_path = cfg.experiment.checkpoint_path
    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    start_epoch = 0

    if os.path.isfile(checkpoint_path):
        if is_main_process():
            print(f"Found checkpoint at {checkpoint_path}. Loading and resuming...")
        ckpt = torch.load(checkpoint_path, map_location=device)

        # Load weights (DDP wraps module)
        if isinstance(token_count_predictor, torch.nn.parallel.DistributedDataParallel):
            token_count_predictor.module.load_state_dict(ckpt["model_state_dict"])
        else:
            token_count_predictor.load_state_dict(ckpt["model_state_dict"])

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        last_loss = ckpt.get("loss", None)

        if is_main_process():
            if last_loss is not None:
                print(f"Resuming from epoch {start_epoch} with last avg loss = {last_loss:.6f}")
            else:
                print(f"Resuming from epoch {start_epoch}")
    else:
        if is_main_process():
            print("No checkpoint found. Starting from epoch 0.")

    # =======================
    # 6) Training loop
    # =======================
    num_epochs = int(cfg.experiment.training.num_epochs)
    token_count_predictor.train()

    # Global step for W&B batch logging
    global_step = start_epoch * len(train_recon_dataloader)

    for epoch in range(start_epoch, num_epochs):
        # Ensure distinct shuffling across processes each epoch
        if isinstance(train_recon_dataloader.sampler, DistributedSampler):
            train_recon_dataloader.sampler.set_epoch(epoch)

        # ---- Sample-weighted epoch accumulators (DDP-friendly) ----
        # We accumulate sums over samples, then divide by total sample count.
        epoch_train_loss_sum = 0.0     # sum over samples of training loss
        epoch_hard_nll_sum = 0.0       # sum over samples of -log p(true_class)
        epoch_count = 0                # number of samples processed

        for b_idx, batch in enumerate(train_recon_dataloader):
            images = batch["image"].to(device, non_blocking=True).float()

            # recon_loss is a model input feature (not the target)
            recon_loss = batch[recon_loss_key].to(device, non_blocking=True).float().unsqueeze(1)

            # k_value is the target token count in [1..C]
            # keep float for Gaussian-soft CE; also keep an int version for indexing
            k_value = batch["k_value"].to(device, non_blocking=True).float().unsqueeze(1)  # [B,1]
            k_int = k_value.long().squeeze(1)  # [B], values in [1..C]

            optimizer.zero_grad(set_to_none=True)

            # Forward:
            #  - classification: logits [B, C]
            #  - regression: pred_y [B, 1] in [0,1] (or unclamped, depending on head)
            token_count_prediction = token_count_predictor(images, recon_loss)
            
            # -----------------------------
            # CLASSIFICATION HEAD (counts 1..C)
            # -----------------------------
            logits = token_count_prediction  # [B, C]

            # 1) Training loss: Gaussian-soft cross-entropy with targets centered at true count
            loss = training_loss(logits, k_value)  # scalar batch mean


            # 3) Diagnostic metric: hard NLL of the true class only
            #    hard_nll = -log p(true_class)
            log_p = F.log_softmax(logits, dim=1)  # [B,C]
            idx = (k_int - 1).clamp(0, num_classes - 1).view(-1, 1)  # [B,1]
            hard_nll = -log_p.gather(1, idx).squeeze(1)  # [B]
            hard_nll_mean = hard_nll.mean()              # scalar batch mean

            # Backprop + update (per batch optimization step)
            loss.backward()
            optimizer.step()

            # -----------------------------
            # Accumulate sample-weighted sums
            # -----------------------------
            bs = int(k_value.size(0))
            epoch_count += bs

            # training loss is a batch mean -> convert to sum over samples
            epoch_train_loss_sum += float(loss.item()) * bs

            # hard_nll_mean is a batch mean (classification) -> sum over samples
            # For regression we stored NaN; skip accumulation in that case.
            if torch.isfinite(hard_nll_mean):
                epoch_hard_nll_sum += float(hard_nll_mean.item()) * bs

            # -----------------------------
            # Optional: batch logging (rank 0)
            # -----------------------------
            if is_main_process():
                # Logging per-batch is useful for debugging but noisy; throttle if desired.
                if global_step % 50 == 0:
                    wandb.log(
                        {
                            "train/batch_loss": float(loss.item()),
                            "train/batch_hard_nll": (
                                float(hard_nll_mean.item()) if torch.isfinite(hard_nll_mean) else None
                            ),
                            "train/epoch": epoch + 1,
                        },
                        step=global_step,
                    )
                print(
                    f"[epoch {epoch+1:03d} | step {global_step:06d}] "
                    f"hard_nll={(hard_nll_mean.item() if torch.isfinite(hard_nll_mean) else float('nan')):.6f}"
                )

            global_step += 1

        # =======================
        # 7) DDP metric reduction (epoch averages)
        # =======================
        # We reduce SUMs and COUNT across all ranks for true global averages.
        loss_sum = torch.tensor(epoch_train_loss_sum, device=device, dtype=torch.float32)
        nll_sum = torch.tensor(epoch_hard_nll_sum, device=device, dtype=torch.float32)
        count = torch.tensor(epoch_count, device=device, dtype=torch.float32)

        if dist_is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(nll_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)

        avg_loss = (loss_sum / count).item()

        # If regression mode, nll_sum stayed 0; report NaN for clarity.
        avg_hard_nll = (nll_sum / count).item() if (not is_reg) else float("nan")

        if is_main_process():
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"AvgLoss(train)={avg_loss:.6f} | "
                f"AvgHardNLL={avg_hard_nll:.6f}"
            )
            wandb.log(
                {
                    "train/epoch_loss": avg_loss,
                    "train/epoch_hard_nll": avg_hard_nll,
                    "train/epoch": epoch + 1,
                }
            )

        # =======================
        # 8) Validation evaluation (optional; no optimization)
        # =======================
        # Switch to eval mode for deterministic behavior in BN/Dropout
        token_count_predictor.eval()

        val_ce_sum = 0.0
        val_nll_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for v_idx, v_batch in enumerate(test_recon_dataloader):
                v_images = v_batch["image"].to(device, non_blocking=True).float()
                v_recon_loss = v_batch[recon_loss_key].to(device, non_blocking=True).float().unsqueeze(1)
                v_k_value = v_batch["k_value"].to(device, non_blocking=True).float().unsqueeze(1)
                v_k_int = v_k_value.long().squeeze(1)

                v_out = token_count_predictor(v_images, v_recon_loss)

                bs = int(v_k_value.size(0))
                val_count += bs

                v_logits = v_out  # [B,C]
                # Cross-entropy (hard labels)
                target_idx = (v_k_int - 1).clamp(0, num_classes - 1)
                ce_mean = F.cross_entropy(v_logits, target_idx, reduction="mean")

                # Hard-NLL for true class
                v_log_p = F.log_softmax(v_logits, dim=1)
                v_idx = target_idx.view(-1, 1)
                v_hard_nll = -v_log_p.gather(1, v_idx).squeeze(1)  # [B]
                nll_mean = v_hard_nll.mean()

                val_ce_sum += float(ce_mean.item()) * bs
                val_nll_sum += float(nll_mean.item()) * bs

        # DDP reduction for validation sums
        val_ce_sum_t = torch.tensor(val_ce_sum, device=device, dtype=torch.float32)
        val_nll_sum_t = torch.tensor(val_nll_sum, device=device, dtype=torch.float32)
        val_count_t = torch.tensor(val_count, device=device, dtype=torch.float32)

        if dist_is_initialized():
            dist.all_reduce(val_ce_sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_nll_sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_count_t, op=dist.ReduceOp.SUM)

        if not is_reg and val_count_t.item() > 0:
            avg_val_ce = (val_ce_sum_t / val_count_t).item()
            avg_val_nll = (val_nll_sum_t / val_count_t).item()
        else:
            avg_val_ce = float("nan")
            avg_val_nll = float("nan")

        if is_main_process():
            print(
                f"Validation Epoch {epoch + 1}/{num_epochs} | "
                f"AvgCrossEntropy(val)={avg_val_ce:.6f} | "
                f"AvgHardNLL(val)={avg_val_nll:.6f}"
            )
            wandb.log(
                {
                    "val/cross_entropy": avg_val_ce,
                    "val/hard_nll": avg_val_nll,
                    "train/epoch": epoch + 1,
                }
            )

        # Return to train mode for next epoch
        token_count_predictor.train()

        # =======================
        # 9) Save checkpoint (rank 0 only)
        # =======================
        if is_main_process():
            state_dict = (
                token_count_predictor.module.state_dict()
                if isinstance(token_count_predictor, torch.nn.parallel.DistributedDataParallel)
                else token_count_predictor.state_dict()
            )
            torch.save(
                {
                    "epoch": epoch + 1,  # next epoch to run
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                f"{checkpoint_path}",
            )
            print(f"Saved checkpoint to: {checkpoint_path}")



if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean up process group (safe to call even if not initialized)
        if dist_is_initialized():
            dist.destroy_process_group()
