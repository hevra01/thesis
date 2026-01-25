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

def setup_distributed_and_device():
    """Setup distributed process group (if launched with torchrun) and return device + local rank."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and not dist_is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    return device, local_rank

def build_optimizer_param_groups(
    model: torch.nn.Module,
    lr_backbone: float,
    lr_head: float,
) -> List[Dict]:
    if lr_backbone is None or lr_head is None:
        raise ValueError("Both lr_backbone and lr_head must be provided.")

    backbone_params = []
    head_like_params = []  # head + cond_mlp

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if name.startswith("backbone."):
            backbone_params.append(p)
        elif name.startswith("head.") or name.startswith("cond_mlp."):
            head_like_params.append(p)
        else:
            # If you later add more modules, decide what to do with them.
            raise ValueError(f"Unrecognized trainable parameter: {name}")

    if not backbone_params:
        raise ValueError("No backbone parameters found (expected names starting with 'backbone.').")
    if not head_like_params:
        raise ValueError("No head/cond_mlp parameters found (expected 'head.' or 'cond_mlp.').")

    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_like_params, "lr": lr_head},
    ]

def init_wandb_if_main(cfg):
    if is_main_process():
        run = wandb.init(
            name=cfg.experiment.experiment_name,
            project=cfg.experiment.project_name,
            group=cfg.experiment.group_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.run.summary["run_id"] = run.id

def log_runtime_ddp_info(device, local_rank):
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
            wandb.config.update(ddp_info, allow_val_change=True)
        except Exception as e:
            print(f"[DDP] Failed to gather/log runtime info: {e}")


def build_recon_dataloader(cfg, reconstruction_path, base_loader_cfg, split="train"):
    """
    Build a DataLoader for the reconstruction dataset.
    """

    # Optional filtering to constrain error range
    filter_key = getattr(cfg.experiment.reconstruction_dataset, "filter_key", None)
    if filter_key is not None:
        min_error  = getattr(cfg.experiment.reconstruction_dataset, "min_error", None)
        max_error  = getattr(cfg.experiment.reconstruction_dataset, "max_error", None)
    else:
        filter_key = min_error = max_error = None

    recon_loss_key = cfg.experiment.reconstruction_dataset.reconstruction_loss

    # reconstruction_data holds per-image errors for multiple K values + token counts.
    with open(reconstruction_path, "r") as f:
        reconstruction_data = json.load(f)

    # This base dataloader supplies images to the ReconstructionDataset implementation.
    base_dataloader = instantiate(base_loader_cfg)

    # for evaluation, we typically do not want to shuffle for the sake of reproducibility
    shuffle = True if (split=="train") else False

    # reconstruction_data (list): List of dicts containing reconstruction metrics.
    # (img, k_value, mse_error, vgg_error).
    recon_dataset = ReconstructionDataset_Neural(
        reconstruction_data=reconstruction_data,
        dataloader=base_dataloader,
        filter_key=filter_key,
        min_error=min_error,
        max_error=max_error,
        error_key=recon_loss_key,
    )

    batch_size = int(cfg.experiment.reconstruction_dataset.batch_size)
    num_workers = int(getattr(cfg.experiment.reconstruction_dataset, "num_workers", 4))

    # In DDP, DistributedSampler shards the dataset across processes.
    sampler = (
        DistributedSampler(
            recon_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            drop_last=False,
        )
        if dist_is_initialized()
        else None
    )

    recon_dataloader = DataLoader(
        recon_dataset,
        batch_size=batch_size,
        shuffle=(False if sampler is not None else shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return recon_dataloader


def load_checkpoint_if_exists(ckp_path, model, optimizer, device):
    """
    Load model and optimizer state from a checkpoint file if it exists.
    """
   # Create directory if checkpoint path includes one
    ckpt_dir = os.path.dirname(ckp_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 0
    last_loss = None

    if os.path.isfile(ckp_path):
        if is_main_process():
            print(f"Found checkpoint at {ckp_path}. Loading and resuming...")
        ckpt = torch.load(ckp_path, map_location=device)

        # Load weights (DDP wraps module)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])

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

    return start_epoch, last_loss


def save_checkpoint(path, epoch_next, model, optimizer, avg_loss):
    if not is_main_process():
        return
    target = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    torch.save(
        {
            "epoch": epoch_next,
            "model_state_dict": target.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        },
        path,
    )


def compute_hard_nll_mean(logits: torch.Tensor, k_int: torch.Tensor) -> torch.Tensor:
    C = logits.size(1)
    log_p = F.log_softmax(logits, dim=1)
    idx = (k_int - 1).clamp(0, C - 1).view(-1, 1)
    hard_nll = -log_p.gather(1, idx).squeeze(1)
    return hard_nll.mean()


def train_one_epoch(model, loader, optimizer, training_loss, device, recon_loss_key, num_classes, epoch, global_step, task_type="classification"):
    model.train()

    # we will accumulate the total loss for each epoch. the accumulated total loss is composed of a sum of the losses over all samples
    # that each gpu has seen.
    loss_sum = 0.0
    nll_sum = 0.0
    count = 0 # this count is the number of samples seen by each GPU. would've been equal to the dataset size if not for DDP sharding.
    max_logk = 8.0  # since log2(256) = 8
    # to mix up the data each epoch in DDP, we need to set the epoch for the DistributedSampler
    # this ensures that each process gets a different shard of the data each epoch.
    # which prevents overfitting to a specific data order.
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    for batch in loader:
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

        bs = int(images.size(0))
        count += bs
        
        if task_type == "classification":
            cond = recon_loss
            logits = model(images, cond)  # [B,C]
            loss = training_loss(logits, k_value)  # your Gaussian-soft CE expects float targets

            k_int = k_value.long().squeeze(1)
            hard_nll_mean = compute_hard_nll_mean(logits, k_int)
            nll_sum += float(hard_nll_mean.item()) * bs

        elif task_type == "regression":
            cond = torch.log2(k_value) / max_logk
            pred = model(images, cond)  # [B,1]
            target = batch[recon_loss_key].to(device, non_blocking=True).float().unsqueeze(1)  # [B,1]
            loss = training_loss(pred, target)  # MAE/SmoothL1/MSE

            # no classification NLL for regression
            # (keep nll_sum as 0 so reducer is happy)
            hard_nll_mean = None

        else:
            raise ValueError(f"Unknown task: {task_type}")

        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item()) * bs

        if is_main_process() and global_step % 50 == 0:
            log_dict = {"train/batch_loss": float(loss.item())}
            if task_type == "classification":
                log_dict["train/batch_hard_nll"] = float(hard_nll_mean.item())
            wandb.log(log_dict, step=global_step)
        global_step += 1

    return {"main_sum": loss_sum, "nll_sum": nll_sum, "count": count, "global_step": global_step}


def ddp_reduce_epoch_metrics(metrics, device):
    main_sum = torch.tensor(metrics["main_sum"], device=device, dtype=torch.float32)
    nll_sum  = torch.tensor(metrics["nll_sum"],  device=device, dtype=torch.float32)
    count    = torch.tensor(metrics["count"],    device=device, dtype=torch.float32)

    if dist_is_initialized():
        dist.all_reduce(main_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(nll_sum,  op=dist.ReduceOp.SUM)
        dist.all_reduce(count,    op=dist.ReduceOp.SUM)

    avg_main = (main_sum / count).item() if count.item() > 0 else float("nan")
    avg_nll  = (nll_sum  / count).item() if count.item() > 0 else float("nan")
    return {"avg_main": avg_main, "avg_nll": avg_nll, "count": count.item()}

def validate_one_epoch(model, dataloader, device, recon_loss_key, num_classes, training_loss, task_type="classification"):
    """
    Runs validation for one epoch.

    Returns a dict with SUMs and COUNT (DDP-friendly):
      - ce_sum: sum of CE over samples
      - nll_sum: sum of hard NLL over samples
      - count: number of samples
    """
    model.eval()

    ce_sum = 0.0
    nll_sum = 0.0
    count = 0

    max_logk = 8.0  # since log2(256) = 8
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True).float()
            recon_loss = batch[recon_loss_key].to(device, non_blocking=True).float().unsqueeze(1)
            k_value = batch["k_value"].to(device, non_blocking=True).float().unsqueeze(1)
            k_int = k_value.long().squeeze(1)  # for indexing/classes

            bs = int(images.size(0))
            count += bs

            if task_type == "classification":
                # condition on reconstruction loss, predict token count
                cond = recon_loss
                logits = model(images, cond)  # [B,C]

                k_int = k_value.long().squeeze(1)

                # use hard CE for validation
                C = logits.size(1)
                target_idx = (k_int - 1).clamp(0, C - 1)
                ce_mean = F.cross_entropy(logits, target_idx, reduction="mean")

                hard_nll_mean = compute_hard_nll_mean(logits, k_int)

                ce_sum += float(ce_mean.item()) * bs
                nll_sum += float(hard_nll_mean.item()) * bs

            elif task_type == "regression":
                # condition on token count, predict reconstruction loss
                cond = torch.log2(k_value) / max_logk
                pred = model(images, cond)  # [B,1]
                target = batch[recon_loss_key].to(device, non_blocking=True).float().unsqueeze(1)

                loss = training_loss(pred, target)  # MAE/SmoothL1/MSE

                ce_sum += float(loss.item()) * bs
                # keep nll_sum at 0
            else:
                raise ValueError(f"Unknown task: {task_type}")

    return {"main_sum": ce_sum, "nll_sum": nll_sum, "count": count}

@hydra.main(version_base=None, config_path="../conf", config_name="neural_baseline_training")
def main(cfg: DictConfig):
    # =====================================
    # 1) Device + Distributed Initialization
    # =====================================
    device, local_rank = setup_distributed_and_device()

    # =======================
    # 2) Initialize Weights&Biases
    # =======================
    init_wandb_if_main(cfg)

    # Log runtime / DDP info (rank 0 only)
    log_runtime_ddp_info(device, local_rank)

    # 3) DataLoader for train set
    train_recon_dataloader = build_recon_dataloader(
        cfg,
        reconstruction_path=cfg.experiment.reconstruction_dataset.reconstruction_train_data_path,
        base_loader_cfg=cfg.experiment.dataset_train,
        split="train"
    )

    # 4) DataLoader for test set
    test_recon_dataloader = build_recon_dataloader(
        cfg,
        reconstruction_path=cfg.experiment.reconstruction_dataset.reconstruction_test_data_path,
        base_loader_cfg=cfg.experiment.dataset_val,
        split="eval"
    )
    
    # =======================
    # 4) Model, optimizer, losses
    # =======================
    # Classification head for predicting token counts: model outputs logits [B, C] (counts 1..C)
    # Regression head for predicting reconstruction loss: model outputs scalar [B, 1] 
    token_count_predictor = instantiate(cfg.experiment.model).to(device)

    # determine task type
    task_type = cfg.experiment.task_type
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

    # while fine-tuning resnet, we typically use a lower learning rate for the backbone.
    # hence, we explicitly build parameter groups.
    lr_backbone = cfg.experiment.optimizer_lr.get("lr_backbone", None)
    lr_head     = cfg.experiment.optimizer_lr.get("lr_head", None)

    param_groups = build_optimizer_param_groups(
        model=token_count_predictor.module if isinstance(token_count_predictor, torch.nn.parallel.DistributedDataParallel) else token_count_predictor,
        lr_backbone=lr_backbone,
        lr_head=lr_head
    )

    optimizer = instantiate(
        cfg.experiment.optimizer,
        params=param_groups,
        _convert_="all",
    )

    # training_loss:
    #  - classification: Gaussian-soft cross-entropy between token classes
    #  - regression: L1 loss
    training_loss = instantiate(cfg.experiment.training.loss_training_classification if task_type == "classification" else cfg.experiment.training.loss_training_regression)
    recon_loss_key = cfg.experiment.reconstruction_dataset.reconstruction_loss

    num_classes = int(getattr(cfg.experiment.model, "num_classes", 256))

    # 5) Resume checkpoint if it exists
    start_epoch, _ = load_checkpoint_if_exists(cfg.experiment.checkpoint_path, token_count_predictor, optimizer, device)

    # =======================
    # 6) Training loop
    # =======================
    num_epochs = int(cfg.experiment.training.num_epochs)

    # Global step for W&B batch logging
    global_step = start_epoch * len(train_recon_dataloader)
    # Track the best (lowest) validation loss seen so far; used to decide
    # whether to overwrite the checkpoint. Initialized to +inf.
    best_val_main = float("inf")

    for epoch in range(start_epoch, num_epochs):
        # train for an epoch
        train_metrics = train_one_epoch(token_count_predictor, train_recon_dataloader, optimizer, training_loss, device, recon_loss_key, num_classes, epoch, global_step, task_type=task_type)

        global_step = train_metrics["global_step"]

        # reduce train metrics across DDP processes
        reduced = ddp_reduce_epoch_metrics(train_metrics, device)

        # log epoch metrics for train
        if is_main_process():
            if task_type == "classification":
                wandb.log({"train/cross_entropy": reduced["avg_main"], "train/hard_nll": reduced["avg_nll"]})
            else:
                wandb.log({"train/mae": reduced["avg_main"], "train/hard_nll": None})
            # NOTE: Do not save checkpoint here. We only save after validation
            # if the validation loss has improved compared to previous epochs.

        # validate for one epoch
        val_metrics = validate_one_epoch(
            token_count_predictor,
            test_recon_dataloader,
            device,
            recon_loss_key,
            training_loss=training_loss,
            num_classes=num_classes,
            task_type=task_type
        )
        # reduce val metrics across DDP processes
        val_reduced = ddp_reduce_epoch_metrics(val_metrics, device)

        if is_main_process():
            if task_type == "classification":
                wandb.log({"val/cross_entropy": val_reduced["avg_main"], "val/hard_nll": val_reduced["avg_nll"], "epoch": epoch})
            else:
                wandb.log({"val/mae": val_reduced["avg_main"], "val/hard_nll": None, "epoch": epoch})
            # Save checkpoint ONLY if validation loss decreased vs previous best
            # This ensures we overwrite checkpoints only on improvement.
            if val_reduced["avg_main"] < best_val_main:
                best_val_main = val_reduced["avg_main"]
                save_checkpoint(
                    cfg.experiment.checkpoint_path,
                    epoch + 1,
                    token_count_predictor,
                    optimizer,
                    best_val_main,
                )

if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean up process group (safe to call even if not initialized)
        if dist_is_initialized():
            dist.destroy_process_group()
