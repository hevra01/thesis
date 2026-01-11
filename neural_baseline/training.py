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
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.run.summary["run_id"] = run.id

    # Log runtime / DDP info (rank 0 only)
    if is_main_process():
        try:
            num_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
            gpu_names = [torch.cuda.get_device_name(i) for i in range(num_visible)] if num_visible > 0 else []
            ddp_info = {
                "ddp/enabled": dist_is_initialized(),
                "ddp/world_size": get_world_size(),
                "ddp/rank": get_rank(),
                "ddp/local_rank": local_rank,
                "cuda/available": torch.cuda.is_available(),
                "cuda/num_visible_gpus": num_visible,
                "cuda/device": str(device),
                "cuda/visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "nccl/debug": os.environ.get("NCCL_DEBUG", ""),
                "nccl/async_error_handling": os.environ.get("NCCL_ASYNC_ERROR_HANDLING", ""),
                "omp/num_threads": os.environ.get("OMP_NUM_THREADS", ""),
                "cudnn/benchmark": bool(torch.backends.cudnn.benchmark),
                "torch/version": torch.__version__,
                "cuda/gpu_names": "; ".join(gpu_names) if gpu_names else "",
            }
            print("[DDP] Runtime info:")
            for k, v in ddp_info.items():
                print(f"  - {k}: {v}")
            wandb.config.update(ddp_info)
        except Exception as e:
            print(f"[DDP] Failed to gather/log runtime info: {e}")

    # =======================
    # 3) Load data + DataLoader
    # =======================
    # reconstruction_data holds per-image errors for multiple K values + token counts.
    with open(cfg.experiment.reconstruction_dataset.reconstruction_data_path, "r") as f:
        reconstruction_data = json.load(f)

    # This base dataloader supplies images to the ReconstructionDataset implementation.
    base_dataloader = instantiate(cfg.experiment.dataset)

    shuffle = bool(cfg.experiment.reconstruction_dataset.shuffle)

    # Optional filtering to constrain error range
    filt_cfg = getattr(cfg.experiment.reconstruction_dataset, "filter_key", None)
    if filt_cfg is not None:
        filter_key = getattr(filt_cfg, "key", None)
        min_error = getattr(filt_cfg, "min", None)
        max_error = getattr(filt_cfg, "max", None)
    else:
        filter_key = min_error = max_error = None

    recon_loss_key = cfg.experiment.reconstruction_dataset.reconstruction_loss

    # NOTE: ReconstructionDataset_Neural is assumed to exist in your codebase
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
        print("len(recon_dataloader):", len(recon_dataloader))
        print("len(recon_dataloader.dataset):", len(recon_dataloader.dataset))
        print("model:\n", token_count_predictor)

    optimizer = instantiate(
    cfg.experiment.optimizer,
        params=[p for p in token_count_predictor.parameters() if p.requires_grad]
    )


    # training_loss:
    #  - classification: Gaussian-soft cross-entropy (expects counts in [1..C])
    #  - regression: e.g., L1/L2 on scaled target y in [0, 1]
    training_loss = instantiate(cfg.experiment.training.loss_training)

    # mae_loss returns a scalar batch-mean MAE (your MAELoss implementation)
    mae_loss = instantiate(cfg.experiment.training.loss_analysis)

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
    global_step = start_epoch * len(recon_dataloader)

    for epoch in range(start_epoch, num_epochs):
        # Ensure distinct shuffling across processes each epoch
        if isinstance(recon_dataloader.sampler, DistributedSampler):
            recon_dataloader.sampler.set_epoch(epoch)

        # ---- Sample-weighted epoch accumulators (DDP-friendly) ----
        # We accumulate sums over samples, then divide by total sample count.
        epoch_train_loss_sum = 0.0     # sum over samples of training loss
        epoch_mae_sum = 0.0            # sum over samples of |E[count]-true|
        epoch_hard_nll_sum = 0.0       # sum over samples of -log p(true_class)
        epoch_count = 0                # number of samples processed

        for b_idx, batch in enumerate(recon_dataloader):
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
            out = token_count_predictor(images, recon_loss)

            if is_reg:
                # -----------------------------
                # REGRESSION HEAD
                # -----------------------------
                # Scale target to [0,1]: y = (K-1)/(C-1)
                y = (k_value - 1.0) / float(num_classes - 1)  # [B,1]

                # Training loss in regression space (e.g., L1/L2 on y)
                loss = training_loss(out, y)

                # Analysis metric: MAE in original count space
                # Convert predicted y -> predicted K in [1..C]
                pred_y = out.clamp(0.0, 1.0)  # [B,1]
                pred_k = (pred_y * (num_classes - 1) + 1.0)  # [B,1]
                mae_expected = mae_loss(pred_k, k_value)     # scalar batch mean

                # Hard-NLL is not meaningful for pure regression unless you define a likelihood model.
                # We log NaN for consistency.
                hard_nll_mean = torch.tensor(float("nan"), device=device)

            else:
                # -----------------------------
                # CLASSIFICATION HEAD (counts 1..C)
                # -----------------------------
                logits = out  # [B, C]

                # 1) Training loss: Gaussian-soft cross-entropy with targets centered at true count
                loss = training_loss(logits, k_value)  # scalar batch mean

                # 2) Analysis metric: expected-count MAE
                prob = torch.softmax(logits, dim=1)  # [B,C]
                classes = torch.arange(
                    1, num_classes + 1, device=logits.device, dtype=prob.dtype
                ).unsqueeze(0)  # [1,C]
                expected_count = (prob * classes).sum(dim=1, keepdim=True)  # [B,1]
                mae_expected = mae_loss(expected_count, k_value)  # scalar batch mean

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

            # mae_expected is a batch mean -> convert to sum over samples
            epoch_mae_sum += float(mae_expected.item()) * bs

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
                            "train/batch_mae_expected": float(mae_expected.item()),
                            "train/batch_hard_nll": (
                                float(hard_nll_mean.item()) if torch.isfinite(hard_nll_mean) else None
                            ),
                            "train/epoch": epoch + 1,
                        },
                        step=global_step,
                    )
                print(
                    f"[epoch {epoch+1:03d} | step {global_step:06d}] "
                    f"loss={loss.item():.6f} mae(E[count])={mae_expected.item():.6f} "
                    f"hard_nll={(hard_nll_mean.item() if torch.isfinite(hard_nll_mean) else float('nan')):.6f}"
                )

            global_step += 1

        # =======================
        # 7) DDP metric reduction (epoch averages)
        # =======================
        # We reduce SUMs and COUNT across all ranks for true global averages.
        loss_sum = torch.tensor(epoch_train_loss_sum, device=device, dtype=torch.float32)
        mae_sum = torch.tensor(epoch_mae_sum, device=device, dtype=torch.float32)
        nll_sum = torch.tensor(epoch_hard_nll_sum, device=device, dtype=torch.float32)
        count = torch.tensor(epoch_count, device=device, dtype=torch.float32)

        if dist_is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(mae_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(nll_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)

        avg_loss = (loss_sum / count).item()
        avg_mae = (mae_sum / count).item()

        # If regression mode, nll_sum stayed 0; report NaN for clarity.
        avg_hard_nll = (nll_sum / count).item() if (not is_reg) else float("nan")

        if is_main_process():
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"AvgLoss(train)={avg_loss:.6f} | "
                f"AvgMAE(E[count])={avg_mae:.6f} | "
                f"AvgHardNLL={avg_hard_nll:.6f}"
            )
            wandb.log(
                {
                    "train/epoch_loss": avg_loss,
                    "train/epoch_mae_expected": avg_mae,
                    "train/epoch_hard_nll": avg_hard_nll,
                    "train/epoch": epoch + 1,
                }
            )

        # =======================
        # 8) Save checkpoint (rank 0 only)
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
                f"{checkpoint_path}_minerr{min_error:.2f}.pt",
            )
            print(f"Saved checkpoint to: {checkpoint_path}_minerr{min_error:.2f}.pt")



if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean up process group (safe to call even if not initialized)
        if dist_is_initialized():
            dist.destroy_process_group()
