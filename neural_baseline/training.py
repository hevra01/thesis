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

    # -------------------------------------
    # Device and distributed initialization
    # -------------------------------------
    # If launched via torchrun, LOCAL_RANK, RANK, WORLD_SIZE are set.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    # Initialize process group if torchrun env vars are present.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist_is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    # Select device: per-process GPU in DDP or fallback to cfg.device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else cfg.device)
    torch.backends.cudnn.benchmark = True

    # Initialize W&B and dump Hydra config
    if is_main_process():
        wandb.init(
            project="dataset_prep",
            name=cfg.experiment.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Load JSONs
    # this data holds images, mse_errors, vgg_errors, token counts for different k_values
    with open(cfg.experiment.reconstruction_dataset.reconstruction_data_path, "r") as f:
        reconstruction_data = json.load(f)

    # this is just to get the images from the dataloader to be used by ReconstructionDataset
    # we will need them for the compression_rate_predictor which will get the latents of the images to make bpp predictions. 
    dataloader = instantiate(cfg.experiment.dataset)

    shuffle = cfg.experiment.reconstruction_dataset.shuffle

    # This dataset holds the mse_errors, vgg_errors for all the images for different 
    # values of k_values and the  bpp.
    # Optional filtering to constrain error range (e.g., near-constant difficulty)
    filt_cfg = getattr(cfg.experiment.reconstruction_dataset, "filter", "vgg_error")
    if filt_cfg is not None:
        filter_key = getattr(filt_cfg, "key", None)
        min_error = getattr(filt_cfg, "min", None)
        max_error = getattr(filt_cfg, "max", None)
    else:
        filter_key = min_error = max_error = None

    recon_dataset = ReconstructionDataset_Neural(
        reconstruction_data=reconstruction_data,
        dataloader=dataloader,
        filter_key=filter_key,
        min_error=min_error,
        max_error=max_error,
    )

    batch_size = cfg.experiment.reconstruction_dataset.batch_size
    

    # Convert recon_dataset into a DataLoader with optional DistributedSampler
    # In DDP, each process sees a distinct shard. Sampler handles shuffling per-epoch.
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
    num_workers = getattr(cfg.experiment.reconstruction_dataset, "num_workers", 4)
    recon_dataloader = DataLoader(
        recon_dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers and num_workers > 0 else False,
    )

    # ---------------------------
    # Model, optimizer, loss
    # ---------------------------
    token_count_predictor = instantiate(cfg.experiment.model).to(device)
    # Wrap model with DDP so each process operates on its device shard
    if dist_is_initialized():
        token_count_predictor = torch.nn.parallel.DistributedDataParallel(
            token_count_predictor,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    # Count only trainable parameters
    num_params = sum(p.numel() for p in token_count_predictor.parameters() if p.requires_grad)
    if is_main_process():
        print(f"Number of trainable parameters: {num_params:,}")

    optimizer = instantiate(cfg.experiment.optimizer, params=token_count_predictor.parameters())
    training_loss = instantiate(cfg.experiment.training.loss_training)
    mae_loss = instantiate(cfg.experiment.training.loss_analysis)

    # check if regression or classification
    is_reg = getattr(cfg.experiment.model, "head", "classification") == "regression"
    num_classes = getattr(cfg.experiment.model, "num_classes", 256)


    # ---------------------------
    # SIMPLE CHECKPOINT RESUME
    #   - We use ONE file only (e.g., ".../last.pt").
    #   - If it exists, load it and continue from its saved epoch.
    #   - If not, start from epoch 0.
    # ---------------------------
    # Put a single path in your config, e.g.:
    # cfg.experiment.checkpoint_path = "checkpoints/last.pt"
    checkpoint_path = cfg.experiment.checkpoint_path
    start_epoch = 0  # default: start from scratch

    if os.path.isfile(checkpoint_path):
        if is_main_process():
            print(f"Found checkpoint at {checkpoint_path}. Loading and resuming...")
        ckpt = torch.load(checkpoint_path, map_location=device)

        # Restore model/optimizer and figure out where to resume
        # If using DDP, the underlying module holds the actual weights
        if isinstance(token_count_predictor, torch.nn.parallel.DistributedDataParallel):
            token_count_predictor.module.load_state_dict(ckpt["model_state_dict"])
        else:
            token_count_predictor.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))  # next epoch will be start_epoch
        last_loss = ckpt.get("loss", None)
        if is_main_process():
            if last_loss is not None:
                print(f"Resuming from epoch {start_epoch} with last avg loss = {last_loss:.6f}")
            else:
                print(f"Resuming from epoch {start_epoch}")
    else:
        if is_main_process():
            print("No checkpoint found. Starting from epoch 0.")

    # ---------------------------
    # Training loop (resume-aware)
    # ---------------------------
    num_epochs = cfg.experiment.training.num_epochs

    # set the model in training mode
    token_count_predictor.train()

    if is_main_process():
        print("len(recon_dataloader):", len(recon_dataloader))
        dataset_size = len(recon_dataloader.dataset)
        print("dataset_size:", dataset_size)
    else:
        dataset_size = len(recon_dataloader.dataset)

    # Log filtering info if applied
    if is_main_process() and hasattr(recon_dataset, "filter_key") and recon_dataset.filter_key is not None:
        print(
            f"Applied filtering on '{recon_dataset.filter_key}' in range {recon_dataset.filter_bounds}; "
            f"kept {recon_dataset.num_kept}/{recon_dataset.num_original} samples (missing_key={recon_dataset.missing_key_count})"
        )
        wandb.log(
            {
                "filter/key": recon_dataset.filter_key,
                "filter/min": recon_dataset.filter_bounds[0],
                "filter/max": recon_dataset.filter_bounds[1],
                "filter/kept": recon_dataset.num_kept,
                "filter/original": recon_dataset.num_original,
                "filter/missing_key": recon_dataset.missing_key_count,
            }
        )
    
    for epoch in range(start_epoch, num_epochs):
        # Ensure distinct shuffling across processes each epoch
        if isinstance(recon_dataloader.sampler, DistributedSampler):
            recon_dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_loss_analysis = 0.0

        # Accumulators for epoch-wide MAE metrics
        total_mae_expected_sum = 0.0  # classification (expected-count)
        total_mae_argmax_sum = 0.0    # classification (argmax)
        grad_norm_sampled = None

        for b_idx, batch in enumerate(recon_dataloader):
            images = batch["image"].to(device).float()
            vgg_error = batch["vgg_error"].to(device).float().unsqueeze(1)
            k_value = batch["k_value"].to(device).float().unsqueeze(1)  

            optimizer.zero_grad()

            # Forward:
            #  - classification: logits [B, C] where classes 1..C map to counts 1..C
            #  - regression: output [B, 1] with scaled target
            logits = token_count_predictor(images, vgg_error)

            if is_reg:
                # Scale target to [0,1]: y = (K-1)/(C-1)
                y = (k_value - 1.0) / float(num_classes - 1)

                # loss is MAE between model output and scaled target
                # logits here are actually a scalar output [B,1]
                loss = training_loss(logits, y)

                # Use the model prediction (not the target!) to compute token-count MAE
                pred_y = logits.clamp(0.0, 1.0)  # [B,1] → scaled prediction
                pred_k = (pred_y * (num_classes - 1) + 1.0).round().clamp(1, num_classes)

                # Per-sample MAE and epoch accumulation
                loss_analysis = mae_loss(pred_k, k_value)

            else:
                # Your Gaussian-CE expects float K; keep as-is
                loss = training_loss(logits, k_value)
                # For analysis, prefer expected-count MAE which aligns with soft targets
                prob = torch.softmax(logits, dim=1)
                classes = torch.arange(1, num_classes + 1, device=logits.device, dtype=prob.dtype).unsqueeze(0)
                expected_count = (prob * classes).sum(dim=1)

                # Also compute argmax-based MAE for reference
                predicted_token_count = token_count_predictor.logits_to_token_count(logits).float()
                per_mae_expected = mae_loss(expected_count, k_value)  # [B]
                per_mae_argmax = mae_loss(predicted_token_count, k_value)  # [B]

                # Use expected-count MAE as the main analysis metric (batch mean)
                loss_analysis = per_mae_expected

                total_mae_expected_sum += float(per_mae_expected.sum().item())
                total_mae_argmax_sum += float(per_mae_argmax.sum().item())
                

            # Backprop + optimization step
            loss.backward()

            # Sample gradient norm once per epoch (first batch after backward)
            if grad_norm_sampled is None:
                total_norm = 0.0
                for p in token_count_predictor.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += float(param_norm.item() ** 2)
                grad_norm_sampled = (total_norm ** 0.5) if total_norm > 0 else 0.0
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_loss_analysis += float(loss_analysis.item())


        # ---------------------------
        # Global average across processes (optional but recommended)
        # We reduce sums over ranks and divide by total number of batches.
        # ---------------------------
        loss_sum = torch.tensor([epoch_loss], device=device)
        lossA_sum = torch.tensor([epoch_loss_analysis], device=device)
        n_batches = torch.tensor([len(recon_dataloader)], device=device)
        if dist_is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(lossA_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_batches, op=dist.ReduceOp.SUM)
        avg_loss = (loss_sum / n_batches).item()
        avg_loss_analysis = (lossA_sum / n_batches).item()

        if is_main_process():
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.6f}, "
                f"Avg Loss Analysis (MAE): {avg_loss_analysis:.6f}"
            )

        # ---------------------------
        # ✅ Log to Weights & Biases
        # ---------------------------
        # Log to W&B, include epoch-averaged MAEs
        log_dict = {
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "avg_loss_analysis": avg_loss_analysis,
        }
        if grad_norm_sampled is not None:
            log_dict["grad_norm_sampled"] = float(grad_norm_sampled)
        # if total_samples > 0:
        #     if is_reg:
        #         log_dict.update({
        #             "mae_epoch": total_mae_reg_sum / float(dataset_size),
        #         })
        #     else:
        #         log_dict.update({
        #             "mae_expected_epoch": total_mae_expected_sum / float(dataset_size),
        #             "mae_argmax_epoch": total_mae_argmax_sum / float(dataset_size),
        #         })
        if is_main_process():
            wandb.log(log_dict)

        # ---------------------------
        # Save/overwrite the single checkpoint file
        #   - We store 'epoch' as the index of the *next* epoch to run.
        #     That way, if we load later, training continues at the correct loop index.
        # ---------------------------
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
                checkpoint_path,
            )
            print(f"Saved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean up process group (safe to call even if not initialized)
        if dist_is_initialized():
            dist.destroy_process_group()