
import json
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
import wandb
from hydra.utils import instantiate
from torch.utils.data.distributed import DistributedSampler
from data.utils.dataloaders import ReconstructionDataset_Neural
from torch.utils.data import DataLoader

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


@hydra.main(version_base=None, config_path="../conf", config_name="experiment/token_estimator_given_loss")
def main(cfg: DictConfig):

    
    # ---------------- Device & DDP init ---------------- #
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        # Map this process to "its" GPU (relative to CUDA_VISIBLE_DEVICES)
        torch.cuda.set_device(local_rank)

    # Initialize DDP if launched via torchrun
    if ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    # Pick device (per-process GPU under DDP, else cfg.device e.g. "cpu")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else cfg.experiment.device)

    # Let cuDNN benchmark conv algorithms (great if shapes are stable)
    torch.backends.cudnn.benchmark = True

    # Initialize W&B and dump Hydra config
    if is_main_process():
        run = wandb.init(
            name=cfg.experiment.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.run.summary["run_id"] = run.id

    # ---------------------------
    # Log Distributed / GPU runtime info (once on rank 0)
    # ---------------------------
    try:
        num_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_visible)] if num_visible > 0 else []
        ddp_info = {
            "ddp/enabled": dist_is_initialized(), # Checks whether PyTorch Distributed (DDP) has been initialized.
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
        if is_main_process():
            print("[DDP] Runtime info:")
            for k, v in ddp_info.items():
                print(f"  - {k}: {v}")
            # Log once at step 0
            wandb.config.update(ddp_info)
    except Exception as e:
        if is_main_process():
            print(f"[DDP] Failed to gather/log runtime info: {e}")

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
    # Choose which reconstruction loss field to learn from; default to 'vgg_error' if not set
    recon_loss_key = cfg.experiment.reconstruction_dataset.reconstruction_loss
    recon_dataset = ReconstructionDataset_Neural(
        reconstruction_data=reconstruction_data,
        base_dataset=dataloader,
        error_key=recon_loss_key
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
    print("model:\n", token_count_predictor)
    # Count only trainable parameters
    num_params = sum(p.numel() for p in token_count_predictor.parameters() if p.requires_grad)
    if is_main_process():
        print(f"Number of trainable parameters: {num_params:,}")
        print("len(recon_dataloader):", len(recon_dataloader))
        print("len(recon_dataloader.dataset):", len(recon_dataloader.dataset))
        print("model:\n", token_count_predictor)

    optimizer = instantiate(cfg.experiment.optimizer, params=token_count_predictor.parameters())
    training_loss = instantiate(cfg.experiment.training.loss_training)
    analysis_loss = instantiate(cfg.experiment.training.loss_analysis)

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

    for epoch in range(start_epoch, num_epochs):
        # Ensure distinct shuffling across processes each epoch
        if isinstance(recon_dataloader.sampler, DistributedSampler):
            recon_dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_loss_analysis = 0.0
        grad_norm_sampled = None

        for b_idx, batch in enumerate(recon_dataloader):
            # Model input: scalar reconstruction error per sample
            reconstruction_loss_value = batch[recon_loss_key].to(device).float().unsqueeze(1)
            # Training labels for GaussianCrossEntropyLoss: integer class indices in [1..C]
            k_value = batch["k_value"].to(device).long().view(-1)

            optimizer.zero_grad()

            # Forward:
            #  - classification: logits [B, C] where classes 1..C map to counts 1..C
            #  - regression: output [B, 1] with scaled target
            logits = token_count_predictor(reconstruction_loss_value)

            # Cross-entropy-like loss (Gaussian soft targets) for training
            loss = training_loss(logits, k_value)

            # Analysis metric: absolute error in token counts (MAE)
            # Predict class via argmax over logits. Our class space is 1..C, so add 1.
            pred_class = logits.argmax(dim=1) + 1  # [B]
            loss_analysis = analysis_loss(pred_class.float(), k_value.float())
                
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
        # Global average across processes (DDP metric reduction)
        # Why this is here:
        # - In DDP, each process ("rank") sees only a shard of data. A local
        #   average (epoch_loss / len(local_loader)) reflects only that shard.
        # - To report a true job-wide metric, we sum numerators and denominators
        #   across all ranks with all_reduce(SUM), then divide.
        # Terms:
        # - rank: process id in {0 .. world_size-1}; rank 0 is the main process.
        # - world_size: total number of processes (typically = number of GPUs).
        # Mechanics:
        # - dist.all_reduce(t, SUM) replaces each rank's tensor t with the sum
        #   over all ranks (in-place, synchronized). After that call, all ranks
        #   hold the same global sum.
        # Semantics we keep here:
        # - "mean-of-batches": (sum of per-batch losses) / (total number of batches
        #   across all ranks). This matches your original single-GPU definition.
        # Notes:
        # - Training correctness does not depend on this block; DDP already
        #   synchronizes gradients during backward. This is for accurate logging.
        # - If you prefer a sample-weighted mean, accumulate inside the loop:
        #     sum_loss += loss.item() * batch_size; sum_count += batch_size
        #   then all_reduce those two scalars and divide.
        # - On single GPU (no dist), this block just uses the local values.
        # ---------------------------
        loss_sum = torch.tensor(epoch_loss, device=device, dtype=torch.float32)
        analysis_sum = torch.tensor(epoch_loss_analysis, device=device, dtype=torch.float32)
        n_batches = torch.tensor(len(recon_dataloader), device=device, dtype=torch.float32)
        if dist_is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(analysis_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_batches, op=dist.ReduceOp.SUM)
        avg_loss = (loss_sum / n_batches).item()
        avg_loss_analysis = (analysis_sum / n_batches).item()

        if is_main_process():
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.6f}, Avg Loss Analysis: {avg_loss_analysis:.6f}")
        

        # ---------------------------
        # ✅ Log to Weights & Biases
        # ---------------------------
        # Log to W&B, include epoch-averaged MAEs
        log_dict = {
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "avg_loss_analysis": avg_loss_analysis,  # average absolute error in token counts
        }
        if grad_norm_sampled is not None:
            log_dict["grad_norm_sampled"] = float(grad_norm_sampled)
        
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



