import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import wandb
from data.utils.dataloaders import ReconstructionDataset_Heuristic
import json
from hydra.utils import instantiate


# ------------------------------
# Distributed utilities (DDP)
# ------------------------------
def dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if dist_is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist_is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


@hydra.main(version_base=None, config_path="../conf", config_name="heuristic_baseline_training")
def main(cfg: DictConfig):

    # -------------------------------------
    # Device and distributed initialization
    # -------------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist_is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else cfg.device)
    torch.backends.cudnn.benchmark = True

    # Initialize W&B and dump Hydra config (main process only)
    if is_main_process():
        wandb.init(
            name=cfg.experiment.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ---------------------------
    # Optional: log DDP / CUDA runtime info
    # ---------------------------
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
        }
        if is_main_process():
            print("[DDP] Runtime info:")
            for k, v in ddp_info.items():
                print(f"  - {k}: {v}")
            wandb.log(ddp_info, step=0)
    except Exception as e:
        if is_main_process():
            print(f"[DDP] Failed to gather/log runtime info: {e}")

    # Load JSONs
    with open(cfg.experiment.reconstruction_dataset.reconstruction_data_path, "r") as f:
        reconstruction_data = json.load(f)

    # for the heuristic baseline we need the edge ratio information as a proxy for complexity
    with open(cfg.experiment.reconstruction_dataset.edge_ratio_path, "r") as f:
        edge_ratio_information = json.load(f)

    # this is just to get the images from the dataloader to be used by ReconstructionDataset
    dataloader = instantiate(cfg.experiment.dataset)

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

    # Dataset
    recon_dataset = ReconstructionDataset_Heuristic(
        reconstruction_data=reconstruction_data,
        edge_ratio_information=edge_ratio_information,
        filter_key=filter_key,
        min_error=min_error,
        max_error=max_error,
    )

    # DataLoader with optional DistributedSampler
    batch_size = cfg.experiment.reconstruction_dataset.batch_size
    shuffle = False  # keep order deterministic; sampler will handle sharding
    num_workers = getattr(cfg.experiment.reconstruction_dataset, "num_workers", 4)
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
    if dist_is_initialized():
        token_count_predictor = torch.nn.parallel.DistributedDataParallel(
            token_count_predictor,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    optimizer = instantiate(cfg.experiment.optimizer, params=token_count_predictor.parameters())
    loss = instantiate(cfg.experiment.training.loss)

    # ----- Scale configuration for loss -----
    # We keep the model output range as configured (typically [1, 256]) but
    # compute the loss in a normalized [0,1] space for numerical stability.
    #   y01 = (y - k_min) / (k_max - k_min)
    # Read the min/max from the model to avoid hard-coding.
    model_ref = token_count_predictor.module if isinstance(token_count_predictor, torch.nn.parallel.DistributedDataParallel) else token_count_predictor
    k_min = float(getattr(model_ref, "output_min", 1.0))
    k_max = float(getattr(model_ref, "output_max", 256.0))
    k_range = max(k_max - k_min, 1.0)

    # ---------------------------
    # SIMPLE CHECKPOINT RESUME (rank 0 handles IO)
    # ---------------------------
    checkpoint_path = cfg.experiment.checkpoint_path
    start_epoch = 0

    if os.path.isfile(checkpoint_path):
        if is_main_process():
            print(f"Found checkpoint at {checkpoint_path}. Loading and resuming...")
        ckpt = torch.load(checkpoint_path, map_location=device)
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

    if is_main_process():
        print("len(recon_dataloader):", len(recon_dataloader))
        dataset_size = len(recon_dataloader.dataset)
        print("dataset_size:", dataset_size)
    else:
        dataset_size = len(recon_dataloader.dataset)

    # ---------------------------
    # Training loop (resume-aware)
    # ---------------------------
    num_epochs = cfg.experiment.training.num_epochs
    for epoch in range(start_epoch, num_epochs):
        if isinstance(recon_dataloader.sampler, DistributedSampler):
            recon_dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        token_count_predictor.train()

        for batch in recon_dataloader:
            vgg_error = batch["vgg_error"].to(device).float().unsqueeze(1)
            true_token_count = batch["k_value"].to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            edge_ratio = batch["edge_ratio"].to(device).float().unsqueeze(1)

            # Forward: model returns token-count prediction in [k_min, k_max]
            token_count_prediction = token_count_predictor(edge_ratio, vgg_error)

            # ----- Compute loss on scaled [0,1] values -----
            # Why: reduces dynamic range, stabilizes gradients and learning rates,
            # and makes the optimization less sensitive to absolute scale.
            # Scale both prediction and target to [0,1]:
            #   y01 = (y - k_min) / (k_max - k_min)
            pred_token_scaled = ((token_count_prediction - k_min) / k_range).clamp(0.0, 1.0)
            true_token_scaled = ((true_token_count - k_min) / k_range).clamp(0.0, 1.0)

            # Loss (e.g., MAE) on scaled values
            current_loss = loss(pred_token_scaled, true_token_scaled).mean()

            current_loss.backward()
            optimizer.step()

            epoch_loss += float(current_loss.item())

        # Global mean-of-batches across ranks
        loss_sum = torch.tensor(epoch_loss, device=device, dtype=torch.float32)
        n_batches = torch.tensor(len(recon_dataloader), device=device, dtype=torch.float32)
        if dist_is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_batches, op=dist.ReduceOp.SUM)
        avg_loss = (loss_sum / n_batches).item()

        if is_main_process():
            print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.6f}")

            # ✅ Log to Weights & Biases
            wandb.log({
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
            })

            # Save/overwrite the single checkpoint file
            state_dict = (
                token_count_predictor.module.state_dict()
                if isinstance(token_count_predictor, torch.nn.parallel.DistributedDataParallel)
                else token_count_predictor.state_dict()
            )
            torch.save(
                {
                    "epoch": epoch + 1,
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
        if dist_is_initialized():
            dist.destroy_process_group()