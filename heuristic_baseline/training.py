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

    # Feature toggles and optional JSON loads
    rd_cfg = cfg.experiment.reconstruction_dataset
    use_edge = bool(getattr(rd_cfg, "use_edge_ratio", False))
    use_lid = bool(getattr(rd_cfg, "use_lid", False))
    use_ld = bool(getattr(rd_cfg, "use_local_density", False))

    edge_ratio_information = None
    lid_information = None
    local_density_information = None

    if use_edge:
        with open(rd_cfg.edge_ratio_path, "r") as f:
            edge_ratio_information = json.load(f)
    if use_lid:
        with open(rd_cfg.lid_path, "r") as f:
            lid_information = json.load(f)
    if use_ld:
        with open(rd_cfg.local_density_path, "r") as f:
            local_density_information = json.load(f)

    # this is just to get the images from the dataloader to be used by ReconstructionDataset
    dataloader = instantiate(cfg.experiment.dataset)

    # Dataset
    recon_loss_key = rd_cfg.reconstruction_loss
    recon_dataset = ReconstructionDataset_Heuristic(
        reconstruction_data=reconstruction_data,
        edge_ratio_information=edge_ratio_information if use_edge else None,
        lid_information=lid_information if use_lid else None,
        local_density_information=local_density_information if use_ld else None,
        filter_key=getattr(rd_cfg, "filter", {}).get("key", None) if hasattr(rd_cfg, "filter") else None,
        min_error=getattr(rd_cfg, "filter", {}).get("min", None) if hasattr(rd_cfg, "filter") else None,
        max_error=getattr(rd_cfg, "filter", {}).get("max", None) if hasattr(rd_cfg, "filter") else None,
        error_key=recon_loss_key,
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
    training_loss = instantiate(cfg.experiment.training.loss_training)
    analysis_loss = instantiate(cfg.experiment.training.loss_analysis)

    # Determine expected in_dim from toggles (1 for recon loss + enabled features)
    expected_in_dim = 1 + int(use_edge) + int(use_lid) + int(use_ld)
    model_ref = token_count_predictor.module if isinstance(token_count_predictor, torch.nn.parallel.DistributedDataParallel) else token_count_predictor
    if hasattr(cfg.experiment, "model") and hasattr(cfg.experiment.model, "in_dim"):
        cfg_in_dim = int(cfg.experiment.model.in_dim)
        if cfg_in_dim != expected_in_dim:
            if is_main_process():
                print(f"Warning: model.in_dim ({cfg_in_dim}) != expected_in_dim from toggles ({expected_in_dim}). Proceeding anyway.")

    # ----- Scale configuration for loss -----
    # We keep the model output range as configured (typically [1, 256]) but
    # compute the loss in a normalized [0,1] space for numerical stability.
    #   y01 = (y - k_min) / (k_max - k_min)
    # Read the min/max from the model to avoid hard-coding.
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
        print("model", token_count_predictor)
        # Count only trainable parameters
        num_params = sum(p.numel() for p in token_count_predictor.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params:,}")
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
        epoch_analysis_loss = 0.0
        token_count_predictor.train()

        for batch in recon_dataloader:
            # Build features [B, in_dim] from selected signals
            feats = [batch[recon_loss_key].to(device).float().unsqueeze(1)]
            if use_edge:
                feats.append(batch["edge_ratio"].to(device).float().unsqueeze(1))
            if use_lid:
                feats.append(batch["lid"].to(device).float().unsqueeze(1))
            if use_ld:
                feats.append(batch["local_density"].to(device).float().unsqueeze(1))
            x = torch.cat(feats, dim=1)

            true_token_count = batch["k_value"].to(device).float().view(-1)

            optimizer.zero_grad()

            # Forward: classifier logits over [1..C]
            logits = token_count_predictor(x)

            # Training loss (Gaussian CE): counts expected [B] or [B,1]
            current_loss = training_loss(logits, true_token_count)

            # Analysis metric: MAE on counts via argmax
            pred_class = logits.argmax(dim=1) + 1
            loss_analysis = analysis_loss(pred_class.float(), true_token_count.float())

            current_loss.backward()
            optimizer.step()

            epoch_loss += float(current_loss.item())
            epoch_analysis_loss += float(loss_analysis.item())

        # Global mean-of-batches across ranks
        loss_sum = torch.tensor(epoch_loss, device=device, dtype=torch.float32)
        n_batches = torch.tensor(len(recon_dataloader), device=device, dtype=torch.float32)
        epoch_analysis_loss = torch.tensor(epoch_analysis_loss, device=device, dtype=torch.float32)
        if dist_is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_batches, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_analysis_loss, op=dist.ReduceOp.SUM)
        avg_loss = (loss_sum / n_batches).item()
        avg_analysis_loss = (epoch_analysis_loss / n_batches).item()

        if is_main_process():
            print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.6f}, Avg Analysis Loss: {avg_analysis_loss:.6f}")

            # ✅ Log to Weights & Biases
            wandb.log({
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "avg_loss_analysis": avg_analysis_loss,
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
                    "analysis_loss": avg_analysis_loss,
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