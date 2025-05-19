import json
from pathlib import Path
import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from training.score_matching_trainer import ScoreMatchingTrainer
import wandb

@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig):

    # Initialize wandb if enabled
    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True)  # Logs full config
        )
    
    # Set device
    device = torch.device(cfg.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your CUDA installation.")
    
    print(f"Using device: {device}")

    # Save final config as JSON
    with open("final_config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    # Configure torch dataloader
    train_loader = instantiate(cfg.train.loader)

    # Configure model
    model = instantiate(cfg.experiment.model).to(device)  # instantiate the model

    # Configure the SDE
    sde = instantiate(cfg.experiment.sde).to(device)

    # Configure optimizer and loss function 
    optimizer = instantiate(cfg.experiment.optimizer, params=model.parameters())
    criterion = instantiate(cfg.experiment.loss)


    # Set the trainer
    trainer = ScoreMatchingTrainer(
        sde=sde,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )

    # Directory to save/load checkpoints
    ckpt_dir = Path(cfg.experiment.checkpoint.dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Path to checkpoint file. This will help if we want to resume training from a checkpoint.
    ckpt_path = ckpt_dir / "latest.pt"

    # start_epoch is used to keep track of the current epoch.
    # If we are resuming training from a checkpoint, we will set this to the epoch of the checkpoint.
    # If we are not resuming training, we will set this to 0.
    start_epoch = 0

    # Try loading existing checkpoint
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Load model weights
        model.load_state_dict(checkpoint["model"])

        # Load epoch
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resumed training from checkpoint at epoch {start_epoch}")



    # start training
    for epoch in range(start_epoch, cfg.experiment.max_epochs):
        loss = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch}: train_loss = {loss:.4f}")

         # Save checkpoint
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": getattr(trainer, "optimizer", None).state_dict() if hasattr(trainer, "optimizer") else None,
                "epoch": epoch,
            },
            ckpt_path
        )

if __name__ == "__main__":

    main()
