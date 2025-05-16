import json
import hydra
import lightning as L
import torch
import torchvision.transforms.functional 
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from training.score_matching_trainer import ScoreMatchingTrainer

@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig):
    
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
    print(model)

    # Configure the SDE
    sde = instantiate(cfg.experiment.sde).to(device)

    # Configure optimizer and loss function
    optimizer = instantiate(cfg.experiment.optimizer, params=model.parameters())
    criterion = instantiate(cfg.experiment.loss)


    trainer = ScoreMatchingTrainer(
        sde=sde,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )

    # start training
    for epoch in range(cfg.experiment.max_epochs):
        loss = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch}: train_loss = {loss:.4f}")

if __name__ == "__main__":

    main()
