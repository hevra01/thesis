import hydra
import lightning as L
import torch
import torchvision.transforms.functional 
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from utils.train_utils import train_one_epoch



@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig):
    print("hev")
    # Set device
    device = torch.device(cfg.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your CUDA installation.")
    
    print(f"Using device: {device}")

    # Save final config as JSON
    with open("final_config.json", "w") as f:
        f.write(OmegaConf.to_json(cfg))

    # Configure dataset
    train_loader = instantiate(cfg.train.loader)
    print(train_loader)

    # Configure model
    model = instantiate(cfg.train.model).to(device)  # instantiate the model
    print(model)

    # Configure optimizer and loss function
    optimizer = instantiate(cfg.train.optimizer, params=model.parameters())
    criterion = instantiate(cfg.train.loss)

    # start training
    for epoch in range(cfg.train.max_epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}: train_loss = {avg_loss:.4f}")

if __name__ == "__main__":

    main()
