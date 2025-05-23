import torch
import torchvision
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from diffusion.diffusion import Diffusion
import hydra
from omegaconf import OmegaConf

def load_model_and_diffusion(cfg):
    # Instantiate model and SDE 
    device = torch.device(cfg.device)
    model = instantiate(cfg.experiment.model, data_dim=cfg.experiment.data_dim).to(device)
    sde = instantiate(cfg.experiment.sde, score_net=model).to(device)

    # since we are loading the model only to perform sampling, we don't need the optimizer and loss function.
    diffusion = Diffusion(sde=sde, optimizer=None, criterion=None, device=device)

    # Load checkpoint
    checkpoint = torch.load(cfg.experiment.checkpoint_path, map_location=device)
    sde.score_net.load_state_dict(checkpoint["model"])
    model.eval()
    return diffusion

def generate_samples(diffusion, num_samples, sample_shape, timesteps=1000, batch_size=128):
    samples = diffusion.sample(
        num=num_samples,
        sample_shape=sample_shape,
        timesteps=timesteps,
        batch_size=batch_size,
    )
    return samples

@hydra.main(version_base=None, config_path="conf", config_name="sample")
def main(cfg: DictConfig):

    # Load model and diffusion process
    diffusion = load_model_and_diffusion(cfg)

    # Instantiate the sampling transform from config
    sampling_transform = instantiate(cfg.experiment.sampling_transforms)
    
    # image output directory
    output_dir = cfg.out_dir + "/" + cfg.experiment.sample_name + "/samples"

    # Generate samples
    samples = diffusion.sample(
        num=cfg.experiment.num_samples,
        sample_shape=cfg.experiment.sample_shape,
        timesteps=cfg.experiment.timesteps,
        batch_size=cfg.experiment.batch_size,
        sampling_transform=sampling_transform,
    )


    for i, sample in enumerate(samples):
        torchvision.utils.save_image(sample, f"{output_dir}/sample_{i}.png")

if __name__ == "__main__":
    main()