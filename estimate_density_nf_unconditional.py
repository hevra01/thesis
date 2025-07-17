import json
import time
from experiments.image.model.dense_flow import DenseFlow
import hydra
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
from hydra.utils import instantiate


def resize_with_box(tensor_batch, size=(64, 64)):
    resized = []
    for img in tensor_batch:  # assumes shape [B, 3, H, W]
        pil_img = to_pil_image(img)  # converts to PIL Image
        pil_resized = pil_img.resize(size, resample=Image.BOX)
        tensor_resized = to_tensor(pil_resized)  # back to tensor
        resized.append(tensor_resized)
    return torch.stack(resized)


@hydra.main(version_base=None, config_path="conf", config_name="estimate_density")
def main(cfg):
    resize_size = tuple(cfg.experiment.resize)  # (64, 64)
    
    # instantiate the data loader
    dataloader = instantiate(cfg.experiment.dataset)

    # instantiate the density estimator model
    density_estimator_model = instantiate(cfg.experiment.model)

    # Load the pretrained checkpoint
    checkpoint = torch.load(cfg.experiment.checkpoint_path)

    density_estimator_model.load_state_dict(checkpoint['model'])

    # go over the imagenet dataset and estimate the density of the first num_images
    densities = []

    density_estimator_model.eval()  # Set the model to evaluation mode

    # get the output path for saving densities
    output_path = cfg.experiment.output_path

    with torch.no_grad():
        for batch in dataloader:
            start = time.time()
            imgs = batch[0]  # get the images without labels

            imgs_box = resize_with_box(imgs, resize_size)

            current_densities = [density.item() for density in density_estimator_model.log_prob(imgs_box)]
            densities.extend(current_densities)

            # Save the estimated densities to a file
            with open(output_path, "w") as f:
                json.dump(densities, f)
            print(f"Processed batch in {time.time() - start:.2f} seconds. Current densities: {current_densities}")  


if __name__ == "__main__":
    main()