import json
import time
from experiments.image.model.dense_flow import DenseFlow
import hydra
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
from hydra.utils import instantiate
from flextok.utils.misc import detect_bf16_support, get_bf16_context



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
    device = cfg.device
    
    # instantiate the data loader
    dataloader = instantiate(cfg.experiment.dataset)

    # instantiate the density estimator model
    flextok = instantiate(cfg.experiment.model)
    flextok.to(device)


    enable_bf16 = detect_bf16_support()
    print('BF16 enabled:', enable_bf16)

    # go over the imagenet dataset and estimate the density of the first num_images
    densities = []


    # get the output path for saving densities
    output_path = cfg.experiment.output_path

    register_path = cfg.experiment.register_path

    # Load the register token IDs
    with open(register_path, "r") as f:
        all_token_ids = json.load(f)

    batch_size = dataloader.batch_size

    # 👈 choose how many registers to keep (user-defined)
    keep_k = cfg.experiment.keep_k  

    flextok.pipeline.count_decoder_params()  # Print decoder parameters 

    hutchinson_samples = cfg.experiment.hutchinson_samples 

    
    for batch_idx, batch in enumerate(dataloader):

        # Convert JSON lists to torch.Tensors and batch them
        token_ids_list = [
            torch.tensor(register_ids[0][:keep_k]).unsqueeze(0).to(device)
            for register_ids in all_token_ids[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        ]
        start = time.time()
        imgs = batch[0].to(device)  # get the images without labels
        with get_bf16_context(enable_bf16):
            yes = flextok.estimate_log_density(imgs, token_ids_list=token_ids_list, hutchinson_samples=hutchinson_samples)
        current_densities = [density.item() for density in yes]
        densities.extend(current_densities)

        # Save the estimated densities to a file
        with open(output_path, "w") as f:
            json.dump(densities, f)
        print(f"Processed batch in {time.time() - start:.2f} seconds. Current densities: {current_densities}")  


if __name__ == "__main__":
    main()