import torch
from flextok.flextok_wrapper import FlexTokFromHub
from data.utils.dataloaders import get_imagenet_dataloader
import os
import json

# Initialize the FlexTok model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FlexTokFromHub.from_pretrained('EPFL-VILAB/flextok_d18_d28_in1k').eval().to(device)

# Prepare the dataloader
imgnet_dataloader = get_imagenet_dataloader(batch_size=8)

# List to store all registers
all_registers = []

# Process each batch in the dataloader
for batch_idx, (images, _) in enumerate(imgnet_dataloader):
    images = images.to(device)

    # Tokenize the images to get the registers
    registers_list = model.tokenize(images)

    # Extend the list with registers for each image
    all_registers.extend([registers.tolist() for registers in registers_list])
    print(f"Processed batch {batch_idx + 1}/{len(imgnet_dataloader)}")

# Save all registers to a single JSON file
output_path = "all_registers_imagenet.json"
with open(output_path, "w") as f:
    json.dump(all_registers, f)

print("Encoding complete. Registers saved in:", output_path)