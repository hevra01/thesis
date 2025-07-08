import itertools
import torchvision.transforms as T
from data.datasets.huggingface_dataset import HuggingFaceDataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from dahuffman import HuffmanCodec



def get_mnist_dataloader(batch_size=120, split="test", flatten=False, class_filter=None):
    # Load raw MNIST from HuggingFace
    raw_dataset = load_dataset("mnist", split=split)

    transform_list = [
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ]

    if flatten:
        transform_list.append(T.Lambda(lambda x: x.view(-1)))  # flatten to 1D vector


    # Compose transforms (match your config)
    transform = T.Compose(transform_list)
        
    # Wrap with your HuggingFaceDataset class
    dataset = HuggingFaceDataset(dataset=raw_dataset, class_filter=class_filter, transform=transform)
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def get_imagenet_dataloader(
    root="/BS/databases23/imagenet/original/",
    split="val",  # or "train"
    batch_size=64,
    image_size=256,
    class_filter=None,
    num_workers=64,
    shuffle=False
):
    # Define standard ImageNet transforms
    transform = T.Compose([
        # Rescales the shorter side of the image to image_size. Note that this could be down or upsampling, 
        # depending on whether the shorter side is smaller or larger than the img_size.
        # The longer side is then resized to maintain the aspect ratio.
        T.Resize(image_size), 
        # Crops out a square of size (image_size × image_size) from the center of the resized image.
        T.CenterCrop(image_size), 
        # Converts the image from a PIL/numpy array in range [0, 255] to a PyTorch tensor in range [0.0, 1.0]
        T.ToTensor(),
        # Applies per-channel normalization
        # Given that the normal (expected) distribution has this mean/std, 
        # re-center and rescale the image data accordingly so it behaves like zero-mean, unit-variance data.
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset from disk
    dataset = ImageFolder(root=f"{root}/{split}", transform=transform)

    # Optional class filter
    if class_filter is not None:
        class_to_idx = {v: k for k, v in dataset.class_to_idx.items()}
        allowed_idxs = [class_to_idx[c] for c in class_filter if c in class_to_idx]
        dataset.samples = [s for s in dataset.samples if s[1] in allowed_idxs]

    # Wrap in DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

class ReconstructionDataset(Dataset):
    def __init__(self, reconstruction_data, all_registers, num_pixels, dataloader):
        """
        Args:
            reconstruction_data (list): List of dicts containing reconstruction metrics.
                (img, k_value, mse_error, vgg_error).
            all_registers (list): List of lists of register IDs (per image).
            num_pixels (int): Number of pixels in each image (e.g., 256*256).
            dataloader (DataLoader): Dataloader from which images can be fetched.
        """
        self.reconstruction_data = reconstruction_data
        self.all_registers = all_registers
        self.num_pixels = num_pixels
        self.dataloader = dataloader    

        # Create the Huffman codebook from the registers
        # self.codebook will be a dictionary mapping register IDs to (bit_length, code_string)
        self.flattened_all_registers = [r[0] if isinstance(r, list) and len(r) == 1 else r for r in all_registers]
        flat_registers = list(itertools.chain.from_iterable(self.flattened_all_registers))

        self.codebook = self.create_codebook(flat_registers)

    def __len__(self):
        # this will be the number of images * the number of k values for each image.
        # e.g. for imagenet val, it is 50000 * len([1, 2, 4, 8, 16, 32, 64, 128, 180, 256]) = 500000
        return len(self.reconstruction_data)

    def __getitem__(self, idx):
        """
        Returns:
            dict: Contains image ID, k value, MSE error, vgg error (or any other error) and compression rate.
        """
        data_point = self.reconstruction_data[idx]
        image_id = data_point["image_id"]
        k_value = data_point["k_value"]
        mae_error = data_point["mae_error"]
        vgg_error = data_point["vgg_error"]

        # Compute compression rate based on Huffman codes
        # Get the first k registers
        registers = self.flattened_all_registers[image_id][:k_value] # this is bc we have nested list[[[registers]], [[registers]], etc]
        
        # get the lenths of the Huffman codes for these registers
        # self.codebook is a dictionary mapping register IDs to (bit_length, code_string)
        # so we can get the bit lengths for the registers used in this image
        huffman_code_lengths = [self.codebook[register][0] for register in registers]

        # Calculate the average bit length per pixel
        bpp = sum(huffman_code_lengths) / self.num_pixels  

        return {
            "image": self.dataloader.dataset[image_id][0],  # Get the image from the dataloader
            "k_value": k_value,
            "mae_error": mae_error,
            "vgg_error": vgg_error,
            "bpp": bpp,  # bits per pixel
        }
    
    def create_codebook(self, registers):
        """
        Create a Huffman codebook from the given registers.

        Args:
            registers (list): List of register IDs.

        Returns:
            dict: A dictionary mapping register IDs to (bit_length, code_string).
        """
        codec = HuffmanCodec.from_data(registers)

        # Access the codebook directly
        codebook = codec.get_code_table()  # Returns dict: symbol → (bit_length, code_string)

        return codebook

    def get_huffman_code(self, register):
        """
        Get the Huffman code for a given register ID.

        Args:
            register (int): The register ID.
            codebook (dict): Huffman codebook mapping register IDs to (bit_length, code_string).

        Returns:
            str: Huffman code string for the register ID.
        """
        if register in self.codebook:
            return self.codebook[register][1]  # Return the Huffman code string
        else:
            raise ValueError(f"Register ID {register} not found in the Huffman codebook.")