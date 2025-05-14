from torch.utils.data import Dataset as TorchDataset


class HuggingFaceDataset:
    """Wrap the HuggingFace dataset (ArrowDataset) and any PyTorch transform together in one object, 
        so they can be used as a PyTorch-compatible dataset in training pipelines.
        
        E.g.

        # Load raw dataset
        raw_dataset = load_dataset("mnist")['train']

        # Define Torch transform
        my_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Wrap HuggingFace dataset
        dataset = HuggingFaceDataset(dataset=raw_dataset, transform=my_transform)
        """

    def __init__(self, dataset, subset_size=None, class_filter=None, transform=None):
        # a HuggingFace dataset (e.g., from datasets.load_dataset("mnist")['train'])
        # load_dataset returns datasets.arrow_dataset.Dataset
        # This object holds your actual data (images, labels, etc.), 
        # but it doesn't naturally support PyTorch-style transforms (e.g., ToTensor(), Normalize()).
        self.dataset = dataset

        # in case we want to filter the dataset. e.g. Imagine you’re using MNIST (digits 0–9), but you only want to get 0 vs 1:
        if class_filter is not None:
            self.dataset = dataset.filter(lambda row: row["label"] in class_filter)

        if transform is not None:
            self.transform = self.get_hf_transform(transform)
            self.dataset.set_transform(self.transform)
        else:
            self.transform = None

        if subset_size is not None:
            split = self.dataset.train_test_split(test_size=1, train_size=subset_size)
            self.dataset = split["train"]

    def __getattr__(self, attr):
        """
        "If someone calls dataset.shuffle() or dataset[0] and the HuggingFaceDataset class doesn't have that method, 
        try forwarding it to self.dataset (the actual HuggingFace dataset underneath)." This is called delegation.
        This is how I can do dataset[0] even though indexing is not defined in the HuggingFaceDataset class, it is
        defined in the underlying dataset.
        """
        return getattr(self.dataset, attr)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    # returns a callable (hf_transform)
    def get_hf_transform(transform):
        """Turn a torchvision-style transform into one for a huggingface dataset"""

        def hf_transform(datarow):
            # A datarow (from a huggingface dataset) is dict-like and sometimes has
            # different names for the image column - try a couple.
            # Looks inside a HuggingFace data row for keys like "image" or "img"
            img_keys = {"img", "image"}
            for img_key in img_keys:
                if img_key in datarow:
                    image_pils = datarow[img_key]

            # Apply preprocessing transform  specified in config to the images
            image_tensors = [transform(image.convert("RGB")) for image in image_pils]
            return {"images": image_tensors}

        return hf_transform


class HuggingFaceSubset(HuggingFaceDataset):
    def __init__(self, dataset, size, transform=None):
        super().__init__(dataset, transform=None)
        self.dataset = self.dataset.train_test_split(test_size=0, train_size=size)["train"]


class TorchHuggingFaceDatasetWrapper(TorchDataset):
    """
    Turn the huggingface dataset into a torch dataset that only returns the images
    in the huggingface dataset. This is useful to pass on to other methods that
    explicitly require a torch dataset. such as LID estimation.
    """

    def __init__(
        self,
        hugging_face_dataset: HuggingFaceDataset,
    ):
        # perform the subsampling using the train_test_split function
        self.hugging_face_dataset = hugging_face_dataset

    def __len__(self):
        return len(self.hugging_face_dataset)

    def __getitem__(self, idx):
        return self.hugging_face_dataset.dataset[idx]["images"]
 