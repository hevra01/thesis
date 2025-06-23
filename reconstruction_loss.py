import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF

class VGGPerceptualLoss(nn.Module):
    """
    Computes perceptual loss using VGG19 features.
    This loss is based on the L1 distance between feature maps
    extracted from specific layers of the VGG19 model.
    Different layers capture features at different scale.
    """
    def __init__(self, hook_layers=('9', '18', '27')):
        """
        The hook layers correspond to activations from:
        - '9': MaxPool2d (after relu2_2)
        - '18': MaxPool2d (after relu3_4)
        - '27': MaxPool2d (after relu4_4)
        """
        super().__init__()

        # Load pretrained VGG19 (features only), freeze it and set to eval mode
        vgg = models.vgg19(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False  # No gradients for VGG during loss computation

        # ImageNet normalization parameters
        # These are the mean and std for RGB channels used in VGG preprocessing
        # The first None: adds the batch dimension. The second and third None: add spatial dimensions.
        self.mean = torch.Tensor([0.485, 0.456, 0.406])[None,:,None,None]
        self.std = torch.Tensor([0.229, 0.224, 0.225])[None,:,None,None]

        self.layers = {}           # dict to store hooked feature maps
        self.hooks = []            # store hook handles to remove later

        # Register forward hooks on specific VGG layers to extract activations
        for idx, module in vgg._modules.items():
            if idx in ['9','18','27']:
                self.hooks.append(module.register_forward_hook(self._capture(idx)))
        self.vgg = vgg

    def _capture(self, idx):
        """
        Returns a forward-hook function that saves the module output
        into self.layers[idx] during the forward pass.
        """
        def hook(module, inp, out):
            self.layers[idx] = out
        return hook

    def forward(self, x, y, preprocess=False, resize=True):
        # if the input is not having the VGG expected input size, change it.
        if resize:
            # 1️⃣ Resize & center-crop to 224×224 (maintain aspect ratio)
            x = TF.resize(x, size=224)            # shorter side → 224 px
            x = TF.center_crop(x, output_size=(224, 224))

            y = TF.resize(y, size=224)
            y = TF.center_crop(y, output_size=(224, 224))

        # Normalize inputs to ImageNet RGB statistics if not already normalized.
        if preprocess:
            x = (x - self.mean.to(x)) / self.std.to(x)
            y = (y - self.mean.to(y)) / self.std.to(y)

        # resets stored features before each forward pass, keeping activations relevant per input.
        self.layers.clear()
        # Run forward pass on both images and capture activations
        _ = self.vgg(x)
        feats_x = {k: v for k,v in self.layers.items()}

        self.layers.clear()
        _ = self.vgg(y)
        feats_y = {k: v for k,v in self.layers.items()}

        # Compute L1 distance between corresponding feature maps
        loss = 0.0
        for k in feats_x:
            loss += nn.functional.l1_loss(feats_x[k], feats_y[k])
        return loss


class MAELoss:
    def __call__(self, reconstructed_img, original_img):
        """
        Compute Mean Absolute Error (MAE) between two tensors.

        Args:
            reconstructed_img: torch.Tensor with any shape (e.g., [B, C, H, W])
            original_img: torch.Tensor of the same shape

        Returns:
            torch.Tensor: scalar loss (averaged over all elements)
        """
        # Compute absolute differences
        diff = (reconstructed_img - original_img).abs()
        # Return mean loss
        return diff.mean()
    
def reconstruction_error(reconstructed_img, original_img, loss_fns, loss_weights=None):
    """
    Computes a weighted sum of reconstruction losses between pred and target.
    
    Args:
        reconstructed_img (torch.Tensor): The reconstructed image.
        original_img (torch.Tensor): The ground truth.
        loss_fns (list of callables): List of loss functions (e.g., [nn.MSELoss(), VGGLoss(), ...]).
        loss_weights (list of floats, optional): Weights for each loss function. If None, all weights are 1.
    
    Returns:
        torch.Tensor: The total (weighted) reconstruction error.
        dict: A dictionary of individual loss values for logging/debugging.
    """
    # If no weights provided, use 1 for each loss
    if loss_weights is None:
        loss_weights = [1.0] * len(loss_fns)

    total_loss = 0.0
    loss_dict = {}

    # Compute each loss and accumulate
    # zip pairs together elements from multiple lists
    for fn, w in zip(loss_fns, loss_weights):
        loss_name = fn.__class__.__name__
        # fn(...) calls fn.__call__(...), which calls fn.forward(...)
        loss_val = fn(reconstructed_img, original_img)
        total_loss += w * loss_val
        loss_dict[loss_name] = loss_val.item() if hasattr(loss_val, 'item') else float(loss_val)

    return total_loss, loss_dict