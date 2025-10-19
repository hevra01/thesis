import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from typing import Optional
import lpips


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
            loss += torch.abs(feats_x[k] - feats_y[k]).mean(dim=[1, 2, 3])  # → shape: [B]

            #loss += nn.functional.l1_loss(feats_x[k], feats_y[k])
        return loss
    
    
class MSELoss:
    def __call__(self, input, target):
        return torch.nn.functional.mse_loss(input, target)



class MAELoss:
    def __call__(self, pred, target):
        """
        Mean Absolute Error between pred and target.

        - Supports scalars (no batch) or batched tensors ([B, ...]).
        - Returns a scalar: average error across batch and all dimensions.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
            )

        # absolute error per sample
        per_sample = torch.abs(pred - target)

        # average over all non-batch dimensions
        if per_sample.ndim > 1:
            per_sample = per_sample.view(per_sample.size(0), -1).mean(dim=1)

        # now reduce across batch (or just return scalar if no batch)
        return per_sample.mean()

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

    # Compute each loss and accumulate. Some loss functions may return auxiliary
    # information (e.g., LPIPS with retPerLayer=True). In that case, we:
    # - Use only the first returned value for aggregation into total_loss
    # - Store both values in loss_dict for LPIPS specifically as [scalar, per_layer_list]
    #   to match the requested format.
    for fn, w in zip(loss_fns, loss_weights):
        loss_name = fn.__class__.__name__
        is_lpips = loss_name.lower().startswith("lpips")

        # Call the loss once; always pass retPerLayer for LPIPS, never for others
        if is_lpips:
            result = fn(reconstructed_img, original_img, retPerLayer=True)
            primary, extras = result[0], result[1]
            # For LPIPS: keep both [aggregate, per_layer]
            loss_dict[loss_name] = [primary, extras]
        else:
            result = fn(reconstructed_img, original_img)
            primary, extras = result, None
            loss_dict[loss_name] = primary
            
        # Accumulate only the primary value
        total_loss += w * primary
    return total_loss, loss_dict


def reconstructionLoss_vs_compressionRate(model, images, k_keep_list, loss_fns, device, loss_weights=None):
    """
    Computes reconstruction losses for different compression rates.
    
    Args:
        model: FlexTok model instance.
        images: Tensor of shape (B, 3, H, W) with input images.
        k_keep_list: List of integers representing the number of tokens to keep.
        loss_fns: List of loss functions to compute.
        loss_weights: List of weights for each loss function.

    Returns:
        dict: Dictionary with compression rates as keys and total losses as values.
    """
    results = {}

    # First tokenize the images into register tokens, which already handles the VAE mapping to latents.
    tokens_list = model.tokenize(images.to(device))

    # ImageNet normalization constants
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # Unnormalize the original input images and scale to [0, 1]
    images_unnorm = images * imagenet_std + imagenet_mean  # [0, 1]
    

    # we want to see how the reconstruction error changes with different compression rates.
    # lower k_keep means (fewer registers) more compression, higher k_keep means less compression.
    for k_keep in k_keep_list:
        # the first ":" means for all images in the batch, the second refers to the register tokens.
        # we keep only the first k_keep tokens.
        tokens_list_filtered = [t[:,:k_keep] for t in tokens_list]
        
        reconst = model.detokenize(
            tokens_list_filtered,
            timesteps=20,
            guidance_scale=7.5,
            perform_norm_guidance=True,
        )

    
        # Convert reconstructed image from [-1, 1] to [0, 1]
        reconst_scaled = (reconst.clamp(-1, 1) + 1) / 2  # --> [0, 1]

        total_loss, loss_dict = reconstruction_error(reconst_scaled, images_unnorm, 
            loss_fns=loss_fns,
            loss_weights=loss_weights
        )
        print(f"Compression rate k_keep={k_keep}: Total Loss = {loss_dict.values()}")
        
        results[k_keep] = [loss_dict, reconst]
    
    return results


class GaussianCrossEntropyLoss(nn.Module):
    """
    Distance-aware classification loss for ordinal targets in {1..C}.
    Builds a Gaussian target distribution centered at the true count so
    near misses are penalized far less than distant ones.

    1) Target class (e.g. 50) => Ground-truth label is just a single integer.
    2) Compute squared distance => Purpose: quantify how far every possible class is from the true one. Small = close, big = far.
    3) Apply Gaussian kernel = Purpose: convert “distance” into a score that decays smoothly with distance.
        Negative sign → close classes get larger scores, far classes get smaller.
        sigma controls how wide the “tolerance zone” is.
    4) Softmax over logits => Purpose: turn scores into a valid probability distribution (sums to 1). 
       Classes near the true one get higher probability; far ones get near-zero.

    Args:
        num_classes (int): number of classes (e.g., 256)
        sigma (float): stddev of the Gaussian over class index space
    """
    def __init__(self, num_classes: int = 256, sigma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.sigma = float(sigma)


    @torch.no_grad()
    def _soft_targets(self, token_counts: torch.Tensor) -> torch.Tensor:
        """
        token_counts: [B] integer labels in [1..C], where B is the batch size.
                      so token_count basically holds the true number of tokens 
                      used for reconstruction.

        The aim is to turn these integer labels into soft targets.               
        
        
        Broadcasting explanation:
            - centers: [B,1] (true counts for each example)
            - classes: [1,C] (all possible counts)
            - classes - centers triggers broadcasting:
                PyTorch automatically expands centers to [B,C] 
                and classes to [B,C], then performs element-wise subtraction.
        """

        # Step 1: Convert labels to column vector [B,1] and float
        centers = token_counts.view(-1, 1).float()  

        # Step 2: Row vector of class indices [1,C]
        # this is just all the possible classes/token counts
        classes = torch.arange(1, self.num_classes + 1, device=token_counts.device).view(1, -1)  

        # Step 3: Squared distance from true count (broadcasting happens here)
        dist2 = (classes - centers).pow(2)  # shape [B,C] due to broadcasting

        # Step 4: Convert distances to Gaussian logits
        # Formula: exp(- (x - mu)^2 / (2*sigma^2)) in log-space
        # logits are unnormalized scores: higher for classes closer to the true count
        logits = -dist2 / (2.0 * self.sigma ** 2)

        # Step 5: Convert logits to probabilities using softmax
        # Softmax: P_i = exp(logit_i) / sum_j exp(logit_j)
        # Ensures probabilities sum to 1 per row
        # Closer classes have higher probabilities, distant classes have near-zero
        targets = torch.softmax(logits, dim=1)

        return targets


    def forward(self, logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """
        Compute distance-aware cross-entropy loss using Gaussian soft targets.

        Args:
            logits: [B, C] raw class logits from the model, where B is batch size and C is num_classes
            counts: [B] integer labels (token counts) in [1..C] 
        
        Returns:
            scalar loss
        """

        # Step 1: Convert integer labels into Gaussian soft targets
        # Each row is a probability distribution over classes
        soft_t = self._soft_targets(counts)  # [B, C]

        # Step 2: Convert model logits to log-probabilities
        # log_softmax = log(softmax(logits))
        # Using log directly is numerically more stable than log(softmax(logits))
        log_p = F.log_softmax(logits, dim=1)  # [B, C]

        # Step 3: Cross-entropy: -sum(target * log(predicted)) per example
        loss = -(soft_t * log_p).sum(dim=1)  # [B]

        # Step 4: Average over batch
        return loss.mean()
