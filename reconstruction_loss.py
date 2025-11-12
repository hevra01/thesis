import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from typing import Optional
import lpips
from transformers import AutoModel, AutoImageProcessor


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
    

class DINOv2FeatureLoss(nn.Module):
    """
    DINOv2FeatureLoss
    ------------------
    Computes a feature-based (semantic) loss between two batches of images
    using pretrained DINOv2 embeddings.

    This loss captures *semantic* similarity — i.e., how similar the images
    are in DINOv2's learned feature space — rather than just pixel-level
    similarity.

    The model uses cosine similarity (recommended) or optionally L1/MSE.
    """

    def __init__(self,
                 dinov2_dir: str,          # local directory with the DINOv2 weights
                 use_pooler=True,          # if True, use pooler_output (CLS after projection)
                 normalize=True,           # whether to L2-normalize embeddings
                 loss_type="cosine",       # 'cosine', 'mse', or 'l1'
                 device=None):             # GPU/CPU selection
        super().__init__()

        # ---------------------------------------------------------------------
        # 1) Load pretrained model and preprocessor from local folder
        # ---------------------------------------------------------------------
        # AutoImageProcessor: handles preprocessing (resize, crop, normalize)
        # AutoModel: loads the transformer backbone itself (ViT trained with DINOv2)
        self.processor = AutoImageProcessor.from_pretrained(
            dinov2_dir, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            dinov2_dir, local_files_only=True
        )

        # Set model to evaluation mode (important: disables dropout, etc.)
        self.model.eval()

        # Freeze all parameters (no gradient updates)
        for p in self.model.parameters():
            p.requires_grad = False

        # Store settings
        self.normalize = normalize
        self.loss_type = loss_type
        self.use_pooler = use_pooler

        # Pick the compute device
        # If no device was passed, prefer GPU if available
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to chosen device
        self.model.to(self.device)

    # -------------------------------------------------------------------------
    # 2) Embedding extractor
    # -------------------------------------------------------------------------
    def _embed(self, x01):
        """
        Computes DINOv2 embeddings for a batch of images.

        Args:
            x01 : torch.Tensor [B, 3, H, W]
                RGB images in [0, 1] range.

        Returns:
            emb : torch.Tensor [B, D]
                Feature embeddings for each image (D=768 for DINOv2-Base).
        """

        # The DINOv2 processor expects a *list* of images (not a single batch tensor)
        # Each image can be a tensor, numpy array, or PIL Image.
        # 'do_rescale=False' -> we already have values in [0,1], not [0,255].
        inputs = self.processor(
            images=[t for t in x01],   # convert batch tensor -> list of images
            return_tensors="pt",       # output PyTorch tensors
            do_rescale=False           # skip scaling 0–255 → 0–1
        )

        # Move the resulting tensors to GPU/CPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Disable gradient computation (faster + no memory overhead)
        with torch.no_grad():
            # Forward pass through DINOv2 model
            # Output is a dict-like object with keys such as:
            #   'last_hidden_state': [B, num_tokens, D]
            #   'pooler_output'    : [B, D] (optional projection of CLS token)
            out = self.model(**inputs)

        # ---------------------------------------------------------------------
        # Extract embeddings
        # ---------------------------------------------------------------------
        # DINOv2 outputs two useful representations:
        #   - out.last_hidden_state[:, 0]  -> CLS token (global feature)
        #   - out.pooler_output             -> projected CLS (after linear+tanh)
        # The pooler output is typically more stable for similarity comparisons.
        emb = (
            out.pooler_output
            if self.use_pooler and getattr(out, "pooler_output", None) is not None
            else out.last_hidden_state[:, 0]
        )

        # Optional: L2-normalize embeddings so that each vector has unit length.
        # This makes comparisons focus purely on direction (semantic content)
        # instead of magnitude (feature scale).
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)

        # Shape: [B, D]
        return emb

    # -------------------------------------------------------------------------
    # 3) Forward: compute the loss between two image batches
    # -------------------------------------------------------------------------
    def forward(self, x01, y01):
        """
        Compute DINOv2 feature loss between two batches of images.

        Args:
            x01 : torch.Tensor [B, 3, H, W]
                First batch (e.g., original images) in [0,1].
            y01 : torch.Tensor [B, 3, H, W]
                Second batch (e.g., reconstructed images) in [0,1].

        Returns:
            loss : torch.ScalarTensor
                Feature-space distance between x and y.
        """

        # Extract embeddings for both sets of images
        fx = self._embed(x01)
        fy = self._embed(y01)

        # ---------------------------------------------------------------------
        # Choose loss type
        # ---------------------------------------------------------------------

        # 1️⃣ Mean Squared Error — penalizes both direction and magnitude differences.
        #    Use if you're doing *feature regression* (not just semantic similarity).
        if self.loss_type == "mse":
            return F.mse_loss(fx, fy)

        # 2️⃣ L1 (Mean Absolute Error) — less sensitive to outliers.
        #    Also includes magnitude information.
        if self.loss_type == "l1":
            return F.l1_loss(fx, fy)

        # 3️⃣ Cosine distance — focuses only on *directional similarity* of features.
        #    This is the correct choice for semantic similarity (DINO embeddings
        #    were trained on normalized features, so scale has no meaning).
        #    The output is in [0, 2] range (0 = identical, 1 = orthogonal).
        return (1 - F.cosine_similarity(fx, fy, dim=1))

    
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

def reconstruction_error(reconstructed_img, original_img, loss_fns, loss_weights=None, device='cuda'):
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
        reconstructed_img = reconstructed_img.to(device)
        original_img = original_img.to(device)
        loss_name = fn.__class__.__name__
        is_lpips = loss_name.lower().startswith("lpips")

        # Call the loss once; always pass retPerLayer for LPIPS, never for others
        if is_lpips:
            result = fn(reconstructed_img, original_img, retPerLayer=True)
            primary, extras = result[0], result[1]
            # For LPIPS: keep both [aggregate, per_layer]
            loss_dict[loss_name] = [primary, extras]
        else:
            result = fn(reconstructed_img, original_img).flatten(1).mean(dim=1)
            primary, extras = result.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), None
            loss_dict[loss_name] = primary  # keep shape consistent
            
        # Accumulate only the primary value
        total_loss += primary
    return total_loss, loss_dict



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
