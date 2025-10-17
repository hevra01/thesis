from typing import Iterable, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_act(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "gelu":
        return nn.GELU
    if name == "silu":
        return nn.SiLU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"Unsupported activation: {name}")


class HeuristicTokenCountPredictor(nn.Module):
    """
    Generic MLP with optional preprocessing for token-count prediction tasks.

    - in_dim: number of input features (e.g., 1 for scalar loss; 2 for [edge_ratio, vgg_error]).
    - mode: 'regression' or 'classification'.
      * regression: outputs a single value; optional output activation + [min,max] mapping.
      * classification: outputs logits over num_classes.
    - preprocessing: log1p and learnable affine per feature.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Iterable[int] = (64, 64),
        activation: str = "silu",
        dropout: float = 0.0,
        mode: str = "regression",  # or 'classification'
        num_classes: int = 256,
        # regression head controls
        output_activation: str = "sigmoid",  # linear|sigmoid|relu|softplus
        output_min: Optional[float] = None,
        output_max: Optional[float] = None,
        # preprocessing
    # You can pass either a single bool or a list[bool] length==in_dim.
    # If a list is provided, the transform is applied per feature.
    use_log1p: bool | Iterable[bool] = True,
        # If True, applies a signed log transform per feature: sign(x) * log1p(|x|).
        # This is robust to heavy-tailed distributions and preserves information for
        # negative-valued features like LID (where plain log1p with clamp would lose sign).
    use_signed_log1p: bool | Iterable[bool] = False,
        # After (optional) log transform, optionally standardize per feature:
        # x := (x - mean) / (std + eps). Provide means/stds via standardize_mean/std
        # (length == in_dim). If not provided, set standardize=False or set later via
        # set_standardization_stats(...).
        standardize: bool = False,
        standardize_mean: Optional[Iterable[float]] = None,
        standardize_std: Optional[Iterable[float]] = None,
        affine_preact: bool = True,
    ) -> None:
        super().__init__()
        # if in_dim is 1, it means that we are mapping the recon loss to the token count directly,
        # however, if it is 2, it means that we are using some heuristic features (e.g. edge ratio, LID value, local density, etc)
        assert in_dim >= 1, "in_dim must be >= 1"
        self.in_dim = int(in_dim)
        # Input transforms: order is (log transform) -> (standardize) -> (learnable affine)
        # Normalize transform specs to boolean masks of length F (in_dim)
        def _to_bool_mask(spec, F, name: str) -> torch.Tensor:
            """Convert a bool or iterable of bools to a torch.bool mask [F]."""
            if isinstance(spec, bool):
                return torch.tensor([spec] * F, dtype=torch.bool)
            # Accept list/tuple/iterable; cast to list of bools
            try:
                vals = [bool(v) for v in list(spec)]
            except Exception as e:
                raise ValueError(f"{name} must be a bool or an iterable of bools (len==in_dim). Error: {e}")
            if len(vals) != F:
                raise ValueError(f"{name} length {len(vals)} != in_dim {F}")
            return torch.tensor(vals, dtype=torch.bool)

        self.use_log1p_mask = _to_bool_mask(use_log1p, self.in_dim, "use_log1p")
        self.use_signed_log1p_mask = _to_bool_mask(use_signed_log1p, self.in_dim, "use_signed_log1p")
        # Signed log has precedence over unsigned log1p per feature
        self.use_log1p_effective = self.use_log1p_mask & (~self.use_signed_log1p_mask)
        # Register masks as buffers so they move with .to(device)
        self.register_buffer("_mask_log1p", self.use_log1p_effective)
        self.register_buffer("_mask_signed", self.use_signed_log1p_mask)
        self.standardize = bool(standardize)
        self.affine_preact = bool(affine_preact)
        self.mode = mode.lower()
        assert self.mode in {"regression", "classification"}

        self.num_classes = int(num_classes)
        self.output_activation = output_activation.lower()
        self.output_min = output_min
        self.output_max = output_max

        # Learnable affine per feature (scale and bias vectors)
        if self.affine_preact:
            self.input_scale = nn.Parameter(torch.ones(self.in_dim))
            self.input_bias = nn.Parameter(torch.zeros(self.in_dim))
        else:
            self.register_parameter("input_scale", None)
            self.register_parameter("input_bias", None)

        # Optional fixed standardization stats (registered as buffers so they move with .to(device))
        # These are applied AFTER log/log1p transforms, BEFORE learnable affine.
        if self.standardize:
            # Initialize from provided values or sensible defaults.
            if standardize_mean is None:
                mean_tensor = torch.zeros(self.in_dim, dtype=torch.float32)
            else:
                mean_tensor = torch.tensor(list(standardize_mean), dtype=torch.float32)
                assert mean_tensor.numel() == self.in_dim, "standardize_mean length must match in_dim"

            if standardize_std is None:
                std_tensor = torch.ones(self.in_dim, dtype=torch.float32)
            else:
                std_tensor = torch.tensor(list(standardize_std), dtype=torch.float32)
                assert std_tensor.numel() == self.in_dim, "standardize_std length must match in_dim"

            self.register_buffer("norm_mean", mean_tensor)
            self.register_buffer("norm_std", std_tensor)
        else:
            self.register_buffer("norm_mean", None)
            self.register_buffer("norm_std", None)

        # Build MLP
        Act = _make_act(activation)
        layers: List[nn.Module] = []
        in_features = self.in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_features, h))
            layers.append(Act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_features = h

        if self.mode == "classification":
            layers.append(nn.Linear(in_features, self.num_classes))
        else: # regression
            layers.append(nn.Linear(in_features, 1))
        self.mlp = nn.Sequential(*layers)

        # Post head activation for regression
        if self.mode == "regression":
            if self.output_activation == "sigmoid":
                self._post = nn.Sigmoid()
                if self.output_min is None or self.output_max is None:
                    raise ValueError("Set output_min/output_max when output_activation='sigmoid'")
            else:
                raise ValueError(f"Unsupported regression output_activation: {self.output_activation}")
        else:
            # if we are doing classification, we don't need any post processing.
            # we return the logits directly. bc we will use cross-entropy loss which
            # combines softmax + log internally.
            self._post = None  # not used

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess raw feature tensor before feeding the MLP.

        Supported input shapes (given in_dim = F):
          1) [B]            when F == 1 (single scalar feature per sample)
          2) [F]            treated as single sample → reshaped to [1, F]
          3) [B, F]         already batched features
          4) [..., F]       any tensor whose LAST dimension matches F; all preceding dims
                             are treated as a batch and collapsed to B = prod(preceding).

        Why collapse higher-rank inputs? It keeps things general (e.g., if later you pass
        a stack produced by some map operation). If you want to preserve structure you can
        always reshape back after the model, but for a plain MLP a 2-D [B,F] matrix is ideal.

        Steps:
          a. Shape normalization → [B, F]
          b. Optional log1p (per-feature, helps with heavy-tailed positive features)
          c. Optional learnable affine per feature (scale & bias vectors of length F)

        Returns:
            Tensor of shape [B, F]
        """
        F = self.in_dim

        # --- Shape normalization -------------------------------------------------
        if x.dim() == 1:
            if F == 1:
                # Single feature per sample; x is [B] -> make it [B,1]
                x = x.unsqueeze(1)
            elif x.numel() == F:  # returns the number of elements in the tensor
                # Vector of features representing ONE sample -> [1, F]
                x = x.view(1, F)
            else:
                raise ValueError(
                    f"Received 1D tensor of length {x.numel()} but in_dim={F}. "
                    "Provide either [B,F] or flatten only when F==1."
                )
        elif x.dim() == 2:
            if x.size(1) != F:
                raise ValueError(f"Expected second dim == {F}, got {x.size(1)} for shape {tuple(x.shape)}")
            # already [B,F]
        else:
            # Higher-rank: collapse all but last if last matches F
            if x.size(-1) != F:
                raise ValueError(
                    f"Last dimension {x.size(-1)} != in_dim {F}; cannot reshape {tuple(x.shape)}"
                )
            # Collapse batch-like leading dims; keep features
            x = x.reshape(-1, F)  # [B,F]

        # --- Feature-wise transforms --------------------------------------------
        # 1) Log transforms (optional, per feature)
        #    We support two masks:
        #      - _mask_signed: apply signed log1p: sign(x) * log1p(|x|)
        #      - _mask_log1p:  apply unsigned log1p: log1p(clamp(x, 0))
        #    Signed log takes precedence; i.e., if a feature is in _mask_signed,
        #    it will NOT also receive unsigned log1p.
        Fmask = (slice(None), slice(None))  # placeholder for readability
        if self._mask_signed.any():
            signed = torch.sign(x) * torch.log1p(torch.abs(x))
            x = torch.where(self._mask_signed.view(1, F), signed, x)
        if self._mask_log1p.any():
            safe = torch.log1p(torch.clamp(x, min=0))
            x = torch.where(self._mask_log1p.view(1, F), safe, x)

        # 2) Standardization (optional): x := (x - mean) / (std + eps)
        if self.standardize:
            if self.norm_mean is None or self.norm_std is None:
                raise RuntimeError(
                    "standardize=True but normalization stats are not set. "
                    "Provide standardize_mean/std in the constructor or call "
                    "set_standardization_stats(mean, std)."
                )
            eps = 1e-8
            x = (x - self.norm_mean.view(1, F)) / (self.norm_std.view(1, F) + eps)

        # Learnable per-feature affine: y_i = scale_i * x_i + bias_i.
        # Broadcasting via (1, F) ensures each column gets its own parameters.
        if self.affine_preact:
            # Ensure parameter shapes align: input_scale/input_bias are registered as [F]
            x = x * self.input_scale.view(1, F) + self.input_bias.view(1, F)

        return x

    @torch.no_grad()
    def set_standardization_stats(self, mean: Iterable[float], std: Iterable[float]):
        """Set per-feature normalization stats to be used when standardize=True.

        Call this if you want to compute mean/std offline (e.g., on the training set)
        and then bake them into the model for reproducible scaling.

        Args:
            mean: iterable of length in_dim with per-feature means (after any log transform you intend to use).
            std:  iterable of length in_dim with per-feature stddevs (non-zero; we add eps internally).
        """
        if not self.standardize:
            raise RuntimeError("Enable standardize=True to use standardization stats.")
        mean_t = torch.tensor(list(mean), dtype=torch.float32, device=self.input_scale.device if self.affine_preact else None)
        std_t = torch.tensor(list(std), dtype=torch.float32, device=self.input_scale.device if self.affine_preact else None)
        assert mean_t.numel() == self.in_dim and std_t.numel() == self.in_dim, "Stats length must match in_dim"
        # Register/overwrite buffers so they move with .to(device)
        self.register_buffer("norm_mean", mean_t)
        self.register_buffer("norm_std", std_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)
        out = self.mlp(x)
        if self.mode == "classification":
            return out  # logits [B, C]
        # regression
        y = self._post(out)
        if self.output_activation == "sigmoid":
            # map [0,1] -> [min,max]
            return self.output_min + (self.output_max - self.output_min) * y
        return y
