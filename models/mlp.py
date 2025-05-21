import torch.nn as nn
from typing import List, Tuple
import torch
import math

def _get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class MLPUnet(nn.Module):
    
    def __init__(
        self,
        data_dim,
        hidden_sizes,
        time_embedding_dim,
    ):
        super().__init__()

        # initially the layers are empty
        self.layers = []
        self.time_embedding_dim = time_embedding_dim 

        # layer_info is a list of tuples, where each tuple contains the size of the layer and the index of the reference layer
        # for the layer. The reference layer is the layer that is concatenated with the current layer.
        # The first layer is the input layer, which takes the data_dim and time_embedding_dim as input.
        self.layer_info = [(self.time_embedding_dim + data_dim, -1)]

        # initially the layer info contains the downsampling layers
        for size in hidden_sizes:
            # When downsampling, there is no reference layer, so the index is -1.
            self.layer_info.append((size, -1))

        ref_layer = len(self.layer_info) - 1

        # now we add the upsampling layers, which are concatenated with the reference layer, to the layer info.
        # hidden sizes contain the size (number of neurons) of the downsampling layers. We reverse the list so
        # that we can start constructing the upsampling layers from the last downsampling layer.
        # Note that we don't concatenate the bottleneck layer with the reference layer, so we start from the 
        # second last downsampling layer.
        for size in hidden_sizes[::-1][1:]:
            ref_layer -= 1
            self.layer_info.append((size, ref_layer))

        # the last layer = input layer dim - time embedding dim = data_dim
        # because we want to find the score ("noise") for each pixel.
        self.layer_info.append((data_dim, 0))

        for i in range(1, len(self.layer_info)):
            layer_sz, ref_layer_idx = self.layer_info[i]

            # last_layer_sz is the num of neurons in the previous layer
            last_layer_sz, _ = self.layer_info[i - 1]
            if ref_layer_idx == -1:
                # downsampling layer
                self.layers.append(nn.Linear(last_layer_sz, layer_sz))
            else:
                # upsampling layer; where we also perform concatenation like UNet
                self.layers.append(
                    nn.Linear(last_layer_sz + self.layer_info[ref_layer_idx][0], layer_sz)
                )
        self.layers = nn.ModuleList(self.layers)

        # define the activation function
        self.activation = nn.SiLU()

    def forward(self, x, t):
        if self.time_embedding_dim > 1:
            # getting the timestep embedding is basically mapping the timestep to a higher dimensional space
            t = _get_timestep_embedding(t, self.time_embedding_dim)

        # note that we will keep track of the embeddings for each layer
        # because we will need them for the upsampling layers where we will need 
        # them for concatenation.
        embeddings = []
        # the first layer is the input layer, which takes the data_dim and time_embedding_dim as input.
        # we need to concatenate the data and the timestep embedding.
        first_embedding = torch.cat([x, t], dim=-1)
        embeddings.append(first_embedding)


        # start forwarding through the layers
        for i, layer in enumerate(self.layers):
            if self.layer_info[i + 1][1] == -1:
                # downsampling layers
                interim = layer(embeddings[-1])
            else:
                # upsampling layers. layer_info[any][1] is the index of the layer that we 
                # will use for concatenation.
                interim = layer(
                    torch.cat([embeddings[self.layer_info[i + 1][1]], embeddings[-1]], dim=-1)
                )
            
            # apply activation function except for the last layer
            if i < len(self.layers) - 1:
                interim = self.activation(interim)

            embeddings.append(interim)
        
        # remember that we were storing all the embeddings, so we only need to return the last one
        return embeddings[-1]

