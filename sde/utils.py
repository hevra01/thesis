import math
import numbers
from typing import Callable, Literal, Optional
import torch
import tqdm

# A threshold for the dimension of the data, if the dimension is above this threshold, the hutchinson method is used
HUTCHINSON_DATA_DIM_THRESHOLD = 3500


def copy_tensor_or_create(t, **kwargs):
    """
    Returns a copy of the input tensor or creates a new tensor because we are working with tensors from the input if it is a number.
    **kwargs is for any additional arguments to pass to torch.tensor such as data type, device, etc. 
    It's like unpacking a dictionary of named arguments.

    """
    # check if t is a number or not
    if isinstance(t, numbers.Number):
        return torch.tensor(t, **kwargs)
    elif isinstance(t, torch.Tensor):
        # This gives you a clean copy without messing up autograd or in-place issues.
        return t.clone().detach()
    else:
        raise ValueError(f"Cannot copy object of type {type(t)}")
    


def batch_linspace(start, end, steps):
    """
    Batched linspace function. If start and end are numbers, it will return
    a linspace from start to end of size (steps, ). If start and end are
    torch.Tensors that are of shape (batch_size, ) it will return a batch of
    linspaces from start to end of size (steps, batch_size).

    e.g. Example 1: Scalar input: batch_linspace(0.0, 1.0, 5), then returns tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])  # shape [5].
    However, Example 2: Batched input

    start = torch.tensor([0.0, 1.0])
    end = torch.tensor([1.0, 3.0])
    batch_linspace(start, end, 5)

    tensor([
    [0.0000, 1.0000],
    [0.2500, 1.5000],
    [0.5000, 2.0000],
    [0.7500, 2.5000],
    [1.0000, 3.0000],
    ])  # shape [5, 2] → 5 time steps for 2 different linspaces


    Args:
        start: (torch.Tensor or float) The start of the linspace.
        end: (torch.Tensor or float) The end of the linspace.
        steps: (int) The number of steps in the linspace.
    Returns:
        (torch.Tensor) The linspace or batch of linspaces.
    """
    # Normal linspace behaviour
    if isinstance(start, numbers.Number) or start.ndim == 0:
        return torch.linspace(start, end, steps)

    # Batched linspace behaviour
    def linspace(start, end):
        return start + torch.arange(0, steps).to(start.device) * (end - start) / (steps - 1)

    return torch.vmap(linspace)(start, end).mT

def compute_trace_of_jacobian(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    method: Literal["hutchinson_gaussian", "hutchinson_rademacher", "deterministic"] | None = None,
    hutchinson_sample_count: int = HUTCHINSON_DATA_DIM_THRESHOLD,
    chunk_size: int = 128,
    seed: int = 42,
    verbose: bool = False,
):
    """
    fn is a function mapping \R^d to \R^d, this function computes the trace of the Jacobian of fn at x.
        In our use case, fn is the score function of the diffusion model, which takes in a batch of data
        and returns a batch of scores for the data, each dimension of the data has a score (the noise that
        needs to be removed from the data to bring it closer to the data distribution).

    To do so, there are different methods implemented:

    1. The Hutchinson estimator:
        This is a stochastic estimator that uses random vector to estimate the trace.
        These random vectors can either come from the Gaussian distribution (if method=`hutchinson_gaussian` is specified)
        or from the Rademacher distribution (if method=`hutchinson_rademacher` is specified).
    2. The deterministic method:
        This is not an estimator and computes the trace by taking all the x.dim() canonical basis vectors times $\sqrt{d}$ (?)
        and taking the average of their quadratic forms. For data with small dimension, the deterministic method
        is the best.

    The implementation of all of these is as follows:
        A set of vectors of the same dimension as data are sampled and the value [v^T \\nabla_x v^T fn(x)] is
        computed using jvp. Finally, all of these values are averaged.

    Args:
        fn (Callable[[torch.Tensor], torch.Tensor]):
            A function that takes in a tensor of size [batch_size, *data_shape] and returns a tensor of size [batch_size, *data_shape]
        x (torch.Tensor): a batch of inputs [batch_size, input_dim]
        method (str, optional):
            chooses between the types of methods to evaluate trace.
            it defaults to None, in which case the most appropriate method is chosen based on the dimension of the data.
        hutchinson_sample_count (int):
            The number of samples for the stochastic methods, if deterministic is chosen, this is ignored.
        chunk_size (int):
            Jacobian vector products can be done in parallel for better speed-up, this is the size of the parallel batch.
    Returns:
        traces (torch.Tensor): A tensor of size [batch_size,] where traces[i] is the trace computed for the i'th batch of data
    """
    # use seed to make sure that the same random vectors are used for the same data
    # NOTE: maybe creating a fork of the random number generator is a better idea here!
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        # save batch size and data dimension and shape
        batch_size = x.shape[0]
        data_shape = x.shape[1:]
        ambient_dim = x.numel() // x.shape[0]

        if ambient_dim > HUTCHINSON_DATA_DIM_THRESHOLD:
            method = method or "hutchinson_gaussian"
        else:
            method = method or "deterministic"

        # The general implementation is to compute the quadratic forms of [v^T \\nabla_x v^T score(x, t)] in a list and then take the average
        all_quadratic_forms = []
        sample_count = hutchinson_sample_count if method != "deterministic" else ambient_dim
        # all_v is a tensor of size [batch_size * sample_count, *data_shape] where each row is an appropriate vector for the quadratic forms
        if method == "hutchinson_gaussian":
            all_v = torch.randn(size=(batch_size * sample_count, *data_shape)).cpu().float()
        elif method == "hutchinson_rademacher":
            all_v = (
                torch.randint(size=(batch_size * sample_count, *data_shape), low=0, high=2)
                .cpu()
                .float()
                * 2
                - 1.0
            )
        elif method == "deterministic":
            all_v = torch.eye(ambient_dim).cpu().float() * math.sqrt(ambient_dim)
            # the canonical basis vectors times sqrt(d) the sqrt(d) coefficient is applied so that when the
            # quadratic form is computed, the average of the quadratic forms is the trace rather than their sum
            all_v = all_v.repeat_interleave(batch_size, dim=0).reshape(
                (batch_size * sample_count, *data_shape)
            )
        else:
            raise ValueError(f"Method {method} for trace computation not defined!")
        # x is also duplicated as much as needed for the computation
        all_x = (
            x.cpu()
            .unsqueeze(0)
            .repeat(sample_count, *[1 for _ in range(x.dim())])
            .reshape(batch_size * sample_count, *data_shape)
        )

        all_quadratic_forms = []
        rng = list(zip(all_v.split(chunk_size), all_x.split(chunk_size)))
        # compute chunks separately
        rng = tqdm(rng, desc="Computing the quadratic forms") if verbose else rng
        idx_dbg = 0
        for vx in rng:
            idx_dbg += 1

            v_batch, x_batch = vx
            v_batch = v_batch.to(x.device)
            x_batch = x_batch.to(x.device)

            all_quadratic_forms.append(
                torch.sum(
                    v_batch * torch.func.jvp(fn, (x_batch,), tangents=(v_batch,))[1],
                    dim=tuple(range(1, x.dim())),
                ).cpu()
            )
    # concatenate all the chunks
    all_quadratic_forms = torch.cat(all_quadratic_forms)
    # reshape so that the quadratic forms are separated by batch
    all_quadratic_forms = all_quadratic_forms.reshape((sample_count, x.shape[0]))
    # take the average of the quadratic forms for each batch
    return all_quadratic_forms.mean(dim=0).to(x.device)
