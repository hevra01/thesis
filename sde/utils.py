import numbers
import torch


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
