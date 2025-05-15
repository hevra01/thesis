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
        return t.clone().detach()
    else:
        raise ValueError(f"Cannot copy object of type {type(t)}")