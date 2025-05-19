import torch.nn as nn
import torch
from abc import ABC, abstractmethod
from .utils import batch_linspace, copy_tensor_or_create

class SDE(ABC, nn.Module):
    """
    This class is an abstract base class for Stochastic Differential Equations (SDEs).
    It provides a template for implementing specific SDEs by defining the methods.

    This closely follows the math described in Song et al. (2020).  (Available here
    https://arxiv.org/abs/2011.13456). Equation numbers in the comments throughout
    this file refer to the equations in the paper.

    It implements a general SDE, as given by equation (5),
        dx = f(x, t)dt + g(t)dw,
    and an approximation of its reverse SDE, the true form of which is given by equation (6):
        dx = [f(x, t) - g(t)^2 grad_x log p_t(x)]dt + g(t) dw',
    where w' is a reverse Brownian motion.

    """

    def __init__(self, score_net: nn.Module):
        """
        We need to pass the score network to the SDE class, so that we can use it to
        predict the score (the gradient of the log probability density function) at each time step.
        """
        super().__init__()
        self.score_net = score_net

    @abstractmethod
    def drift(self, x, t):
        """The drift coefficient f(x, t) of the forward SDE."""

    @abstractmethod
    def diff(self, t):
        """The diffusion coefficient g(t) of the forward SDE."""

    @abstractmethod
    def sigma(self, t_end, t_start=0):
        """
        The standard deviation of x(t_end) | x(t_start).
        It is the scale (standard deviation) of the noise added during that interval of time (t_start, t_end).
        """


    @torch.no_grad()
    def _solve(
        self,
        x_start: torch.Tensor,
        t_start: float = 1.0,
        t_end: float = 1e-4,
        steps: int = 1000,
        stochastic: bool = True,
        **score_kwargs,
    ):
        """Solve the SDE or ODE with an Euler(-Maruyama) solver.

        Note that this can be used for either the forward or backward solve, depending on whether
        t_start < t_end (forward) or t_start > t_end (reverse). Note that this method is not
        appropriate for the forward SDE; the forward SDE should have an analytical solution.

        TODO: Add predictor-corrector steps.

        Args:
            x_start (Tensor of shape (batch_size, ...)): The starting point
            t_start: The starting time
            t_end: The final time (best not set to zero for numerical stability)
            steps: The number of steps for the solver
            stochastic: Whether to use the SDE (True) or ODE (False)

        Returns:
            x_end: (Tensor of shape (batch_size, ...))
        """
        device = x_start.device
        x = x_start.detach().clone()

        ts = batch_linspace(t_start, t_end, steps=steps).to(device)
        delta_t = copy_tensor_or_create((t_end - t_start) / (steps - 1))  # Negative in reverse time

        for t in ts:
            score = self.score(x, t, **score_kwargs)
            drift = self.drift(x, t)
            diff = self.diff(t)

            if t.ndim > 0:  # diff is batched, so add dimensions for broadcasting
                new_dims = x.ndim - t.ndim
                diff = diff.reshape(x.shape[:1] + (1,) * new_dims)
                delta_t = delta_t.reshape(x.shape[:1] + (1,) * new_dims)

            if stochastic:
                # Perform an Euler-Maruyama step on the reverse SDE from equation (6)
                delta_w = delta_t.abs().sqrt() * torch.randn(x.shape).to(device)
                dx = (drift - diff**2 * score) * delta_t + diff * delta_w
            else:
                # Compute an Euler step on the reverse ODE from equation (13)
                dx = (drift - diff**2 * score / 2) * delta_t

            x += dx
        return x

class VpSDE(SDE):
    """The variance-preserving SDE described by Song et al. (2020) in equation (11).

    Here, the SDE is given by
    dx = -(1/2) beta(t) x dt + sqrt(beta(t)) dw;
    ie., f(x, t) = -(1/2)beta(t)x and g(t) = sqrt(beta(t)); where f(x, t) is the drift term and g(t) is the diffusion term.

    For beta(t), we use a linear schedule: beta(t) = (beta_max - beta_min)*(t/T) + beta_min.

    So, at t=0, beta(t) = beta_min and at t=T, beta(t) = beta_max. 

    Beta controls how fast noise is added and signal decays. So in the initial stages of forward diffusion,
    the noise is small and the signal is large. As time progresses, the noise increases and the signal decays.
    """

    def __init__(
        self, 
        score_net: nn.Module,
        beta_min: float = 0.1,
        beta_max: float = 20,
        t_max: float = 1.0):

        self.beta_min = torch.tensor(beta_min)
        self.beta_max = torch.tensor(beta_max)
        self.t_max = torch.tensor(t_max)

        super().__init__(score_net)

        
    def drift(self, x, t):
        """
        The drift coefficient f(x, t) of the forward SDE.
        f(x, t) = -(1/2) beta(t) x
        """
        beta_t = self.beta(t)
        # it is negative because we are removing signal.
        return -0.5 * beta_t * x
    

    def diff(self, t):
        """
        The diffusion coefficient g(t) of the forward SDE.
        g(t) = sqrt(beta(t))
        """
        # it is positive because we are adding noise.
        return torch.sqrt(self.beta(t))
    

    def beta(self, t):
        """
        The variance schedule beta(t) of the forward SDE, which is a linear one here.
        It determines the amount of signal decay and noise addition at each time step.
        beta(t) = (beta_max - beta_min) * (t / t_max) + beta_min
        """
        return (self.beta_max - self.beta_min) * t / self.t_max + self.beta_min
    
    
    def beta_integral(self, t_start, t_end):
        """
        The integral of beta(t) from t_start to t_end.
        This is used to compute the standard deviation (sigma).
        """
        return (self.beta(t_end) - self.beta(t_start)) * (t_end - t_start) / 2
    

    def sigma(self, t_end, t_start=0):
        """The standard deviation of x(t_end) | x(t_start).
        It is the scale (standard deviation) of the noise added during that interval of time (t_start, t_end).
        This helps with computing the closed form solution of the forward SDE.
        You use the sigma() function whenever you want to simulate the forward SDE in closed form,
        i.e. not step by step in a numeric way.
        This corresponds to the diffusion coefficient in the integrated form of the SDE: xt = μ*x0 + σϵ
        """
        return torch.sqrt(1.0 - torch.exp(-self.beta_integral(t_start, t_end)))
    
    def mu_scale(self, t_end, t_start=0.0):
        """Scaling factor for the mean of x(t_end) | x(t_start).

        The mean should equal mu_scale(t_end, t_start) * x(t_start).
        this is the drift term in the integrated form of the SDE: xt = μ*x0 + σϵ
        """
        return torch.exp(-self.beta_integral(t_start, t_end) / 2)
    
    @staticmethod
    def _match_timestep_shapes(t_start, t_end):
        t_start = copy_tensor_or_create(t_start)
        t_end = copy_tensor_or_create(t_end)
        if t_start.ndim > t_end.ndim:
            t_end = torch.full_like(t_start, fill_value=t_end)
        elif t_start.ndim < t_end.ndim:
            t_start = torch.full_like(t_end, fill_value=t_start)
        return t_start, t_end
    
    def solve_forward_sde(self, x_start, t_end=1.0, t_start=0.0, return_eps=False):
        """
        Solve the SDE forward from time t_start to t_end.
            Forward SDE:
            x_t = mu(t) * x_0 + sigma(t) * eps
        """

        t_start, t_end = self._match_timestep_shapes(t_start, t_end)
        t_start, t_end = t_start.to(x_start.device), t_end.to(x_start.device)
        assert torch.all(t_start <= t_end)

        mu_scale = self.mu_scale(t_start=t_start, t_end=t_end)
        sigma_end = self.sigma(t_start=t_start, t_end=t_end)
        eps = torch.randn_like(x_start) # Same shape and device as x_start

        if mu_scale.ndim > 0:  # Add a broadcasting dimensions to the scalars
            new_dims = x_start.ndim - mu_scale.ndim
            mu_scale = mu_scale.reshape(x_start.shape[:1] + (1,) * new_dims)
            sigma_end = sigma_end.reshape(x_start.shape[:1] + (1,) * new_dims)

        # The forward SDE is given by:
        # x_t = mu(t) * x_0 + sigma(t) * eps
        x_end = mu_scale * x_start + sigma_end * eps

        if return_eps:  # epsilon, the random noise value, may be needed for training
            return x_end, eps
        else:
            return x_end