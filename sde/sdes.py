import torch.nn as nn
import torch
from abc import ABC, abstractmethod

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


    def score(self, x, t):
        """"
        """