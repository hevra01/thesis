import functools
import numbers
from typing import Literal
import torch
from model_based_estimator import ModelBasedLIDEstimator
from sde.sdes import VpSDE
from sde.utils import compute_trace_of_jacobian, HUTCHINSON_DATA_DIM_THRESHOLD

class FokkerPlanckEstimator(ModelBasedLIDEstimator):
    """
    Fokker-Planck Estimator for LID estimation. Note that this is a subclass of ModelBasedLIDEstimator.
    In other words, there could be other estimators that are not based on the Fokker-Planck equation.
    The Fokker-Planck equation describes the time evolution of the probability density function of a stochastic process.
    This estimator only works with Variance Preserving SDEs.
    """
    def __init__(self, ambient_dim: int, model: torch.nn.Module, device: torch.device):
        super().__init__(ambient_dim, model, device)

        assert isinstance(self.model, VpSDE), "The model should be a VpSDE object."
        self.VpSDE: VpSDE = self.model

    def _get_laplacian_term(
        self,
        x: torch.Tensor,
        t: float,
        coeff: float,
        method: (
            Literal["hutchinson_gaussian", "hutchinson_rademacher", "deterministic"] | None
        ) = None,
        # The number of samples if one opts for estimation methods to save time:
        hutchinson_sample_count: int = HUTCHINSON_DATA_DIM_THRESHOLD,
        chunk_size: int = 128,
        seed: int = 42,
        verbose: int = 0,
        **score_kwargs,
    ):
        """
        Computes the Laplacian term of the Fokker-Planck equation using Hutchinson's method 
        or the deterministic method. 
        
        The laplacian term is computed as the trace of the Jacobian of the score function.
        The score function is the first derivative of the log probability density function with respect to the input x.
        The jacobian of the score, is the second derivative of the log probability density function with respect to the input x.
        Jacobian gives the derivative of every score value wrt every input dimension. However, we only need the derivative
        of the score value wrt its own dimension. E.g. if the first dimension of the score is 0.5, we need to compute the
        derivative of 0.5 wrt x[0].

        For high dimensional data, the laplacian term will be small negative. However, for low dimensional data, the 
        laplacian term will be large negative. The values are negative because we have converging score vectors close to
        the manifold. 
        """


        def score_fn(x, t: float):
            """Computes the score function for the given input x, where x is a noisy image as a data point, and time t."""
            if isinstance(t, numbers.Number):
                t = torch.tensor(t).float()
            t: torch.Tensor
            t_repeated = t.repeat(x.shape[0]).to(x.device)
            return self.vpsde.score_net(x, t=t_repeated, **score_kwargs)

        laplacian_term = compute_trace_of_jacobian(
            fn=functools.partial(score_fn, t=t),
            x=coeff * x,
            method=method,
            hutchinson_sample_count=hutchinson_sample_count,
            chunk_size=chunk_size,
            seed=seed,
            verbose=verbose,
        )
        return laplacian_term

    def _get_score_norm_term(self, x, t: float, coeff: float, **score_kwargs):
        """
        FIXME: I am not sure how the value of the score norm will change based on high or low dim data.
        I suspect it is high for low dim and low for high dim data. 
        """
        if isinstance(t, numbers.Number):
            t = torch.tensor(t).float()
        t: torch.Tensor
        t_repeated = t.repeat(x.shape[0]).to(self.device)
        scores_flattened = self.vpsde.score_net(coeff * x, t=t_repeated, **score_kwargs).reshape(
            x.shape[0], -1
        )
        score_norm_term = torch.sum(scores_flattened * scores_flattened, dim=1)
        return score_norm_term



class FlipdEstimator(FokkerPlanckEstimator):
    """
    An LID estimator based on the connection made between marginal probabilities
    and Gaussian convolution + running the singular value decomposition.

    This is the fastest model-based LID estimator available in the library.

    Args:
        sde: An Sde object containing a trained diffusion model
        ambient_dim: Corresponds to d in the paper. Inferred by estimate_id if not
            specified here.
    """

    @torch.no_grad
    # we use torch.no_grad because we will call the score estimator model 
    # only for inference.
    def _estimate_lid(
        self,
        x: torch.Tensor, # a point (or batch of points) where LID is to be estimated.
        t: float | None = None, # the noise level
        method: (
            Literal["hutchinson_gaussian", "hutchinson_rademacher", "deterministic"] | None
        ) = None, # how to compute the Laplacian (trace of score Jacobian)
        # The number of samples if one opts for estimation methods to save time:
        # number of random vectors to estimate the trace.
        hutchinson_sample_count: int = HUTCHINSON_DATA_DIM_THRESHOLD,
        chunk_size: int = 128,
        seed: int = 42, # for reproducibility in random sampling.
        verbose: int = 0,
        **score_kwargs,
    ) -> torch.Tensor:
        x = x.to(self.device)
        # sigma_t is the standard deviation of noise at time t.
        t, sigma_t, coeff = self._get_all_math_terms(
            t=t,
            sigma_t=None,
            coeff=None,
        )

        # First, get the contribution from the laplacian term, which is the trace of the Jacobian of the score.
        laplacian_term = self._get_laplacian_term(
            x=x,
            t=t,
            coeff=coeff,
            method=method,
            hutchinson_sample_count=hutchinson_sample_count,
            chunk_size=chunk_size,
            seed=seed,
            verbose=verbose,
            **score_kwargs,
        )

        # Second, get the contribution of the score norm.
        score_norm_term = self._get_score_norm_term(x=x, t=t, coeff=coeff, **score_kwargs)
        return self.ambient_dim + sigma_t * laplacian_term + score_norm_term
