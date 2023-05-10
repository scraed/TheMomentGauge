import jax.numpy as jnp
import numpy as np
from jax.lax import switch
from jax import vmap, jit


class BaseSampler():
    def __init__(self, constant):
        r"""The base class for sampler.
        A sampler is a probability distribution :math:`f(\mathbf{u};\boldsymbol{\beta})` parametrized by :math:`\boldsymbol{\beta}` from which we could draw samples and compute likelihoods.

        Parameters
        ----------
        constant : dictionary
            a dictionary with necessary constants provided as key-value pairs.

        Attributes
        ----------
        constant : dict
            a dictionary with necessary constants provided as key-value pairs.
        """
        #self.pi = jnp.pi
        #self.m = constant["m"]
        #self.kB = constant["kB"]
        self.constant = constant  # a dictionary with necessary constants provided as key-value pairs

    def sample(self, betas):
        r"""Generate N samples :math:`\mathbf{u}_i` from the distribution :math:`f(\mathbf{u})` with proper weights :math:`w_i` such that
        
        .. math::
            :nowrap:

            \begin{equation}
            \int \phi(\mathbf{u}) f(\mathbf{u}) d \mathbf{u} \approx \sum_{i=1}^N w_i \phi(\mathbf{u}_i),
            \end{equation}

        in whic N depends on the particular implementation of the sampler.

        Parameters
        ----------
        betas : array of shape (n)
            the n-dim parameter :math:`\boldsymbol{\beta}` specifying the distributions

        Returns
        -------
        Tuple
            A tuple containing

                **samples**: *array of shape (N,d)* - N samples of d-dim vectors :math:`\mathbf{u}_i` draw from the distribution. 

                **weights**: *array of shape (N)* - non-negative weights :math:`w_i` for each samples. The summation of weights equals to 1.
        
                **log_likelihoods**: *array of shape (N)* - the log-likelihoods :math:`\log f(\mathbf{u}_i)` for each samples

        Raises
        ------
        NotImplementedError
            This method is not implemented
        """
        raise NotImplementedError