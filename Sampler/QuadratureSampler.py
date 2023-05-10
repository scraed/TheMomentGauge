from MomentGauge.Sampler.Base import BaseSampler
import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
from sympy.integrals.quadrature import gauss_hermite, gauss_laguerre, gauss_jacobi, gauss_legendre, gauss_gen_laguerre
from MomentGauge.Sampler.SamplerUtility import Gauss_Legendre_Quadrature_2D_Block


class Gauss_Legendre_Sampler2D(BaseSampler):
    def __init__(self, n_x=8, n_y=8, B_x=16, B_y=16):
        r"""A sampler of the 2D uniform distribution on a square area :math:`[a_x, b_x]*[a_y, b_y]` based on the Block-wise Gauss_Legendre quadrature 
        
        See :func:`MomentGauge.Sampler.SamplerUtility.Gauss_Legendre_Quadrature_2D_Block` for detailed information for the Block-wise Gauss_Legendre quadrature.

        **domain**: :math:`\mathbf{u} \in [a_x, b_x]*[a_y, b_y]`   
        
        **likelihood**: :math:`f(\mathbf{u}) = \frac{1}{(b_x-a_x)(b_y-a_y)}`

        Parameters
        ----------
        n_x : int 
            the order of Gauss Legendre quadrature in x dimension, default = 8
        n_y : int 
            the order of Gauss Legendre quadrature in y dimension, default = 8
        B_x : int 
            how many blocks are splitted in the x dimension, default = 16
        B_y : int 
            how many blocks are splitted in the y dimension, default = 16

        Attributes
        ----------
        constant : dict
            an empty dictionary.
        """
        super().__init__({})
        self.__n_x = n_x
        self.__n_y = n_y
        self.__B_x = B_x
        self.__B_y = B_y
    def __hash__(self):
        """Redefine the hash method to include the class attributes. It helps jax.jit to correctly identify class instances"""
        return hash(("Gauss_Legendre_Sampler2D",self.__n_x,self.__n_y,self.__B_x,self.__B_y))

    def __eq__(self, other):
        """Redefine the eq method to include the class attributes."""
        return (isinstance(other, Gauss_Legendre_Sampler2D) and
                (self.__n_x,self.__n_y,self.__B_x,self.__B_y) == (other.__n_x,other.__n_y,other.__B_x,other.__B_y))

    @partial(jax.jit,
             static_argnums=0,
             static_argnames=["n_x", "n_y", "B_x", "B_y"])
    def __sample(self, betas, n_x=8, n_y=8, B_x=16, B_y=16):
        r"""
        Generate samples from the uniform distribution on :math:`(a_x, b_x)\times(a_y, b_y)` with proper weights :math:`w_i` such that

        .. math::
            :nowrap:

            \begin{equation}
            \int_{a_y}^{b_y}\int_{a_x}^{b_x} \phi(u_x,u_y) f(u_x,u_y) d u_x d u_y \approx \sum_{i=1}^N w_i \phi(\mathbf{u}_i); \quad \mathbf{u}_i=\{u_{x,i},u_{y,i}\},
            \end{equation}

        in which :math:`N = n_x\times n_y\times B_x\times B_y`

        Parameters
        ----------
        betas : float array of shape (4)
            an array (a_x, b_x, a_y, b_y), in which 

                **a_x** : *float* - lower integration limit in x dimension

                **b_x** : *float* - upper integration limit in x dimension

                **a_y** : *float* - lower integration limit in y dimension

                **b_y** : *float* - upper integration limit in y dimension

        n_x : int 
            the order of Gauss Legendre quadrature in x dimension, default = 8, static for jax.jit
        n_y : int 
            the order of Gauss Legendre quadrature in y dimension, default = 8, static for jax.jit
        B_x : int 
            how many blocks are splitted in the x dimension, default = 16, static for jax.jit
        B_y : int 
            how many blocks are splitted in the y dimension, default = 16, static for jax.jit

        Returns
        -------
        Tuple
            A tuple containing

                **samples**: *float array of shape (N,2)* - N  samples of 2-dim vectors :math:`\mathbf{u}_i` draw from the distribution. 

                **weights**: *float array of shape (N)* - N non-negative weights :math:`w_i` for each samples. The summation of weights equals to 1.
        
                **log_likelihoods**: *float array of shape (N)* - N the log-likelihoods :math:`\log f(\mathbf{u}_i)` for each samples
            in which N = n_x*n_y*B_x*B_y
        """
        a_x, b_x, a_y, b_y = betas
        x, w = Gauss_Legendre_Quadrature_2D_Block(a_x, b_x, n_x, B_x, a_y, b_y,
                                                  n_y, B_y)
        volume = (b_y - a_y) * (b_x - a_x)
        x = x.reshape(-1, 2)
        w = w.reshape(-1) / volume
        log_likeilihoods = jnp.broadcast_to(jnp.array([jnp.log(1 / volume)]),
                                            w.shape)
        return x, w, log_likeilihoods
    @partial(jax.jit,
             static_argnums=0)
    def sample(self, betas):
        r"""
        Generate samples from the uniform distribution on :math:`(a_x, b_x)\times(a_y, b_y)` with proper weights :math:`w_i` such that

        .. math::
            :nowrap:

            \begin{equation}
            \int_{a_y}^{b_y}\int_{a_x}^{b_x} \phi(u_x,u_y) f(u_x,u_y) d u_x d u_y \approx \sum_{i=1}^N w_i \phi(\mathbf{u}_i); \quad \mathbf{u}_i=\{u_{x,i},u_{y,i}\},
            \end{equation}

        in which :math:`N = n_x\times n_y\times B_x\times B_y`

        Parameters
        ----------
        betas : float array of shape (4)
            an array (a_x, b_x, a_y, b_y), in which 

                **a_x** : *float* - lower integration limit in x dimension

                **b_x** : *float* - upper integration limit in x dimension

                **a_y** : *float* - lower integration limit in y dimension

                **b_y** : *float* - upper integration limit in y dimension

        Returns
        -------
        Tuple
            A tuple containing

                **samples**: *float array of shape (N,2)* - N  samples of 2-dim vectors :math:`\mathbf{u}_i` draw from the distribution. 

                **weights**: *float array of shape (N)* - N non-negative weights :math:`w_i` for each samples. The summation of weights equals to 1.
        
                **log_likelihoods**: *float array of shape (N)* - N the log-likelihoods :math:`\log f(\mathbf{u}_i)` for each samples
            in which N = n_x*n_y*B_x*B_y
        """
        return self.__sample(betas, n_x=self.__n_x, n_y=self.__n_y, B_x=self.__B_x, B_y=self.__B_y)

class Gauss_Legendre_Sampler_Polar2D(BaseSampler):
    def __init__(self, n_x=8, n_r=8, B_x=16, B_r=16):
        r"""A sampler of the 2D distribution :math:`f(u_x,u_r) \propto u_r` on a square area :math:`[a_x, b_x]*[0, b_r]` based on importance sampling.
        
        The importance sampling is w.r.t :func:`Sampler.QuadratureSampler.Gauss_Legendre_Sampler2D`.

        **domain**: :math:`\mathbf{u} \in \{ (u_x,u_r) \ | \ u_x \in [a_x, b_x], \ u_r \in [0, b_r] \}`   
        
        **likelihood**: :math:`f(\mathbf{u}) = \frac{2 u_r}{(b_x-a_x)b_r^2}`

        Parameters
        ----------
        n_x : int 
            the order of Gauss Legendre quadrature in x dimension, default = 8
        n_r : int 
            the order of Gauss Legendre quadrature in r dimension, default = 8
        B_x : int 
            how many blocks are splitted in the x dimension, default = 16
        B_r : int 
            how many blocks are splitted in the r dimension, default = 16

        Attributes
        ----------
        constant : dict
            an empty dictionary.
        """
        super().__init__({})
        
        self.__BaseSampler = Gauss_Legendre_Sampler2D(n_x = n_x,n_y = n_r, B_x = B_x, B_y = B_r)
        self.__n_x = n_x
        self.__n_r = n_r
        self.__B_x = B_x
        self.__B_r = B_r
    def __hash__(self):
        """Redefine the hash method to include the class attributes. It helps jax.jit to correctly identify class instances"""
        return hash(("Gauss_Legendre_Sampler_Polar2D",self.__n_x,self.__n_r,self.__B_x,self.__B_r))

    def __eq__(self, other):
        """Redefine the eq method to include the class attributes."""
        return (isinstance(other, Gauss_Legendre_Sampler_Polar2D) and
                (self.__n_x,self.__n_r,self.__B_x,self.__B_r) == (other.__n_x,other.__n_r,other.__B_x,other.__B_r))

    @partial(jax.jit,
             static_argnums=0,
             static_argnames=["n_x", "n_r", "B_x", "B_r"])
    def __sample(self, betas, n_x=8, n_r=8, B_x=16, B_r=16):
        r"""Generate samples from the distribution :math:`f(\mathbf{u})` with proper weights :math:`w_i` such that

        .. math::
            :nowrap:

            \begin{equation}
            \int_{0}^{b_r} \int_{a_x}^{b_x} \phi_r(u_x,u_r) f(u_x,u_r) du_x du_r \approx \sum_{i=1}^N w_i \phi_r(\mathbf{u}_i); \quad \mathbf{u}_i=\{u_{x,i},u_{r,i}\},
            \end{equation}

        in which :math:`w_i, \mathbf{u}_i` are weights and  :math:`N = n_x\times n_r\times B_x\times B_r`. 

        Parameters
        ----------
        betas : float array of shape (3)
            an array (a_x, b_x, b_r), in which 

                **a_x** : *float* - lower integration limit in x dimension

                **b_x** : *float* - upper integration limit in x dimension

                **b_r** : *float* - upper integration limit in r dimension

        n_x : int 
            the order of Gauss Legendre quadrature in x dimension, default = 8, static for jax.jit
        n_r : int 
            the order of Gauss Legendre quadrature in r dimension, default = 8, static for jax.jit
        B_x : int 
            how many blocks are splitted in the x dimension, default = 16, static for jax.jit
        B_r : int 
            how many blocks are splitted in the r dimension, default = 16, static for jax.jit

        Returns
        -------
        Tuple
            A tuple containing

                **samples**: *float array of shape (N,2)* - N  samples of 2-dim vectors :math:`\mathbf{u}_i` draw from the distribution. 

                **weights**: *float array of shape (N)* - N non-negative weights :math:`w_i` for each samples. The summation of weights equals to 1.
        
                **log_likelihoods**: *float array of shape (N)* - N the log-likelihoods :math:`\log f(\mathbf{u}_i)` for each samples
            in which N = n_x*n_y*B_x*B_y
        """
        a_x, b_x, b_r = betas
        a_r = 0.
        x, w, logli = self.__BaseSampler.sample(jnp.array([a_x, b_x, a_r, b_r]))

        volume = b_r**2 * (b_x - a_x)
        log_likelihood = jnp.log(2 * x[..., 1] / volume)
        w_polar = w * jnp.exp(log_likelihood - logli)

        #logli_polar = jnp.log( 2*jnp.pi*x[...,1]/volume )
        #logli_polar = jnp.log( 1/volume )
        return x, w_polar, log_likelihood
    @partial(jax.jit,
             static_argnums=0)
    def sample(self, betas):
        r"""Generate samples from the distribution :math:`f(\mathbf{u})` with proper weights :math:`w_i` such that

        .. math::
            :nowrap:

            \begin{equation}
            \int_{0}^{b_r} \int_{a_x}^{b_x} \phi_r(u_x,u_r) f(u_x,u_r) du_x du_r \approx \sum_{i=1}^N w_i \phi_r(\mathbf{u}_i); \quad \mathbf{u}_i=\{u_{x,i},u_{r,i}\},
            \end{equation}

        in which :math:`w_i, \mathbf{u}_i` are weights and  :math:`N = n_x\times n_r\times B_x\times B_r`. 

        Parameters
        ----------
        betas : float array of shape (3)
            an array (a_x, b_x, b_r), in which 

                **a_x** : *float* - lower integration limit in x dimension

                **b_x** : *float* - upper integration limit in x dimension

                **b_r** : *float* - upper integration limit in r dimension

        Returns
        -------
        Tuple
            A tuple containing

                **samples**: *float array of shape (N,2)* - N  samples of 2-dim vectors :math:`\mathbf{u}_i` draw from the distribution. 

                **weights**: *float array of shape (N)* - N non-negative weights :math:`w_i` for each samples. The summation of weights equals to 1.
        
                **log_likelihoods**: *float array of shape (N)* - N the log-likelihoods :math:`\log f(\mathbf{u}_i)` for each samples
            in which N = n_x*n_y*B_x*B_y
        """
        return self.__sample(betas, n_x=self.__n_x, n_r=self.__n_r, B_x=self.__B_x, B_r=self.__B_r)

class Gauss_Legendre_Sampler_Polar3D(Gauss_Legendre_Sampler_Polar2D):
    def __init__(self, n_x=8, n_r=8, B_x=16, B_r=16):
        r"""A sampler of the 3D uniform distribution on a cylinder.
        
        The sampling based on the Block-wise Gauss_Legendre quadrature and importance sampling. 
        
        See :func:`Sampler.SamplerUtility.Gauss_Legendre_Quadrature_2D_Block` for detailed information for the Block-wise Gauss_Legendre quadrature.

        **domain**: :math:`\mathbf{u} \in \{ (u_x,u_y,u_z) \ | \ u_x \in [a_x, b_x], u_y^2 + u_z^2 \le b_r^2 \}`   
        
        **likelihood**: :math:`f(\mathbf{u}) = \frac{1}{\pi(b_x-a_x)b_r^2}`

        Parameters
        ----------
        n_x : int 
            the order of Gauss Legendre quadrature in x dimension, default = 8
        n_r : int 
            the order of Gauss Legendre quadrature in r dimension, default = 8
        B_x : int 
            how many blocks are splitted in the x dimension, default = 16
        B_r : int 
            how many blocks are splitted in the r dimension, default = 16

        Attributes
        ----------
        constant : dict
            an empty dictionary.
        """
        super().__init__(n_x=n_x, n_r=n_r, B_x=B_x, B_r=B_r)
        self.__n_x = n_x
        self.__n_r = n_r
        self.__B_x = B_x
        self.__B_r = B_r

    def __hash__(self):
        """Redefine the hash method to include the class attributes. It helps jax.jit to correctly identify class instances"""
        return hash(("Gauss_Legendre_Sampler_Polar3D",self.__n_x,self.__n_r,self.__B_x,self.__B_r))

    def __eq__(self, other):
        """Redefine the eq method to include the class attributes."""
        return (isinstance(other, Gauss_Legendre_Sampler_Polar3D) and
                (self.__n_x,self.__n_r,self.__B_x,self.__B_r) == (other.__n_x,other.__n_r,other.__B_x,other.__B_r))


    @partial(jax.jit,
             static_argnums=0)
    def sample(self, betas):
        r"""Generate samples from the uniform distribution on the cylinder :math:`\mathbf{D} = \{ (u_x,u_y,u_z) \ | \ u_x \in [a_x, b_x], u_y^2 + u_z^2 \le b_r^2 \}` with proper weights :math:`w_i` such that

        .. math::
            :nowrap:

            \begin{equation}
            \begin{split}
            \int_{\mathbf{D}} \phi(\mathbf{u}) f(\mathbf{u}) d^3 \mathbf{u} &= \int_{0}^{b_r} \int_{a_x}^{b_x} 2\pi u_r \phi_r(u_x,u_r) f_r(u_x,u_r) du_x du_r\\
            &\approx \sum_{i=1}^N w_i \phi(\mathbf{u}_i); \quad \mathbf{u}_i=\{u_{x,i},u_{y,i},u_{z,i}\},
            \end{split}
            \end{equation}

        in which

        .. math::
            :nowrap:

            \begin{equation}
            \begin{split}
            \phi(\mathbf{u}) &= \phi(u_x, u_y, u_z) = \phi_r(u_x, u_r(u_y,u_z)) \\
            f(\mathbf{u}) &= f(u_x, u_y, u_z) = f_r(u_x, u_r(u_y,u_z)); \quad u_r(u_y,u_z) = \sqrt{u_y^2+u_z^2} \\
            \end{split}
            \end{equation}
        
        are both polar symmetric w.r.t the x axis, and :math:`N = n_x\times n_r\times B_x\times B_r`.

        Parameters
        ----------
        betas : float array of shape (3)
            an array (a_x, b_x, b_r), in which 

                **a_x** : *float* - lower integration limit in x dimension

                **b_x** : *float* - upper integration limit in x dimension

                **b_r** : *float* - upper integration limit in r dimension

        Returns
        -------
        Tuple
            A tuple containing

                **samples**: *float array of shape (N,3)* - N  samples of 3-dim vectors :math:`\mathbf{u}_i=\{u_{x,i},u_{y,i},u_{z,i}\}` draw from the distribution. Notably these samples have :math:`u_{y,i} = u_{z,i}` due to polar symmetry. 

                **weights**: *float array of shape (N)* - N non-negative weights :math:`w_i` for each samples. The summation of weights equals to 1.
        
                **log_likelihoods**: *float array of shape (N)* - N the log-likelihoods :math:`\log f(\mathbf{u}_i)` for each samples
            in which N = n_x*n_y*B_x*B_y
        """
        a_x, b_x, b_r = betas
        volume = jnp.pi * b_r**2 * (b_x - a_x)
        u, w, logli = super().sample(betas)

        u_r_decompose = u[..., 1][..., jnp.newaxis] / 2**0.5
        u3d = jnp.concatenate(
            (u[..., 0][..., jnp.newaxis], u_r_decompose, u_r_decompose),
            axis=-1)

        #logli_polar = jnp.log( 2*jnp.pi*x[...,1]/volume )
        log_likeilihoods = jnp.broadcast_to(jnp.array([jnp.log(1 / volume)]),
                                            w.shape)
        return u3d, w, log_likeilihoods


if __name__ == "__main__":
    pass
    '''
    constant = {"m": 1., "kB": 1.}
    sampler = Gauss_Legendre_Sampler_Polar3D(n_x = 8,n_r = 8, B_x = 16, B_r = 16)

    br = 2
    l = 3
    x, w, logli = sampler.sample([0, l, br])
    #print(w.shape)
    #print(jnp.sum(w))
    #print(x[:,:,1])
    #x,w,logli = sampler.sample([1.0,0.0], n_x = 8, n_r = 8, B_x = 16, B_r = 16)
    print(8 * 16 * 8 * 16)
    print(x.shape)
    #print( jnp.sum(w/x[...,1]) -2/br)
    print(jnp.exp(logli) - 1 / (jnp.pi * br**2 * l))
    #print( jnp.sum(w*x[:,0]) )
    print(jnp.sum(w * x[:, 0] * (x[:, 1]**2 + x[:, 2]**2)**0.5))
    #print( jnp.sum(w*x[:,1]**2) )
    '''