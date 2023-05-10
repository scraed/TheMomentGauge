from MomentGauge.Sampler.Base import BaseSampler
import jax.numpy as jnp
import jax
from jax.nn import softmax
from jax import vmap
from jax.scipy.special import logsumexp
from functools import partial
class ExponentialFamilySampler(BaseSampler):
    def __init__(self,suff_stats):
        r"""The base class for exponential family sampler.
        A sampler of the probability distribution :math:`f(\mathbf{u};\boldsymbol{\beta})` parametrized by :math:`\boldsymbol{\beta}` from which we could draw samples and compute likelihoods. 
        Specifically, the distribution has the form

        .. math::
            :nowrap:

            \begin{equation}
            f(\mathbf{u};\boldsymbol{\beta},\mathbf{g}) = \exp\left( \sum_{i=0}^M \beta_i \phi_i(\mathbf{u},\mathbf{g}) \right)
            \end{equation}

        in which :math:`\{\phi_i,i=0,\cdots,M\}` are sufficient statistics, :math:`\boldsymbol{\beta}` is the natural parameter of the distribution, :math:`\phi_0(\mathbf{u},\mathbf{g}) = 1`, and :math:`\mathbf{g}` is extra gauge parameters that may or may not be requested by the moments :math:`\phi_i`.


        Parameters
        ----------
        suff_stats : list
            a list of moment functions [:math:`\phi_i,i=0,\cdots,M`], in which each :math:`\phi_i` is a function :math:`\phi_i` ( **u** , :math:`*` **gauge_paras** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector 

                    :math:`*` **gauge_paras** : - Arbitrary many extra parameters. The :math:`*` refers to the unpacking operator in python.

                **Returns**: 
                
                    float -- the moment value
            
            The lengh of the list may vary. Its first element must satisfy :math:`\phi_0` ( **u** , :math:`*` **gauge_paras** ) = 1


        Attributes
        ----------
        constant : dict
            an empty dict
        num_statistics : int
            The number of sufficient statistics
        """
        super().__init__({})
        num_statistics = len(suff_stats)
        self._suff_stats_list = suff_stats
        self.num_statistics = num_statistics
        #self.sufficient_statistic_list = suff_stats
        sufficient_statistic = lambda i,u_and_args : jax.lax.switch( i, suff_stats, *u_and_args  )
        #self.sufficient_statistic = sufficient_statistic
        self._sufficient_statistics = lambda u_and_args: vmap( sufficient_statistic, in_axes = (0,None) )(jnp.arange(num_statistics),u_and_args)

    def __hash__(self):
        """Redefine the hash method to include the class attributes. It helps jax.jit to correctly identify class instances"""
        return hash(("ExponentialFamilySampler",*self._suff_stats_list,))

    def __eq__(self, other):
        """Redefine the eq method to include the class attributes."""
        return (isinstance(other, ExponentialFamilySampler) and
                (*self._suff_stats_list,) == (*other._suff_stats_list,))

    @partial(jax.jit,
             static_argnums=0)
    def suff_statistics(self, u, gauge_paras=()):
        r"""Compute the value of sufficient statistics at certain :math:`\mathbf{u}`

        Parameters
        ----------
        u : float array of shape (3)
            The 3D sample vector 
        gauge_paras : tuple
            A tuple ( para1, para2, ... ) containing arbitrary many extra parameters, i.e :math:`\mathbf{g}` , required by sufficient statistics :math:`\{\phi_i\}` as **gauge_paras**. Defaut is (), an empty tuple.

        Returns
        -------
        float array of shape (M+1)
            The array :math:`\{ \phi_0(\mathbf{u}), \cdots, \phi_M(\mathbf{u}) \}`

        """
        phi_values = self._sufficient_statistics((u,*gauge_paras))

        return phi_values
    @partial(jax.jit,
             static_argnums=0)
    def sample(self,betas, gauge_paras = ()):
        r"""Generate N samples :math:`\mathbf{u}_i` from the distribution :math:`f(\mathbf{u})` with proper weights :math:`w_i` such that
        
        .. math::
            :nowrap:

            \begin{equation}
            \int \psi(\mathbf{u}) f(\mathbf{u};\boldsymbol{\beta},\mathbf{g}) d \mathbf{u} \approx \sum_{i=1}^N w_i(\boldsymbol{\beta},\mathbf{g}) \psi(\mathbf{u}_i (\boldsymbol{\beta},\mathbf{g}) ),
            \end{equation}

        in which :math:`\psi` is arbitrary test function, :math:`\boldsymbol{\beta}` is the natural parameter, :math:`\mathbf{g}` is extra gauge parameters that may or may not be requested, and N depends on the particular implementation of the sampler.

        Parameters
        ----------
        betas : float array of shape (M+1)
            the natural parameter :math:`\boldsymbol{\beta}` of the distribution
        gauge_paras : tuple
            A tuple ( para1, para2, ... ) containing arbitrary many extra parameters, i.e :math:`\mathbf{g}` , required by sufficient statistics :math:`\{\phi_i\}` as **gauge_paras**. Defaut is (), an empty tuple.

        Returns
        -------
        Tuple
            A tuple containing

                **samples**: *float array of shape (N,3)* - N  samples of 3-dim vectors :math:`\mathbf{u}_i` draw from the distribution. 

                **weights**: *float array of shape (N)* - N non-negative weights :math:`w_i` for each samples.
        
                **log_likelihoods**: *float array of shape (N)* - N the log-likelihoods :math:`\log f(\mathbf{u}_i)` for each samples
            
            in which N is determined by the specific implementation.
        """
        raise NotImplementedError
    @partial(jax.jit,
             static_argnums=0)
    def LogProb(self, betas, u, gauge_paras = ()):
        r"""
        Calculate the log-likelihood of the distribution :math:`\log f(\mathbf{u};\boldsymbol{\beta},\mathbf{g})` at a certain sample point :math:`\mathbf{u}`

        Parameters
        ----------
        betas : float array of shape (M+1)
            the natural parameter :math:`\boldsymbol{\beta}` of the distribution
        u : float array of shape (3)
            the sample point :math:`\mathbf{u}` at which the likelihood is evaluated.
        gauge_paras : tuple
            A tuple ( para1, para2, ... ) containing arbitrary many extra parameters, i.e :math:`\mathbf{g}` , required by sufficient statistics :math:`\{\phi_i\}` as **gauge_paras**. Defaut is (), an empty tuple.

        Returns
        -------
        float
            the likelihood :math:`\sum_{i=0}^M \beta_i \phi_i(\mathbf{u},\mathbf{g})`
        """
        para = betas
        phi_values = self.suff_statistics(u,gauge_paras = gauge_paras) # The value of each phi_i(u) for phi_i in sufficient_statistics
        #self._sufficient_statistics((u,*gauge_paras)) 
        #print(phi_values.shape)
        #print(para.shape)
        return phi_values.dot(para)
    def sample_Loss(self, betas, moments, gauge_paras = (), base_args = ()):
        r"""
        The optimization objective loss as a function of parameters :math:`\boldsymbol{\beta}`, moments of sufficient statistics :math:`\mathbf{M}` and necessary gauge parameters. 
        
        Minimizing the optimization objective given a set of moments yields the corresponding parameters :math:`\boldsymbol{\beta}`.

        Parameters
        ----------
        betas : float array of shape (M+1)
            the natural parameter :math:`\boldsymbol{\beta}` of the distribution
        moments : float array of shape (M+1)
            the target moments we wish the distribution to have as moments of sufficient statistics.
        gauge_paras : tuple
            A tuple ( para1, para2, ... ) containing arbitrary many extra parameters, i.e :math:`\mathbf{g}` , required by sufficient statistics :math:`\{\phi_i\}` as **gauge_paras**. Defaut is (), an empty tuple.
        base_args: tuple
            A tuple ( para1, para2, ... ) containing arbitrary many parameters required by **BaseSampler.sample**. Defaut is (), an empty tuple.

        Returns
        -------
        float
            the loss value :math:`L`
        """
        raise NotImplementedError

class ExpFamilyImportanceSampler(ExponentialFamilySampler):
    def __init__(self,suff_stats, baseSampler: BaseSampler):
        r"""The sampler for exponential family by importance sampling.
        A sampler of the probability distribution :math:`f(\mathbf{u};\boldsymbol{\beta})` parametrized by :math:`\boldsymbol{\beta}` from which we could draw samples and compute likelihoods. 
        Specifically, the distribution has the form

        .. math::
            :nowrap:

            \begin{equation}
            f(\mathbf{u};\boldsymbol{\beta},\mathbf{g}) = \exp\left( \sum_{i=0}^M \beta_i \phi_i(\mathbf{u},\mathbf{g}) \right)
            \end{equation}

        in which :math:`\{\phi_i\}` are sufficient statistics, :math:`\boldsymbol{\beta}` is the natural parameter of the distribution, :math:`\phi_0(\mathbf{u},\mathbf{g}) = 1`, and :math:`\mathbf{g}` is extra gauge parameters that may or may not be requested by the moments :math:`\phi_i`.


        Parameters
        ----------
        suff_stats : list
            a list of moment functions [:math:`\phi_i,i=0,\cdots,M`], in which each :math:`\phi_i` is a function :math:`\phi_i` ( **u** , :math:`*` **gauge_paras** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector 

                    :math:`*` **gauge_paras** : - Arbitrary many extra parameters. The :math:`*` refers to the unpacking operator in python.

                **Returns**: 
                
                    float -- the moment value
            
            The lengh of the list may vary. Its first element must satisfy :math:`\phi_0` ( **u** , :math:`*` **gauge_paras** ) = 1
        baseSampler : An instance of :class:`Sampler.Base.BaseSampler`
            The primary sampler used by importance sampling to generate samples. Its method :meth:`sample` must have benn implemented.

        Attributes
        ----------
        constant : dict
            an empty dict
        num_statistics : int
            The number of sufficient statistics
        """
        super().__init__(suff_stats)
        self.__BaseSampler = baseSampler

    def __hash__(self):
        """Redefine the hash method to include the class attributes. It helps jax.jit to correctly identify class instances"""
        return hash(("ExpFamilyImportanceSampler",*self._suff_stats_list, self.__BaseSampler))

    def __eq__(self, other):
        """Redefine the eq method to include the class attributes."""
        return (isinstance(other, ExpFamilyImportanceSampler) and
                (*self._suff_stats_list, self.__BaseSampler) == (*other._suff_stats_list, other.__BaseSampler))
    @partial(jax.jit,
             static_argnums=0)
    def sample(self, betas, gauge_paras = (), base_args = ()):
        r"""Generate N samples :math:`\mathbf{u}_i` by importance sampling from the distribution :math:`f(\mathbf{u})` with proper weights :math:`w_i` such that
        
        .. math::
            :nowrap:

            \begin{equation}
            \int \psi(\mathbf{u}) f(\mathbf{u};\boldsymbol{\beta},\mathbf{g}) d \mathbf{u} \approx \sum_{i=1}^N w_i(\boldsymbol{\beta},\mathbf{g}) \psi(\mathbf{u}_i (\boldsymbol{\beta},\mathbf{g}) ),
            \end{equation}

        in which :math:`\psi` is arbitrary test function, :math:`\boldsymbol{\beta}` is the natural parameter, :math:`\mathbf{g}` is extra gauge parameters that may or may not be requested, and N depends on the particular implementation of the sampler.

        Parameters
        ----------
        betas : float array of shape (M+1)
            the natural parameter :math:`\boldsymbol{\beta}` of the distribution
        gauge_paras : tuple
            A tuple ( para1, para2, ... ) containing arbitrary many extra parameters, i.e :math:`\mathbf{g}` , required by sufficient statistics :math:`\{\phi_i\}` as **gauge_paras**. Defaut is (), an empty tuple.
        base_args: tuple
            A tuple ( para1, para2, ... ) containing arbitrary many parameters required by **BaseSampler.sample**. Defaut is (), an empty tuple.

        Returns
        -------
        Tuple
            A tuple containing

                **samples**: *float array of shape (N,3)* - N  samples of 3-dim vectors :math:`\mathbf{u}_i` draw from the distribution. 

                **weights**: *float array of shape (N)* - N non-negative weights :math:`w_i` for each samples.
        
                **log_likelihoods**: *float array of shape (N)* - N the log-likelihoods :math:`\log f(\mathbf{u}_i)` for each samples
            
            in which N is determined by the specific implementation.
        """
        u3d, base_w, base_logli = self.__BaseSampler.sample(*base_args)

        ######Compute the log_likelihood of u##############
        log_likelihood = vmap( lambda betas, u3d, gauge_paras: self.LogProb(betas,u3d,gauge_paras=gauge_paras), in_axes=(None,0,None) )( betas,  u3d , gauge_paras)
        ###################################################
        w = base_w*jnp.exp( log_likelihood - base_logli )

        return u3d, w, log_likelihood
    @partial(jax.jit,
             static_argnums=0)
    def sample_Loss(self, betas, moments, gauge_paras = (), base_args = ()):
        r"""
        Approximate the entropy loss of exponential family distribution

        .. math::
            :nowrap:

            \begin{equation}
            \begin{split}
            L(\boldsymbol{\beta},\mathbf{g}, \mathbf{M}) &= \int f(\mathbf{u};\boldsymbol{\beta},\mathbf{g}) d\mathbf{u} - \sum_{i=0}^M \beta_i M_i\\
                &\approx \sum_{i=1}^{N} w_i(\boldsymbol{\beta},\mathbf{g}) - \sum_{i=0}^M \beta_i M_i
            \end{split}
            \end{equation}

        by sampling weights :math:`w_i`, provided the natural parameters :math:`\boldsymbol{\beta}`, the moments :math:`\mathbf{M}` and necessary gauge parameters.

        Parameters
        ----------
        betas : float array of shape (M+1)
            the natural parameter :math:`\boldsymbol{\beta}` of the distribution
        moments : float array of shape (M+1)
            the target moments we wish the distribution to have as moments of sufficient statistics.
        gauge_paras : tuple
            A tuple ( para1, para2, ... ) containing arbitrary many extra parameters, i.e :math:`\mathbf{g}` , required by sufficient statistics :math:`\{\phi_i\}` as **gauge_paras**. Defaut is (), an empty tuple.
        base_args: tuple
            A tuple ( para1, para2, ... ) containing arbitrary many parameters required by **BaseSampler.sample**. Defaut is (), an empty tuple.

        Returns
        -------
        float
            the loss value :math:`L`
        """
        samples, weights,_ = self.sample( betas, gauge_paras = gauge_paras, base_args = base_args)
        return jnp.sum( weights ) - betas.dot(moments)


'''
if __name__ == "__main__":
    from QuadratureSampler import Gauss_Legendre_Sampler_Polar3D
    Qsampler = Gauss_Legendre_Sampler_Polar3D(n_x = 8,n_r = 8, B_x = 16, B_r = 16)
    suff_moments = [lambda u: 1., lambda u: u[0], lambda u: u[0]**2]
    Mom = jnp.array([1,0,1.])
    beta = jnp.array( [1.,0,0] )
    sampler = ExpFamilyImportanceSampler(suff_moments, Qsampler )
    x,w,logli = sampler.sample(beta, gauge_paras = (), base_args = (jnp.array([0., 1, 1]),))
    L= sampler.sample_Loss(beta, Mom, gauge_paras = (), base_args = (jnp.array([0., 1, 1]),))
    print(x.shape)
    print(w.shape)
    print(logli)
    print(L)
'''
