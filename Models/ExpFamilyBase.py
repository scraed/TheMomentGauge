from MomentGauge.Sampler.QuadratureSampler import Gauss_Legendre_Sampler_Polar3D
from MomentGauge.Sampler.CanonicalExpFamilySampler import CanonicalExpImportanceSampler,CanonicalExpSampler
from MomentGauge.Sampler.ExponentialFamilySampler import ExponentialFamilySampler, ExpFamilyImportanceSampler
from MomentGauge.Statistic.PolyStatistics import PolyStatistics, Maxwellian_1D_stats
from MomentGauge.Statistic.PolyGaugedStatistics import M35_1D_gauged_stats, PolyGaugedStatistics
import jax
import jax.numpy as jnp
from MomentGauge.Estimator.Estimator import BaseEstimator, EstimatorPolar2D
from MomentGauge.Optim.NewtonOptimizer import Newton_Optimizer
from MomentGauge.Optim.BaseOptimizer import BaseOptimizer
from collections import OrderedDict
from functools import partial
class BaseExpFamilyModel():
    def __init__(self, constant):
        r"""The base class for exponential family model.

        Specifically, the distribution has the form

        .. math::
            :nowrap:

            \begin{equation}
            f(\mathbf{u};\boldsymbol{\beta},\mathbf{g}) = \exp\left( \sum_{i=0}^M \beta_i \phi_i(\mathbf{u},\mathbf{g}) \right)
            \end{equation}

        in which :math:`\{\phi_i,i=0,\cdots,M\}` are sufficient statistics, :math:`\boldsymbol{\beta}` is the natural parameter of the distribution, :math:`\phi_0(\mathbf{u},\mathbf{g}) = 1`, and :math:`\mathbf{g}` is extra gauge parameters that may or may not be requested by the moments :math:`\phi_i`.


        Parameters
        ----------
        constant : dict
            dictionary with the following keys

                **'m'** : float - the mass of particle considered

                **'kB'** : float - the Boltzmann constant
        Attributes
        ----------
        m : float
            the mass of particle considered
        kB : float
            the Boltzmann constant
        constant : dict
            dictionary with the keys containing **'m'** and **'kB'**
        """
        
        self.m = constant["m"]
        self.kB = constant["kB"]
        self.constant = OrderedDict(sorted(constant.items()))  # a dictionary with necessary constants provided as key-value pairs
        self._statistics = PolyStatistics()
        self._sampler = ExponentialFamilySampler(self._statistics.suff_stats)
        self._estimator = BaseEstimator(constant)
        self._optimizer = BaseOptimizer(self._Loss)

    def __hash__(self):
        """Redefine the hash method to include the class attributes. It helps jax.jit to correctly identify class instances"""
        return hash(("BaseExpFamilyModel",*self.constant.items(), self.m,self.kB, self._statistics, self._sampler, self._estimator, self._optimizer))

    def __eq__(self, other):
        """Redefine the eq method to include the class attributes."""
        return (isinstance(other, BaseExpFamilyModel) and
                (*self.constant.items(), self.m,self.kB, self._statistics, self._sampler, self._estimator, self._optimizer) == (*other.constant.items(), other.m,other.kB, other._statistics, other._sampler, other._estimator, other._optimizer))
    @partial(jax.jit,
             static_argnums=0)
    def _Loss(self, betas, moments, gauge_paras, base_args):
        r"""Compute the sample loss given specific moments for solving natural parameters. The natural parameters corresponding to the to specific moments minimize the loss. 
        
        Specific implementation (Children) of BaseExpFamilyModel should replace gauge_paras and base_args to explicit arguments.

        This function is necessary to construct the Newton optimizer. Therefore it must be overwrite for each specific implementation (Children) of BaseExpFamilyModel to implement :meth:`moments_to_natural_paras`

        Parameters
        ----------
        betas : float array of shape (M+1)
            A set of proposed natural parameter :math:`\boldsymbol{\beta}`
        moments : float array of shape (M+1)
            the target moments that we expect exponential familty with the natural parameter :math:`\boldsymbol{\beta}` to have.
        gauge_paras : tuple
            A tuple ( para1, para2, ... ) containing arbitrary many extra parameters, i.e :math:`\mathbf{g}` , required by sufficient statistics :math:`\{\phi_i\}` as **gauge_paras**.
        base_args: tuple
            A tuple ( para1, para2, ... ) containing arbitrary many parameters required by **BaseSampler.sample**.

        Returns
        -------
        float 
            The value of loss
        """
        return self._sampler.sample_Loss( betas, moments, gauge_paras = gauge_paras, base_args = base_args )
    @partial(jax.jit,
             static_argnums=0)
    def _ori_to_canon_parameter_convertion(self, betas, gauge_paras = (), base_args = ()):
        r"""Convert the natural parameter to parameters of other forms
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
        float array of shape (M+1)
            The parameters array :math:`\{ M_0, \cdots, M_M \}` for exponential family model in canonical forms

        """
        return betas
    @partial(jax.jit,
             static_argnums=0)
    def _canon_to_ori_parameter_convertion(self, betas, gauge_paras = (), base_args = ()):
        r"""Convert the natural parameter to parameters of other forms.

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
        float array of shape (M+1)
            The parameters array :math:`\{ M_0, \cdots, M_M \}` for exponential family model in canonical forms

        """
        return betas


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
        phi_values = self._sampler.suff_statistics(u, gauge_paras = gauge_paras)

        return phi_values
    @partial(jax.jit,
             static_argnums=0)
    def moments_to_natural_paras(self, betas_ini, moments, gauge_paras = (), base_args = ()):
        r"""Compute the natural parameters :math:`\boldsymbol{\beta}` from the moments of sufficient statistics.
        
        Specific implementation (Children) of BaseExpFamilyModel should replace gauge_paras and base_args to explicit arguments.

        Parameters
        ----------
        betas : float array of shape (M+1)
            A set of proposed natural parameter :math:`\boldsymbol{\beta}`
        moments : float array of shape (M+1)
            the target moments that we expect exponential familty with the natural parameter :math:`\boldsymbol{\beta}` to have.
        gauge_paras : tuple
            A tuple ( para1, para2, ... ) containing arbitrary many extra parameters, i.e :math:`\mathbf{g}` , required by sufficient statistics :math:`\{\phi_i\}` as **gauge_paras**. Defaut is (), an empty tuple.
        base_args: tuple
            A tuple ( para1, para2, ... ) containing arbitrary many parameters required by **BaseSampler.sample**. Defaut is (), an empty tuple.

        Returns
        -------
        Tuple
            A tuple containing

                **beta**: *float array of shape (M+1)* - The natural parameters :math:`\boldsymbol{\beta}`

                **opt_info**: *tuple* - A tuple containing other information of the optimization process

                    **values**: *float* - the optimal value of target_function.
            
                    **residuals**: *float* - the residual of the optimization. 

                    **step**: *float* - the total number of Newton's step iteration.

                    **bsteps**: *float* - the total number of Backtracking step.
        """       
        aux_paras = (moments, gauge_paras, base_args)
        beta, opt_info = self._optimizer.optimize( betas_ini, *aux_paras )
        return beta, opt_info
    @partial(jax.jit,
             static_argnums=0)
    def natural_paras_to_moments(self, betas, gauge_paras = (), base_args = ()):
        r"""Compute the moments of sufficient statistics given natural parameters. 
        
        Specific implementation of BaseExpFamilyModel should replace gauge_paras and base_args to explicit arguments.

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
        float array of shape (M+1)
            The array :math:`\{ M_0, \cdots, M_M \}`

        """
        suff_statistics = lambda u,*gauge_paras: self._sampler.suff_statistics( u, gauge_paras = gauge_paras )
        x,w,logli = self._sampler.sample(betas, gauge_paras = gauge_paras, base_args = base_args )
        moments = self._estimator.get_sample_moment(suff_statistics, x, w, gauge_paras=gauge_paras)
        return moments
    def natural_paras_to_custom_moments(self, betas, statistics, gauge_paras = (), base_args = (), stats_gauge_paras = ()):
        r"""Compute the moments of custom statistics given natural parameters. 
        
        Specific implementation of BaseExpFamilyModel should replace gauge_paras and base_args to explicit arguments.

        Parameters
        ----------
        betas : float array of shape (M+1)
            the natural parameter :math:`\boldsymbol{\beta}` of the distribution
        statistics : function
            A float-valued or tensor-valued function :math:`\phi_{i_1,\cdots,i_k}` ( **u** , :math:`*` **gauge_paras** ) with
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector :math:`\mathbf{u}`

                    :math:`*` **gauge_paras** : - Arbitrary many extra parameters such as :math:`\mathbf{g}`. The :math:`*` refers to the unpacking operator in python.

                **Returns**: 
                
                    float or array of arbitrary shape :math:`(d_1,\cdots,d_k)` -- the value of the statistic :math:`\phi_{i_1,\cdots,i_k}(\mathbf{u},\mathbf{g})` 
        gauge_paras : tuple
            A tuple ( para1, para2, ... ) containing arbitrary many extra parameters, i.e :math:`\mathbf{g}` , required by sufficient statistics as **gauge_paras**. Defaut is (), an empty tuple.
        stats_gauge_paras : tuple
            A tuple ( para1, para2, ... ) containing arbitrary many extra parameters the custom statistics function :math:`\phi_{i_1,\cdots,i_k}` required as **gauge_paras**. Defaut is (), an empty tuple.
        base_args: tuple
            A tuple ( para1, para2, ... ) containing arbitrary many parameters required by **BaseSampler.sample**. Defaut is (), an empty tuple.



        Returns
        -------
        float or array of arbitrary shape :math:`(d_1,\cdots,d_k)`
            The moment value :math:`M_{i_1,\cdots,i_k}(\mathbf{g})`

        """
        #cus_statistics = lambda u,*stat_gauge_paras: statistics( u, gauge_paras = stat_gauge_paras )
        x,w,logli = self._sampler.sample(betas, gauge_paras = gauge_paras, base_args = base_args )
        moments = self._estimator.get_sample_moment(statistics, x, w, gauge_paras=stats_gauge_paras)
        return moments
    @partial(jax.jit,
             static_argnums=0)
    def natural_paras_to_fluid_properties(self, betas, gauge_paras = (), base_args = ()):
        r"""
        Compute the fluid properties including number density :math:`n`, density :math:`\rho`, flow velocities :math:`\mathbf{v} = \{v_\alpha, \alpha \in \{x,y,z\}\}`, temperature :math:`T`, pressure :math:`p`, stress :math:`\{\sigma_{\alpha \beta}, \alpha, \beta \in \{x,y,z\}\}` and heat flux :math:`\{q_{\alpha}, \alpha \in \{x,y,z\}\}`.

        .. math::
            :nowrap:

            \begin{equation} 
            \begin{split}
            n &= \frac{\rho}{m}= \int f(\mathbf{u}) d^3 \mathbf{u} \\
            v_\alpha &= \frac{1}{n}\int u_\alpha f(\mathbf{u}) d^3 \mathbf{u}\\
            p  &= n k_B T = \frac{1}{3} \int m c_\alpha c_\alpha f(\mathbf{u}) d^3 \mathbf{u} \\
            \sigma_{\alpha\beta} &= \int m c_\alpha c_\beta f(\mathbf{u}) d^3 \mathbf{u} - p \delta_{\alpha\beta} \\
            \epsilon  &= \frac{3}{2} k_B T  = \frac{1}{n}\int \frac{m}{2} \mathbf{c}^2 f(\mathbf{u}) d^3 \mathbf{u} \\
            q_\alpha &= \int \frac{m}{2} c_\alpha \mathbf{c}^2 f(\mathbf{u}) d^3 \mathbf{u}; \quad \alpha, \beta \in \{x,y,z\}
            \end{split}
            \end{equation}

        in which :math:`m` is the mass of gas molecule.


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
        float array of shape (16)
            Array containing macroscopic quantities :math:`\{ \rho, n, v_x, v_y, v_z, T, p, \sigma_{xx}, \sigma_{xy}, \sigma_{xz}, \sigma_{yy}, \sigma_{yz}, \sigma_{zz}, q_x, q_y, q_z \}`
        """
        suff_statistics = lambda u,*gauge_paras: self._sampler.suff_statistics( u, gauge_paras = gauge_paras )
        x,w,logli = self._sampler.sample(betas, gauge_paras = gauge_paras, base_args = base_args )
        
        props = self._estimator.cal_macro_quant(x, w)
        return props

class CanonicalExpFamilyModel(BaseExpFamilyModel):
    def __init__(self, constant):
        r"""The base class for exponential family model using canonical form samplers. 
        
        During sampling, the natural parameter are first converted to fit canonical forms.

        Specifically, the distribution has the form

        .. math::
            :nowrap:

            \begin{equation}
            \begin{split}
            f(\mathbf{u};\boldsymbol{\beta},\mathbf{g}) &= \frac{\beta_0}{Z(\boldsymbol{\beta},\mathbf{g})} \exp\left( \sum_{i=1}^M \beta_i \phi_i(\mathbf{u},\mathbf{g})  \right)\\
            Z(\boldsymbol{\beta},\mathbf{g}) &= \int \exp\left( \sum_{i=1}^M \beta_i \phi_i(\mathbf{u},\mathbf{g})  \right) d\mathbf{u}
            \end{split}
            \end{equation}

        in which :math:`\{\phi_i,i=0,\cdots,M\}` are sufficient statistics, :math:`\boldsymbol{\beta}` is the natural parameter of the distribution, :math:`\phi_0(\mathbf{u},\mathbf{g}) = 1`, and :math:`\mathbf{g}` is extra gauge parameters that may or may not be requested by the moments :math:`\phi_i`.


        Parameters
        ----------
        constant : dict
            dictionary with the following keys

                **'m'** : float - the mass of particle considered

                **'kB'** : float - the Boltzmann constant
        Attributes
        ----------
        m : float
            the mass of particle considered
        kB : float
            the Boltzmann constant
        constant : dict
            dictionary with the keys containing **'m'** and **'kB'**
        """
        super().__init__(constant)
    @partial(jax.jit,
             static_argnums=0)
    def _ori_to_canon_parameter_convertion(self, betas, gauge_paras = (), base_args = ()):
        r"""Convert the natural parameter to parameters of canonical forms, if exp family model in canonical form :class:`MomentGauge.Sampler.CanonicalExpFamilySampler` are used

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
        float array of shape (M+1)
            The parameters array :math:`\{ M_0, \cdots, M_M \}` for exponential family model in canonical forms

        """
        logZ = self._sampler.LogPartition( betas.at[0].set(0), gauge_paras = gauge_paras,  base_args = base_args )
        #print("logZ",logZ)
        #jax.debug.print("logZ: {x}", x=logZ, ordered=True)
        betas = betas.at[0].set( jnp.exp( betas[0] + logZ ) )
        return betas
    @partial(jax.jit,
             static_argnums=0)
    def _canon_to_ori_parameter_convertion(self, betas, gauge_paras = (), base_args = ()):
        r"""Convert the natural parameter to parameters of canonical forms, if exp family model in canonical form :class:`MomentGauge.Sampler.CanonicalExpFamilySampler` are used

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
        float array of shape (M+1)
            The parameters array :math:`\{ M_0, \cdots, M_M \}` for exponential family model in canonical forms

        """
        n = betas[0]
        logZ = self._sampler.LogPartition( betas.at[0].set(0), gauge_paras = gauge_paras,  base_args = base_args )
        betas = betas.at[0].set( jnp.log(n) - logZ )
        return betas
    @partial(jax.jit,
             static_argnums=0)
    def moments_to_natural_paras(self, betas_ini, moments, gauge_paras = (), base_args = ()):
        r"""Compute the natural parameters :math:`\boldsymbol{\beta}` from the moments of sufficient statistics.
        
        Specific implementation (Children) of BaseExpFamilyModel should replace gauge_paras and base_args to explicit arguments.

        Parameters
        ----------
        betas : float array of shape (M+1)
            A set of proposed natural parameter :math:`\boldsymbol{\beta}`
        moments : float array of shape (M+1)
            the target moments that we expect exponential familty with the natural parameter :math:`\boldsymbol{\beta}` to have.
        gauge_paras : tuple
            A tuple ( para1, para2, ... ) containing arbitrary many extra parameters, i.e :math:`\mathbf{g}` , required by sufficient statistics :math:`\{\phi_i\}` as **gauge_paras**. Defaut is (), an empty tuple.
        base_args: tuple
            A tuple ( para1, para2, ... ) containing arbitrary many parameters required by **BaseSampler.sample**. Defaut is (), an empty tuple.

        Returns
        -------
        Tuple
            A tuple containing

                **beta**: *float array of shape (M+1)* - The natural parameters :math:`\boldsymbol{\beta}`

                **opt_info**: *tuple* - A tuple containing other information of the optimization process

                    **values**: *float* - the optimal value of target_function.
            
                    **residuals**: *float* - the residual of the optimization. 

                    **step**: *float* - the total number of Newton's step iteration.

                    **bsteps**: *float* - the total number of Backtracking step.
        """       
        #betas_ini = self._ori_to_canon_parameter_convertion(betas_ini, gauge_paras = gauge_paras, base_args = base_args)
        #print(betas_ini)
        beta, opt_info = super().moments_to_natural_paras(betas_ini.at[0].set(moments[0]), moments, gauge_paras = gauge_paras, base_args = base_args)
        #print(beta)
        #beta = self._canon_to_ori_parameter_convertion(beta, gauge_paras = gauge_paras, base_args = base_args)
        return beta, opt_info

class GaugedExpFamilyModel(BaseExpFamilyModel):
    def __init__(self, constant):
        r"""The base class for exponential family model with Gauge transformations. 
        
        It add two methods :meth:`moments_gauge_transformation` and :meth:`natural_paras_gauge_transformation` to :class:`BaseExpFamilyModel`

        Parameters
        ----------
        constant : dict
            dictionary with the following keys

                **'m'** : float - the mass of particle considered

                **'kB'** : float - the Boltzmann constant
        Attributes
        ----------
        m : float
            the mass of particle considered
        kB : float
            the Boltzmann constant
        constant : dict
            dictionary with the keys containing **'m'** and **'kB'**
        """
        super().__init__(constant)
        self._statistics = PolyGaugedStatistics(self._statistics)
    @partial(jax.jit,
             static_argnums=0)
    def moments_gauge_transformation(self,moments, gauge_para2 = (), gauge_para1 = (), base_args = ()):
        r"""Gauge transformation for moments of sufficient statistics. The transformation is defined as
        
        .. math::
            :nowrap:

            \begin{equation}
                M_i(\mathbf{g}' )= T_{ij}(\mathbf{g}',\mathbf{g}) M_j(\mathbf{g}); \quad i,j = 0, \cdots, M
            \end{equation}       

        which is induced from the gauge transformation :math:`T_{ij}(\mathbf{g}',\mathbf{g})` between sufficient statistics :math:`\phi_i(\mathbf{u},\mathbf{g})` such that

        .. math::
            :nowrap:

            \begin{equation}
                \phi_i(\mathbf{u}, \mathbf{g}' )= T_{ij}(\mathbf{g}',\mathbf{g}) \phi_j(\mathbf{u}, \mathbf{g}); \quad i,j = 0, \cdots, M
            \end{equation}

        with :math:`\phi_i(\mathbf{u},\mathbf{g})` is sufficient statistics parameterized by gauge parameters :math:`\mathbf{g}`.

        Parameters
        ----------
        moments : float array of shape (M+1)
            the moments :math:`M_i(\mathbf{g})` of sufficient statistics
        gauge_para2 : tuple
            Tuple containing arbitrary many extra gauge parameters such as :math:`\mathbf{g}'`
        gauge_para1 : tuple
            Tuple containing arbitrary many extra gauge parameters such as :math:`\mathbf{g}`
        base_args: tuple
            A tuple ( para1, para2, ... ) containing arbitrary many parameters required by **BaseSampler.sample**. Defaut is (), an empty tuple.



        Returns
        -------
        float array of shape (M+1)
            the moments :math:`M_i(\mathbf{g}')` of sufficient statistics
        """
        Tmat = self._statistics.gauge_transformation_matrix(gauge_para2 = gauge_para2, gauge_para1 = gauge_para1)
        return Tmat.dot(moments)
    @partial(jax.jit,
             static_argnums=0)
    def natural_paras_gauge_transformation(self,betas, gauge_para2 = (), gauge_para1 = (), base_args = ()):
        r"""Gauge transformation for natural parameters. The transformation is defined as
        
        .. math::
            :nowrap:

            \begin{equation}
                \beta_i(\mathbf{g}' )= T_{ij}^T(\mathbf{g},\mathbf{g}') \beta_j(\mathbf{g}); \quad i,j = 0, \cdots, M
            \end{equation}       

        which is induced from the gauge transformation :math:`T_{ij}(\mathbf{g}',\mathbf{g})` between sufficient statistics :math:`\phi_i(\mathbf{u},\mathbf{g})` such that

        .. math::
            :nowrap:

            \begin{equation}
                \phi_i(\mathbf{u}, \mathbf{g}' )= T_{ij}(\mathbf{g}',\mathbf{g}) \phi_j(\mathbf{u}, \mathbf{g}); \quad i,j = 0, \cdots, M
            \end{equation}

        with :math:`\phi_i(\mathbf{u},\mathbf{g})` is sufficient statistics parameterized by gauge parameters :math:`\mathbf{g}`.

        Parameters
        ----------
        betas : float array of shape (M+1)
            the moments :math:`\beta_i(\mathbf{g})` of sufficient statistics
        gauge_para2 : tuple
            Tuple containing arbitrary many extra gauge parameters such as :math:`\mathbf{g}'`
        gauge_para1 : tuple
            Tuple containing arbitrary many extra gauge parameters such as :math:`\mathbf{g}`
        base_args: tuple
            A tuple ( para1, para2, ... ) containing arbitrary many parameters required by **BaseSampler.sample**. Defaut is (), an empty tuple.


        Returns
        -------
        float array of shape (M+1)
            the moments :math:`\beta_i(\mathbf{g}')` of sufficient statistics
        """

        Tmat = self._statistics.gauge_transformation_matrix(gauge_para2 = gauge_para1, gauge_para1 = gauge_para2)

        betas = self._canon_to_ori_parameter_convertion(betas, gauge_paras = gauge_para1, base_args = base_args)

        betas = Tmat.T.dot(betas)

        betas = self._ori_to_canon_parameter_convertion(betas, gauge_paras = gauge_para2, base_args = base_args)
        
        return betas
    @partial(jax.jit,
             static_argnums=0)
    def standard_gauge_para_from_moments(self, moments, gauge_para = ()):
        r"""the standard gauge parameters :math:`\mathbf{g}` prefered among all possible gauges.
        
        Parameters
        ----------
        moments : float array of shape (N)
            The array containing moments of sufficient statistics given the gauge parameters :math:`\mathbf{g}`
        gauge_para : tuple
            Tuple containing arbitrary many extra gauge parameters such as :math:`\mathbf{g}`

        Returns
        -------
        float array
            the standard gauge parameters :math:`\mathbf{g}` prefered among all possible gauges.
        """
        
        return self._statistics.standard_gauge_paras(moments, gauge_para=gauge_para)
