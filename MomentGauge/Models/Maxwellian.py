from MomentGauge.Sampler.QuadratureSampler import Gauss_Legendre_Sampler_Polar3D
from MomentGauge.Sampler.CanonicalExpFamilySampler import CanonicalExpImportanceSampler,CanonicalExpSampler
from MomentGauge.Sampler.ExponentialFamilySampler import ExpFamilyImportanceSampler
from MomentGauge.Statistic.PolyStatistics import Maxwellian_1D_stats
from MomentGauge.Statistic.PolyGaugedStatistics import PolyGaugedStatistics, M35_1D_gauged_stats, Maxwellian_1D_gauged_stats
from MomentGauge.Models.ExpFamilyBase import BaseExpFamilyModel, CanonicalExpFamilyModel, GaugedExpFamilyModel
import jax
import jax.numpy as jnp
from MomentGauge.Estimator.Estimator import EstimatorPolar2D
from MomentGauge.Optim.NewtonOptimizer import Newton_Optimizer
from MomentGauge.Optim.BaseOptimizer import BaseOptimizer
from functools import partial


class Maxwell_Legendre_1D(BaseExpFamilyModel):
    def __init__(self, constant):
        r"""The 1D Maxwell distribution.

        .. math::
            :nowrap:

            \begin{equation}
            \begin{split}
            f_0(\mathbf{u}) &= \exp\left( \beta_0 \phi_0(\mathbf{u}) + \beta_1 \phi_1(\mathbf{u}) + \beta_2 \phi_2(\mathbf{u} ) \right)\\
                    &= \frac{\rho}{m} \exp\left( \frac{m  v_x}{k_B T}\phi_1(\mathbf{u}) -\frac{m}{2 k_B T} \phi_2(\mathbf{u}) -  \left(\frac{m v_x^2}{2k_B T}+ \log \left(\frac{2\pi  k_B T}{m}\right)^{\frac{3}{2}}\right) \right),
            \end{split} 
            \end{equation}

        in which :math:`\rho` is the number density, :math:`m` is the molecule velocity, :math:`v_x` is the flow velocity at the x direction, :math:`T` is the temperature, and

            :math:`\phi_0` (**u** ) = 1.

            :math:`\phi_1` (**u** ) = :math:`u_x`
            
            :math:`\phi_2` (**u** ) = :math:`u_x^2 + u_y^2 + u_z^2`.

        Parameters
        ----------
        constant : dict
            dictionary with the following keys

                **'m'** : float - the mass of particle considered

                **'kB'** : float - the Boltzmann constant

                **'n_x'** : int - the order of Gauss Legendre quadrature in x dimension. Required by :class:`MomentGauge.Sampler.QuadratureSampler.Gauss_Legendre_Sampler_Polar3D`

                **'n_r'** : int - the order of Gauss Legendre quadrature in r dimension.

                **'B_x'** : int - how many blocks are splitted in the x dimension for Gauss Legendre quadrature.

                **'B_r'** : int - how many blocks are splitted in the r dimension for Gauss Legendre quadrature.

                **'alpha'**: float - the initial step size used in backtracking line search, default = 1. Required by :class:`MomentGauge.Optim.NewtonOptimizer.Newton_Optimizer`

                **'beta'**: float - the decreasing factor of the step size used in backtracking line search, default = 0.5

                **'c'**: float - the parameter used in the Armijo's condiiton, must lies in (0,1), default = 5e-4

                **'atol'**: float - the absolute error tolerance of the Armijo's condiiton since we use -(atol + rtol*abs(next_value)) instead of 0 to handle single precision numerical error.  default = 5e-6. 

                **'rtol'**: float - the relative error tolerance of the Armijo's condition since we use -(atol + rtol*abs(next_value)) instead of 0 to handle single precision numerical error.  default = 1e-5. 
                
                **'max_iter'**: int - the maximal iteration allowed for the Netwon's method, default = 100

                **'max_back_tracking_iter'**: int - the maximal iteration allowed for the backtracking line search, default = 25

                **'tol'**: float - the tolerance for residual, below which the optimization stops.

                **'min_step_size'**: float - the minimum step size given by back tracking, below which the optimization stops, default = 1e-6.

                **'reg_hessian'**: bool - Regularize the Hessian if the Cholesky decomposition failed. Default = True

                **'debug'**: bool - print debug information if True.

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
        self._Qsampler = Gauss_Legendre_Sampler_Polar3D(n_x = constant['n_x'],n_r = constant['n_r'], B_x = constant['B_x'], B_r = constant['B_r'])
        self._statistics = Maxwellian_1D_stats()
        self._estimator = EstimatorPolar2D(constant)
        self._sampler = ExpFamilyImportanceSampler(self._statistics.suff_stats, self._Qsampler )
        self._optimizer = Newton_Optimizer(self._Loss,alpha = constant["alpha"], beta = constant["beta"], c = constant["c"], atol = constant["atol"], rtol = constant["rtol"],max_iter=constant["max_iter"], max_back_tracking_iter = constant["max_back_tracking_iter"], tol = constant["tol"], min_step_size = constant["min_step_size"], reg_hessian = constant["reg_hessian"], debug = constant["debug"] )
    @partial(jax.jit,
             static_argnums=0)
    def suff_statistics(self, u):
        r"""Compute the value of sufficient statistics at certain :math:`\mathbf{u}`

        Parameters
        ----------
        u : float array of shape (3)
            The 3D sample vector 

        Returns
        -------
        float array of shape (3)
            The array :math:`\{ \phi_0(\mathbf{u}),  \phi_1(\mathbf{u}),  \phi_2(\mathbf{u}) \}`

        """
        gauge_paras = ()
        return super().suff_statistics(u, gauge_paras=gauge_paras)
    @partial(jax.jit,
             static_argnums=0)
    def natural_paras_to_moments(self, betas, domain):
        r"""Compute the moments of sufficient statistics given natural parameters`

        Parameters
        ----------
        betas : float array of shape (3)
            the natural parameter :math:`\boldsymbol{\beta}` of the distribution
        domain : float array of shape (3)
            an array (a_x, b_x, b_r), in which 

                **a_x** : *float* - lower integration limit in x dimension for the Legendre quadrature

                **b_x** : *float* - upper integration limit in x dimension for the Legendre quadrature

                **b_r** : *float* - upper integration limit in r dimension for the Legendre quadrature
            
        Returns
        -------
        float array of shape (M+1)
            The array :math:`\{ M_0, \cdots, M_M \}`

        """
        gauge_paras = ()
        base_args = (domain,)

        return super().natural_paras_to_moments(betas, gauge_paras = gauge_paras, base_args = base_args )
    @partial(jax.jit,
             static_argnums=0)
    def moments_to_natural_paras(self, betas_ini, moments, domain):
        r"""Compute the natural parameters :math:`\boldsymbol{\beta}` from the moments of sufficient statistics.

        Parameters
        ----------
        betas_ini : float array of shape (3)
            A set of proposed natural parameter :math:`\boldsymbol{\beta}`
        moments : float array of shape (3)
            the target moments that we expect exponential familty with the natural parameter :math:`\boldsymbol{\beta}` to have.
        domain : float array of shape (3)
            an array (a_x, b_x, b_r), in which 

                **a_x** : *float* - lower integration limit in x dimension for the Legendre quadrature

                **b_x** : *float* - upper integration limit in x dimension for the Legendre quadrature

                **b_r** : *float* - upper integration limit in r dimension for the Legendre quadrature
            
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
        gauge_paras = ()
        base_args = (domain,)        
        beta, opt_info = super().moments_to_natural_paras( betas_ini, moments, gauge_paras = gauge_paras, base_args = base_args)
        return beta, opt_info
    def natural_paras_to_custom_moments(self, betas, domain, statistics, stats_gauge_paras = ()):
        r"""Compute the moments of custom statistics given natural parameters. 
        
        Specific implementation of BaseExpFamilyModel should replace gauge_paras and base_args to explicit arguments.

        Parameters
        ----------
        betas : float array of shape (3)
            the natural parameter :math:`\boldsymbol{\beta}` of the distribution
        domain : float array of shape (3)
            an array (a_x, b_x, b_r), in which 

                **a_x** : *float* - lower integration limit in x dimension for the Legendre quadrature

                **b_x** : *float* - upper integration limit in x dimension for the Legendre quadrature

                **b_r** : *float* - upper integration limit in r dimension for the Legendre quadrature
        statistics : function
            A float-valued or tensor-valued function :math:`\phi_{i_1,\cdots,i_k}` ( **u** , :math:`*` **gauge_paras** ) with
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector :math:`\mathbf{u}`

                    :math:`*` **gauge_paras** : - Arbitrary many extra parameters such as :math:`\mathbf{g}`. The :math:`*` refers to the unpacking operator in python.

                **Returns**: 
                
                    float or array of arbitrary shape :math:`(d_1,\cdots,d_k)` -- the value of the statistic :math:`\phi_{i_1,\cdots,i_k}(\mathbf{u},\mathbf{g})` 
        stats_gauge_paras : tuple
            A tuple ( para1, para2, ... ) containing arbitrary many extra parameters the custom statistics function :math:`\phi_{i_1,\cdots,i_k}` required. Defaut is (), an empty tuple.
            

        Returns
        -------
        float or array of arbitrary shape :math:`(d_1,\cdots,d_k)`
            The moment value :math:`M_{i_1,\cdots,i_k}(\mathbf{g})`

        """
        gauge_paras = ()
        base_args = (domain,)    
        return super().natural_paras_to_custom_moments(betas, statistics, gauge_paras=gauge_paras, base_args=base_args, stats_gauge_paras=stats_gauge_paras)
    @partial(jax.jit,
             static_argnums=0)
    def natural_paras_to_fluid_properties(self, betas, domain):
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
        betas : float array of shape (3)
            the natural parameter :math:`\boldsymbol{\beta}` of the distribution
        domain : float array of shape (3)
            an array (a_x, b_x, b_r), in which 

                **a_x** : *float* - lower integration limit in x dimension for the Legendre quadrature

                **b_x** : *float* - upper integration limit in x dimension for the Legendre quadrature

                **b_r** : *float* - upper integration limit in r dimension for the Legendre quadrature
            

        Returns
        -------
        float array of shape (16)
            Array containing macroscopic quantities :math:`\{ \rho, n, v_x, v_y, v_z, T, p, \sigma_{xx}, \sigma_{xy}, \sigma_{xz}, \sigma_{yy}, \sigma_{yz}, \sigma_{zz}, q_x, q_y, q_z \}`
        """
        gauge_paras = ()
        base_args = (domain,)
        return super().natural_paras_to_fluid_properties(betas, gauge_paras = gauge_paras, base_args = base_args)
    @partial(jax.jit,
             static_argnums=0)
    def rhoVT_to_natural_paras(self, rhoVT, domain):
        r"""Compute the natural parameters :math:`\boldsymbol{\beta}` from the density :math:`\rho`, flow velocity :math:`v_x` and temperature :math:`T`

        Parameters
        ----------
        rhoVT : float array of shape (3)
            A float array ( :math:`\rho`, :math:`v_x`, :math:`T` ) 

        Returns
        -------
        float array of shape (3)
           The natural parameters :math:`\boldsymbol{\beta}`
        """      
        gauge_paras = ()
        base_args = (domain,)  
        rho, vx, T = rhoVT
        m = self.constant["m"]
        kB = self.constant["kB"]
        n = rho/m
        beta0 = jnp.log(n) -( m*vx**2/(2*kB*T) +3/2*jnp.log(2*jnp.pi*kB*T/m)  )
        beta1 = m*vx/kB/T
        beta2 = -m/(2*kB*T)
        betas = jnp.asarray([beta0, beta1, beta2])

        return betas



class Maxwell_Canonical_Legendre_1D(Maxwell_Legendre_1D,CanonicalExpFamilyModel):
    def __init__(self, constant):
        super().__init__(constant)
        self._sampler = CanonicalExpImportanceSampler(self._statistics.suff_stats, self._Qsampler )
    @partial(jax.jit,
             static_argnums=0)
    def rhoVT_to_natural_paras(self, rhoVT, domain):
        r"""Compute the natural parameters :math:`\boldsymbol{\beta}` from the density :math:`\rho`, flow velocity :math:`v_x` and temperature :math:`T`

        Parameters
        ----------
        rhoVT : float array of shape (3)
            A float array ( :math:`\rho`, :math:`v_x`, :math:`T` ) 

        Returns
        -------
        float array of shape (3)
           The natural parameters :math:`\boldsymbol{\beta}`
        """      
        gauge_paras = ()
        base_args = (domain,)  
        betas = super().rhoVT_to_natural_paras(rhoVT, domain)
        betas = self._ori_to_canon_parameter_convertion(betas, gauge_paras = gauge_paras, base_args = base_args )
        return betas




class Maxwell_Gauged_Legendre_1D(GaugedExpFamilyModel, BaseExpFamilyModel):
    def __init__(self, constant):
        r"""The 1D Maxwell distribution with Gauge transformation.

        .. math::
            :nowrap:

            \begin{equation}
            \begin{split}
            f_0(\mathbf{u}) &= \exp\left( \beta_0 \phi_0(\mathbf{u},s, w_x) + \beta_1 \phi_1(\mathbf{u},s, w_x) + \beta_2 \phi_2(\mathbf{u} ,s, w_x) \right)\\
            \end{split} 
            \end{equation}

        in which :math:`\rho` is the number density, :math:`m` is the molecule velocity, :math:`v_x` is the flow velocity at the x direction, :math:`T` is the temperature, and

            Specifically,

            :math:`\phi_0` (**u**, :math:`s, w_x` ) = 1.

            :math:`\phi_1` (**u**, :math:`s, w_x` ) = :math:`\bar{u}_x`
            
            :math:`\phi_2` (**u**, :math:`s, w_x` ) = :math:`\bar{u}_x^2 + \bar{u}_y^2 + \bar{u}_z^2`.
            
            in which :math:`\bar{u}_x = \frac{u_x - w_x}{s}`, :math:`\bar{u}_y = \frac{u_y}{s}`, :math:`\bar{u}_z = \frac{u_z}{s}`

        Parameters
        ----------
        constant : dict
            dictionary with the following keys

                **'m'** : float - the mass of particle considered

                **'kB'** : float - the Boltzmann constant

                **'n_x'** : int - the order of Gauss Legendre quadrature in x dimension. Required by :class:`MomentGauge.Sampler.QuadratureSampler.Gauss_Legendre_Sampler_Polar3D`

                **'n_r'** : int - the order of Gauss Legendre quadrature in r dimension.

                **'B_x'** : int - how many blocks are splitted in the x dimension for Gauss Legendre quadrature.

                **'B_r'** : int - how many blocks are splitted in the r dimension for Gauss Legendre quadrature.

                **'alpha'**: float - the initial step size used in backtracking line search, default = 1. Required by :class:`MomentGauge.Optim.NewtonOptimizer.Newton_Optimizer`

                **'beta'**: float - the decreasing factor of the step size used in backtracking line search, default = 0.5

                **'c'**: float - the parameter used in the Armijo's condiiton, must lies in (0,1), default = 5e-4

                **'atol'**: float - the absolute error tolerance of the Armijo's condiiton since we use -(atol + rtol*abs(next_value)) instead of 0 to handle single precision numerical error.  default = 5e-6. 

                **'rtol'**: float - the relative error tolerance of the Armijo's condition since we use -(atol + rtol*abs(next_value)) instead of 0 to handle single precision numerical error.  default = 1e-5. 
                
                **'max_iter'**: int - the maximal iteration allowed for the Netwon's method, default = 100

                **'max_back_tracking_iter'**: int - the maximal iteration allowed for the backtracking line search, default = 25

                **'tol'**: float - the tolerance for residual, below which the optimization stops.

                **'min_step_size'**: float - the minimum step size given by back tracking, below which the optimization stops, default = 1e-6.

                **'reg_hessian'**: bool - Regularize the Hessian if the Cholesky decomposition failed. Default = True

                **'debug'**: bool - print debug information if True.

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
        self._Qsampler = Gauss_Legendre_Sampler_Polar3D(n_x = constant['n_x'],n_r = constant['n_r'], B_x = constant['B_x'], B_r = constant['B_r'])
        self._statistics = Maxwellian_1D_gauged_stats()
        self._estimator = EstimatorPolar2D(constant)
        self._sampler = ExpFamilyImportanceSampler(self._statistics.suff_stats, self._Qsampler )
        self._optimizer = Newton_Optimizer(self._Loss,alpha = constant["alpha"], beta = constant["beta"], c = constant["c"], atol = constant["atol"], rtol = constant["rtol"],max_iter=constant["max_iter"], max_back_tracking_iter = constant["max_back_tracking_iter"], tol = constant["tol"], min_step_size = constant["min_step_size"], reg_hessian = constant["reg_hessian"], debug = constant["debug"] )
    @partial(jax.jit,
             static_argnums=0)
    def suff_statistics(self, u, gauge_paras):
        r"""Compute the value of sufficient statistics at certain :math:`\mathbf{u}`

        Parameters
        ----------
        u : float array of shape (3)
            The 3D sample vector 
        gauge_paras : float array of shape (2)  
            The array (:math:`s`, :math:`w_x`).

        Returns
        -------
        float array of shape (3)
            The array :math:`\{ \phi_0(\mathbf{u}, s, w_x),  \phi_1(\mathbf{u}, s, w_x),  \phi_2(\mathbf{u}, s, w_x) \}`

        """
        gauge_paras = (gauge_paras,)
        return super().suff_statistics(u, gauge_paras=gauge_paras)
    @partial(jax.jit,
             static_argnums=0)
    def natural_paras_to_moments(self, betas, gauge_paras, domain):
        r"""Compute the moments of sufficient statistics given natural parameters`

        Parameters
        ----------
        betas : float array of shape (3)
            the natural parameter :math:`\boldsymbol{\beta}` of the distribution
        gauge_paras : float array of shape (2)  
            The array (:math:`s`,:math:`w_x`).
        domain : float array of shape (3)
            an array (a_x, b_x, b_r), in which 

                **a_x** : *float* - lower integration limit in x dimension for the Legendre quadrature

                **b_x** : *float* - upper integration limit in x dimension for the Legendre quadrature

                **b_r** : *float* - upper integration limit in r dimension for the Legendre quadrature
            
        Returns
        -------
        float array of shape (3)
            The array :math:`\{ M_0, \cdots, M_M \}`

        """
        gauge_paras = (gauge_paras,)
        base_args = (domain,)

        return super().natural_paras_to_moments(betas, gauge_paras = gauge_paras, base_args = base_args )
    @partial(jax.jit,
             static_argnums=0)
    def moments_to_natural_paras(self, betas_ini, moments, gauge_paras, domain):
        r"""Compute the natural parameters :math:`\boldsymbol{\beta}` from the moments of sufficient statistics.

        Parameters
        ----------
        betas_ini : float array of shape (3)
            A set of proposed natural parameter :math:`\boldsymbol{\beta}`
        moments : float array of shape (3)
            the target moments that we expect exponential familty with the natural parameter :math:`\boldsymbol{\beta}` to have.
        gauge_paras : float array of shape (2)  
            The array (:math:`s`, :math:`w_x`).
        domain : float array of shape (3)
            an array (a_x, b_x, b_r), in which 

                **a_x** : *float* - lower integration limit in x dimension for the Legendre quadrature

                **b_x** : *float* - upper integration limit in x dimension for the Legendre quadrature

                **b_r** : *float* - upper integration limit in r dimension for the Legendre quadrature
            
        Returns
        -------
        Tuple
            A tuple containing

                **beta**: *float array of shape (3)* - The natural parameters :math:`\boldsymbol{\beta}`

                **opt_info**: *tuple* - A tuple containing other information of the optimization process

                    **values**: *float* - the optimal value of target_function.
            
                    **residuals**: *float* - the residual of the optimization. 

                    **step**: *float* - the total number of Newton's step iteration.

                    **bsteps**: *float* - the total number of Backtracking step.
        """       
        gauge_paras = (gauge_paras,)
        base_args = (domain,)        
        beta, opt_info = super().moments_to_natural_paras( betas_ini, moments, gauge_paras = gauge_paras, base_args = base_args)
        return beta, opt_info
    def natural_paras_to_custom_moments(self, betas, gauge_paras, domain, statistics, stats_gauge_paras = ()):
        r"""Compute the moments of custom statistics given natural parameters. 
    
        Parameters
        ----------
        betas : float array of shape (3)
            the natural parameter :math:`\boldsymbol{\beta}` of the distribution
        gauge_paras : float array of shape (2)  
            The array (:math:`s`, :math:`w_x`).
        domain : float array of shape (3)
            an array (a_x, b_x, b_r), in which 

                **a_x** : *float* - lower integration limit in x dimension for the Legendre quadrature

                **b_x** : *float* - upper integration limit in x dimension for the Legendre quadrature

                **b_r** : *float* - upper integration limit in r dimension for the Legendre quadrature
        statistics : function
            A float-valued or tensor-valued function :math:`\phi_{i_1,\cdots,i_k}` ( **u** , :math:`*` **gauge_paras** ) with
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector :math:`\mathbf{u}`

                    :math:`*` **gauge_paras** : - Arbitrary many extra parameters such as :math:`\mathbf{g}`. The :math:`*` refers to the unpacking operator in python.

                **Returns**: 
                
                    float or array of arbitrary shape :math:`(d_1,\cdots,d_k)` -- the value of the statistic :math:`\phi_{i_1,\cdots,i_k}(\mathbf{u},\mathbf{g})` 
        stats_gauge_paras : tuple
            A tuple ( para1, para2, ... ) containing arbitrary many extra parameters the custom statistics function :math:`\phi_{i_1,\cdots,i_k}` required. Defaut is (), an empty tuple.
            

        Returns
        -------
        float or array of arbitrary shape :math:`(d_1,\cdots,d_k)`
            The moment value :math:`M_{i_1,\cdots,i_k}(\mathbf{g})`

        """
        gauge_paras = (gauge_paras,)
        base_args = (domain,)    
        return super().natural_paras_to_custom_moments(betas, statistics, gauge_paras=gauge_paras, base_args=base_args, stats_gauge_paras=stats_gauge_paras)
    def moments_gauge_transformation(self,moments, gauge_para2, gauge_para1, domain):
        r"""Gauge transformation for moments of sufficient statistics. The transformation is defined as
        
        .. math::
            :nowrap:

            \begin{equation}
                M_i( s', w_x' )= T_{ij}(s', w_x', s, w_x) M_j(s, w_x); \quad i,j = 0, \cdots, 2
            \end{equation}       

        which is induced from the gauge transformation :math:`T_{ij}(\mathbf{g}',\mathbf{g})` between sufficient statistics :math:`\phi_i(\mathbf{u},(s, w_x))` such that

        .. math::
            :nowrap:

            \begin{equation}
                \phi_i(\mathbf{u}, (s',w_x') )= T_{ij}(s', w_x',s, w_x) \phi_j(\mathbf{u}, (s, w_x)); \quad i,j = 0, \cdots, 2
            \end{equation}


        with :math:`\phi_i(\mathbf{u},(s, w_x))` is sufficient statistics parameterized by gauge parameters :math:`(s, w_x)`.

        Parameters
        ----------
        moments : float array of shape (3)
            the moments :math:`M_i(s, w_x)` of sufficient statistics
        gauge_para2 : float array of shape (2)
            The array (:math:`s'`, :math:`w_x'`).
        gauge_para1 : float array of shape (2)
            The array (:math:`s`,  :math:`w_x`).
        domain : float array of shape (3)
            an array (a_x, b_x, b_r), in which 

                **a_x** : *float* - lower integration limit in x dimension for the Legendre quadrature

                **b_x** : *float* - upper integration limit in x dimension for the Legendre quadrature

                **b_r** : *float* - upper integration limit in r dimension for the Legendre quadrature


        Returns
        -------
        float array of shape (3)
            the moments :math:`M_i(s', w_x')` of sufficient statistics
        """
        gauge_para2 = (gauge_para2,)
        gauge_para1 = (gauge_para1,)
        base_args = (domain,)
        return super().moments_gauge_transformation(moments,gauge_para2 = gauge_para2, gauge_para1 = gauge_para1, base_args = base_args)
    @partial(jax.jit,
             static_argnums=0)
    def natural_paras_gauge_transformation(self,betas, gauge_para2, gauge_para1, domain):
        r"""Gauge transformation for natural parameters. The transformation is defined as
        
        .. math::
            :nowrap:

            \begin{equation}
                \beta_i( s', w_x' )= T_{ij}^T(s, w_x, s', w_x') \beta_j(s, w_x); \quad i,j = 0, \cdots, 2
            \end{equation}        

        which is induced from the gauge transformation :math:`T_{ij}(\mathbf{g}',\mathbf{g})` between sufficient statistics :math:`\phi_i(\mathbf{u},\mathbf{g})` such that

        .. math::
            :nowrap:

            \begin{equation}
                \phi_i(\mathbf{u}, (s', w_x') )= T_{ij}(s', w_x',s, w_x) \phi_j(\mathbf{u}, (s, w_x)); \quad i,j = 0, \cdots, 2
            \end{equation}

        with :math:`\phi_i(\mathbf{u},(s, w_x))` is sufficient statistics parameterized by gauge parameters :math:`(s, w_x)`.

        Parameters
        ----------
        betas : float array of shape (3)
            the moments :math:`\beta_i(s, w_x)` of sufficient statistics
        gauge_para2 : float array of shape (2)
            The array (:math:`s'`, :math:`w_x'`).
        gauge_para1 : float array of shape (2)
            The array (:math:`s`, :math:`w_x`).

        Returns
        -------
        float array of shape (M+1)
            the moments :math:`\beta_i(s', w_x')` of sufficient statistics
        """
        gauge_para2 = (gauge_para2,)
        gauge_para1 = (gauge_para1,)
        base_args = (domain,)
        return super().natural_paras_gauge_transformation(betas,gauge_para2 = gauge_para2, gauge_para1 = gauge_para1, base_args = base_args )

    @partial(jax.jit,
             static_argnums=0)
    def natural_paras_to_fluid_properties(self, betas, gauge_paras, domain):
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
        betas : float array of shape (3)
            the natural parameter :math:`\boldsymbol{\beta}` of the distribution
        gauge_paras : float array of shape (2)  
            The array (:math:`s`, :math:`w_x`).
        domain : float array of shape (3)
            an array (a_x, b_x, b_r), in which 

                **a_x** : *float* - lower integration limit in x dimension for the Legendre quadrature

                **b_x** : *float* - upper integration limit in x dimension for the Legendre quadrature

                **b_r** : *float* - upper integration limit in r dimension for the Legendre quadrature
            

        Returns
        -------
        float array of shape (16)
            Array containing macroscopic quantities :math:`\{ \rho, n, v_x, v_y, v_z, T, p, \sigma_{xx}, \sigma_{xy}, \sigma_{xz}, \sigma_{yy}, \sigma_{yz}, \sigma_{zz}, q_x, q_y, q_z \}`
        """
        gauge_paras = (gauge_paras,)
        base_args = (domain,)

        return super().natural_paras_to_fluid_properties(betas, gauge_paras = gauge_paras, base_args = base_args)

    @partial(jax.jit,
             static_argnums=0)
    def rhoVT_to_natural_paras(self, rhoVT, gauge_paras, domain):
        r"""Compute the natural parameters :math:`\boldsymbol{\beta}` from the density :math:`\rho`, flow velocity :math:`v_x` and temperature :math:`T`

        Parameters
        ----------
        rhoVT : float array of shape (3)
            A float array ( :math:`\rho`, :math:`v_x`, :math:`T` ) 
        gauge_paras : float array of shape (2)  
            The array (:math:`s`, :math:`w_x`).
        domain : float array of shape (3)
            an array (a_x, b_x, b_r), in which 

                **a_x** : *float* - lower integration limit in x dimension for the Legendre quadrature

                **b_x** : *float* - upper integration limit in x dimension for the Legendre quadrature

                **b_r** : *float* - upper integration limit in r dimension for the Legendre quadrature
            

        Returns
        -------
        float array of shape (3)
           The natural parameters :math:`\boldsymbol{\beta}`
        """      
        rho, vx, T = rhoVT
        m = self.constant["m"]
        kB = self.constant["kB"]
        n = rho/m
        beta0 = jnp.log(n) -( m*vx**2/(2*kB*T) +3/2*jnp.log(2*jnp.pi*kB*T/m)  )
        beta1 = m*vx/kB/T
        beta2 = -m/(2*kB*T)

        betas = jnp.asarray([beta0, beta1, beta2])
        gauge_ori = jnp.asarray([1,0.])

        #betas = self._ori_to_canon_parameter_convertion(betas,  gauge_paras = (gauge_ori,), base_args = (domain,) )


        return self.natural_paras_gauge_transformation( betas, gauge_paras, gauge_ori, domain )
    @partial(jax.jit,
             static_argnums=0)
    def standard_gauge_para_from_moments(self, moments, gauge_paras):
        r"""the standard gauge parameters :math:`\mathbf{g}` prefered among all possible gauges.
        
        Parameters
        ----------
        moments : float array of shape (3)
            the moments :math:`M_i(s, w_x)` of sufficient statistics
        gauge_paras : float array of shape (2)  
            The array (:math:`s`, :math:`w_x`).

        Returns
        -------
        float array of shape (2)  
            The gauge parameters (:math:`s`, :math:`w_x`) in the Hermite gauge.
        """
        gauge_paras = (gauge_paras,)
        return self._statistics.standard_gauge_paras(moments, gauge_para=gauge_paras)


class Maxwell_Canonical_Gauged_Legendre_1D(Maxwell_Gauged_Legendre_1D,CanonicalExpFamilyModel):
    def __init__(self, constant):
        super().__init__(constant)
        self._sampler = CanonicalExpImportanceSampler(self._statistics.suff_stats, self._Qsampler )
    @partial(jax.jit,
             static_argnums=0)
    def rhoVT_to_natural_paras(self, rhoVT, gauge_paras, domain):
        r"""Compute the natural parameters :math:`\boldsymbol{\beta}` from the density :math:`\rho`, flow velocity :math:`v_x` and temperature :math:`T`

        Parameters
        ----------
        rhoVT : float array of shape (3)
            A float array ( :math:`\rho`, :math:`v_x`, :math:`T` ) 
        gauge_paras : float array of shape (2)  
            The array (:math:`s`, :math:`w_x`).
        domain : float array of shape (3)
            an array (a_x, b_x, b_r), in which 

                **a_x** : *float* - lower integration limit in x dimension for the Legendre quadrature

                **b_x** : *float* - upper integration limit in x dimension for the Legendre quadrature

                **b_r** : *float* - upper integration limit in r dimension for the Legendre quadrature
            

        Returns
        -------
        float array of shape (3)
           The natural parameters :math:`\boldsymbol{\beta}`
        """      
        rho, vx, T = rhoVT
        m = self.constant["m"]
        kB = self.constant["kB"]
        n = rho/m
        beta0 = jnp.log(n) -( m*vx**2/(2*kB*T) +3/2*jnp.log(2*jnp.pi*kB*T/m)  )
        beta1 = m*vx/kB/T
        beta2 = -m/(2*kB*T)

        betas = jnp.asarray([beta0, beta1, beta2])
        gauge_ori = jnp.asarray([1,0.])

        betas = self._ori_to_canon_parameter_convertion(betas,  gauge_paras = (gauge_ori,), base_args = (domain,) )


        return self.natural_paras_gauge_transformation( betas, gauge_paras, gauge_ori, domain )


if __name__ == "__main__":
    from MomentGauge.Models.Moment35 import M35_Gauged_Legendre_1D
    constant = {"m": 1., "kB": 1.,"n_x": 8, "n_r": 8, "B_x": 16, "B_r": 16 ,
                "alpha" : 1., "beta" : 0.5, "c": 5e-4, "atol": 5e-6, "rtol":1e-5,
                "max_iter": 100, "max_back_tracking_iter": 25, "tol": 1e-8, "min_step_size": 1e-6,
                "reg_hessian": True, "debug" : False }
    Ebeta = jnp.array( [-7.25,3,-1/2] )
    domain_para = jnp.array([-15,15,15])   
    gauge = jnp.array([1,0.])       
    Maxwell = Maxwell_Legendre_1D(constant)
    Maxwell2 = Maxwell_Canonical_Legendre_1D(constant)
    Maxwell3 = Maxwell_Gauged_Legendre_1D(constant)
    Maxwell4 = Maxwell_Canonical_Gauged_Legendre_1D(constant)
    rhoVT = jnp.array([1,0,3])
    beta1 = Maxwell.rhoVT_to_natural_paras(rhoVT,domain_para)
    print(beta1)
    beta2 = Maxwell2.rhoVT_to_natural_paras(rhoVT,domain_para)
    print(beta2)
    beta3 = Maxwell3.rhoVT_to_natural_paras(rhoVT,gauge,domain_para)
    print(beta3)
    beta4 = Maxwell4.rhoVT_to_natural_paras(rhoVT,gauge,domain_para)
    print(beta4)

    '''

    moments = Maxwell.natural_paras_to_moments(Ebeta, domain_para )
    print(moments)
    Obeta,_ = Maxwell.moments_to_natural_paras(Ebeta,moments, domain_para ) 
    print(Obeta)
    gauge_para = jnp.array( [ 1,0. ] )
    M35G = M35_Gauged_Legendre_1D(constant)
    Maxwell = Maxwell_Gauged_Legendre_1D(constant)
    mom = Maxwell.natural_paras_to_moments(Obeta, gauge_para, domain_para )
    print(mom)
    print(Obeta.dot(mom))
    gauge_para2 = jnp.array( [ 1.1,3. ] )
    mom_G2 = Maxwell.moments_gauge_transformation(mom, gauge_para2, gauge_para, )
    beta_G2 = Maxwell.natural_paras_gauge_transformation(Obeta, gauge_para2, gauge_para, )

    print(beta_G2.dot(mom_G2))

    mom2 = Maxwell.natural_paras_to_custom_moments(beta_G2, gauge_para2, domain_para, Maxwell.suff_statistics, stats_gauge_paras=(gauge_para,) )

    print(mom2)

    fp = Maxwell.natural_paras_to_fluid_properties(beta_G2,gauge_para2,domain_para)
    rho, vx, T = fp[0],fp[2],fp[5]
    Abeta2 = Maxwell.rhoVT_to_natural_paras(jnp.asarray([rho, vx, T]),gauge_para2)
    print("Abeta", Abeta2, beta_G2 )
    Abeta1 = Maxwell.rhoVT_to_natural_paras(jnp.asarray([rho, vx, T]),gauge_para)
    print("Abeta", Abeta1, Obeta )
    '''

    """
    max35moments = Maxwell.natural_paras_to_custom_moments( Ebeta, domain_para, M35G.suff_statistics , stats_gauge_paras= ( gauge_para,) )
    print(max35moments)
    beta = jnp.array( [-7.25,3,-1/2,0,0,0,0,0,0] )
    betaG, optinfo = M35G.moments_to_natural_paras(beta,max35moments,gauge_para, domain_para ) 
    print(optinfo)
    max35Rmom = M35G.natural_paras_to_moments(betaG, gauge_para, domain_para )
    print(max35Rmom)
    """
    
    """
    Qsampler = Gauss_Legendre_Sampler_Polar3D(n_x = 8,n_r = 8, B_x = 16, B_r = 16)
    Meq = Maxwellian_1D_stats()
    M35moments = M35_1D_gauged_stats()
    sampler = CanonicalExpImportanceSampler(M35moments.suff_stats, Qsampler )
    Est = EstimatorPolar2D(constant)
    beta = jnp.array( [1.,0,0,0,0,0,0,0,0] )
    gauge_para = jnp.array([1,1,0.])
    domain_para = jnp.array([-15,15,15])
    suff_statistics = lambda u, gauge_para: sampler.suff_statistics( u, gauge_paras=(gauge_para,) )

    #print(moments)

    Esampler = CanonicalExpImportanceSampler(Meq.suff_stats, Qsampler )
    Est = EstimatorPolar2D(constant)
    Ebeta = jnp.array( [1.,3,-1/2] )
    x,w,logli = Esampler.sample(Ebeta, gauge_paras = (), base_args = (domain_para,))


    Esuff_statistics = lambda u: Esampler.suff_statistics( u, gauge_paras=() )
    Emoments = Est.get_sample_moment(Esuff_statistics, x, w, gauge_paras=())

    moments = Est.get_sample_moment(suff_statistics, x, w, gauge_paras=(gauge_para,))

    #print(Emoments)

    print(moments)

    


    opt_Loss = lambda para, moments, gauge_para: sampler.sample_Loss( para, moments, gauge_paras = (gauge_para,), base_args = (domain_para,) )
    beta, values, residual, iter, biter = Newton_Backtracking_Optimizer_JIT(opt_Loss, beta, moments, gauge_para ,debug=False, alpha=1.0, beta=0.5, c=5e-4, max_iter = 400, max_back_tracking_iter = 55, tol=1e-9, atol=5e-6, rtol=1e-5,reg_hessian=True )

    #print(beta)
    print("iter", iter, "biter", biter)

    x,w,logli = sampler.sample(beta, gauge_paras = (gauge_para,), base_args = (domain_para,))



    moments = Est.get_sample_moment(suff_statistics, x, w, gauge_paras=(gauge_para,))

    print(moments)
    """