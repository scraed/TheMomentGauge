from MomentGauge.Sampler.QuadratureSampler import Gauss_Legendre_Sampler_Polar3D
from MomentGauge.Sampler.CanonicalExpFamilySampler import CanonicalExpImportanceSampler,CanonicalExpSampler
from MomentGauge.Sampler.ExponentialFamilySampler import ExpFamilyImportanceSampler
from MomentGauge.Statistic.PolyStatistics import Maxwellian_1D_stats
from MomentGauge.Statistic.PolyGaugedStatistics import PolyGaugedStatistics, M35_1D_gauged_stats
from MomentGauge.Models.ExpFamilyBase import BaseExpFamilyModel, CanonicalExpFamilyModel, GaugedExpFamilyModel
import jax
import jax.numpy as jnp
from MomentGauge.Estimator.Estimator import EstimatorPolar2D
from MomentGauge.Optim.NewtonOptimizer import Newton_Optimizer
from MomentGauge.Optim.BaseOptimizer import BaseOptimizer
from functools import partial




class M35_Gauged_Legendre_1D(GaugedExpFamilyModel,BaseExpFamilyModel):
    def __init__(self, constant):
        r"""The exponential family distribution model for M35 gauged statistics.
        
        The gauged statistics are defined in :class:`MomentGauge.Statistic.PolyGaugedStatistics.M35_1D_gauged_stats` .

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
        self._statistics = M35_1D_gauged_stats()
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
        gauge_paras : float array of shape (3)  
            The array (:math:`s_r`, :math:`s_x`, :math:`w_x`).

        Returns
        -------
        float array of shape (3)
            The array :math:`\{ \phi_0(\mathbf{u}),  \phi_1(\mathbf{u}),  \phi_2(\mathbf{u}) \}`

        """
        gauge_paras = (gauge_paras,)
        return super().suff_statistics(u, gauge_paras=gauge_paras)
    @partial(jax.jit,
             static_argnums=0)
    def natural_paras_to_moments(self, betas, gauge_paras, domain):
        r"""Compute the moments of sufficient statistics given natural parameters`

        Parameters
        ----------
        betas : float array of shape (9)
            the natural parameter :math:`\boldsymbol{\beta}` of the distribution
        gauge_paras : float array of shape (3)  
            The array (:math:`s_r`, :math:`s_x`, :math:`w_x`).
        domain : float array of shape (3)
            an array (a_x, b_x, b_r), in which 

                **a_x** : *float* - lower integration limit in x dimension for the Legendre quadrature

                **b_x** : *float* - upper integration limit in x dimension for the Legendre quadrature

                **b_r** : *float* - upper integration limit in r dimension for the Legendre quadrature
            
        Returns
        -------
        float array of shape (9)
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
        betas_ini : float array of shape (9)
            A set of proposed natural parameter :math:`\boldsymbol{\beta}`
        moments : float array of shape (9)
            the target moments that we expect exponential familty with the natural parameter :math:`\boldsymbol{\beta}` to have.
        gauge_paras : float array of shape (3)  
            The array (:math:`s_r`, :math:`s_x`, :math:`w_x`).
        domain : float array of shape (3)
            an array (a_x, b_x, b_r), in which 

                **a_x** : *float* - lower integration limit in x dimension for the Legendre quadrature

                **b_x** : *float* - upper integration limit in x dimension for the Legendre quadrature

                **b_r** : *float* - upper integration limit in r dimension for the Legendre quadrature
            
        Returns
        -------
        Tuple
            A tuple containing

                **beta**: *float array of shape (9)* - The natural parameters :math:`\boldsymbol{\beta}`

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
        betas : float array of shape (9)
            the natural parameter :math:`\boldsymbol{\beta}` of the distribution
        gauge_paras : float array of shape (3)  
            The array (:math:`s_r`, :math:`s_x`, :math:`w_x`).
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
    @partial(jax.jit,
             static_argnums=0)
    def moments_gauge_transformation(self,moments, gauge_para2, gauge_para1, domain):
        r"""Gauge transformation for moments of sufficient statistics. The transformation is defined as
        
        .. math::
            :nowrap:

            \begin{equation}
                M_i( s_r', s_x', w_x' )= T_{ij}(s_r', s_x', w_x',s_r, s_x, w_x) M_j(s_r, s_x, w_x); \quad i,j = 0, \cdots, 8
            \end{equation}       

        which is induced from the gauge transformation :math:`T_{ij}(\mathbf{g}',\mathbf{g})` between sufficient statistics :math:`\phi_i(\mathbf{u},(s_r, s_x, w_x))` such that

        .. math::
            :nowrap:

            \begin{equation}
                \phi_i(\mathbf{u}, (s_r', s_x', w_x') )= T_{ij}(s_r', s_x', w_x',s_r, s_x, w_x) \phi_j(\mathbf{u}, (s_r, s_x, w_x)); \quad i,j = 0, \cdots, 8
            \end{equation}


        with :math:`\phi_i(\mathbf{u},(s_r, s_x, w_x))` is sufficient statistics parameterized by gauge parameters :math:`(s_r, s_x, w_x)`.

        Parameters
        ----------
        moments : float array of shape (9)
            the moments :math:`M_i(s_r, s_x, w_x)` of sufficient statistics
        gauge_para2 : float array of shape (3)
            The array (:math:`s_r'`, :math:`s_x'`, :math:`w_x'`).
        gauge_para1 : float array of shape (3)
            The array (:math:`s_r`, :math:`s_x`, :math:`w_x`).
        domain : float array of shape (3)
            an array (a_x, b_x, b_r), in which 

                **a_x** : *float* - lower integration limit in x dimension for the Legendre quadrature

                **b_x** : *float* - upper integration limit in x dimension for the Legendre quadrature

                **b_r** : *float* - upper integration limit in r dimension for the Legendre quadrature


        Returns
        -------
        float array of shape (M+1)
            the moments :math:`M_i(s_r', s_x', w_x')` of sufficient statistics
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
                \beta_i( s_r', s_x', w_x' )= T_{ij}^T(s_r, s_x, w_x,s_r', s_x', w_x') \beta_j(s_r, s_x, w_x); \quad i,j = 0, \cdots, 8
            \end{equation}        

        which is induced from the gauge transformation :math:`T_{ij}(\mathbf{g}',\mathbf{g})` between sufficient statistics :math:`\phi_i(\mathbf{u},\mathbf{g})` such that

        .. math::
            :nowrap:

            \begin{equation}
                \phi_i(\mathbf{u}, (s_r', s_x', w_x') )= T_{ij}(s_r', s_x', w_x',s_r, s_x, w_x) \phi_j(\mathbf{u}, (s_r, s_x, w_x)); \quad i,j = 0, \cdots, 8
            \end{equation}

        with :math:`\phi_i(\mathbf{u},(s_r, s_x, w_x))` is sufficient statistics parameterized by gauge parameters :math:`(s_r, s_x, w_x)`.

        Parameters
        ----------
        betas : float array of shape (M+1)
            the moments :math:`\beta_i(s_r, s_x, w_x)` of sufficient statistics
        gauge_para2 : float array of shape (3)
            The array (:math:`s_r'`, :math:`s_x'`, :math:`w_x'`).
        gauge_para1 : float array of shape (3)
            The array (:math:`s_r`, :math:`s_x`, :math:`w_x`).
        domain : float array of shape (3)
            an array (a_x, b_x, b_r), in which 

                **a_x** : *float* - lower integration limit in x dimension for the Legendre quadrature

                **b_x** : *float* - upper integration limit in x dimension for the Legendre quadrature

                **b_r** : *float* - upper integration limit in r dimension for the Legendre quadrature


        Returns
        -------
        float array of shape (M+1)
            the moments :math:`\beta_i(s_r', s_x', w_x')` of sufficient statistics
        """
        gauge_para2 = (gauge_para2,)
        gauge_para1 = (gauge_para1,)
        base_args = (domain,)
        return super().natural_paras_gauge_transformation(betas,gauge_para2 = gauge_para2, gauge_para1 = gauge_para1, base_args = base_args)
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
        betas : float array of shape (9)
            the natural parameter :math:`\boldsymbol{\beta}` of the distribution
        gauge_paras : float array of shape (3)  
            The array (:math:`s_r`, :math:`s_x`, :math:`w_x`).
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
    def standard_gauge_para_from_moments(self, moments, gauge_paras):
        r"""the standard gauge parameters :math:`\mathbf{g}` prefered among all possible gauges.
        
        Parameters
        ----------
        moments : float array of shape (9)
            the moments :math:`M_i(s_r, s_x, w_x)` of sufficient statistics
        gauge_paras : float array of shape (3)  
            The array (:math:`s_r`, :math:`s_x`, :math:`w_x`).

        Returns
        -------
        float array of shape (3)  
            The gauge parameters (:math:`s_r`, :math:`s_x`, :math:`w_x`) in the Hermite gauge.
        """
        gauge_paras = (gauge_paras,)
        return self._statistics.standard_gauge_paras(moments, gauge_para=gauge_paras)



class M35_Gauged_Canonical_Legendre_1D(M35_Gauged_Legendre_1D,CanonicalExpFamilyModel):
    def __init__(self, constant):
        r"""The exponential family distribution model using the canonical form for M35 gauged statistics. 
        
        The gauged statistics are defined in :class:`MomentGauge.Statistic.PolyGaugedStatistics.M35_1D_gauged_stats` .

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
        self._sampler = CanonicalExpImportanceSampler(self._statistics.suff_stats, self._Qsampler )



if __name__ == "__main__":

    from MomentGauge.Models.Maxwellian import Maxwell_Canonical_Legendre_1D

    print(M35_Gauged_Canonical_Legendre_1D.__mro__)
    #print(Maxwell._sampler)
    
    constant = {"m": 1., "kB": 1.,"n_x": 8, "n_r": 8, "B_x": 16, "B_r": 16 ,
                "alpha" : 1., "beta" : 0.5, "c": 5e-4, "atol": 5e-6, "rtol":1e-5,
                "max_iter": 100, "max_back_tracking_iter": 25, "tol": 1e-8, "min_step_size": 1e-6,
                "reg_hessian": True, "debug" : False }
    Ebeta = jnp.array( [-7.25,3,-1/2] )
    domain_para = jnp.array([-25,25,25])
    Maxwell = Maxwell_Canonical_Legendre_1D(constant)
    moments = Maxwell.natural_paras_to_moments(Ebeta, domain_para )
    print(moments)
    Obeta,_ = Maxwell.moments_to_natural_paras(Ebeta,moments, domain_para ) 
    print("Obeta", Obeta)
    print( Maxwell.natural_paras_to_moments(Obeta, domain_para ) )
    gauge_para = jnp.array( [ 1,1,0. ] )
    M35G = M35_Gauged_Canonical_Legendre_1D(constant)
    fp = Maxwell.natural_paras_to_fluid_properties(Obeta,  domain_para )
    #print("maxwell fp",fp)
    rho, vx, T = fp[0],fp[2],fp[5]
    Abeta = Maxwell.rhoVT_to_natural_paras(jnp.asarray([rho, vx, T]))
    print("Abeta",Abeta)




    """
    max35moments = Maxwell.natural_paras_to_custom_moments( Ebeta, domain_para, M35G.suff_statistics , stats_gauge_paras= ( gauge_para,) )
    print(max35moments)
    beta = jnp.array( [-7.25,3,-1/2,0,0,0,0,0,0] )
    #beta = jnp.array( [-4.2500000e+00, -3.8508006e-08, -7.0710653e-01, -9.9999988e-01,-2.7186616e-08 , 2.2246968e-07, -5.7010766e-08, -8.8830348e-09,1.5636033e-07])
    betaG, optinfo = M35G.moments_to_natural_paras(beta,max35moments,gauge_para, domain_para ) 
    print(optinfo)
    print(betaG)
    max35Rmom = M35G.natural_paras_to_moments(betaG, gauge_para, domain_para )
    print(max35Rmom)
    gauge_para2 = jnp.array( [ 1.1,1.2,3. ] )

    print("product", betaG.dot(max35Rmom))

    max35Rmom_G2 = M35G.moments_gauge_transformation(max35Rmom, gauge_para2, gauge_para, )
    print("moment2",max35Rmom_G2)
    beta_G2 = M35G.natural_paras_gauge_transformation(betaG, gauge_para2, gauge_para, )
    print("moment2",beta_G2)
    max35Rmom = M35G.natural_paras_to_custom_moments( betaG,gauge_para, domain_para, M35G.suff_statistics , stats_gauge_paras= ( gauge_para,) )
    print(max35Rmom)
    max35Rmom = M35G.natural_paras_to_custom_moments( beta_G2,gauge_para2, domain_para, M35G.suff_statistics , stats_gauge_paras= ( gauge_para,) )
    print(max35Rmom)
    fp = M35G.natural_paras_to_fluid_properties(betaG, gauge_para, domain_para )
    print(fp)
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

    from MomentGauge.Optim.NewtonOptimizer import Newton_Backtracking_Optimizer_JIT


    opt_Loss = lambda para, moments, gauge_para: sampler.sample_Loss( para, moments, gauge_paras = (gauge_para,), base_args = (domain_para,) )
    beta, values, residual, iter, biter = Newton_Backtracking_Optimizer_JIT(opt_Loss, beta, moments, gauge_para ,debug=False, alpha=1.0, beta=0.5, c=5e-4, max_iter = 400, max_back_tracking_iter = 55, tol=1e-9, atol=5e-6, rtol=1e-5,reg_hessian=True )

    #print(beta)
    print("iter", iter, "biter", biter)

    x,w,logli = sampler.sample(beta, gauge_paras = (gauge_para,), base_args = (domain_para,))



    moments = Est.get_sample_moment(suff_statistics, x, w, gauge_paras=(gauge_para,))

    print(moments)
    """