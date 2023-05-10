import jax.numpy as jnp
import jax
from jax import vmap,jit
from functools import partial
from collections import OrderedDict
#from Utility import convert_matrix_from_sufficient_statistics_to_conserved_statistics
class BaseEstimator():
    def __init__(self,constant):
        """The Base Class for Estimators

        Estimate moments from samples draw from distributions.

        Parameters
        ----------
        constant : dict
            dictionary with the following keys

                **'m'** : float - the mass of particle considered

                **'kB'** : float - the Boltzmann constant
        Attributes
        ----------
        pi : float
            the value of :math:`\pi`
        m : float
            the mass of particle considered
        kB : float
            the Boltzmann constant
        constant : dict
            dictionary with the keys containing **'m'** and **'kB'**
        """
        self.pi = jnp.pi
        self.m = constant["m"]
        self.kB = constant["kB"]
        self.constant = OrderedDict(sorted(constant.items()))
    def __hash__(self):
        """Redefine the hash method to include the class attributes. It helps jax.jit to correctly identify class instances"""
        return hash(("BaseEstimator",*self.constant.items()))

    def __eq__(self, other):
        """Redefine the eq method to include the class attributes."""
        return (isinstance(other, BaseEstimator) and
                (self.constant,) == (other.constant,))
    def get_sample_moment(self,statistics,samples, weights, gauge_paras = ()):
        r"""
        Calculate the moment provided samples :math:`\mathbf{u}_i` and weiths :math:`w_i` of the distribution as follows

        .. math::
            :nowrap:

            \begin{equation}
            M_{i_1,\cdots,i_k}(\mathbf{g})=\int \phi_{i_1,\cdots,i_k}(\mathbf{u}, \mathbf{g}) f(\mathbf{u}) d \mathbf{u} \approx \sum_{i=1}^N w_i \phi_{i_1,\cdots,i_k}(\mathbf{u}_i, \mathbf{g}),
            \end{equation}

        in which :math:`\phi_{i_1,\cdots,i_k}` is a float-valued or tensor-valued function, :math:`\mathbf{g}` is extra gauge parameters that may or may not be requested by the moments :math:`\phi`.


        Parameters
        ----------
        statistics : function
            A float-valued or tensor-valued function :math:`\phi_{i_1,\cdots,i_k}` ( **u** , :math:`*` **gauge_paras** ) with
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector :math:`\mathbf{u}`

                    :math:`*` **gauge_paras** : - Arbitrary many extra parameters such as :math:`\mathbf{g}`. The :math:`*` refers to the unpacking operator in python.

                **Returns**: 
                
                    float or array of arbitrary shape :math:`(d_1,\cdots,d_k)` -- the value of the statistic :math:`\phi_{i_1,\cdots,i_k}(\mathbf{u},\mathbf{g})` 
        samples : float array of shape (N,3)
            N  samples of 3-dim vectors :math:`\mathbf{u}_i` draw from the distribution. 
        weights : float array of shape (N) 
            N non-negative weights :math:`w_i` for each samples.
        gauge_paras : tuple
            A tuple ( para1, para2, ... ) containing arbitrary many extra parameters the statistics function :math:`\phi_{i_1,\cdots,i_k}` required. Defaut is (), an empty tuple.

        Returns
        -------
        float or array of arbitrary shape :math:`(d_1,\cdots,d_k)`
            The moment value :math:`M_{i_1,\cdots,i_k}(\mathbf{g})`
        """
        statistics_gauge = lambda u: statistics(u,*gauge_paras)
        return jnp.sum( vmap(statistics_gauge,in_axes=0,out_axes=-1)( samples )*weights,axis=-1 )
    @partial(jax.jit,
             static_argnums=0)
    def cal_macro_quant(self, samples, weights):
        r"""
        Compute the macroscopic quantities of distribution including number density :math:`n`, density :math:`\rho`, flow velocities :math:`\mathbf{v} = \{v_\alpha, \alpha \in \{x,y,z\}\}`, temperature :math:`T`, pressure :math:`p`, stress :math:`\{\sigma_{\alpha \beta}, \alpha, \beta \in \{x,y,z\}\}` and heat flux :math:`\{q_{\alpha}, \alpha \in \{x,y,z\}\}`.

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
        samples : float array of shape (N,3)
            N  samples of 3-dim vectors :math:`\mathbf{u}_i` draw from the distribution. 
        weights : float array of shape (N) 
            N non-negative weights :math:`w_i` for each samples.

        Returns
        -------
        float array of shape (16)
            Array containing macroscopic quantities :math:`\{ \rho, n, v_x, v_y, v_z, T, p, \sigma_{xx}, \sigma_{xy}, \sigma_{xz}, \sigma_{yy}, \sigma_{yz}, \sigma_{zz}, q_x, q_y, q_z \}`
        """
        raise NotImplementedError




class EstimatorPolar2D(BaseEstimator):
    def __init__(self,constant):
        """The Estimators for 3D distribution with polar symmetry.

        Estimate moments from samples draw from distributions.

        Parameters
        ----------
        constant : dict
            dictionary with the following keys

                **'m'** : float - the mass of particle considered

                **'kB'** : float - the Boltzmann constant
        Attributes
        ----------
        pi : float
            the value of :math:`\pi`
        m : float
            the mass of particle considered
        kB : float
            the Boltzmann constant
        constant : dict
            dictionary with the keys containing **'m'** and **'kB'**
        """
        super().__init__(constant)
    def __hash__(self):
        """Redefine the hash method to include the class attributes. It helps jax.jit to correctly identify class instances"""
        return hash(("EstimatorPolar2D",*self.constant.items()))

    def __eq__(self, other):
        """Redefine the eq method to include the class attributes."""
        return (isinstance(other, EstimatorPolar2D) and
                (self.constant,) == (other.constant,))
    @partial(jax.jit,
             static_argnums=0)
    def cal_macro_quant(self, samples, weights):
        r"""
        Compute the macroscopic quantities of distribution including number density :math:`n`, density :math:`\rho`, flow velocities :math:`\mathbf{v} = \{v_\alpha, \alpha \in \{x,y,z\}\}`, temperature :math:`T`, pressure :math:`p`, stress :math:`\{\sigma_{\alpha \beta}, \alpha, \beta \in \{x,y,z\}\}` and heat flux :math:`\{q_{\alpha}, \alpha \in \{x,y,z\}\}`.

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
        samples : float array of shape (N,3)
            N  samples of 3-dim vectors :math:`\mathbf{u}_i` draw from the distribution. 
        weights : float array of shape (N) 
            N non-negative weights :math:`w_i` for each samples.

        Returns
        -------
        float array of shape (16)
            Array containing macroscopic quantities :math:`\{ \rho, n, v_x, v_y, v_z, T, p, \sigma_{xx}, \sigma_{xy}, \sigma_{xz}, \sigma_{yy}, \sigma_{yz}, \sigma_{zz}, q_x, q_y, q_z \}`
        """
        get_moment = lambda moment: self.get_sample_moment( moment, samples, weights )
        #macros={}
        m0 = lambda u: 1.
        mu1 = lambda u: u[0]
        mu2 = lambda u: u[1]
        mu3 = lambda u: u[2]
        n = get_moment(m0)
        rho = n*self.m
        vx = get_moment(mu1)/n
        vy = 0.
        vz = 0.
        cx = lambda u: u[0]-vx
        cy = lambda u: u[1]-vy
        cz = lambda u: u[2]-vz
        mcsq = lambda u: cx(u)**2+cy(u)**2+cz(u)**2

        mcxx = lambda u: cx(u)*cx(u)
        mcxy = lambda u: 0.
        mcxz = lambda u: 0.
        mcyy = lambda u: ( cy(u)*cy(u) + cz(u)*cz(u) )/2
        mcyz = lambda u: 0.
        mczz = lambda u: ( cy(u)*cy(u) + cz(u)*cz(u) )/2

        mcxcsq = lambda u: cx(u)*( cx(u)**2+cy(u)**2+cz(u)**2 )
        mcycsq = lambda u: 0.
        mczcsq = lambda u: 0.

        T = self.m/2*get_moment(mcsq)/n/self.kB/3*2
        p = n*self.kB*T
        sigma_xx = self.m*get_moment(mcxx) - p
        sigma_xy = self.m*get_moment(mcxy)
        sigma_xz = self.m*get_moment(mcxz)
        sigma_yy = self.m*get_moment(mcyy) - p
        sigma_yz = self.m*get_moment(mcyz)
        sigma_zz = self.m*get_moment(mczz) - p

        q_x = self.m/2*get_moment(mcxcsq)
        q_y = self.m/2*get_moment(mcycsq)
        q_z = self.m/2*get_moment(mczcsq)
    

        return jnp.array([ rho, n, vx, vy, vz, T, p, sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz, q_x, q_y, q_z  ])