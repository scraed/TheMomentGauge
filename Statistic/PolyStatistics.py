import sympy
import numpy as np
from sympy import symbols, Matrix, RR, expand_complex, nsimplify, N
from sympy import Poly, pprint
from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key
from sympy.abc import x, y, z
from sympy.solvers.solvers import solve_linear_system
from sympy import integrate, Symbol,symbols, oo, exp, Poly, summation
from sympy import assuming, Q, ask, diff, solve, simplify, lambdify, factor, collect
from sympy.polys.domainmatrix import DomainMatrix
from sympy.interactive import init_printing
import jax.numpy as jnp
import warnings
import jax                              
from MomentGauge.Statistic.SymbolicPolynomials import _get_convert_mat_between_polynomial_lists
#init_printing(pretty_print=True) 
######################### The basic symbols for velocities ################



class PolyStatistics:
    def __init__(self):
        r"""The base class for store pre-defined polynomial statistics. 
        
        Attributes
        ----------
        suff_stats : None
            a list of statistics [:math:`\phi_i,i=0,\cdots,M`], in which each :math:`\phi_i` is a polynomial function :math:`\phi_i` ( **u** , :math:`*` **gauge_paras** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector 

                    :math:`*` **gauge_paras** : - Arbitrary many extra parameters. The :math:`*` refers to the unpacking operator in python.

                **Returns**: 
                
                    float -- the moment value
            
            The lengh of the list may vary. Its first element must satisfy :math:`\phi_0` ( **u** , :math:`*` **gauge_paras** ) = 1


        """

        self.suff_stats = [] # A list of sufficient statistics, must be polynomials


    def __hash__(self):
        """Redefine the hash method to include the class attributes. It helps jax.jit to correctly identify class instances"""
        return hash(("PolyStatistics",*self.suff_stats))

    def __eq__(self, other):
        """Redefine the eq method to include the class attributes."""
        return (isinstance(other, PolyStatistics) and
                (*self.suff_stats,) == (*other.suff_stats,))





class Maxwellian_1D_stats(PolyStatistics):
    def __init__(self):
        r"""The polynomial statistics for 1D Maxwellian distribution. 
        
        Attributes
        ----------
        suff_stats : list of length (3)
            a list of moment functions [:math:`\phi_i,i=0,\cdots,2`] in which each :math:`\phi_i` is a polynomial function :math:`\phi_i` ( **u** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector (:math:`u_x`, :math:`u_y`, :math:`u_z`)

                **Returns**: 
                
                    float -- the moment value
            
            Specifically,

            :math:`\phi_0` (**u** ) = 1.

            :math:`\phi_1` (**u** ) = :math:`u_x`
            
            :math:`\phi_2` (**u** ) = :math:`u_x^2 + u_y^2 + u_z^2`.
        """
        super().__init__()
        #####Define the sufficient statistics with gauge parameters#####
        self.suff_stats = [lambda u: 1., 
                                lambda u: u[0]**1,          
                                lambda u: (u[0]**2+u[1]**2+u[2]**2),
                                #lambda u: (u[0]**2+u[1]**2+u[2]**2)/6**0.5- (3/2)**0.5,
                                ]
    

class ESBGK_1D_stats(PolyStatistics):
    def __init__(self):
        r"""The polynomial statistics for 1D ESBGK distribution. 
        
        Attributes
        ----------
        suff_stats : list of length (4)
            a list of moment functions [:math:`\phi_i,i=0,\cdots,3`] in which each :math:`\phi_i` is a polynomial function :math:`\phi_i` ( **u** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector (:math:`u_x`, :math:`u_y`, :math:`u_z`)

                **Returns**: 
                
                    float -- the moment value
            
            Specifically,

            :math:`\phi_0` (**u** ) = 1.

            :math:`\phi_1` (**u** ) = :math:`u_x`
            
            :math:`\phi_2` (**u** ) = :math:`u_x^2`

            :math:`\phi_2` (**u** ) = :math:`u_y^2 + u_z^2`.
        """
        super().__init__()
        #####Define the sufficient statistics with gauge parameters#####
        self.suff_stats = [lambda u: 1., 
                                lambda u: u[0]**1,          
                                lambda u: u[0]**2,
                                lambda u: u[1]**2+u[2]**2
                                ]


class M35_1D_stats(PolyStatistics):
    def __init__(self):
        r"""The 1D version of polynomial statistics for 35 moments. 
        
        Attributes
        ----------
        suff_stats : list of length (9)
            a list of moment functions [:math:`\phi_i,i=0,\cdots,8`] in which each :math:`\phi_i` is a polynomial function :math:`\phi_i` ( **u** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector (:math:`u_x`, :math:`u_y`, :math:`u_z`)

                **Returns**: 
                
                    float -- the moment value
            
            Specifically,

            .. math::
                :nowrap:

                \begin{equation}
                \{\phi_i,i=0,\cdots,8\} = \left\{1, {u}_x,\frac{ {u}_x^2-1}{\sqrt{2}},\frac{ {u}_r^2}{2} -1,\frac{ {u}_x^3-3 {u}_x}{\sqrt{6}},\frac{ {u}_x^4-6 {u}_x^2+3}{2 \sqrt{6}},\frac{1}{8}  {u}_r^4- {u}_r^2+1,\frac{1}{2}  {u}_x ( {u}_r^2-1),\frac{(  {u}_x^2 -1)(  {u}_r^2-2)}{2 \sqrt{2}}\right\}
                \end{equation}
            
            in which :math:`u_r = \sqrt{u_y^2+u_z^2}`
        """
        super().__init__()
        Sqrt = lambda x: x**0.5
        self.suff_stats =  [lambda u: u[0]**0, 
                                  lambda u: u[0]**1,
                                  lambda u: -(1/2**0.5)+u[0]**2/2**0.5,
                                  lambda u: -1+1/2*(u[1]**2+u[2]**2),
                                  lambda u: -(3/2)**0.5*u[0]+u[0]**3/6**0.5,
                                  lambda u: Sqrt(3/2)/2-Sqrt(3/2)*u[0]**2+u[0]**4/(2*Sqrt(6)),
                                  lambda u: 1-u[1]**2-u[2]**2+1/8*(u[1]**2+u[2]**2)**2,
                                  lambda u: -u[0]+1/2*u[0]*(u[1]**2+u[2]**2),
                                  lambda u: 1/2**0.5-u[0]**2/2**0.5-(u[1]**2+u[2]**2)/(2*2**0.5)+(u[0]**2*(u[1]**2+u[2]**2))/(2*2**0.5),
                                  ]

class M35_P2_1D_stats(PolyStatistics):
    def __init__(self):
        r"""The 1D version of polynomial statistics for 35 moments with . 
        
        Attributes
        ----------
        suff_stats : list of length (11)
            a list of moment functions [:math:`\phi_i,i=0,\cdots,10`] in which each :math:`\phi_i` is a polynomial function :math:`\phi_i` ( **u** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector (:math:`u_x`, :math:`u_y`, :math:`u_z`)

                **Returns**: 
                
                    float -- the moment value
            
            Specifically,

            .. math::
                :nowrap:

                \begin{equation}
                \{\phi_i,i=0,\cdots,10\} = \left\{1, {u}_x,\frac{ {u}_x^2-1}{\sqrt{2}},\frac{ {u}_r^2}{2} -1,\frac{ {u}_x^3-3 {u}_x}{\sqrt{6}},\frac{ {u}_x^4-6 {u}_x^2+3}{2 \sqrt{6}},\frac{1}{8}  {u}_r^4- {u}_r^2+1,\frac{1}{2}  {u}_x ( {u}_r^2-1),\frac{(  {u}_x^2 -1)(  {u}_r^2-2)}{2 \sqrt{2}} , \frac{u_x^5}{2 \sqrt{30}}-\sqrt{\frac{5}{6}} u_x^3+\frac{1}{2} \sqrt{\frac{15}{2}} u_x, \frac{u_x^6}{12 \sqrt{5}}-\frac{\sqrt{5} u_x^4}{4}+\frac{3 \sqrt{5} u_x^2}{4}-\frac{\sqrt{5}}{4}  \right\}
                \end{equation}
            
            in which :math:`u_r = \sqrt{u_y^2+u_z^2}`
        """
        super().__init__()
        Sqrt = lambda x: x**0.5
        self.suff_stats =  [lambda u: u[0]**0, 
                                  lambda u: u[0]**1,
                                  lambda u: -(1/2**0.5)+u[0]**2/2**0.5,
                                  lambda u: -1+1/2*(u[1]**2+u[2]**2),
                                  lambda u: -(3/2)**0.5*u[0]+u[0]**3/6**0.5,
                                  lambda u: Sqrt(3/2)/2-Sqrt(3/2)*u[0]**2+u[0]**4/(2*Sqrt(6)),
                                  lambda u: 1-u[1]**2-u[2]**2+1/8*(u[1]**2+u[2]**2)**2,
                                  lambda u: -u[0]+1/2*u[0]*(u[1]**2+u[2]**2),
                                  lambda u: 1/2**0.5-u[0]**2/2**0.5-(u[1]**2+u[2]**2)/(2*2**0.5)+(u[0]**2*(u[1]**2+u[2]**2))/(2*2**0.5),
                                  lambda u: 1/2*Sqrt(15/2)*u[0]-Sqrt(5/6)*u[0]**3+u[0]**5/(2*Sqrt(30)),
                                  lambda u: -(Sqrt(5)/4)+(3*Sqrt(5)*u[0]**2)/4-(Sqrt(5)*u[0]**4)/4+u[0]**6/(12*Sqrt(5)),
                                  #lambda u: -1+3/2*(u[1]**2+u[2]**2)-3/8*(u[1]**2+u[2]**2)**2+1/48*(u[1]**2+u[2]**2)**3
                                  ]

if __name__ == "__main__":
    pass

"""
    stat = M35_1D_gauged_stats()
    stat._ini_sympy_symbols()
    print(stat._negative_canonical_symbols)
    Tmat = stat.gauge_transformation_matrix(jnp.array([3.,1.,2.]), jnp.array([1,1,0]))
    #Tmat1 = stat.gauge_transformation_matrix( jnp.array([2,2,1]), jnp.array([3.,1.,2.]))
    print(Tmat)
    #print(Tmat1)
    #stat.natural_parameters_from_moments(verbose = True)
"""