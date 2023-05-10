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
from MomentGauge.Statistic.PolyStatistics import PolyStatistics, M35_1D_stats, Maxwellian_1D_stats, ESBGK_1D_stats, M35_P2_1D_stats
from MomentGauge.Statistic.SymbolicPolynomials import SympyGaugeTransformation
#init_printing(pretty_print=True) 
######################### The basic symbols for velocities ################

     

class PolyGaugedStatistics(PolyStatistics):
    def __init__(self, base_statistics: PolyStatistics):
        r"""The base class for store pre-defined gauged polynomial statistics. 
        
        The gauged statistics are transformation of ordinary polynomial statistics by a gauge transformation :math:`A_{ij}(\mathbf{g})`. 
        
        Specifically, given a set of polynomial statistics :math:`\phi_i(\mathbf{u})`, their gauged version is a set of statistics :math:`\phi_i(\mathbf{u}, \mathbf{g})` admits extra gauge parameters :math:`\mathbf{g}` such that

        .. math::
            :nowrap:

            \begin{equation}
            \phi_i(\mathbf{u}, \mathbf{g}) = A_{ij}(\mathbf{g})\phi_j(\mathbf{u}) 
            \end{equation}

        Moreover we have gauge transformation between different gauge parameters

        .. math::
            :nowrap:

            \begin{equation}
            \phi_i(\mathbf{u}, \mathbf{g}') = T_{ij}(\mathbf{g}',\mathbf{g})\phi_j(\mathbf{u}, \mathbf{g}) 
            \end{equation}

        Attributes
        ----------
        suff_stats : list
            a list of moment functions [:math:`\phi_i`, i=0,\cdots,N-1] in which each :math:`\phi_i` is a polynomial function :math:`\phi_i` ( **u**, :math:`*` **gauge_paras** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector (:math:`u_x`, :math:`u_y`, :math:`u_z`)

                    :math:`*` **gauge_paras** : - Arbitrary many extra parameters. The :math:`*` refers to the unpacking operator in python.

                **Returns**: 
                
                    float -- the moment value

        """
        self.base_statistics = base_statistics
        self.suff_stats = [ self.gauge(func) for func in self.base_statistics.suff_stats   ]

    def gauge(self,func):
        r"""Convert the functions of :math:`\mathbf{u}` into function of :math:`\mathbf{u}` and gauge parameters

        Parameters
        ----------
        func : function
            a polynomial function :math:`\phi` ( **u** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector 

                **Returns**: 
                
                    float -- the moment value

        Returns
        -------
        function
            a polynomial function :math:`\phi` ( **u**, :math:`*` **gauge_paras** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector 

                    :math:`*` **gauge_paras** : - Arbitrary many extra parameters. The :math:`*` refers to the unpacking operator in python.

                **Returns**: 
                
                    float -- the moment value
        """
        raise NotImplementedError

    def gauge_transformation_matrix( self, gauge_para2 = (), gauge_para1 = () ):
        r"""Compute the gauge transformation matrix :math:`T_{ij}(\mathbf{g}',\mathbf{g})` between different gauge parameters

        .. math::
            :nowrap:

            \begin{equation}
            \phi_i(\mathbf{u}, \mathbf{g}') = T_{ij}(\mathbf{g}',\mathbf{g})\phi_j(\mathbf{u}, \mathbf{g}) 
            \end{equation}

        Parameters
        ----------
        gauge_para2 : tuple
            Tuple containing arbitrary many extra gauge parameters such as :math:`\mathbf{g}'`
        gauge_para1 : tuple
            Tuple containing arbitrary many extra gauge parameters such as :math:`\mathbf{g}`

        Returns
        -------
        float array of shape (N,N)
            the matrix :math:`T_{ij}(\mathbf{g}',\mathbf{g})`
        """
        raise NotImplementedError
    def standard_gauge_paras( self, moments, gauge_para = () ):
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
        gauge_para = gauge_para[0]

        Tmat = self.__get_gauge_para_mat( (*moments, *gauge_para) )
        w_x, sx2, sr2 = Tmat.dot(moments)

        return jnp.asarray( [sr2**0.5, sx2**0.5, w_x])

class Maxwellian_1D_gauged_stats(PolyGaugedStatistics):
    def __init__(self):
        r"""The 1D version of polynomial statistics for 35 moments with gauge transformation. 
        
        Attributes
        ----------
        suff_stats : list of length (9)
            a list of moment functions [:math:`\phi_i,i=0,\cdots,2`] in which each :math:`\phi_i` is a polynomial function :math:`\phi_i` ( **u**, **gauge_paras** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector (:math:`u_x`, :math:`u_y`, :math:`u_z`)

                    **gauge_paras** : float array of shape (3) - The array (:math:`s`,:math:`w_x`).

                **Returns**: 
                
                    float -- the moment value
            
            Specifically,

            :math:`\phi_0` (**u** ) = 1.

            :math:`\phi_1` (**u** ) = :math:`\bar{u}_x`
            
            :math:`\phi_2` (**u** ) = :math:`\bar{u}_x^2 + \bar{u}_y^2 + \bar{u}_z^2`.
            
            in which :math:`\bar{u}_x = \frac{u_x - w_x}{s}`, :math:`\bar{u}_y = \frac{u_y}{s}`, :math:`\bar{u}_z = \frac{u_z}{s}`
        """
        super().__init__(Maxwellian_1D_stats())
        self.SymGauge = SympyGaugeTransformation(self.suff_stats)
        self._wx = Symbol('w_x')
        self._s = Symbol('s', positive=True)
        self._dwx = Symbol('dw_x')
        self._rs = Symbol('rs', positive=True)
        gauge_symbols_source = ( [ self._s, self._wx], ) 
        gauge_symbols_target =  ( [ self._s*self._rs, self._wx + self._dwx ], )
        GaugeTransinput =  [self._rs,self._dwx,self._s,self._wx]
        self.__gauge_transformation = jax.jit( self.SymGauge.get_gauge_transformation(self.suff_stats,gauge_symbols_source, gauge_symbols_target, GaugeTransinput ) )
        self.__get_gauge_para_mat = jax.jit( self.SymGauge.get_gauge_paras_s_wx_1D(self.suff_stats,gauge_symbols_source, [ *self.SymGauge._sufficient_statistics_symbols, *gauge_symbols_source[0]] ) )
    def gauge(self,func):
        r"""Convert the functions of :math:`\mathbf{u}` into function of :math:`\mathbf{u}` and gauge parameters (:math:`s`, :math:`w_x`).

        Parameters
        ----------
        func : function
            a polynomial function :math:`\phi` ( **u** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector 

                **Returns**: 
                
                    float -- the moment value

        Returns
        -------
        function
            a polynomial function :math:`\phi` ( **u**, **gauge_paras** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector 

                    **gauge_paras** : float array of shape (3) - The array (:math:`s`, :math:`w_x`).

                **Returns**: 
                
                    float -- the moment value
        """
        def gauged_func(u, gauge_paras):
            s, wx = gauge_paras
            return func( [ (u[0]-wx)/s, u[1]/s, u[2]/s ] )
        return gauged_func

    def gauge_transformation_matrix( self, gauge_para2 = (), gauge_para1 = () ):
        r"""Compute the gauge transformation matrix between different gauge parameters

        .. math::
            :nowrap:

            \begin{equation}
                \phi_i(\mathbf{u}, (s', w_x') )= T_{ij}(s', w_x',s, w_x) \phi_j(\mathbf{u}, (s, w_x)); \quad i,j = 0, \cdots, 2
            \end{equation}

        Parameters
        ----------
        gauge_para2 : tuple
            A tuple containing the array (:math:`s'`, :math:`w_x'`) which is a float array of shape (2).
        gauge_para1 : 
            A tuple containing the array (:math:`s`, :math:`w_x`) which is a float array of shape (2).

        Returns
        -------
         float array of shape (3,3)
            the matrix :math:`T_{ij}(s', w_x',s, w_x)`
        """
        gauge_para2 = gauge_para2[0]
        gauge_para1 = gauge_para1[0]
        
        s2, wx2, s1, wx1 = (*gauge_para2, *gauge_para1)
        return self.__gauge_transformation( (s2/s1, wx2-wx1 , s1, wx1 ) )
    def standard_gauge_paras( self, moments, gauge_para = () ):
        r"""Compute the Hermite gauge parameters

        Parameters
        ----------
        moments : float array of shape (3)
            The array containing moments of sufficient statistics :math:`(M_0(s, w_x), M_1(s, w_x), M_2(s, w_x))`
        gauge_para : tuple
            A tuple containing the array (:math:`s`, :math:`w_x`) which is a float array of shape (2).

        Returns
        -------
        float array of shape (2)
            the Hermite gauge parameters (:math:`s`, :math:`w_x`)
        """
        gauge_para = gauge_para[0]

        Tmat = self.__get_gauge_para_mat( (*moments, *gauge_para) )
        w_x, s2 = Tmat.dot(moments)

        delta = jnp.finfo(s2.dtype).resolution
        s2=jnp.maximum(s2,delta)

        return jnp.asarray( [s2**0.5, w_x])


class PolyGaugedStatistics_sr_sx_wx(PolyGaugedStatistics):
    def __init__(self, base_statistics: PolyStatistics):
        r"""The base class for store pre-defined gauged polynomial statistics with gauge parameter :math:`s_r`, :math:`s_x`, and :math:`w_x`
        
        The gauged statistics are transformation of ordinary polynomial statistics by a gauge transformation :math:`A_{ij}(\mathbf{g})`. 
        
        Specifically, given a set of polynomial statistics :math:`\phi_i(\mathbf{u})`, their gauged version is a set of statistics :math:`\phi_i(\mathbf{u}, \mathbf{g})` admits extra gauge parameters :math:`\mathbf{g}` such that

        .. math::
            :nowrap:

            \begin{equation}
            \phi_i(\mathbf{u}, \mathbf{g}) = A_{ij}(\mathbf{g})\phi_j(\mathbf{u}) = \phi_i(\bar{\mathbf{u}})
            \end{equation}

        such that :math:`\bar{\mathbf{u}}={ \bar{u}_x, \bar{u}_y, \bar{u}_z }`, :math:`\bar{u}_x = \frac{u_x - w_x}{s_x}`, :math:`\bar{u}_y = \frac{u_y}{s_r}`, :math:`\bar{u}_z = \frac{u_z}{s_r}`.

        Attributes
        ----------
        suff_stats : list
            a list of moment functions [:math:`\phi_i`, i=0,\cdots,N-1] in which each :math:`\phi_i` is a polynomial function :math:`\phi_i` ( **u**, :math:`*` **gauge_paras** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector (:math:`u_x`, :math:`u_y`, :math:`u_z`)

                    :math:`*` **gauge_paras** : - Arbitrary many extra parameters. The :math:`*` refers to the unpacking operator in python.

                **Returns**: 
                
                    float -- the moment value

        """
        super().__init__(base_statistics)
        self.SymGauge = SympyGaugeTransformation(self.suff_stats)

        self._wx = Symbol('w_x')
        self._sx = Symbol('s_x', positive=True)
        self._sr = Symbol('s_r', positive=True)
        self._dwx = Symbol('dw_x')
        self._rsx = Symbol('rs_x', positive=True)
        self._rsr = Symbol('rs_r', positive=True)
        gauge_symbols_source = ( [ self._sr, self._sx, self._wx], ) 
        gauge_symbols_target =  ( [ self._sr*self._rsr, self._sx*self._rsx, self._wx + self._dwx ], )
        GaugeTransinput =  [self._rsr,self._rsx,self._dwx,self._sr,self._sx,self._wx]

        self.__gauge_transformation = jax.jit( self.SymGauge.get_gauge_transformation(self.suff_stats,gauge_symbols_source, gauge_symbols_target, GaugeTransinput ) )
        self.__get_gauge_para_mat = jax.jit( self.SymGauge.get_gauge_paras_sr_sx_wx_1D(self.suff_stats,gauge_symbols_source, [ *self.SymGauge._sufficient_statistics_symbols, *gauge_symbols_source[0]] ) )
    def gauge(self,func):
        r"""Convert the functions of :math:`\mathbf{u}` into function of :math:`\mathbf{u}` and gauge parameters (:math:`s_r`, :math:`s_x`, :math:`w_x`).

        Parameters
        ----------
        func : function
            a polynomial function :math:`\phi` ( **u** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector 

                **Returns**: 
                
                    float -- the moment value

        Returns
        -------
        function
            a polynomial function :math:`\phi` ( **u**, **gauge_paras** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector 

                    **gauge_paras** : float array of shape (3) - The array (:math:`s_r`, :math:`s_x`, :math:`w_x`).

                **Returns**: 
                
                    float -- the moment value
        """
        def gauged_func(u, gauge_paras):
            sr, sx, wx = gauge_paras
            return func( [ (u[0]-wx)/sx, u[1]/sr, u[2]/sr ] )
        return gauged_func

    def gauge_transformation_matrix( self, gauge_para2 = (), gauge_para1 = () ):
        r"""Compute the gauge transformation matrix between different gauge parameters

        .. math::
            :nowrap:

            \begin{equation}
                \phi_i(\mathbf{u}, (s_r', s_x', w_x') )= T_{ij}(s_r', s_x', w_x',s_r, s_x, w_x) \phi_j(\mathbf{u}, (s_r, s_x, w_x)); \quad i,j = 0, \cdots, M
            \end{equation}
        in which M is the number of sufficient statistics

        Parameters
        ----------
        gauge_para2 : tuple
            A tuple containing the array (:math:`s_r'`, :math:`s_x'`, :math:`w_x'`) which is a float array of shape (3).
        gauge_para1 : 
            A tuple containing the array (:math:`s_r`, :math:`s_x`, :math:`w_x`) which is a float array of shape (3).

        Returns
        -------
         float array of shape (M,M)
            the matrix :math:`T_{ij}(s_r', s_x', w_x',s_r, s_x, w_x)`
        """
        gauge_para2 = gauge_para2[0]
        gauge_para1 = gauge_para1[0]
        
        sr2, sx2, wx2, sr1, sx1, wx1 = (*gauge_para2, *gauge_para1)
        return self.__gauge_transformation( (sr2/sr1, sx2/sx1, wx2-wx1 , sr1, sx1,wx1 ) )
    def standard_gauge_paras( self, moments, gauge_para = () ):
        r"""Compute the Hermite gauge parameters

        Parameters
        ----------
        moments : float array of shape (9)
            The array containing moments of sufficient statistics :math:`(M_0(s_r, s_x, w_x), \cdots, M_8(s_r, s_x, w_x))`
        gauge_para : tuple
            A tuple containing the array (:math:`s_r`, :math:`s_x`, :math:`w_x`) which is a float array of shape (2).

        Returns
        -------
        float array of shape (3)
            the Hermite gauge parameters (:math:`s_r`, :math:`s_x`, :math:`w_x`)
        """
        gauge_para = gauge_para[0]

        Tmat = self.__get_gauge_para_mat( (*moments, *gauge_para) )
        w_x, sx2, sr2 = Tmat.dot(moments)
        #jax.debug.print("residuals: {x}", x=(w_x, sx2, sr2), ordered=True)

        
        delta = jnp.finfo(sx2.dtype).resolution
        sx2=jnp.maximum(sx2,delta)
        sr2=jnp.maximum(sr2,delta)

        

        return jnp.asarray( [sr2**0.5, sx2**0.5, w_x])



class ESBGK_1D_gauged_stats(PolyGaugedStatistics_sr_sx_wx):
    def __init__(self):
        r"""The 1D version of polynomial statistics for ESBGK moments with gauge transformation. 
        
        Attributes
        ----------
        suff_stats : list of length (4)
            a list of moment functions [:math:`\phi_i,i=0,\cdots,3`] in which each :math:`\phi_i` is a polynomial function :math:`\phi_i` ( **u**, **gauge_paras** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector (:math:`u_x`, :math:`u_y`, :math:`u_z`)

                    **gauge_paras** : float array of shape (3) - The array (:math:`s_r`, :math:`s_x`, :math:`w_x`).

                **Returns**: 
                
                    float -- the moment value
            
            Specifically,

            .. math::
                :nowrap:

                \begin{equation}
                \{\phi_i(\mathbf{u}, (s_r, s_x, w_x) ),i=0,\cdots,3\} = \left\{1,\bar{u}_x,\bar{u}_x^2,\bar{u}_r^2,\right\}
                \end{equation}
            
            in which :math:`\bar{u}_x = \frac{u_x - w_x}{s_x}`, :math:`\bar{u}_r = \frac{u_r}{s_r}`, :math:`u_r = \sqrt{u_y^2+u_z^2}`
        """
        super().__init__(ESBGK_1D_stats())


class M35_1D_gauged_stats(PolyGaugedStatistics_sr_sx_wx):
    def __init__(self):
        r"""The 1D version of polynomial statistics for 35 moments with gauge transformation. 
        
        Attributes
        ----------
        suff_stats : list of length (9)
            a list of moment functions [:math:`\phi_i,i=0,\cdots,8`] in which each :math:`\phi_i` is a polynomial function :math:`\phi_i` ( **u**, **gauge_paras** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector (:math:`u_x`, :math:`u_y`, :math:`u_z`)

                    **gauge_paras** : float array of shape (3) - The array (:math:`s_r`, :math:`s_x`, :math:`w_x`).

                **Returns**: 
                
                    float -- the moment value
            
            Specifically,

            .. math::
                :nowrap:

                \begin{equation}
                \{\phi_i(\mathbf{u}, (s_r, s_x, w_x) ),i=0,\cdots,8\} = \left\{1,\bar{u}_x,\frac{\bar{u}_x^2-1}{\sqrt{2}},\frac{\bar{u}_r^2}{2} -1,\frac{\bar{u}_x^3-3\bar{u}_x}{\sqrt{6}},\frac{\bar{u}_x^4-6\bar{u}_x^2+3}{2 \sqrt{6}},\frac{1}{8} \bar{u}_r^4-\bar{u}_r^2+1,\frac{1}{2} \bar{u}_x (\bar{u}_r^2-1),\frac{( \bar{u}_x^2 -1)( \bar{u}_r^2-2)}{2 \sqrt{2}}\right\}
                \end{equation}
            
            in which :math:`\bar{u}_x = \frac{u_x - w_x}{s_x}`, :math:`\bar{u}_r = \frac{u_r}{s_r}`, :math:`u_r = \sqrt{u_y^2+u_z^2}`
        """
        super().__init__(M35_1D_stats())
        self._wx = Symbol('w_x')
        self._sx = Symbol('s_x', positive=True)
        self._sr = Symbol('s_r', positive=True)
        self._dwx = Symbol('dw_x')
        self._rsx = Symbol('rs_x', positive=True)
        self._rsr = Symbol('rs_r', positive=True)
        gauge_symbols_source = ( [ self._sr, self._sx, self._wx], ) 
        gauge_symbols_target =  ( [ self._sr*self._rsr, self._sx*self._rsx, self._wx + self._dwx ], )
        GaugeTransinput =  [self._rsr,self._rsx,self._dwx,self._sr,self._sx,self._wx]
        self.__conservation_proj_mat = self.SymGauge.conservation_projection_1D(self.suff_stats,gauge_symbols_source, [*gauge_symbols_source[0]] )


    def conservative_decomposition( self, moments, gauge_para = () ):
        r"""Decompose the moments as the summation of the conserved part and the non-conserved part

        Parameters
        ----------
        moments : float array of shape (9)
            The array containing moments of sufficient statistics :math:`(M_0(s_r, s_x, w_x), \cdots, M_8(s_r, s_x, w_x))`
        gauge_para : tuple
            A tuple containing the array (:math:`s_r`, :math:`s_x`, :math:`w_x`) which is a float array of shape (2).

        Returns
        -------
        float array of shape (9)
            the conservative part of the moments. The non conservative part is moments - conservative part.
        """
        gauge_para = gauge_para[0]

        Tmat = self.__conservation_proj_mat( (*gauge_para,) )
        conserved_part = Tmat.dot(moments)
        

        return conserved_part

class M35_P2_1D_gauged_stats(PolyGaugedStatistics_sr_sx_wx):
    def __init__(self):
        r"""The 1D version of polynomial statistics for 35 moments with gauge transformation. 
        
        Attributes
        ----------
        suff_stats : list of length (11)
            a list of moment functions [:math:`\phi_i,i=0,\cdots,8`] in which each :math:`\phi_i` is a polynomial function :math:`\phi_i` ( **u**, **gauge_paras** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector (:math:`u_x`, :math:`u_y`, :math:`u_z`)

                    **gauge_paras** : float array of shape (3) - The array (:math:`s_r`, :math:`s_x`, :math:`w_x`).

                **Returns**: 
                
                    float -- the moment value
            
            Specifically,

            .. math::
                :nowrap:

                \begin{equation}
                \{\phi_i(\mathbf{u}, (s_r, s_x, w_x) ),i=0,\cdots,10\} = \left\{1,\bar{u}_x,\frac{\bar{u}_x^2-1}{\sqrt{2}},\frac{\bar{u}_r^2}{2} -1,\frac{\bar{u}_x^3-3\bar{u}_x}{\sqrt{6}},\frac{\bar{u}_x^4-6\bar{u}_x^2+3}{2 \sqrt{6}},\frac{1}{8} \bar{u}_r^4-\bar{u}_r^2+1,\frac{1}{2} \bar{u}_x (\bar{u}_r^2-1),\frac{( \bar{u}_x^2 -1)( \bar{u}_r^2-2)}{2 \sqrt{2}} , \frac{\bar{u}_x^5}{2 \sqrt{30}}-\sqrt{\frac{5}{6}} \bar{u}_x^3+\frac{1}{2} \sqrt{\frac{15}{2}} \bar{u}_x, \frac{\bar{u}_x^6}{12 \sqrt{5}}-\frac{\sqrt{5} \bar{u}_x^4}{4}+\frac{3 \sqrt{5} \bar{u}_x^2}{4}-\frac{\sqrt{5}}{4}   \right\}
                \end{equation}
            
            in which :math:`\bar{u}_x = \frac{u_x - w_x}{s_x}`, :math:`\bar{u}_r = \frac{u_r}{s_r}`, :math:`u_r = \sqrt{u_y^2+u_z^2}`
        """
        super().__init__(M35_P2_1D_stats())

if __name__ == "__main__":
    


    
    '''
    stat = M35_1D_gauged_stats()
    u = jnp.array( [1.,2,3] )
    gauge1 = jnp.array([1,1,0])
    gauge2 = jnp.array([3,2,1])

    def stats(u, gauge1):
        return jnp.asarray( [ st(u,gauge1) for st in stat.suff_stats ] )

    suff1 = stats(u,gauge1 )
    suff2 = stats(u,gauge2 )
    Tmat = stat.gauge_transformation_matrix(gauge_para2 =(gauge2, ),gauge_para1 = (gauge1, ) )


    #print( jnp.max(jnp.abs(suff2 - Tmat.dot(suff1) )) )
    '''
    '''
    gauge1 = jnp.array([1,0])
    gauge2 = jnp.array([3,1])

    stat = Maxwellian_1D_gauged_stats()

    def stats(u, gauge1):
        return jnp.asarray( [ st(u,gauge1) for st in stat.suff_stats ] )

    suff1 = stats(u,gauge1 )
    suff2 = stats(u,gauge2 )

    Tmat = stat.gauge_transformation_matrix(gauge_para2 =(gauge2, ),gauge_para1 = (gauge1, ) )
    print(Tmat)
    print( jnp.max(jnp.abs(suff2 - Tmat.dot(suff1) )) )

    stat.default_gauge_paras(gauge_paras = ())
    '''
    #Tmat1 = stat.gauge_transformation_matrix( jnp.array([2,2,1]), jnp.array([3.,1.,2.]))
    #print(Tmat)
    #print(Tmat1)
    #stat.natural_parameters_from_moments(verbose = True)
    """
    gauge1 = jnp.array([1,0])
    gauge2 = jnp.array([3,1])
    moments = jnp.array([1,2,4])

    stat = Maxwellian_1D_gauged_stats()
    default_gauge = stat.standard_gauge_paras(moments,gauge_para = (gauge1,))
    print(default_gauge)
    """
    """
    gauge1 = jnp.array([1,1,0])
    stat = M35_1D_gauged_stats()
    moments = jnp.array([1,2,3,4,5,6,7,8,9.])
    #default_gauge = stat.standard_gauge_paras(moments,gauge_para = (gauge1,))
    #print(default_gauge)

    m_conserve = stat.conservative_decomposition(moments,gauge_para = (gauge1,) )
    print(m_conserve)
    """
    pass
