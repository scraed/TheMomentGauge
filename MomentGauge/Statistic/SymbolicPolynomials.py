import sympy
import numpy as np
from sympy import symbols, Matrix, RR, expand_complex, N
from sympy import Poly, pprint
from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key
from sympy.abc import x, y, z
from sympy.solvers.solvers import solve_linear_system
from sympy import integrate, Symbol,symbols, oo, exp, Poly, summation
from sympy import assuming, Q, ask, diff, solve, simplify, lambdify, expand
from sympy.polys.domainmatrix import DomainMatrix
from sympy.interactive import init_printing
from sympy.core.sorting import default_sort_key
from sympy.core.traversal import bottom_up
def _get_all_monomials( polynomial_list, symbol_list ):
    """
    Get all monomials used in the polynomials provided in the polynomial_list

    Args:
        polynomial_list: a list of sympy polynomials
        symbol_list: a list of sympy symbols which are variables of the polynomials in the polynomial_list

    Returns:
        all_monoms_poly: a list of monomials
    """
    all_monoms = []
    for poly in  polynomial_list:
        all_monoms = all_monoms+poly.monoms() 
    all_monoms = np.unique( np.array( all_monoms ) , axis=0)
    all_monoms_poly = [ sorted(itermonomials(symbol_list, order, order), reverse=True, key=monomial_key('lex', symbol_list))[0] for order in  all_monoms]
    return all_monoms_poly

def _get_convert_mat_between_polynomial_lists( sympy_poly_sub_list, sympy_poly_list, polysymbols ):
    """
    Compute the injection and projection matrix from a list of polynomials to its linear subspaces spaned by a subset.

    Args:
        sympy_poly_sub_list: a list of sympy polynomials to be projected onto from sympy_poly_list
        sympy_poly_list: a list of sympy polynomials to be injected into from sympy_poly_sub_list
        polysymbols: a list of sympy symbols which are variables of the polynomials in the polynomial_list

    Returns:
        A tuple containing

            inj_mat: a matrx of shape(  len(sympy_poly_list), len(sympy_poly_sub_list) ) 
                Matrix that inject a combination of sympy_poly_sub_list to a combination of sympy_poly_list. Explicitly we have inj * sympy_poly_sub_list = projection of sympy_poly_list onto the span of sympy_poly_list
                    
            proj_mat: a matrx of shape( len(sympy_poly_sub_list) , len(sympy_poly_list)) 
                Matrix that  project a combination of sympy_poly_list to a combination of sympy_poly_sub_list. Explicitly we have proj_mat * sympy_poly_list = sympy_poly_sub_list
    """
    monomials = _get_all_monomials(sympy_poly_sub_list + sympy_poly_list, polysymbols)
    polycoefs_poly_list = np.array( [[  poly.coeff_monomial(mono) for mono in monomials ] for poly in sympy_poly_list] )
    polycoefs_sub_list = np.array( [[  poly.coeff_monomial(mono) for mono in monomials ] for poly in sympy_poly_sub_list] )
    coefs_poly_list = DomainMatrix.from_list_sympy( *polycoefs_poly_list.shape, polycoefs_poly_list)
    coefs_sub_list = DomainMatrix.from_list_sympy( *polycoefs_sub_list.shape, polycoefs_sub_list)
    coefs_poly_list = coefs_poly_list.convert_to(coefs_poly_list.domain.get_field())
    coefs_sub_list = coefs_sub_list.convert_to(coefs_sub_list.domain.get_field())
    #print("coefs_sub_list", coefs_sub_list.shape)
    #for coef in list(coefs_sub_list.to_Matrix()):
    #    print(coef)
    assert coefs_poly_list.rank() == coefs_poly_list.shape[0] , "The polynomial list is not linearly independent"
    assert coefs_sub_list.rank() == coefs_sub_list.shape[0] , "The polynomial sublist is not linearly independent"

    def pinv_row_indept_symbolic(mat):
        matmatT = mat*mat.transpose()
        #pprint(matmatT)
        return mat.transpose()*matmatT.inv()
    pinv_coefs_poly_list = pinv_row_indept_symbolic(coefs_poly_list)
    pinv_coefs_sub_list = pinv_row_indept_symbolic(coefs_sub_list)
    proj_mat = coefs_sub_list*pinv_coefs_poly_list
    inj_mat = coefs_poly_list*pinv_coefs_sub_list
    proj_inj_mat = proj_mat*inj_mat
    assert (proj_inj_mat - proj_inj_mat.eye( proj_inj_mat.shape, proj_inj_mat.domain )).is_zero_matrix, "The polynomial sublist is not a linear subspace of polynomial list, the projection is not exact"
    return inj_mat.to_Matrix(), proj_mat.to_Matrix()

def _get_aux_paras(expr, list_of_symbols):
    symbols_in_expr = list(expr.free_symbols)
    symbols_in_expr = [*set(symbols_in_expr)]
    aux_symbols = set(symbols_in_expr) - set(list_of_symbols)
    aux_symbols = list(aux_symbols)
    return sorted( aux_symbols, key=default_sort_key )


class SympyGaugeTransformation():
    def __init__(self, suff_stats):
        r"""The mixin class to generate gauge transformation automatically using sympy symbolic computation.
        Parameters
        ----------
        suff_stats : list
            a list of moment functions [:math:`\phi_i`, i=0,\cdots,N-1] in which each :math:`\phi_i` is a polynomial function :math:`\phi_i` ( **u**, :math:`*` **gauge_paras** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector (:math:`u_x`, :math:`u_y`, :math:`u_z`)

                    :math:`*` **gauge_paras** : - Arbitrary many extra parameters. The :math:`*` refers to the unpacking operator in python.

                **Returns**: 
                
                    float -- the moment value
        """
        self._ini_sympy_symbols(suff_stats)
                
    def _ini_sympy_symbols(self,suff_stats):
        """set sympy symbols necessary to compute the gauge transformation.
        """
        self._ux, self._uy, self._uz = symbols('u_x u_y u_z') # Symbols for moleculer velocities
        ux, uy, uz = self._ux, self._uy, self._uz

        self._m = m = Symbol('m', positive=True) # Symbols for molecular mass
        self._uvelocity = uvelocity = [self._ux, self._uy, self._uz]
        exec( "self._sufficient_statistics_symbols = "+"[" + ",".join(["Symbol('M" + str(i)+"',real=True)" for i in range(len(suff_stats))]) + "]" ) # A list of sympy symbols for moments of sufficient statistics
        exec( "self._negative_canonical_symbols = "+"[" + ",".join(["Symbol('nB" + str(i)+"',real=True)" for i in range(len(suff_stats))]) + "]" ) # A list of sympy symbols for natural parameters of sufficient statistics
        #self._suff_stats_input_symbols = [ self._uvelocity, *gauge_symbols ]

        ############Symbols for flow property #######
        n = Symbol('n', positive=True) # number density
        rho = Symbol('rho', positive=True)
        T = Symbol('T', positive=True)
        vx, vy, vz = symbols('v_x v_y v_z') # Flow velocities
        sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz= symbols('sigma_xx sigma_xy sigma_xz sigma_yy sigma_yz sigma_zz') 
        p = Symbol('p', positive=True) # The pressure p = n kB T
        q_x, q_y, q_z = symbols('q_x q_y q_z')

        self._flow_property_symbols = [ rho, n, vx, vy, vz, T, p, sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz, q_x, q_y, q_z  ]

        pressure_poly = Poly( m*( (ux-vx)*(ux-vx) + (uy-vy)*(uy-vy) + (uz-vz)*(uz-vz) )/3, *uvelocity)

        self._macroscopic_quantities_poly = {rho: Poly(m, *uvelocity), vx: Poly( ux/n, *uvelocity), vy: Poly( uy/n, *uvelocity), vz: Poly( uz/n, *uvelocity) ,
                                        sigma_xx: Poly(m*(ux-vx)*(ux-vx) - pressure_poly.as_expr(), *uvelocity), sigma_xy: Poly( m*(ux-vx)*(uy-vy) , *uvelocity), sigma_xz: Poly(m*(ux-vx)*(uz-vz) , *uvelocity),
                                        sigma_yy: Poly(m*(uy-vy)*(uy-vy) - pressure_poly.as_expr(), *uvelocity), sigma_yz: Poly( m*(uy-vy)*(uz-vz) , *uvelocity), sigma_zz: Poly(m*(uz-vz)*(uz-vz) - pressure_poly.as_expr(), *uvelocity), p: pressure_poly , 
                                        q_x: Poly( (ux-vx)*pressure_poly.as_expr()*3/2, *uvelocity), q_y: Poly( (uy-vy)*pressure_poly.as_expr()*3/2 , *uvelocity) , q_z: Poly( (uz-vz)*pressure_poly.as_expr()*3/2 , *uvelocity) }


    def _get_Sympy_polynomials_statistics(self, poly_list, poly_symbols):
        r"""
        Convert the python functions into sympy polynomials

        Parameters
        ----------
        poly_list : list
            a list of functions [:math:`\phi_i,i=0,\cdots,M`], in which each :math:`\phi_i` is a polynomial function :math:`\phi_i` ( **u** , :math:`*` **gauge_paras** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector 

                    :math:`*` **gauge_paras** : - Arbitrary many extra parameters. The :math:`*` refers to the unpacking operator in python.

                **Returns**: 
                
                    float -- the moment value
            
            The lengh of the list may vary.
        poly_symbols : list or tuple
            A list containig sympy symbols as parameters required by functions in poly_list, such as [ [ux, uy, uz], paras1, paras2, ...  ] in which ux, uy, yz are sympy symbols and paras1, paras2 contain other symbols.

        Returns
        -------
        list 
            A list containing sympy polynomials for each sufficient statistics. Its length equals len(self.sufficient_statistics)
        """
        exprs = [ simplify( poly_list[i](*poly_symbols), rational=True,full=True ) for i in range(len(poly_list)) ]
        #pprint(exprs)
        return [ Poly(expr, self._ux,self._uy,self._uz) for expr in exprs ]
    def get_gauge_transformation(self, suff_stats, gauge_symbols_source, gauge_symbols_target, transformation_inputs):
        r"""Compute the gauge transformation matrix between different gauge parameters

        .. math::
            :nowrap:

            \begin{equation}
                \phi_i(\mathbf{u}, \mathbf{g}' )= T_{ij}(\mathbf{g}',\mathbf{g}) \phi_j(\mathbf{u}, \mathbf{g})
            \end{equation}

        Parameters
        ----------
        suff_stats : list
            a list of moment functions [:math:`\phi_i`, i=0,\cdots,N-1] in which each :math:`\phi_i` is a polynomial function :math:`\phi_i` ( **u**, :math:`*` **gauge_paras** ) whose
                
                **Parameters**:

                    **u** : float array of shape (3) - The 3D sample vector (:math:`u_x`, :math:`u_y`, :math:`u_z`)

                    :math:`*` **gauge_paras** : - Arbitrary many extra parameters. The :math:`*` refers to the unpacking operator in python.

                **Returns**: 
                
                    float -- the moment value
        gauge_symbols_source : tuple or list
            A tuple or list containing sympy symbols of **gauge_paras** before transformation. These symbols must match the input of functions in **suff_stats**.
            
                For example, suppose functions in **suff_stats** takes input ( **u**, (**a**, **b**, **c**) ), the **gauge_symbols_source** should be a 
                nested tuple ( (**a**, **b**, **c**), ), in which **a**, **b**, and **c** are sympy symbols. Alternatively, suppose functions in **suff_stats** takes input ( **u**, **a**, **b**, **c** ), the **gauge_symbols_source** should be a 
                tuple (**a**, **b**, **c**), in which **a**, **b**, and **c** are sympy symbols.
        gauge_symbols_target : tuple or list
            A tuple or list containing sympy symbols of **gauge_paras** after transformation. These symbols must match the input of functions in **suff_stats**.
            
                For example, suppose functions in **suff_stats** takes input ( **u**, (**a**, **b**, **c**) ), the **gauge_symbols_source** should be a 
                nested tuple ( (**a'**, **b'**, **c'**), ), in which **a'**, **b'**, and **c'** are sympy symbols. Alternatively, suppose functions in **suff_stats** takes input ( **u**, **a**, **b**, **c** ), the **gauge_symbols_source** should be a 
                tuple (**a'**, **b'**, **c'**), in which **a'**, **b'**, and **c'** are sympy symbols.
        transformation_inputs: tuple or list
            A tuple or list containing sympy symbols specifying the input of the matrix valued function :math:`T_{ij}`. For example, (**a'**, **b'**, **c'**, **a**, **b**, **c**)
        Returns
        -------
        function
            a matrix valued function :math:`T_{ij}` whose
                
                **Parameters**:

                    **inputs** : float array of shape (len(**transformation_inputs**)) - The array containing values of symbols specified in **transformation_inputs**.

                **Returns**: 
                
                    float array of shape (N,N) -- the matrix :math:`T_{ij}`
        """
        #moments_target = self._get_Sympy_polynomials_statistics(self.suff_stats,self._suff_stats_input_symbols_D)
        #moments_source = self._get_Sympy_polynomials_statistics(self.suff_stats,self._suff_stats_input_symbols)
        suff_stats_input_target = [ self._uvelocity, *gauge_symbols_target ]
        suff_stats_input_source = [ self._uvelocity, *gauge_symbols_source ]
        moments_target = self._get_Sympy_polynomials_statistics(suff_stats,suff_stats_input_target)
        moments_source = self._get_Sympy_polynomials_statistics(suff_stats,suff_stats_input_source)
        #print("solve")
        inj_mat, proj_mat = _get_convert_mat_between_polynomial_lists( moments_target, moments_source, self._uvelocity )

        
        aux_symbols = _get_aux_paras(proj_mat, transformation_inputs)
        assert len(aux_symbols) == 0, "unknown vairables " + str(aux_symbols)  +  " in transformation are not covered by the designed input"
        T_mat_func = lambdify([transformation_inputs], simplify( proj_mat.evalf() ).evalf(), "jax",cse=True)


        #def gauge_transformation( array  ):
        #    sr2, sx2, wx2, sr1, sx1, wx1 = array
        #    return T_mat_func( (sr2/sr2, sx2/sx1, wx2-wx1 , sr1, sx1,wx1 ))
        return T_mat_func   
    def get_gauge_paras_s_wx_1D(self,suff_stats, gauge_symbols, transformation_inputs):

        ###############Prepare symbols#########
        m = self._m
        rho, n, vx, vy, vz, T, p, sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz, q_x, q_y, q_z = self._flow_property_symbols
        uvelocity = self._uvelocity
        substitute = {vy: 0, vz: 0, rho: n*m}

        ###############Get polynomials of sufficient statistics for moments#########
        suff_stats_input = [ self._uvelocity, *gauge_symbols ]
        moments = self._get_Sympy_polynomials_statistics(suff_stats,suff_stats_input)

        ###############Get polynomials of desired flow properties#########
        rho_poly = self._macroscopic_quantities_poly[rho]
        vx_poly = self._macroscopic_quantities_poly[vx]
        p_poly = self._macroscopic_quantities_poly[p]
        
        p_poly = self._macroscopic_quantities_poly[p]
        ###############the number density n in terms of moments and gauge parameters#########
        targets = { "n": Poly( rho_poly.subs(substitute)/m, *uvelocity ), 
                    "v_x": Poly( vx_poly.subs(substitute), *uvelocity )   }
        inj_mat, proj_mat = _get_convert_mat_between_polynomial_lists( [targets["n"],], moments, self._uvelocity )
        n_expr = (proj_mat*Matrix(self._sufficient_statistics_symbols)).__getitem__((0,0))
        ###############the flow velocity vx in terms of moments and gauge parameters#########
        inj_mat, proj_mat = _get_convert_mat_between_polynomial_lists( [targets["v_x"],], moments, self._uvelocity )
        vx_expr = (proj_mat*Matrix(self._sufficient_statistics_symbols)).__getitem__((0,0)).subs({n: n_expr})
        substitute2 = {n: n_expr, vx: vx_expr}


        ###############Get polynomials of desired gauge parameters from flow properties#########
        targets = { "w_x": Poly( vx_poly.subs(substitute), *uvelocity ), 
                    "s^2": Poly( p_poly.subs(substitute)/m/n, *uvelocity )   }

        ###############Get Transformation from sufficient statistics to desired flow properties#########
        inj_mat, proj_mat = _get_convert_mat_between_polynomial_lists( [targets["w_x"], targets["s^2"]], moments, self._uvelocity )
        proj_mat = simplify(proj_mat.subs(substitute2))
        aux_symbols = _get_aux_paras(proj_mat, transformation_inputs)
        assert len(aux_symbols) == 0, "unknown vairables " + str(aux_symbols)  +  " in transformation are not covered by the designed input"

        T_mat_func = lambdify([transformation_inputs], simplify( proj_mat.evalf() ).evalf(), "jax",cse=True)
        return T_mat_func   
    def get_gauge_paras_sr_sx_wx_1D(self,suff_stats, gauge_symbols, transformation_inputs):

        ###############Prepare symbols#########
        m = self._m
        rho, n, vx, vy, vz, T, p, sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz, q_x, q_y, q_z = self._flow_property_symbols
        uvelocity = self._uvelocity
        substitute = {vy: 0, vz: 0, rho: n*m}

        ###############Get polynomials of sufficient statistics for moments#########
        suff_stats_input = [ self._uvelocity, *gauge_symbols ]
        moments = self._get_Sympy_polynomials_statistics(suff_stats,suff_stats_input)

        ###############Get polynomials of desired flow properties#########
        rho_poly = self._macroscopic_quantities_poly[rho]
        vx_poly = self._macroscopic_quantities_poly[vx]
        p_poly = self._macroscopic_quantities_poly[p]
        sigmaxx_poly = self._macroscopic_quantities_poly[sigma_xx]
        sigmayy_poly = self._macroscopic_quantities_poly[sigma_yy]
        sigmazz_poly = self._macroscopic_quantities_poly[sigma_zz]
        
        p_poly = self._macroscopic_quantities_poly[p]
        ###############the number density n in terms of moments and gauge parameters#########
        targets = { "n": Poly( rho_poly.subs(substitute)/m, *uvelocity ), 
                    "v_x": Poly( vx_poly.subs(substitute), *uvelocity )   }
        inj_mat, proj_mat = _get_convert_mat_between_polynomial_lists( [targets["n"],], moments, self._uvelocity )
        n_expr = (proj_mat*Matrix(self._sufficient_statistics_symbols)).__getitem__((0,0))
        ###############the flow velocity vx in terms of moments and gauge parameters#########
        inj_mat, proj_mat = _get_convert_mat_between_polynomial_lists( [targets["v_x"],], moments, self._uvelocity )
        vx_expr = (proj_mat*Matrix(self._sufficient_statistics_symbols)).__getitem__((0,0)).subs({n: n_expr})
        substitute2 = {n: n_expr, vx: vx_expr}


        ###############Get polynomials of desired gauge parameters from flow properties#########
        targets = { "w_x": Poly( vx_poly.subs(substitute), *uvelocity ), 
                    "sx^2": Poly( (p_poly + sigmaxx_poly).subs(substitute)/m/n, *uvelocity ),
                    "sr^2": Poly( (p_poly + (sigmayy_poly + sigmazz_poly)/2).subs(substitute)/m/n, *uvelocity )   }

        ###############Get Transformation from sufficient statistics to desired flow properties#########
        inj_mat, proj_mat = _get_convert_mat_between_polynomial_lists( [targets["w_x"], targets["sx^2"], targets["sr^2"]], moments, self._uvelocity )
        proj_mat = simplify(proj_mat.subs(substitute2))
        aux_symbols = _get_aux_paras(proj_mat, transformation_inputs)
        assert len(aux_symbols) == 0, "unknown vairables " + str(aux_symbols)  +  " in transformation are not covered by the designed input"

        T_mat_func = lambdify([transformation_inputs], simplify( proj_mat.evalf() ).evalf(), "jax",cse=True)
        return T_mat_func   
    def conservation_projection_1D(self,suff_stats, gauge_symbols, transformation_inputs):

        ###############Prepare symbols#########
        m = self._m
        rho, n, vx, vy, vz, T, p, sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz, q_x, q_y, q_z = self._flow_property_symbols
        uvelocity = self._uvelocity
        ux, uy, uz = uvelocity
        substitute = {vy: 0, vz: 0, rho: n*m}

        ###############Get polynomials of sufficient statistics for moments#########
        suff_stats_input = [ self._uvelocity, *gauge_symbols ]
        moments = self._get_Sympy_polynomials_statistics(suff_stats,suff_stats_input)

        conserved_poly = [Poly(1, *uvelocity), Poly(ux, *uvelocity), Poly(ux*ux + uy*uy + uz*uz, *uvelocity)]

        ###############Get Transformation from sufficient statistics to desired flow properties#########
        inj_mat, proj_mat = _get_convert_mat_between_polynomial_lists( conserved_poly, moments, self._uvelocity )
        
        inj_proj_mat = simplify(inj_mat*proj_mat)

        aux_symbols = _get_aux_paras(inj_proj_mat, transformation_inputs)
        assert len(aux_symbols) == 0, "unknown vairables " + str(aux_symbols)  +  " in transformation are not covered by the designed input"

        T_mat_func = lambdify([transformation_inputs], simplify( inj_proj_mat.evalf() ).evalf(), "jax",cse=True)
        return T_mat_func   






