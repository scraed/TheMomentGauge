#from Quadratures import Gauss_Legendre_Quadrature_2D_Block, Gauss_Hermite_Quadrature_3D
#from ExponentialFamilyDistributions import Maxwellian, MomentSystem35_2D,MomentSystem35_Maxwell_2D
#from NewtonSolver import Batch_Newton_Iteration_Delta, Batch_Newton_Solver, Newton_Iteration_Delta
from MomentGauge.Utility import generalized_eigh
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jit,vmap
import numpy as np
from functools import partial
from MomentGauge.ShockConfig import PhyConstants, Shock_Grid_info, Shock_Thick_Predict
from MomentGauge.FluxEstimation import Local_Lax_Friedrichs_Flux
from MomentGauge.Models.Moment35 import M35_Gauged_Canonical_Legendre_1D
from MomentGauge.Models.Maxwellian import Maxwell_Canonical_Legendre_1D, Maxwell_Canonical_Gauged_Legendre_1D
from MomentGauge.Models.ESBGK import ESBGK_Canonical_Gauged_Legendre_1D
import numpy as np
import matplotlib.pyplot as plt
import os
#import h5py

#jax.config.update('jax_platform_name', 'cpu')
#jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
#jax.config.update("jax_debug_nans", True) 


def allClose(array1, array2):
    return jnp.allclose( array1, array2, atol=1e-3, rtol=1e-3 )
def interpolate_boundary( initial_parameters_boundary, thickness, cell_centers ):
    value1 = initial_parameters_boundary[0]
    value2 = initial_parameters_boundary[-1]
    #tanhCurve = jnp.heaviside( cell_centers/PhyConst.estimated_thickness,0.5 ) 
    #tanhCurve = 2*(tanhCurve-tanhCurve[0])/(tanhCurve[-1] - tanhCurve[0]) + -1
    tanhCurve = jnp.tanh( cell_centers/( thickness/2 ) ) 
    tanhCurve = 2*(tanhCurve-tanhCurve[0])/(tanhCurve[-1] - tanhCurve[0]) + -1
    para_ini = tanhCurve[:,jnp.newaxis]* ( value2-value1 )/2 + ( value1+value2 )/2
    return para_ini

PhyConst = PhyConstants(Ma=9.925,velocity_std_ratio=(5,5,5))

constant = {"m": PhyConst.m, "kB": PhyConst.kB,"n_x": 8, "n_r": 8, "B_x": 6, "B_r": 4,
            "alpha" : 1., "beta" : 0.5, "c": 5e-4, "atol": 5e-6, "rtol":1e-5,
            "max_iter": 400, "max_back_tracking_iter": 25, "tol": 1e-9, "min_step_size": 1e-6,
            "reg_hessian": "SVD", "debug" : False }
M35 = M35_Gauged_Canonical_Legendre_1D(constant)
ESBGK = ESBGK_Canonical_Gauged_Legendre_1D(constant)
Maxwell = Maxwell_Canonical_Gauged_Legendre_1D(constant) 

Grid_info = Shock_Grid_info(PhyConst, 150)

delta_x = Grid_info.delta_x
cell_interfaces = Grid_info.cell_interfaces
cell_centers = Grid_info.cell_interfaces

CFL_ratio = 0.5 # If nan appears, make it smaller or shrink the velocity domain



#################################################################################
# Utilities
#################################################################################


@jax.jit
def flux_statistics(u, gauge_paras):
    """Statistics for computing fluxes of moments
    """
    stat_values = M35.suff_statistics(u, gauge_paras)
    return u[0]*stat_values
@jax.jit
def cov_statistics(u, gauge_paras):
    """Statistics for covariance among moments
    """
    stat_values = M35.suff_statistics(u, gauge_paras)
    return stat_values[:,jnp.newaxis]*stat_values
@jax.jit
def flux_cov_statistics(u, gauge_paras):
    """Statistics for fluxes of covariance among moments
    """
    return u[0]*cov_statistics(u, gauge_paras)
@jax.jit
def centered_flux_cov_statistics(u, gauge_paras):
    """Statistics for fluxes of covariance among moments
    """
    sr, sx, wx = gauge_paras
    return (u[0]-wx)/sx*cov_statistics(u, gauge_paras)
@jax.jit
def compute_flux(betas, gauge_paras, domain):
    """Computing fluxes of moments
    """
    return M35.natural_paras_to_custom_moments(betas, gauge_paras, domain, flux_statistics, stats_gauge_paras = (gauge_paras,))
@jax.jit
def compute_cov(betas, gauge_paras, domain):
    """Computing covariance of moments
    """
    return M35.natural_paras_to_custom_moments(betas, gauge_paras, domain, cov_statistics, stats_gauge_paras = (gauge_paras,))
@jax.jit
def compute_flux_cov(betas, gauge_paras, domain):
    """Computing fluxes of covariance of moments
    """
    return M35.natural_paras_to_custom_moments(betas, gauge_paras, domain, flux_cov_statistics, stats_gauge_paras = (gauge_paras,))
@jax.jit
def compute_centered_flux_cov(betas, gauge_paras, domain):
    """Computing fluxes of covariance of moments
    """
    return M35.natural_paras_to_custom_moments(betas, gauge_paras, domain, centered_flux_cov_statistics, stats_gauge_paras = (gauge_paras,))
@jax.jit
def compute_chara_speed(betas, gauge_paras, domain):
    jac_M = compute_cov(betas, gauge_paras, domain)
    jac_F = compute_flux_cov(betas, gauge_paras, domain)
    eig_FM, eig_vec_FM = generalized_eigh( jac_F, jac_M )
    max_speed = jnp.max( jnp.abs( eig_FM) )
    return max_speed
@jax.jit
def compute_chara_speed_centered(betas, gauge_paras, domain):
    sr, sx, wx = gauge_paras
    jac_M = compute_cov(betas, gauge_paras, domain)
    jac_F = compute_centered_flux_cov(betas, gauge_paras, domain)
    eig_FM, eig_vec_FM = generalized_eigh( jac_F, jac_M )
    #max_speed = jnp.max( jnp.abs( sx*eig_FM + wx) )
    return sx*eig_FM + wx
@jax.jit
def Local_Lax_Friedrichs_Flux( values_left, values_right):
    """
    Calculate the Lax_Friedrichs_Flux at a cell interface.

    Args:
        values_left: the quantities in the cell at the left of the interface.
                    a tuple (moment, flux, speed) in which moment is an jnp array of shape (m,), 
                    fluxes is an jnp array of shape (m,), speed is a float number.
                n+1 must equal to the number of sufficient statistics.
        values_right: the quantities in the cell at the left of the interface. Similar to values_left.

    Returns:
        The Lax_Friedrichs_flux, float
    """
    moment_left, flux_left, speed_left = values_left
    moment_right, flux_right, speed_right = values_right
    Lax_Friedrichs_speed = jnp.max( jnp.array([speed_left , speed_right]) )
    Lax_Friedrichs_flux = 0.5*( flux_left + flux_right ) + Lax_Friedrichs_speed/2*( moment_left - moment_right  )
    return Lax_Friedrichs_flux

@jax.jit
def gauged_Local_Lax_Friedrichs_Flux( values_left, values_middle, values_right):
    """
    Calculate the Lax_Friedrichs_Flux at a cell interface with gauge transformation considered.
    """
    paras_left, moment_left, flux_left, gauges_left, domain_left, speed_left = values_left
    paras_middle, moment_middle, flux_middle, gauges_middle, domain_middle, speed_middle = values_middle
    paras_right, moment_right, flux_right, gauges_right, domain_right, speed_right = values_right

    # Transform moments and fluxes of the right cell to the gauge of left cell
    moment_right = M35.moments_gauge_transformation( moment_right, gauges_middle, gauges_right, domain_right )
    flux_right = M35.moments_gauge_transformation( flux_right, gauges_middle, gauges_right, domain_right )
    # Transform moments and fluxes of the left cell to the gauge of right cell
    moment_left = M35.moments_gauge_transformation( moment_left, gauges_middle, gauges_left, domain_left )
    flux_left = M35.moments_gauge_transformation( flux_left, gauges_middle, gauges_left, domain_left)        

    F_left = Local_Lax_Friedrichs_Flux( (moment_left, flux_left, speed_left), (moment_middle, flux_middle, speed_middle) )
    F_right = Local_Lax_Friedrichs_Flux( (moment_middle, flux_middle, speed_middle), (moment_right, flux_right, speed_right) )

    return F_left,F_right


@jax.jit
def VanLeerLimiter(r):
    """
    Calculate the VanLeerLimiter at a cell interface.
    """
    return (r + jnp.abs(r))/(1 + jnp.abs(r))
@jax.jit
def SuperBeeLimiter(r):
    """
    Calculate the VanLeerLimiter at a cell interface.
    """
    r1 = jnp.minimum( 2*r, 1. )
    r2 = jnp.minimum( r, 2. )
    r3 = jnp.maximum(r1, r2  )

    return jnp.maximum( 0., r3 )
@jax.jit
def MinmodLimiter(r):
    """
    Calculate the VanLeerLimiter at a cell interface.
    """
    r1 = jnp.minimum( r, 1. )

    return jnp.maximum( 0., r1 )
@jax.jit
def KorenLimiter(r):
    """
    Calculate the VanLeerLimiter at a cell interface.
    """
    r1 = jnp.minimum( (1+2*r)/3, 2. )
    r2 = jnp.minimum( 2*r, r1 )

    return jnp.maximum( 0., r2 )




@jax.jit
def gauged_Limiter( values_left, values_middle, values_right):
    """
    Calculate the Lax_Friedrichs_Flux at a cell interface with gauge transformation considered.
    """
    paras_left, moment_left, flux_left, gauges_left, domain_left, speed_left = values_left
    paras_middle, moment_middle, flux_middle, gauges_middle, domain_middle, speed_middle = values_middle
    paras_right, moment_right, flux_right, gauges_right, domain_right, speed_right = values_right

    # Transform moments and fluxes of the right cell to the gauge of left cell
    moment_right = M35.moments_gauge_transformation( moment_right, gauges_middle, gauges_right, domain_right )
    # Transform moments and fluxes of the left cell to the gauge of right cell
    moment_left = M35.moments_gauge_transformation( moment_left, gauges_middle, gauges_left, domain_left )   

    rvec = jnp.sign(moment_right - moment_middle)*(moment_middle - moment_left)/(  jnp.maximum(  jnp.abs(moment_right - moment_middle) , 1e-6 )  )

    phi_vec = vmap( VanLeerLimiter )(rvec)

    return phi_vec


@jax.jit
def Richtmyer_Intermediate( values_left, values_right, delta_x, delta_t):
    """
    Calculate the Richtmyer_Intermediate at a cell interface.

    Args:
        values_left: the quantities in the cell at the left of the interface.
                    a tuple (moment, flux, speed) in which moment is an jnp array of shape (m,), 
                    fluxes is an jnp array of shape (m,), speed is a float number.
                n+1 must equal to the number of sufficient statistics.
        values_right: the quantities in the cell at the left of the interface. Similar to values_left.

    Returns:
        The Lax_Friedrichs_flux, float
    """
    moment_left, flux_left, speed_left = values_left
    moment_right, flux_right, speed_right = values_right
    Local_speed = delta_x/delta_t

    moment_intermediate = (moment_left + moment_right)/2 + 1/Local_speed/2*(  flux_left - flux_right )

    return moment_intermediate




@jax.jit
def gauged_Richtmyer_fluxes( moment_middle, paras_middle, gauges_middle, values_left, values_right, delta_x, delta_t):
    """
    Calculate the Lax_Friedrichs_Flux at a cell interface with gauge transformation considered.
    """
    paras_left, moment_left, flux_left, gauges_left, domain_left, speed_left = values_left
    paras_right, moment_right, flux_right, gauges_right, domain_right, speed_right = values_right


    
    #gauges_middle = (gauges_left+gauges_right)/2
    domain_middle = domain_left

    # Transform moments and fluxes of the right cell to the gauge of gauges_middle
    moment_right = M35.moments_gauge_transformation( moment_right, gauges_middle, gauges_right, domain_right )
    flux_right = M35.moments_gauge_transformation( flux_right, gauges_middle, gauges_right, domain_right )
    paras_right = M35.natural_paras_gauge_transformation( paras_right , gauges_middle, gauges_right, domain_right)
    # Transform moments and fluxes of the left cell to the gauge of gauges_middle
    moment_left = M35.moments_gauge_transformation( moment_left, gauges_middle, gauges_left, domain_left )
    flux_left = M35.moments_gauge_transformation( flux_left, gauges_middle, gauges_left, domain_left)      
    paras_left = M35.natural_paras_gauge_transformation( paras_left , gauges_middle, gauges_left, domain_left)
    moment_middle = Richtmyer_Intermediate( (moment_left, flux_left, speed_left) , (moment_right, flux_right, speed_right), delta_x, delta_t )

    #delta_m = (moment_middle - moment_left).dot( moment_right - moment_left  )/ (jnp.sum( (moment_right - moment_left )**2 ) + 1e-6 )
    #paras_middle = paras_left + delta_m*(paras_right - paras_left  )

    # Transform into the Hermite Gauge
    gauges, moments, paras = gauges_middle, moment_middle, paras_middle
    gauges_H = M35.standard_gauge_para_from_moments( moments, gauges )
    moments_H = M35.moments_gauge_transformation( moments, gauges_H, gauges , domain_middle)
    paras_H = M35.natural_paras_gauge_transformation( paras, gauges_H, gauges , domain_middle)
    gauges_middle, moment_middle, paras_middle = gauges_H,moments_H,paras_H
    # Optimize the parameters
    paras_middle, opt_info = M35.moments_to_natural_paras( paras_middle, moment_middle, gauges_middle ,domain_middle)

    flux_middle = compute_flux( paras_middle, gauges_middle, domain_middle )

    flux_middle_left = M35.moments_gauge_transformation( flux_middle, gauges_left, gauges_middle, domain_middle)
    flux_middle_right = M35.moments_gauge_transformation( flux_middle, gauges_right, gauges_middle, domain_middle)

    return moment_middle, paras_middle, gauges_middle, flux_middle_left, flux_middle_right, opt_info




@jax.jit
def Hermite_gauge_para_to_rhoVT(moments_H, gauges_H):
    sr, sx, wx = gauges_H[:,0], gauges_H[:,1], gauges_H[:,2]
    m, kB = M35.constant["m"], M35.constant["kB"]
    n = moments_H[:,0]
    rho = n*m
    vx = wx
    T = (2*sr**2 + sx**2)*m/3/kB
    return jnp.asarray( [rho,vx,T] ).T

#################################################################################
# A time step
#################################################################################

@jax.jit
def ToHermiteGauge(moments, paras, gauges, domain):
    gauges_H = vmap(M35.standard_gauge_para_from_moments, in_axes = 0)( moments, gauges )
    moments_H = vmap(M35.moments_gauge_transformation, in_axes = (0,0,0,None))( moments, gauges_H, gauges , domain)
    paras_H = vmap(M35.natural_paras_gauge_transformation, in_axes = (0,0,0,None))( paras, gauges_H, gauges , domain)
    moments, paras, gauges = moments_H, paras_H, gauges_H
    return moments, paras, gauges, domain
@jax.jit
def GaugeTransformation(moments, paras, gauges2, gauges, domain):
    moments2 = vmap(M35.moments_gauge_transformation, in_axes = (0,0,0,None))( moments, gauges2, gauges , domain)
    paras2 = vmap(M35.natural_paras_gauge_transformation, in_axes = (0,0,0,None))( paras, gauges2, gauges , domain)
    return moments2, paras2, gauges2, domain
@partial(jax.jit, static_argnames=['verbose'])
def transport(moments, paras, gauges, domain, delta_t, moment_middle, paras_middle, gauges_middle, verbose = False):
    moments_O, paras_O, gauges_O, domain_O = moments, paras, gauges, domain
    # Update natural parameters
    moments, paras, gauges, domain = ToHermiteGauge(moments, paras, gauges, domain)
    paras, opt_info = vmap( M35.moments_to_natural_paras, in_axes = (0,0,0,None) )( paras, moments, gauges ,domain)
    moments, paras, gauges, domain = GaugeTransformation(moments, paras, gauges_O, gauges, domain)
    
    if verbose:
        jax.debug.print("M35 transport optimization iteration steps: Avg {x}, Min {y}, Max {z}", x=jnp.average( opt_info[2]), y=jnp.min( opt_info[2]), z=jnp.max( opt_info[2]) )

    # Compute fluxes of moments
    fluxes = vmap(compute_flux, in_axes = (0,0,None))( paras, gauges, domain )
    # Compute characteristic speeds of moments
    #speeds = vmap(compute_chara_speed, in_axes = (0,0,None))( paras, gauges, domain )
    #max_chara_speed = lambda betas, gauge_paras, domain: jnp.max( jnp.abs( compute_chara_speed_centered(betas, gauge_paras, domain)) )
    speeds = vmap(max_chara_speed, in_axes = (0,0,None))( paras, gauges, domain )
    #delta_t = 1e-1
    #speeds = delta_x/delta_t*np.ones(len(moments))


    values_left = ( paras[:-1], moments[:-1], fluxes[:-1],gauges[:-1], domain, speeds[:-1] )
    values_right = ( paras[1:], moments[1:], fluxes[1:],gauges[1:], domain, speeds[1:] )

    moment_middle, paras_middle, gauges_middle, flux_middle_left, flux_middle_right, opt_info = vmap( gauged_Richtmyer_fluxes, in_axes =  ( 0,0,0, (0,0,0,0,None,0),(0,0,0,0,None,0), None, None ) )(moment_middle, paras_middle, gauges_middle, values_left, values_right, delta_x, delta_t)
    RI_fluxes_minus_half, RI_fluxes_plus_half = flux_middle_right[:-1], flux_middle_left[1:]
    if verbose:
        jax.debug.print("M35 transport intermediate optimization iteration steps: Avg {x}, Min {y}, Max {z}", x=jnp.average( opt_info[2]), y=jnp.min( opt_info[2]), z=jnp.max( opt_info[2]) )


    # Compute the Lax Friedrichs numerical fluxes 
    left_left_index = np.concatenate( ( [0] , np.arange(len(paras)-3) ) , axis = 0)
    right_right_index = np.concatenate( ( np.arange(3, len(paras)), [len(paras)-1] ) , axis = 0)
    values_left_left = ( paras[left_left_index], moments[left_left_index],fluxes[left_left_index],gauges[left_left_index], domain, speeds[left_left_index] )    
    values_left = ( paras[:-2], moments[:-2],fluxes[:-2],gauges[:-2], domain, speeds[:-2] )
    values_middle = ( paras[1:-1], moments[1:-1],fluxes[1:-1],gauges[1:-1], domain, speeds[1:-1] )
    values_right = ( paras[2:], moments[2:],fluxes[2:],gauges[2:], domain, speeds[2:] )
    values_right_right = ( paras[right_right_index], moments[right_right_index],fluxes[right_right_index],gauges[right_right_index], domain, speeds[right_right_index] )
    LF_fluxes_minus_half, LF_fluxes_plus_half = vmap( gauged_Local_Lax_Friedrichs_Flux , in_axes =  ( (0,0,0,0,None,0),(0,0,0,0,None,0),(0,0,0,0,None,0) ) )(values_left, values_middle,values_right)


    Force_fluxes_minus_half = (LF_fluxes_minus_half + RI_fluxes_minus_half)/2
    Force_fluxes_plus_half = (LF_fluxes_plus_half + RI_fluxes_plus_half)/2


    phi_vec_plus_half = vmap(gauged_Limiter, in_axes =  ( (0,0,0,0,None,0),(0,0,0,0,None,0),(0,0,0,0,None,0) ) )(values_left, values_middle,values_right)
    phi_vec_minus_half = vmap(gauged_Limiter, in_axes =  ( (0,0,0,0,None,0),(0,0,0,0,None,0),(0,0,0,0,None,0) ) )(values_left_left, values_left, values_middle)


    fluxes_minus_half = Force_fluxes_minus_half + phi_vec_minus_half*(RI_fluxes_minus_half - Force_fluxes_minus_half)
    fluxes_plus_half = Force_fluxes_plus_half + phi_vec_plus_half*(RI_fluxes_plus_half - Force_fluxes_plus_half)
    # Update the moments according to the transport part of moment equations
    Transport_terms = 1/delta_x*(fluxes_minus_half - fluxes_plus_half)
    #delta_t = CFL_ratio*delta_x/jnp.max(speeds)

    #moments = moments.at[1:-1].set(moments[1:-1]  + Transport_terms )
    # Compute the time step size according to the CFL condition
    
    return Transport_terms, paras, moment_middle, paras_middle, gauges_middle
@partial(jax.jit, static_argnames=['verbose'])
def collision(moments, paras, gauges, domain, delta_t, moment_middle, paras_middle, gauges_middle, verbose = False):
    moments_O, paras_O, gauges_O, domain_O = moments, paras, gauges, domain
    moments, paras, gauges, domain = ToHermiteGauge(moments, paras, gauges, domain)
    paras, opt_info = vmap( M35.moments_to_natural_paras, in_axes = (0,0,0,None) )( paras, moments, gauges ,domain)
    moments, paras, gauges, domain = GaugeTransformation(moments, paras, gauges_O, gauges, domain)    
    if verbose:
        jax.debug.print("M35 collision optimization iteration steps: Avg {x}, Min {y}, Max {z}", x=jnp.average( opt_info[2]), y=jnp.min( opt_info[2]), z=jnp.max( opt_info[2]) )

    compute_flow_props = vmap( M35.natural_paras_to_fluid_properties, in_axes = (0,0,None))
    flow_props = compute_flow_props(paras, gauges, domain)
    Pr = PhyConst.Pr
    get_ESBGK_paras = lambda rhoVTsigma, gauge_paras, domain: ESBGK.rhoVTsigma_to_natural_paras( rhoVTsigma, gauge_paras, domain, Pr = Pr  )
    ESBGK_paras = vmap( get_ESBGK_paras, in_axes=(0,0,None) )( flow_props[:,[0,2,5,7]], gauges, domain )
    # Compute the moments of the Maxwell distribution
    Maxwell_paras_to_M35_moments = lambda Maxwell_beta, Maxwell_gauge, M35_gauge, flow_props: ESBGK.natural_paras_to_custom_moments( Maxwell_beta, Maxwell_gauge, domain, M35.suff_statistics, stats_gauge_paras= ( M35_gauge,) )
    moments_eq = vmap( Maxwell_paras_to_M35_moments, in_axes=0 )(ESBGK_paras, gauges, gauges, flow_props)
    # Compute the relaxation time of the ESBGK model
    taus = PhyConst.tau1*PhyConst.Rho1/moments_eq[:,0]/PhyConst.m/Pr
    if verbose:
        jax.debug.print("delta_t/tau: {x}", x=jnp.max( delta_t/taus) )


    moment_collisions_term = ( (moments_eq - moments)/taus[:,jnp.newaxis] )
    return moment_collisions_term[1:-1], paras, moment_middle, paras_middle, gauges_middle
@partial(jax.jit, static_argnames=['verbose'])
def step( moments, paras, gauges, domain, delta_t, moment_middle, paras_middle, gauges_middle, verbose = False ):

    def RK2( func, moments, paras, gauges, domain, delta_t, moment_middle, paras_middle, gauges_middle, verbose = False ):
        dmoment, paras, moment_middle, paras_middle, gauges_middle = func(moments, paras, gauges, domain, delta_t, moment_middle, paras_middle, gauges_middle, verbose = verbose)
        moments1 = moments.at[1:-1].set(moments[1:-1]  + delta_t*dmoment )
        dmoment1, paras1, moment_middle, paras_middle, gauges_middle = func(moments1, paras, gauges, domain, delta_t, moment_middle, paras_middle, gauges_middle, verbose = verbose)
        moments = moments.at[1:-1].set(moments[1:-1]  + delta_t/2*(dmoment + dmoment1) )
        return moments, paras1, gauges, domain, moment_middle, paras_middle, gauges_middle

    moments, paras, gauges, domain = ToHermiteGauge(moments, paras, gauges, domain)
    moment_middle, paras_middle, gauges_middle, domain = ToHermiteGauge(moment_middle, paras_middle, gauges_middle, domain)
    moments, paras, gauges, domain, moment_middle, paras_middle, gauges_middle = RK2(collision, moments, paras, gauges, domain, delta_t/2, moment_middle, paras_middle, gauges_middle, verbose = verbose)
    moments, paras, gauges, domain, moment_middle, paras_middle, gauges_middle = RK2(transport, moments, paras, gauges, domain, delta_t, moment_middle, paras_middle, gauges_middle, verbose = verbose)
    moments, paras, gauges, domain, moment_middle, paras_middle, gauges_middle = RK2(collision, moments, paras, gauges, domain, delta_t/2, moment_middle, paras_middle, gauges_middle, verbose = verbose)

    max_chara_speed = lambda betas, gauge_paras, domain: jnp.max( jnp.abs( compute_chara_speed_centered(betas, gauge_paras, domain)) )
    speeds = vmap(max_chara_speed, in_axes = (0,0,None))( paras, gauges, domain )
    delta_t2 = CFL_ratio*delta_x/jnp.max(speeds)
    if verbose:
        jax.debug.print("delta_t: {x}", x=delta_t2)

    return moments, paras, gauges, domain, delta_t2, moment_middle, paras_middle, gauges_middle, opt_info

if __name__ == "__main__":

    #################################################################################
    # Initialize the flow field by interpolating boundary conditions. 
    # Distributions in each cell are initialized with Maxwell distribution.
    #################################################################################

    # The initial gauge parameters
    Maxwell_gauge_para_ini = jnp.asarray([1.,0.])
    # The integration domain for Gauss_Legendre quadrature
    domain_para = jnp.asarray([PhyConst.velocity_lowbound,PhyConst.velocity_highbound,PhyConst.velocity_r_bound])
    print(domain_para)
    # Initialize the boundary moments and parameters
    rhoVT_boundary = jnp.asarray( [[PhyConst.Rho1, PhyConst.v1, PhyConst.T1],[PhyConst.Rho2, PhyConst.v2, PhyConst.T2]] )
    maxwell_paras_boundary = vmap( Maxwell.rhoVT_to_natural_paras, in_axes = (0,None,None) )(rhoVT_boundary, Maxwell_gauge_para_ini, domain_para)
    maxwell_moments_boundary = vmap( Maxwell.natural_paras_to_moments, in_axes=(0,None,None) )(maxwell_paras_boundary, Maxwell_gauge_para_ini, domain_para )
    # Interpolate the boundary values
    maxwell_paras = interpolate_boundary( maxwell_paras_boundary, PhyConst.estimated_thickness, cell_centers )
    maxwell_moments = interpolate_boundary( maxwell_moments_boundary, PhyConst.estimated_thickness, cell_centers )

    # Transform maxwell distribution into the Hermite gauge and compute the natural parameters
    maxwell_gauges = vmap( Maxwell.standard_gauge_para_from_moments, in_axes = (0, None) )(maxwell_moments, Maxwell_gauge_para_ini)
    maxwell_moments = vmap(Maxwell.moments_gauge_transformation, in_axes = (0,0,None,None))( maxwell_moments, maxwell_gauges, Maxwell_gauge_para_ini, domain_para)
    maxwell_paras =  vmap(Maxwell.natural_paras_gauge_transformation, in_axes = (0,0,None,None))( maxwell_paras, maxwell_gauges , Maxwell_gauge_para_ini, domain_para )
    maxwell_paras, opt_info = vmap( Maxwell.moments_to_natural_paras, in_axes = (0,0,0,None) )( maxwell_paras, maxwell_moments, maxwell_gauges, domain_para )
    values, residuals, steps, bsteps = opt_info


    # Compute flow properties of all Maxwell distributions
    maxwell_macros = vmap( Maxwell.natural_paras_to_fluid_properties, in_axes = (0,0,None))(maxwell_paras, maxwell_gauges, domain_para)
    # Compute natural parameters analytically from flow properties
    maxwell_paras_analytical = vmap( Maxwell.rhoVT_to_natural_paras, in_axes = (0,0,None) )(maxwell_macros[:,[0,2,5]], maxwell_gauges, domain_para)

    assert allClose( maxwell_paras_analytical, maxwell_paras ), "The analytical solution of maxwell parameter deviates too much from numerical result. Please increase the velocity domain resolution"

    Maxwell_distribution_infos = (maxwell_moments, maxwell_paras, maxwell_gauges)
    #################################################################################
    # Re-initialize the flow field by converting the Maxwell distributions to M35 distributions.
    #################################################################################

    # Initialize the gauge paramters of M35 distribution
    M35_gauges = jnp.asarray( [maxwell_gauges[:,0],maxwell_gauges[:,0],maxwell_gauges[:,1]] ).T
    # Compute the moments of M35 distribution from the Maxwell distributions initialized previously.
    Maxwell_paras_to_M35_moments = lambda Maxwell_beta, Maxwell_gauge, M35_gauge: Maxwell.natural_paras_to_custom_moments( Maxwell_beta, Maxwell_gauge, domain_para, M35.suff_statistics , stats_gauge_paras= ( M35_gauge,) )
    M35_moments = vmap( Maxwell_paras_to_M35_moments, in_axes=0 )(maxwell_paras, maxwell_gauges, M35_gauges)
    # Optimize the natural parameters of M35 distribution from moments
    M35_paras, opt_info = vmap( M35.moments_to_natural_paras, in_axes = (None,0,0,None) )( jnp.array([1.,0,0,0,0,0,0,0,0]), M35_moments, M35_gauges ,domain_para)
    values, residuals, steps, bsteps = opt_info
    # Compute flow properties of M35 distributions.
    M35_macros = vmap( M35.natural_paras_to_fluid_properties, in_axes = (0,0,None))(M35_paras, M35_gauges, domain_para)

    assert allClose( M35_macros[:,[0,2,5]], maxwell_macros[:,[0,2,5]] ), "The optimized M35 distribution does not match the density, velocity and temperature of Maxwell distributions. Try decrease the tolerance of optimization process. "
    M35_distribution_infos = (M35_moments, M35_paras, M35_gauges, domain_para)
    moment_middle, paras_middle, gauges_middle = M35_moments[:-1], M35_paras[:-1], M35_gauges[:-1]


    #################################################################################
    # Main loop
    #################################################################################
    filename = "Mach10.npz"

    compute_flow_props = jax.jit(vmap( M35.natural_paras_to_fluid_properties, in_axes = (0,0,None)))
    flow_properties_hist = []
    time_hist = []
    time=0.
    max_chara_speed = lambda betas, gauge_paras, domain: jnp.max( jnp.abs( compute_chara_speed_centered(betas, gauge_paras, domain)) )
    speeds = vmap(max_chara_speed, in_axes = (0,0,None))( M35_paras, M35_gauges, domain_para )
    delta_t = CFL_ratio*delta_x/jnp.max(speeds)
    print("*********************************************************")
    print("Jit Compile")
    print("*********************************************************")
    step(*M35_distribution_infos, delta_t, moment_middle, paras_middle, gauges_middle, verbose=True)
    step(*M35_distribution_infos, delta_t, moment_middle, paras_middle, gauges_middle, verbose=False)
    print("*********************************************************")
    print("Jit Compile Complete")
    print("*"*100)
    print("Start Computation")
    record_steps = 20
    moments, paras, gauges, domain = M35_distribution_infos
    for i in range(1000000000+1):
        time += delta_t
        if i%record_steps == 0:
            moments, paras, gauges, domain, delta_t, moment_middle, paras_middle, gauges_middle, opt_info = step(moments, paras, gauges, domain, delta_t,moment_middle, paras_middle, gauges_middle, verbose=True)
            M35_distribution_infos = (moments, paras, gauges, domain)
        else:
            moments, paras, gauges, domain, delta_t, moment_middle, paras_middle, gauges_middle, opt_info = step(moments, paras, gauges, domain, delta_t,moment_middle, paras_middle, gauges_middle, verbose=False)
            M35_distribution_infos = (moments, paras, gauges, domain)
        if time > 25:
            break
        if i%record_steps == 0:
            moments, paras, gauges, domain = M35_distribution_infos
            flow_properties_hist.append(  compute_flow_props(paras, gauges, domain) )
            print("tot time", time)
        if i%(record_steps*10) == 0:
            flow_properties = np.array( flow_properties_hist )
            times = np.array( time_hist )
            moments, paras, gauges, domain = [np.array(item) for item in M35_distribution_infos]
            np.savez( filename, cell_centers = cell_centers, flow_properties = flow_properties, times = times, moments = moments, paras = paras, gauges = gauges, domain = domain, PhyConst = PhyConst, constant = constant, Grid_info = Grid_info )
    flow_properties = np.array( flow_properties_hist )
    times = np.array( time_hist )
    moments, paras, gauges, domain = [np.array(item) for item in M35_distribution_infos]
    np.savez( filename, cell_centers = cell_centers, flow_properties = flow_properties, times = times, moments = moments, paras = paras, gauges = gauges, domain = domain, PhyConst = PhyConst, constant = constant, Grid_info = Grid_info )


    """

        moment_change_scales = vmap( M35.moments_gauge_transformation, in_axes = (0,0,0,None) )

        fluxes_in_right_gauge = moment_change_scales( fluxes[:-1], gauges[1:], gauges[:-1], domain )
        fluxes_in_left_gauge = moment_change_scales( fluxes[1:], gauges[:-1], gauges[1:], domain )
        moments_in_right_gauge = moment_change_scales( moments[:-1],  gauges[1:], gauges[:-1], domain )
        moments_in_left_gauge = moment_change_scales( moments[1:], gauges[:-1], gauges[1:], domain )

        left_wall_values_left = ( moments_in_right_gauge[:-1],fluxes_in_right_gauge[:-1],speeds[:-2] )
        left_wall_values_right = ( moments[1:-1],fluxes[1:-1],speeds[1:-1] )
        right_wall_values_left = ( moments[1:-1],fluxes[1:-1],speeds[1:-1] )
        right_wall_values_right = ( moments_in_left_gauge[1:],fluxes_in_left_gauge[1:],speeds[2:] )
        
        LF_fluxes_minus_half = vmap( Lax_Friedrichs_Flux, in_axes =  ( (0),(0) ) )( left_wall_values_left , left_wall_values_right )
        LF_fluxes_plus_half = vmap( Lax_Friedrichs_Flux, in_axes =  ( (0),(0) ) )( right_wall_values_left , right_wall_values_right )


        paras, opt_info = vmap( M35.moments_to_natural_paras, in_axes = (0,0,0,None) )( paras, moments, gauges ,domain)

        macros = vmap(M35.natural_paras_to_fluid_properties, in_axes= (0,0,None) )(paras,gauges,domain)

        paras_H, opt_infoH = vmap( M35.moments_to_natural_paras, in_axes = (0,0,0,None) )( paras_H, moments_H, gauges_H ,domain)

        macros_H = vmap(M35.natural_paras_to_fluid_properties, in_axes= (0,0,None) )(paras_H,gauges_H,domain)


        if verbose:
            jax.debug.print("M35 optimization iteration steps: Avg {x}, Min {y}, Max {z}", x=jnp.average( opt_info[2]), y=jnp.min( opt_info[2]), z=jnp.max( opt_info[2]) )
            jax.debug.print("M35 optimization iteration steps H: Avg {x}, Min {y}, Max {z}", x=jnp.average( opt_infoH[2]), y=jnp.min( opt_infoH[2]), z=jnp.max( opt_infoH[2]) )
            jax.debug.print("rho: {x}", x=jnp.allclose( macros[:,0],rho) )
            jax.debug.print("vx: {x}", x=jnp.allclose( macros[:,2],vx) )
            jax.debug.print("T: {x}", x=jnp.allclose( macros[:,5],T) )
            jax.debug.print("rho: {x}", x=jnp.allclose( macros_H[:,0],rho) )
            jax.debug.print("vx: {x}", x=jnp.allclose( macros_H[:,2],vx) )
            jax.debug.print("T: {x}", x=jnp.allclose( macros_H[:,5],T) )
    """

    momentSolu = np.load("Mach10.npz")

    plt.style.use('bmh')
    plt.figure(dpi=200, figsize=(6,4))
    #fig, ax1 = plt.subplots(dpi=200)
    #fig.style.use('bmh')
    #plt.bar(Mas, steps2, alpha = 0.6, label = "Without Gauge Transformation")
    plt.plot( momentSolu["cell_centers"], momentSolu["flow_properties"][-1,:,0],label="Moment 35")
    plt.ylabel("density")
    plt.xlabel("x")
    plt.legend()
    plt.savefig("./densityM10.png")

    plt.style.use('bmh')
    plt.figure(dpi=200, figsize=(6,4))
    #fig, ax1 = plt.subplots(dpi=200)
    #fig.style.use('bmh')
    #plt.bar(Mas, steps2, alpha = 0.6, label = "Without Gauge Transformation")
    plt.plot( momentSolu["cell_centers"], momentSolu["flow_properties"][-1,:,2],label="Moment 35")
    plt.ylabel("vx")
    plt.xlabel("x")
    plt.legend()
    plt.savefig("./velocityM10.png")

    plt.style.use('bmh')
    plt.figure(dpi=200, figsize=(6,4))
    #fig, ax1 = plt.subplots(dpi=200)
    #fig.style.use('bmh')
    #plt.bar(Mas, steps2, alpha = 0.6, label = "Without Gauge Transformation")
    plt.plot( momentSolu["cell_centers"], momentSolu["flow_properties"][-1,:,5],label="Moment 35")
    plt.ylabel("T")
    plt.xlabel("x")
    plt.legend()
    plt.savefig("./tempM10.png")

    plt.style.use('bmh')
    plt.figure(dpi=200, figsize=(6,4))
    plt.plot( momentSolu["cell_centers"], momentSolu["flow_properties"][-1,:,7],label="Moment 35")
    plt.ylabel("sigma_xx")
    plt.xlabel("x")
    plt.legend()
    plt.savefig("./sigmaxM10.png")



    plt.style.use('bmh')
    plt.figure(dpi=200, figsize=(6,4))
    #fig, ax1 = plt.subplots(dpi=200)
    #fig.style.use('bmh')
    #plt.bar(Mas, steps2, alpha = 0.6, label = "Without Gauge Transformation")
    plt.plot( momentSolu["cell_centers"], momentSolu["flow_properties"][-1,:,13],label="Moment 35")
    plt.ylabel("q_x")
    plt.xlabel("x")
    plt.legend()
    plt.savefig("./qxM10.png")