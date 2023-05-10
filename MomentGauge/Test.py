from MomentGauge.Sampler.QuadratureSampler import Gauss_Legendre_Sampler_Polar3D
from MomentGauge.Sampler.CanonicalExpFamilySampler import CanonicalExpImportanceSampler
import jax
import jax.numpy as jnp
constant = {"m": 1., "kB": 1.}
Qsampler = Gauss_Legendre_Sampler_Polar3D(n_x = 8,n_r = 8, B_x = 16, B_r = 16)
Qsampler2 = Gauss_Legendre_Sampler_Polar3D(n_x = 8,n_r = 8, B_x = 16, B_r = 15)
suff_moments = [lambda u: 1., lambda u: u[0], lambda u: u[0]**2]
Mom = jnp.array([1,0,1.])
beta = jnp.array( [jnp.pi*jnp.exp(1),0,0] )
sampler = CanonicalExpImportanceSampler(suff_moments, Qsampler )
sampler1 = CanonicalExpImportanceSampler(suff_moments, Qsampler )
x,w,logli = sampler.sample(beta, gauge_paras = (), base_args = (jnp.array([0., 1, 1]),))

print(sampler1 ==sampler)

from MomentGauge.Estimator.Estimator import BaseEstimator,EstimatorPolar2D
#moms = [lambda u: 1.,lambda u: u[0]]
#u = [1.,0,0]
#momswitch = lambda i, u: jax.lax.switch(i,moms, u  )
#print(momswitch(1,u))

momswitch = lambda u, *gauge_paras: sampler.suff_statistics(u, gauge_paras = gauge_paras)
def momswitch_cov(u,*gauge_paras):
    phi = sampler.suff_statistics(u, gauge_paras = gauge_paras)
    return phi[:,jnp.newaxis]*phi
#printmoms

#momswitch = lambda u: jax.vmap( lambda i, u: jax.lax.switch(i,moms, u  ) ,in_axes = (0,None) )(jnp.arange(len(moms)),u)

print(momswitch([0.5,0,0]))
est = EstimatorPolar2D(constant)
print(est.get_sample_moment(momswitch,x,w))
print(est.get_sample_moment(momswitch_cov,x,w))
print(est.cal_macro_quant(x,w))
#print(w.shape)
#print(jnp.sum(w))
#print(x[:,:,1])
#x,w,logli = sampler.sample([1.0,0.0], n_x = 8, n_r = 8, B_x = 16, B_r = 16)
#print(8 * 16 * 8 * 16)
#print(x.shape)
#print( jnp.sum(w/x[...,1]) -2/br)
#print(jnp.exp(logli) - 1 / (jnp.pi * br**2 * l))
#print( jnp.sum(w*x[:,0]) )
#print(jnp.sum(w * x[:, 0] * (x[:, 1]**2 + x[:, 2]**2)**0.5))

"""
export PYTHONPATH=$PYTHONPATH:/mnt/c/Users/scraed/Downloads/MomentGaugePackage
"""