import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
from sympy.integrals.quadrature import gauss_hermite,gauss_laguerre,gauss_jacobi,gauss_legendre,gauss_gen_laguerre

def Gauss_Legendre_Quadrature(a,b,n):
    r"""Generate the Legender Quadrature points and corresponding weights for 1D integral

    .. math::
        :nowrap:

        \begin{equation}
        \int_{a}^b \phi(x) d x \approx \sum_{i=i}^n w_i \phi(x_i),
        \end{equation}

    Parameters
    ----------
    a : float
        lower integration limit
    b : float
        upper integration limit, float
    n : int 
        the order of Gauss Legendre quadrature

    Returns
    -------
    Tuple
        A tuple containing

            **x**: *array of shape (n)* - the quadrature points :math:`x_i`

            **w**: *array of shape (n)* - the quadrature weights :math:`w_i`
    """
    x, w = gauss_legendre(n, 21)
    x = (b-a)/2*jnp.asarray( np.array(x,dtype=float) ) +(b+a)/2
    w = (b-a)/2*jnp.asarray( np.array(w,dtype=float) )
    return x,w

def Gauss_Legendre_Quadrature_2D(a_x, b_x, n_x, a_y, b_y, n_y):
    r"""Generate the Legender Quadrature points and corresponding weights for 2D integral 

    .. math::
        :nowrap:

        \begin{equation}
        \int_{a_y}^{b_y} \int_{a_x}^{b_x} \phi(x,y) d x dy \approx \sum_{i=1,j=1}^{n_x,n_y} w_{ij} \phi( \mathbf{x}_{ij}),
        \end{equation}

    Parameters
    ----------
    a_x : float
        lower integration limit in x dimension
    b_x : float
        upper integration limit in x dimension
    n_x : int 
        the order of Gauss Legendre quadrature in x dimension
    a_y : float
        lower integration limit in y dimension
    b_y : float
        upper integration limit in y dimension
    n_y : int 
        the order of Gauss Legendre quadrature in y dimension

    Returns
    -------
    Tuple
        A tuple containing

            **x**: *array of shape (n_x, n_y, 2)* - the quadrature points, x[i,j,:] is the 2D i-jth quadrature points :math:`\mathbf{x}_{ij}`

            **w**: *array of shape (n_x, n_y)* - the quadrature weights w[i,j] is the quadrature weight :math:`w_{ij}`
    """
    x_x, w_x = Gauss_Legendre_Quadrature(a_x, b_x, n_x)
    #print(x_x.shape)
    x_y, w_y = Gauss_Legendre_Quadrature(a_y, b_y, n_y)
    x = jnp.array( jnp.meshgrid(x_x,x_y,indexing='ij') ).transpose(1,2,0)
    w = w_x[:,jnp.newaxis]*w_y[jnp.newaxis,:]
    return x, w
def Gauss_Legendre_Quadrature_2D_Block(a_x, b_x, n_x, B_x, a_y, b_y, n_y, B_y):
    r"""Block-wise the Legender Quadrature points and corresponding weights for 2D integral.

    The integration domain is a square :math:`(a_x, b_x)\times(a_y, b_y)` that is splitted into (b_x*b_y) blocks, in which each block are integrated with 2D Gauss Legendre quadratures as in :func:`Sampler.SamplerUtility.Gauss_Legendre_Quadrature_2D`.

    Specifically, the interval :math:`(a_x, b_x)` is divided into :math:`B_x` piecies: :math:`\{(a_{x,l_x},a_{x,l_x+1})\ |\ l_x = 1,\cdots,B_x; a_{x,1} = a_x, a_{x,B_x+1} = b_x\}`.

    The interval :math:`(b_x, b_y)` is divided into :math:`B_y` piecies: :math:`\{(a_{y,l_y},a_{y,l_y+1})\ |\ l_y = 1,\cdots,B_y; a_{y,1} = a_y, a_{y,B_y+1} = b_y\}`.

    .. math::
        :nowrap:

        \begin{equation}
        \begin{split}
        \int_{a_y}^{b_y} \int_{a_x}^{b_x} \phi(x,y) d x dy &=  \sum_{l_x=1,l_y=1}^{B_x,B_y}  \int_{a_{y,l_y}}^{a_{y,l_y+1}} \int_{a_{x,l_x}}^{a_{x,l_x+1}} \phi(x,y) d x dy \\ 
        & \approx \sum_{l_x=1,l_y=1,i=1,j=1}^{B_x,B_y,n_x,n_y} w_{l_xl_yij} \phi( \mathbf{x}_{l_xl_yij}),
        \end{split}
        \end{equation}

    Parameters
    ----------
    a_x : float
        lower integration limit in x dimension
    b_x : float
        upper integration limit in x dimension
    n_x : int 
        the order of Gauss Legendre quadrature in x dimension
    B_x : int 
        how many blocks are splitted in the x dimension
    a_y : float
        lower integration limit in y dimension
    b_y : float
        upper integration limit in y dimension
    n_y : int 
        the order of Gauss Legendre quadrature in y dimension
    B_y : int 
        how many blocks are splitted in the y dimension

    Returns
    -------
    Tuple
        A tuple containing

            **x**: *array of shape (B_x,B_y,n_x, n_y, 2)* - the quadrature points, x[l_x,l_y,i,j,:] is the i-jth 2D quadrature points :math:`\mathbf{x}_{l_xl_yij}` in the l_x-l_yth block 

            **w**: *array of shape (B_x,B_y,n_x, n_y)* - the quadrature weights w[l_x,l_y,i,j] is the i-jth quadrature weight :math:`w_{l_xl_yij}` in the l_x-l_yth block 
    """

    x_splits = jnp.linspace(a_x,b_x,B_x+1)
    x_centers = (x_splits[1:]+x_splits[:-1])/2
    delta_x = (b_x-a_x)/B_x


    y_splits = jnp.linspace(a_y,b_y,B_y+1)
    y_centers = (y_splits[1:]+y_splits[:-1])/2
    delta_y = (b_y-a_y)/B_y

    xy_centers = jnp.array( jnp.meshgrid(x_centers,y_centers,indexing='ij') ).transpose(1,2,0)


    x_B, w_B = Gauss_Legendre_Quadrature_2D(-delta_x/2, delta_x/2, n_x, -delta_y/2, delta_y/2, n_y)

    x_Full = x_B + xy_centers[:,:,jnp.newaxis,jnp.newaxis,:]
    w_Full = jnp.ones(x_Full.shape[:-1] )*w_B

    return x_Full, w_Full
if __name__ == "__main__":
    x,w = Gauss_Legendre_Quadrature_2D_Block(-10,10,3,5,-5,5,4,6)
    print(w.shape)
    print(jnp.sum(w))