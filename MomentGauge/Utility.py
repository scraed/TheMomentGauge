import jax.numpy as jnp
from jax.lax.linalg import triangular_solve
from jax import vmap
import numpy as np
def generalized_eigh_cholesky(A, B):
    """Compute the generalilzed eignvalue problem :math:`A \mathbf{x} = \lambda B \mathbf{x}` in which A and B are Hermite matrix


    Parameters
    ----------
    A : array of shape (M,M)
        a Hermite matrix of the shape (M, M)
    B : array of shape (M,M)
        a Hermite matrix of the shape (M, M)

    Returns
    -------
    Tuple
        A tuple containing

            **w**: *array of shape (M)* - The eigenvalues in ascending order, each repeated according to its multiplicity.

            **V**: *array of shape (M,M)* - The matrix whose ith column V[:, i] is the normalized eigenvector corresponding to the ith eigenvalue.
    """
    # In case the B fail to be positive definite, we add a small quantity here
    L = jnp.linalg.cholesky(B )
    #fail = jnp.sum(jnp.isnan(L)) > 0
    Linv = lambda x: triangular_solve( L, x, left_side=True, lower=True ,transpose_a=False)
    #L_inv = jnp.linalg.inv(L)
    A = (A + A.T)/2
    LinvA = vmap(Linv,in_axes = 1,out_axes=1)(A)
    LinvALinvT = vmap(Linv,in_axes = 1,out_axes=1)(LinvA.T)
    LinvALinvT = (LinvALinvT + LinvALinvT.T)/2

    #w = triangular_solve( L, -grad, left_side=True, lower=True ,transpose_a=False)
    #delta_input = triangular_solve( L, w, left_side=True, lower=True ,transpose_a=True)
    #A_redo = L_inv.dot(A).dot(L_inv.T)
    #jax.debug.print("Mcond: {x}", x=jnp.allclose(A_redo, LinvALinvT, atol=1e-4))
    return jnp.linalg.eigh(LinvALinvT)
def generalized_eigh(A, B):
    """Compute the generalilzed eignvalue problem :math:`A \mathbf{x} = \lambda B \mathbf{x}` in which A and B are Hermite matrix


    Parameters
    ----------
    A : array of shape (M,M)
        a Hermite matrix of the shape (M, M)
    B : array of shape (M,M)
        a Hermite matrix of the shape (M, M)

    Returns
    -------
    Tuple
        A tuple containing

            **w**: *array of shape (M)* - The eigenvalues in ascending order, each repeated according to its multiplicity.

            **V**: *array of shape (M,M)* - The matrix whose ith column V[:, i] is the normalized eigenvector corresponding to the ith eigenvalue.
    """
    beta = jnp.finfo(B.dtype).resolution**0.5
    u,s,v = jnp.linalg.svd( B , full_matrices=True, hermitian=True )
    s = jnp.maximum( s, beta*np.ones(len(s))  )
    ss_inv = jnp.sqrt(jnp.diag(1/s))

    LinvALinvT = ss_inv.dot(u.T).dot(A).dot(v.T).dot(ss_inv)

    #w = triangular_solve( L, -grad, left_side=True, lower=True ,transpose_a=False)
    #delta_input = triangular_solve( L, w, left_side=True, lower=True ,transpose_a=True)
    #A_redo = L_inv.dot(A).dot(L_inv.T)
    #jax.debug.print("Mcond: {x}", x=jnp.allclose(A_redo, LinvALinvT, atol=1e-4))
    return jnp.linalg.eigh(LinvALinvT)
'''
A = np.random.rand(5,5)
B = np.random.rand(5,5)
A = A.dot(A.T)
B = B.dot(B.T)
print(generalized_eigh2(A,B))
print(generalized_eigh(A,B))
'''
'''
def generalized_eigh(A, B, eps = 0.):
    """Compute the generalilzed eignvalue problem :math:`A \mathbf{x} = \lambda B \mathbf{x}` in which A and B are Hermite matrix


    Parameters
    ----------
    A : array of shape (M,M)
        a Hermite matrix of the shape (M, M)
    B : array of shape (M,M)
        a Hermite matrix of the shape (M, M)

    Returns
    -------
    Tuple
        A tuple containing

            **w**: *array of shape (M)* - The eigenvalues in ascending order, each repeated according to its multiplicity.

            **V**: *array of shape (M,M)* - The matrix whose ith column V[:, i] is the normalized eigenvector corresponding to the ith eigenvalue.
    """
    # In case the B fail to be positive definite, we add a small quantity here
    L = jnp.linalg.cholesky(B + eps*jnp.eye(len(B)))
    #fail = jnp.sum(jnp.isnan(L)) > 0
    L_inv = jnp.linalg.inv(L)
    A_redo = L_inv.dot(A+ eps*jnp.eye(len(B))).dot(L_inv.T)
    return jnp.linalg.eigh(A_redo)
'''
