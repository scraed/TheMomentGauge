
from jax import jacfwd,vmap
import jax.numpy as jnp
import numpy as np
from jax import jit
import functools
import jax
from jax.lax import fori_loop,while_loop,cond
from jax.lax.linalg import triangular_solve
from functools import partial
class BaseOptimizer():
    def __init__(self, target_function, **kwargs ):
        """The base class for optimizer

        Parameters
        ----------
        target_function : function
            the function to be optimized by the Netwons method whose

                    **Parameters**:

                        **input_para** : float array of shape (n) - The parameter to be optimized

                        :math:`*` **aux_paras** : - Arbitrary many extra parameters not to be optimized. The :math:`*` refers to the unpacking operator in python.

                    **Returns**: 
                    
                        float -- the function value
        **kwargs : other key word arguments.
        """
        self._target_function = target_function
        self._kwargs = kwargs
    def optimize(self, input_para, *aux_paras):
        r"""
        optimization of the **target_function**

        Parameters
        ----------
        input_para: float array of shape (n)
            the input parameters for the **target_function** to be optimized. 
        *aux_paras: 
            Arbitrary many extra parameters not to be optimized for the **target_function**.

        Returns
        -------
        Tuple
            A tuple containing

                **opt_para**: *float array of shape (n)* - The optimized parameters.

                **opt_info**: *tuple* - A tuple containing other information
        """   
        raise NotImplementedError
    def __hash__(self):
        """Redefine the hash method to include the class attributes. It helps jax.jit to correctly identify class instances"""
        return hash(( "BaseOptimizer", *list(self._kwargs.items()) ))

    def __eq__(self, other):
        """Redefine the eq method to include the class attributes."""
        return (isinstance(other, BaseOptimizer) and
                (self._target_function, *list(self._kwargs.items())) == ( other._target_function, *list(other._kwargs.items())))