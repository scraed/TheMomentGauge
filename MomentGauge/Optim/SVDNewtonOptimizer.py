
from jax import jacfwd,vmap
import jax.numpy as jnp
import numpy as np
from jax import jit
import functools
import jax
from jax.lax import fori_loop,while_loop,cond
from jax.lax.linalg import triangular_solve
from functools import partial
from MomentGauge.Optim.BaseOptimizer import BaseOptimizer
@partial(jax.jit, static_argnums=0,static_argnames=["reg_hessian"])
def Newton_Optimizer_Iteration_Delta( target_function, input_para, *aux_paras, reg_hessian=True ):
    r"""
    A single iteration step of Newton's method for optimizing the **target_function** ( **input_para** , :math:`*` **aux_paras** ) w.r.t input_para. 

    Parameters
    ----------
    target_function : function
        the function to be optimized by the Netwons method whose

                **Parameters**:

                    **input_para** : float array of shape (n) - The parameter to be optimized

                    :math:`*` **aux_paras** : - Arbitrary many extra parameters not to be optimized. The :math:`*` refers to the unpacking operator in python.

                **Returns**: 
                
                    float -- the function value

    input_para: float array of shape (n)
        the input for the target_function to be optimized. 
    *aux_paras: 
        Arbitrary many extra parameters not to be optimized for the target_function.
    reg_hessian: bool
        Regularize the Hessian if the Cholesky decomposition failed. Default = True


    Returns
    -------
    Tuple
        A tuple containing

            **delta_input**: *float array of shape (n)* - update direction for input_para according to a single step Newton's iteration.

            **value**: *float* - the current value of target_function
    
            **grads**: *float array of shape (n)* - current gradients of the target function w.r.t input_para

            **residual**: *float* - the residual as estimated target_function value change along the delta_input direction

            **hessian**: *float array of shape (n,n)* - current Hessian matrix of the target function w.r.t input_para

            **count**: *int* - The number of regularization applied on Hessian by adding a multiple of the identity
    """
    beta = jnp.finfo(input_para.dtype).resolution**0.5
    f = lambda input_para: target_function( input_para, *aux_paras )
    hessian_f = jax.hessian(f)
    value,grad = jax.value_and_grad(f)(input_para)
    hessian = hessian_f(input_para) 
    hessian_OK = jnp.sum(jnp.isnan(hessian)) == 0
    L = jnp.linalg.cholesky(hessian)
    fail = jnp.sum(jnp.isnan(L)) > 0
    count=0
    def ch_cond( arg ):
        L, beta_scale, fail, count = arg
        return fail & hessian_OK
    def ch_body( arg ):
        L, beta_scale, fail, count = arg
        L = jnp.linalg.cholesky(hessian + beta_scale*beta*jnp.eye(len(hessian)))
        fail = jnp.sum(jnp.isnan(L)) > 0
        return L, beta_scale*10, fail, count+1
    if reg_hessian:
        L, beta_scale, fail, count = while_loop( ch_cond, ch_body, ( L,  1.0, fail, 0) )
    #jax.debug.print( "Hessian decomp failed {x}", x=jnp.sum(jnp.isnan(L)) != 0 )
    #if reg_hessian:
    #    L = cond( jnp.sum(jnp.isnan(L)) == 0, lambda hessian: L , lambda hessian: jnp.linalg.cholesky(hessian + beta*jnp.eye(len(hessian))), hessian )
    w = triangular_solve( L, -grad, left_side=True, lower=True ,transpose_a=False)
    delta_input = triangular_solve( L, w, left_side=True, lower=True ,transpose_a=True)
    residual = -0.5*delta_input.dot(grad)
    return delta_input, value, grad, residual,hessian, count






@partial(jax.jit, static_argnums=0, static_argnames=['c','atol','rtol',"debug"])
def Armijo_condition(target_function, current_value, input_para, update_step, grad_para, *aux_paras, c = 5e-4, atol=5e-6, rtol=1e-5, debug=False):
    """
    Check whether an update step 'update_para' of the Newton's method satisfy the Armijo's condition for sufficiently decreasing in the objective function

    Parameters
    ----------
    target_function : function
        the function to be optimized by the Netwons method whose

                **Parameters**:

                    **input_para** : float array of shape (n) - The parameter to be optimized

                    :math:`*` **aux_paras** : - Arbitrary many extra parameters not to be optimized. The :math:`*` refers to the unpacking operator in python.

                **Returns**: 
                
                    float -- the function value

    current_value : float
        the current value of the target_funciton w.r.t the parameters given in **input_para**
    input_para: float array of shape (n)
        the input for the target_function to be optimized. 
    update_step: float array of shape (n)
        the update direction proposed by the Newton's method, to be checked by the Armijo's condiiton.
    grad_para: float array of shape (n)
        the gradient direction of the target function at input_para, used in the Armijo's condiiton.
    *aux_paras: 
        Arbitrary many extra parameters not to be optimized for the target_function.
    c: float
        the parameter used in the Armijo's condiiton, must lies in (0,1). Smaller c converges faster but less stable. default = 5e-4.
    atol: float 
        the absolute error tolerance of the Armijo's condiiton since we use -(atol + rtol*abs(next_value)) instead of 0 to handle single precision numerical error.  default = 5e-6. 
    rtol: float
        the relative error tolerance of the Armijo's condition since we use -(atol + rtol*abs(next_value)) instead of 0 to handle single precision numerical error.  default = 1e-5. 
    debug: bool
        print debug information if True.

    Returns
    -------
    Tuple
        A tuple containing

            **satisfied**: *bool* - a bool indicating whether the **update_step** satisfy the Armijo's condition

            **delta_value**: *float* - the decrease of target_function at the **update_step** direction.
    
            **grad_delta_value**: *float* - the expected minimal decrease of target function at the **update_step** direction. The Armijo condition is satisfied if **delta_value** is greater than **grad_delta_value**.

            **target_value**: *float* - the value of target_function at the **update_step** direction. It equals **current_value** - **delta_value**
    """    

    grad_delta_value = -c*grad_para.dot(update_step)
    next_input_para = input_para + update_step
    target_value = target_function( next_input_para, *aux_paras ) 
    delta_value = current_value  - target_value
    delta_value = jax.numpy.nan_to_num(delta_value, nan=jnp.nan, posinf=jnp.nan,neginf=jnp.nan )
    #delta_value = jnp.maximum( delta_value,0 )
    
    # If nan exists, satisfied is False
    satisfied = delta_value - grad_delta_value >= -(atol +  rtol*abs(delta_value)    )

    
    return satisfied, delta_value, grad_delta_value, target_value


@partial(jax.jit, static_argnums=0, static_argnames=["alpha","beta", 'c','atol','rtol', "max_iter", "max_back_tracking_iter", "tol", "min_step_size","reg_hessian","debug"])
def Newton_Backtracking_Optimizer_JIT(target_function, input_para, *aux_paras, alpha=1.0, beta=0.5, c=5e-4, atol=5e-6, rtol=1e-5, max_iter = 100, max_back_tracking_iter = 25, tol = 1e-6, min_step_size=1e-6, reg_hessian = True, debug = False ):
    """
    optimization of the target_function using the Newton's method with backtracking line search according to the Armijo's condition.

    Parameters
    ----------
    target_function : function
        the function to be optimized by the Netwons method whose

                **Parameters**:

                    **input_para** : float array of shape (n) - The parameter to be optimized

                    :math:`*` **aux_paras** : - Arbitrary many extra parameters not to be optimized. The :math:`*` refers to the unpacking operator in python.

                **Returns**: 
                
                    float -- the function value
    input_para: float array of shape (n)
        the input parameters for the target_function to be optimized. 
    *aux_paras: 
        Arbitrary many extra parameters not to be optimized for the target_function.
    alpha: float
        the initial step size used in backtracking line search, default = 1
    beta: float
        the decreasing factor of the step size used in backtracking line search, default = 0.5
    c: float
        the parameter used in the Armijo's condiiton, must lies in (0,1), default = 5e-4
    atol: float 
        the absolute error tolerance of the Armijo's condiiton since we use -(atol + rtol*abs(next_value)) instead of 0 to handle single precision numerical error.  default = 5e-6. 
    rtol: float
        the relative error tolerance of the Armijo's condition since we use -(atol + rtol*abs(next_value)) instead of 0 to handle single precision numerical error.  default = 1e-5. 
    max_iter: int
        the maximal iteration allowed for the Netwon's method, default = 100
    max_back_tracking_iter: int
        the maximal iteration allowed for the backtracking line search, default = 25
    tol: float
        the tolerance for residual, below which the optimization stops.
    min_step_size: float
        the minimum step size given by back tracking, below which the optimization stops, default = 1e-6.
    reg_hessian: bool
        Regularize the Hessian if the Cholesky decomposition failed. Default = True
    debug: bool
        print debug information if True.

    Returns
    -------
    Tuple
        A tuple containing

            **opt_para**: *float array of shape (n)* - The optimized parameters.

            **values**: *float* - the optimal value of target_function.
    
            **residuals**: *float* - the residual of the optimization. 

            **step**: *float* - the total number of Newton's step iteration.

            **bsteps**: *float* - the total number of Backtracking step.
    """   
    def back_traking_update(input_para, delta_input, values, grads, hessien, converged, residuals):
        nonlocal max_back_tracking_iter, beta, target_function, aux_paras, c, atol, rtol
        max_delta = jnp.max(jnp.abs(delta_input))
        ##################### Start back_tracking###################
        # The condition for continuing back_tracking
        def Bt_cond( arg ):
            nonlocal max_back_tracking_iter, max_delta

            alpha_0, satisfied, bstep, *debug_args = arg 
            return (bstep < max_back_tracking_iter) & ( satisfied==False ) & ( alpha_0*max_delta > jnp.finfo(delta_input.dtype).resolution  )
        # A back_tracking step
        def Bt_body( arg ):
            nonlocal beta, target_function, values, aux_paras, grads, hessien, c, atol, rtol, converged
            alpha_0, satisfied, bstep, *debug_args = arg

            alpha_0 = alpha_0*beta**(1-satisfied) # Shrink the step size first

            update_para = alpha_0*delta_input
            satisfied,delta_value,grad_delta_value, target_value = Armijo_condition( target_function, values, input_para, update_para,grads, *aux_paras, c = c , atol=atol, rtol=rtol, debug=debug )

            #batch_satisfied,_,_ = Batch_Delta_Armijo_condition( delta_target_func, batch_values, batch_input_para,  batch_update_para, batch_grads,batch_hessien, *batch_output_target, c = c , atol=atol, rtol=rtol )
            # check if the update converged
            #batch_satisfied = jnp.logical_or( batch_converged, batch_satisfied )

            bstep = bstep + 1
            #print(bstep)
            return alpha_0, satisfied, bstep, delta_value, grad_delta_value, target_value
        # We use alpha/beta as initial alpha to compensate the alpha_0 = alpha_0*beta in the function Bt_body
        ######## Loop over shrinked step sizes ########
        alpha_0, satisfied, bstep, delta_value, grad_delta_value, target_value = while_loop( Bt_cond, Bt_body, ( alpha/beta, False , 0, 0., 0., 0.  ) )
        ######## End loop over shrinked step sizes ########
        if debug:
            jax.debug.print("backtrackingstep: {x},\t\t alpha_0: {y},\t\t delta_value: {z},\t\t grad_delta_value: {a},\t\t target_value: {b},\t\t Armijo satisfied: {c},\t\t max_delta: {d}", x=bstep, y=alpha_0, z = delta_value, a = grad_delta_value, b = target_value, c = satisfied, d=max_delta, ordered=True)
        para_update = alpha_0*delta_input 
        return input_para+para_update, para_update, alpha_0, bstep

    def Newton_step_cond( arg  ):
        nonlocal max_iter
        input_para, values, residuals, step, bsteps, converged, grads, hessien, delta_input = arg
        return (step < max_iter) & (  converged==False )
    def Newton_step_body( arg ):
        # Intake necessary quantities
        nonlocal target_function, aux_paras, tol
        input_para, values, residuals, step, bsteps, converged, grads, hessien, delta_input = arg

        ############## Update parameters with Backtracked step #######################
        #No backtracking for the first step since the step direction has not been computed
        if debug:
            jax.debug.print("############BackTracking###########", ordered=True)
        input_para, para_update, alpha_0, bstep= back_traking_update(input_para, delta_input, values, grads, hessien, converged, residuals)  
        #jax.debug.print("############EndBackTracking###########")
        #batch_input_para = back_traking_update(batch_input_para, batch_delta_input, batch_values, batch_grads, batch_converged, batch_residuals)
        ############## End Update parameters with Backtracked step #######################

        # Compute a Netwon step
        if debug:
            jax.debug.print("############Compute Newton Step###########", ordered=True)
        delta_input, values, grads, residuals, hessien, count = Newton_Optimizer_Iteration_Delta(target_function,  input_para, *aux_paras, reg_hessian = reg_hessian )
        if debug:
            jax.debug.print("Loss value: {x},\t\t max_para_update: {y},\t\t residuals: {z},\t\t hessian cond: {a},\t\t update_value_ratio: {b},\t\t hessian_decomp_fail_time: {c}", x=values,y=jnp.max( jnp.abs(para_update)  ), z=residuals,a=jnp.linalg.cond(hessien), b= jnp.linalg.norm(para_update)/jnp.linalg.norm(input_para), c=count, ordered=True)
            #jax.debug.print("residuals: {x}", x=residuals, ordered=True)
            #jax.debug.print("hessian cond: {x}", x=jnp.linalg.cond(hessien), ordered=True)
            #jax.debug.print("update_value_ratio: {x}", x= jnp.linalg.norm(para_update)/jnp.linalg.norm(input_para) , ordered=True)
            jax.debug.print("############End Compute Newton Step###########", ordered=True)
        expected_step_decrease = grads.dot(para_update)
        # Check convergence
        converged = (residuals <= tol) | (alpha_0<=min_step_size) | jnp.sum(jnp.isnan(delta_input))>0

        step = step + 1
        bsteps = bsteps + bstep
        return input_para, values, residuals, step, bsteps, converged, grads, hessien, delta_input
    
    ############## Loop over Newton's steps #######################
    ini_values = ( input_para, 1., 1., 0, 0,  False, jnp.zeros( len(input_para) ), np.eye( len(input_para) ) , jnp.zeros( len(input_para) ) )
    input_para, values, residuals, step, bsteps, converged, grads, hessien, delta_input = while_loop( Newton_step_cond, Newton_step_body, ini_values )
    ############## End Newton's steps #######################
    opt_para = input_para

    return opt_para, values, residuals, step, bsteps

class Newton_Optimizer(BaseOptimizer):
    def __init__(self, target_function, alpha=1.0, beta=0.5, c=5e-4, atol=5e-6, rtol=1e-5, max_iter = 100, max_back_tracking_iter = 25, tol = 1e-6, min_step_size=1e-6, reg_hessian = True, debug = False ):
        """Newton optimizer

        Parameters
        ----------
        target_function : function
            the function to be optimized by the Netwons method whose

                    **Parameters**:

                        **input_para** : float array of shape (n) - The parameter to be optimized

                        :math:`*` **aux_paras** : - Arbitrary many extra parameters not to be optimized. The :math:`*` refers to the unpacking operator in python.

                    **Returns**: 
                    
                        float -- the function value
        alpha: float
            the initial step size used in backtracking line search, default = 1
        beta: float
            the decreasing factor of the step size used in backtracking line search, default = 0.5
        c: float
            the parameter used in the Armijo's condiiton, must lies in (0,1), default = 5e-4
        atol: float 
            the absolute error tolerance of the Armijo's condiiton since we use -(atol + rtol*abs(next_value)) instead of 0 to handle single precision numerical error.  default = 5e-6. 
        rtol: float
            the relative error tolerance of the Armijo's condition since we use -(atol + rtol*abs(next_value)) instead of 0 to handle single precision numerical error.  default = 1e-5. 
        max_iter: int
            the maximal iteration allowed for the Netwon's method, default = 100
        max_back_tracking_iter: int
            the maximal iteration allowed for the backtracking line search, default = 25
        tol: float
            the tolerance for residual, below which the optimization stops.
        min_step_size: float
            the minimum step size given by back tracking, below which the optimization stops, default = 1e-6.
        reg_hessian: bool
            Regularize the Hessian if the Cholesky decomposition failed. Default = True
        debug: bool
            print debug information if True.
        """
        super().__init__(target_function, alpha=alpha, beta=beta, c=c, atol=atol, rtol=rtol, max_iter = max_iter, max_back_tracking_iter = max_back_tracking_iter, tol = tol, min_step_size=min_step_size, reg_hessian = reg_hessian, debug = debug)
    @partial(jax.jit, static_argnums=(0))
    def optimize(self, ini_para, *aux_paras):
        r"""
        optimization of the target_function using the Newton's method with backtracking line search according to the Armijo's condition.

        Parameters
        ----------
        ini_para: float array of shape (n)
            the initial parameters for the target_function to be optimized. 
        *aux_paras: 
            Arbitrary many extra parameters not to be optimized for the target_function.

        Returns
        -------
        Tuple
            A tuple containing

                **opt_para**: *float array of shape (n)* - The optimized parameters.

                **opt_info**: *tuple* - A tuple containing 

                    **values**: *float* - the optimal value of target_function.
            
                    **residuals**: *float* - the residual of the optimization. 

                    **step**: *float* - the total number of Newton's step iteration.

                    **bsteps**: *float* - the total number of Backtracking step.
        """   
        constant = self._kwargs

        beta, values, residual, step, bsteps = Newton_Backtracking_Optimizer_JIT(self._target_function, ini_para, *aux_paras, alpha = constant["alpha"], beta = constant["beta"], c = constant["c"], atol = constant["atol"], rtol = constant["rtol"],max_iter=constant["max_iter"], max_back_tracking_iter = constant["max_back_tracking_iter"], tol = constant["tol"], min_step_size = constant["min_step_size"], reg_hessian = constant["reg_hessian"], debug = constant["debug"] )
        return beta, (values, residual, step, bsteps)
 