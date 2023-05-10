import jax.numpy as jnp
from jax import vmap
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