# jax
import jax.numpy as jnp
import jax

def euler(vfunc, y0, t_array, *args):
    """
    Euler integration for dy/dt = v(y, t, *args).

    Parameters:
        vfunc: Function v(y, t, *args), represents dy/dt.
        y0: Initial condition (scalar).
        t_array: Array of time steps.
        *args: Additional arguments to pass to vfunc.

    Returns:
        Array of y values at each time step.
    """
    dt = jnp.diff(t_array)

    def step(y, t_and_dt):
        """One step of Euler integration."""
        t, dt = t_and_dt
        y_new = y + dt * vfunc(y, t, *args)
        return y_new, y_new

    def integrate(y0):
        """Perform Euler integration for a single initial condition."""
        t_and_dt = jnp.stack([t_array[:-1], dt], axis=-1)
        y_values = jax.lax.scan(lambda y, tdt: step(y, tdt), y0, t_and_dt)[1]
        # y_values = jax.lax.scan(lambda y, tdt: (y, step(y, tdt)), y0, t_and_dt)[1]
        return jnp.concatenate([jnp.array([y0]), y_values])

    # `integrate` is vmap-compatible. You can apply vmap for multiple initial conditions.
    return integrate(y0)