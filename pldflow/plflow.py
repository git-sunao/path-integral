# jax
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit
from jax import grad
from jax import vmap
from jax import jacobian, hessian
from jax import jacfwd, jacrev
from jax import vjp
from jax import jvp
from jax.lax import scan
from tqdm import tqdm
from functools import partial
# plottting
from getdist import MCSamples
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PicardLefschetzModelBaseClass(object):
    """
    Base class for the Picard-Lefschetz model.
    The target is to solve the integral of the form
        Z = \int dx e^{-S(x)}

    Attributes:
        ndim (int): The dimension of the model
        rescale_velocity (bool): Whether to rescale the velocity
        complex_dtype (dtype): The dtype for the complex numbers
    """
    ndim: int = 1
    def __init__(self, ndim=None):
        self.rescale_velocity = False
        self.complex_dtype = jnp.complex64
        if ndim is not None:
            self.ndim = ndim

    # Functions
    def action_s(self, z, *args, **kwargs):
        """
        Action function S(z) = -log(I(z))

        Parameters:
            z (array): The real or complex variable z with shape (ndim,)
            args: Additional arguments to pass to the action function

        Returns:
            s (array): The action function evaluated at z, scalar
        """
        raise NotImplementedError("action_s must be implemented in a subclass")

    @partial(jit, static_argnums=(0,))
    def grad_s(self, z, *args, **kwargs):
        """
        Gradient of the action function dS/dz

        Parameters:
            z (array): The real or complex variable z with shape (ndim,)
            args: Additional arguments to pass to the action function

        Returns:
            g (array): The gradient of the action function evaluated at z with shape (ndim,)
        """
        g = grad(self.action_s, argnums=0, holomorphic=True)
        return g(z, *args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def hessian_s(self, z, *args, **kwargs):
        """
        Hessian of the action function d^2S/dz^2

        Parameters:
            z (array): The real or complex variable z with shape (ndim,)
            args: Additional arguments to pass to the action function
        
        Returns:
            h (array): The Hessian of the action function evaluated at z with shape (ndim, ndim)
        """
        h = hessian(self.action_s, argnums=0, holomorphic=True)
        return h(z, *args, **kwargs)

    def integrand(self, z, *args, **kwargs):
        """
        Integrand function I(z) = e^{-S(z)}

        Parameters:
            z (array): The real or complex variable z with shape (ndim,)
            args: Additional arguments to pass to the action function

        Returns:
            i (array): The integrand function evaluated at z, scalar
        """
        s = self.action_s(z, *args, **kwargs)
        return jnp.exp(-s)

    def velocity(self, z, *args, **kwargs):
        """
        Velocity of the Picard-Leffschetz flow dz/dt = v(z)

        Parameters:
            z (array): The real or complex variable z with shape (ndim,)
            args: Additional arguments to pass to the action function
        
        Returns:
            v (array): The velocity of the flow evaluated at z with shape (ndim,)
        """
        # Velocity for the flow
        v    = jnp.conj(self.grad_s(z, *args, **kwargs))
        # Rescale the velocity
        if self.rescale_velocity:
            # Exponential + polynomial rescaling
            s = self.action_s(z, *args, **kwargs).real
            v = v * 2 / (1+jnp.abs(s)+jnp.exp(s)/1e3)
        return v

    @partial(jit, static_argnums=(0,))
    def velocity_divergence(self, z, *args, **kwargs):
        """
        Special divergence of the Picard-Lefschetz flow velocity,
        which is used to obtain the approximate Jacobian of the flow.
        What will be needed is the trace of the Jacobian where jacobian
        is defined as the derivative of the flow with respect to the 
        real part of the input variable.

        Parameters:
            z (array): The real or complex variable z with shape (ndim,)
            args: Additional arguments to pass to the action function

        Returns:
            d (array): The divergence of the flow velocity evaluated at z, scalar
        """
        z_real = jnp.real(z)
        z_imag = jnp.imag(z)

        def func(x):
            z_add = x + 1j * z_imag
            v_val = self.velocity(z_add, *args, **kwargs)
            return v_val

        j = jacfwd(func)(z_real)
        d = jnp.trace(j)

        return d

    # Flow related functions
    @partial(jit, static_argnums=(0,), static_argnames=("uselast",))
    def flow(self, x, t, *args, uselast=True, **kwargs):
        """
        Flow of the Picard-Lefschetz flow for a given initial condition z

        Parameters:
            x (array): The real variable z with shape (ndim,)
            t (array/scalar): The time array for integration
            args: Additional arguments to pass to the action function
            kwargs: Additional keyword arguments

        Returns:
            z (array): The solution of the flow at each time step with shape (ndim, t.size)

        Algorithm:
            The flow equation dz/dt = v(z) is solved using the Euler method,
            with some modifications to ensure the monotonic increase of the action.
            The naive application of the Euler method derives

            .. math::
                z(t+dt) = z(t) + dt * v(z(t))
            
            However, this may not guarantee the monotonic increase of the action,
            especially when the action has singularity around which the numerical
            flow can be unstable. To avoid this, we apply the following algorithm:

            1.  Compute the next point with the naive Euler method
            2.  If the next point has a larger action, accept it, otherwise 
                keep the current point, and accumulate the number of stay at 
                the position (we call it n in this func).
            3.  In the next loop, we rescale the shift term by 2^n to try smaller
                steps to avoid the singularity.

            This effectively slows down the flow sorresponding to the adoptive 
            time step to avoid the singularity.
        """
        def step(zsn, dt):
            """One step of Euler integration."""
            z, s, n = zsn
            z_tmp = z + dt * self.velocity(z, *args, **kwargs) / 2.0**n
            s_tmp = self.action_s(z_tmp, *args, **kwargs).real
            z_new = jnp.where(s_tmp >= s, z_tmp, z)
            s_new = jnp.where(s_tmp >= s, s_tmp, s)
            n_new = jnp.where(s_tmp >= s, 0, n+1)
            return (z_new, s_new, n_new), (z_new, s_new, n_new)

        def integrate(z0s0n0):
            """Perform Euler integration for a single initial condition."""
            z, s, n = scan(step, z0s0n0, jnp.diff(t))[1]
            z = jnp.concatenate([jnp.array([z0s0n0[0]]), z])
            return z

        z0= x.astype(self.complex_dtype)
        s0= self.action_s(z0, *args, **kwargs).real
        n0= jnp.zeros_like(x)
        z = integrate((z0,s0,n0))

        if uselast:
            z = z[-1]

        return z
    
    @partial(jit, static_argnums=(0,), static_argnames=("uselast",))
    def flow_jacobian(self, x, t, *args, uselast=True, **kwargs):
        """
        Jacobian of the Picard-Lefschetz flow at a given initial condition x.

        Parameters:
            x (array): The real-valued initial condition with shape (ndim,)
            t (array/scalar): The time array for integration
            args: Additional arguments to pass to the action function
            kwargs: Additional keyword arguments

        Returns:
            j (array): The Jacobian of the flow evaluated at x with shape (ndim, ndim)
        """
        def flow_split(x):
            z = self.flow(x, t, *args, uselast=uselast, **kwargs)
            return jnp.real(z), jnp.imag(z)
        j = jacobian(flow_split)(x)
        j = j[0] + 1j * j[1]
        return jnp.linalg.det(j)

    @partial(jit, static_argnums=(0,), static_argnames=("uselast", "withz"))
    def flow_jacobian_approx(self, x, t, *args, uselast=True, withz=True, **kwargs):
        """
        [Warning] This function is not recommended for the general use.

        Approximated Jacobian of the Picard-Lefschetz flow at a given initial condition x.

        Parameters:
            x (array): The real-valued initial condition with shape (ndim,)
            t (array/scalar): The time array for integration
            args: Additional arguments to pass to the action function
            kwargs: Additional keyword arguments

        Returns:
            j (array): The approximated Jacobian of the flow evaluated at x with shape (ndim, ndim)
        """
        def step(zjs, dt):
            """One step of Euler integration."""
            z, j, s = zjs
            z_tmp = z + dt * self.velocity(z, *args, **kwargs)
            s_tmp = self.action_s(z_tmp, *args, **kwargs).real
            j_tmp = j * jnp.exp(dt * self.velocity_divergence(z, *args, **kwargs))
            z_new = jnp.where(s_tmp >= s, z_tmp, z)
            j_new = jnp.where(s_tmp >= s, j_tmp, j)
            s_new = jnp.where(s_tmp >= s, s_tmp, s)
            return (z_new, j_new, s_new), (z_new, j_new, s_new)
        
        def integrate(z0j0s0):
            """Perform Euler integration for a single initial condition."""
            z, j, s = scan(step, z0j0s0, jnp.diff(t))[1]
            z = jnp.concatenate([jnp.array([z0_and_j0[0]]), z])
            j = jnp.concatenate([jnp.array([z0_and_j0[1]]), j])
            return z, j
        
        z0 = x.astype(self.complex_dtype)
        j0 = 1.0
        s0 = self.action_s(z0, *args, **kwargs).real
        z, j = integrate((z0, j0, s0))

        if uselast:
            z = z[-1]
            j = j[-1]

        if withz :
            return z, j
        else:
            return j

    # Integration with samples
    def integrate(self, x_samples, lnp_samples, t, *args, **kwargs):
        """
        Integrate the integrand I(z) = e^{-S(z)} based on the importance 
        sampling using the samples of x provided with the corresponding 
        log-probability lnp_samples.

        Parameters:
            x_samples (array): The samples of x with shape (nsamples, ndim)
            lnp_samples (array): The log-probability of the samples with shape (nsamples,)
            t (array/scalar): The time array for integration
            args: Additional arguments to pass to the action function
            kwargs: Additional keyword arguments
        
        Returns:
            Z (scalar): The integrated value of the integrand

        Notes:
            Acceptable keyword arguments for this function are:
            - error   : Whether to estimate the error of the integral (default is False)
                        This error is the statistical error of the integral due to the 
                        finite number of samples. The systematics error is not included.
            - min_lnp : The minimum log-probability to consider the samples (default is -6)
        """
        error = kwargs.pop('error', False)
        def ij(x_sample):
            z = self.flow(x_sample, t, uselast=True, *args, **kwargs)
            j = self.flow_jacobian(x_sample, t, *args, uselast=True, **kwargs)
            i = self.integrand(z, *args, **kwargs)
            return i, j
        # Integrand
        i, j = vmap(ij)(x_samples)
        integrand = i*j*jnp.exp(-lnp_samples)
        # We should cut samples with very small probability to avoid numerical issues
        min_lnp = kwargs.pop('min_lnp', -6)
        mask = lnp_samples > lnp_samples.max() + min_lnp
        Z  = jnp.mean(integrand, axis=0, where=mask)
        dZ = jnp.std( integrand.real, axis=0, where=mask) + 1j*jnp.std( integrand.imag, axis=0, where=mask)      
        dZ/= jnp.sqrt(jnp.sum(mask, axis=0))
        if error:
            return Z, dZ
        else:
            return Z

    def integrate_approx(self, x_samples, lnp_samples, t, *args, **kwargs):
        """
        Integrate the integrand I(z) = e^{-S(z)} based on the importance 
        sampling using the samples of x provided with the corresponding 
        log-probability lnp_samples.

        Parameters:
            x_samples (array): The samples of x with shape (nsamples, ndim)
            lnp_samples (array): The log-probability of the samples with shape (nsamples,)
            t (array/scalar): The time array for integration
            args: Additional arguments to pass to the action function
            kwargs: Additional keyword arguments
        
        Returns:
            Z (scalar): The integrated value of the integrand
        """
        def ij(x_sample):
            z, j = self.flow_jacobian_approx(x_sample, t, *args, uselast=True, withz=True, **kwargs)
            i = self.integrand(z, *args, **kwargs)
            return i, j
        # We should cut samples with very small probability to avoid numerical issues
        min_lnp = kwargs.pop('min_lnp', -6)
        sel = lnp_samples > lnp_samples.max() + min_lnp
        # Integrate the integrand
        i, j = vmap(ij)(x_samples[sel])
        Z = jnp.mean(i*j*jnp.exp(-lnp_samples[sel]), axis=0)
        return Z

    # Vectorized functions
    def vaction_s(self, z, *args, **kwargs):
        """Vectorized action function"""
        s = vmap(lambda zin: self.action_s(zin, *args, **kwargs), in_axes=0, out_axes=0)
        return s(z)

    def vgrad_s(self, z, *args, **kwargs):
        """Vectorized gradient of the action function"""
        g = vmap(lambda zin: self.grad_s(zin, *args, **kwargs), in_axes=0, out_axes=0)
        return g(z)

    def vhessian_s(self, z, *args, **kwargs):
        """Vectorized Hessian of the action function"""
        h = vmap(lambda zin: self.hessian_s(zin, *args, **kwargs), in_axes=0, out_axes=0)
        return h(z)

    def vintegrand(self, z, *args, **kwargs):
        """Vectorized integrand function"""
        i = vmap(lambda zin: self.integrand(zin, *args, **kwargs), in_axes=0, out_axes=0)
        return i(z)
    
    def vvelocity(self, z, *args, **kwargs):
        """Vectorized flow velocity function"""
        v = vmap(lambda zin: self.velocity(zin, *args, **kwargs), in_axes=0, out_axes=0)
        return v(z)

    def vvelocity_divergence(self, z, *args, **kwargs):
        """Vectorized flow velocity divergence function"""
        d = vmap(lambda zin: self.velocity_divergence(zin, *args, **kwargs), in_axes=0, out_axes=0)
        return d(z)

    def vflow(self, x, t, *args, **kwargs):
        """Vectorized flow function"""
        f = vmap(lambda xin: self.flow(xin, t, *args, **kwargs), in_axes=0, out_axes=0)
        return f(x)

    def vflow_jacobian(self, x, t, *args, **kwargs):
        """Vectorized flow Jacobian function"""
        f = vmap(lambda xin: self.flow_jacobian(xin, t, *args, **kwargs), in_axes=0, out_axes=0)
        return f(x)
    
    def vflow_jacobian_approx(self, z, t, *args, **kwargs):
        """Vectorized approximated flow Jacobian function"""
        f = vmap(lambda zin: self.flow_jacobian_approx(zin, t, *args, **kwargs), in_axes=0, out_axes=0)
        return f(z)

    def vintegrate_with_sampler(self, context, num_samples, sampler):
        i = vmap(lambda context: self.integrate_with_sampler(context, num_samples, sampler), in_axes=0, out_axes=0)
        return i(context)

    # Plotter
    def _plot1d_template(self, fig=None, axes=None):
        """Template for the 1D plotter"""
        fig = plt.figure(figsize=(5,6))
        ax1 = fig.add_axes((.15,.30,.75,.55))
        ax2 = fig.add_axes((.15,.10,.75,.15), sharex=ax1)
        ax1.grid()
        ax1.set_ylabel(r'$\mathcal{I}(z)$')
        ax1.set_xlabel(r'$\mathcal{R}(z)$')
        ax2.axhline(0., color='black', lw=1, ls='--')
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'$I(x) = e^{-S(x)}$')
        return fig, [ax1, ax2]

    def plot1d(self, x, t, *args, t_indices=None, **kwargs):
        """
        Plot the flow of the Picard-Lefschetz flow in 1D

        Parameters:
            x (array): The real-valued initial condition with shape (n,)
            t (array/scalar): The time array for integration
            args: Additional arguments to pass to the action function
            kwargs: Additional keyword arguments

        Returns:
            fig, (ax1, ax2): The figure and axes for the plot

        Notes:
            Acceptable keyword arguments for this function are:
            - dim: The dimension to plot (default is 0)
        """
        # dimension to plot
        dim = kwargs.pop('dim', 0)

        # sort x for better visualization
        x = x[jnp.argsort(x[:, dim]), :]

        if isinstance(t, (int, float)):
            t = jnp.linspace(0, t, 2)
        z = self.vflow(x, t, *args, uselast=False, **kwargs)
        iz= self.vintegrand(z.reshape(-1, self.ndim), *args, **kwargs).reshape(-1, t.size)

        if t_indices is not None:
            t_indices = jnp.array(t_indices)
            t = t[t_indices]
            z = z[:,t_indices]
            iz= iz[:,t_indices]

        color = []
        for i in range(t.size):
            color.append(plt.cm.bwr(i/(t.size-1)))

        fig, (ax1, ax2) = self._plot1d_template()
        # Plot the path
        for i in range(t.size):
            ax1.plot(z[:,i,dim].real, z[:,i,dim].imag, marker='.', color=color[i])
        ax1.set_xlim(x.min(),x.max())
        ax1.set_ylim(x.min(),x.max())
        ax1.axhline(0, color='black', lw=1, ls='--')
        ax1.axvline(0, color='black', lw=1, ls='--')
        # Plot the integrand
        for i in range(t.size):
            ax2.plot(x, jnp.real(iz[:,i]), ls='--', color=color[i])
            ax2.plot(x, jnp.imag(iz[:,i]), ls='-.', color=color[i])
            ax2.plot(x, jnp.abs(iz[:,i]) , ls='-' , color=color[i])
        return fig, (ax1, ax2)

    def plot1dgif(self, fname, x, t, *args, **kwargs):
        """
        Plot the flow of the Picard-Lefschetz flow in 1D and save it as a GIF

        Parameters:
            fname (str): The filename to save the GIF
            x (array): The real-valued initial condition with shape (n,)
            t (array/scalar): The time array for integration
            args: Additional arguments to pass to the action function
            kwargs: Additional keyword arguments

        Notes:
            Acceptable keyword arguments for this function are:
            - dim: The dimension to plot (default is 0)
            - fig: The figure for the plot
            - axes: The axes for the plot
            - color: The color for the plot
        """
        # dimension to plot
        dim = kwargs.pop('dim', 0)

        # fig and axes
        if 'fig' in kwargs and 'axes' in kwargs:
            fig = kwargs.pop('fig')
            (ax1, ax2) = kwargs.pop('axes')
        else:
            fig, (ax1, ax2) = self._plot1d_template()

        # xmin, xmax
        xmin = kwargs.pop('xmin', x.min())
        xmax = kwargs.pop('xmax', x.max())
        ymin = kwargs.pop('ymin', xmin)
        ymax = kwargs.pop('ymax', xmax)

        # color
        color = kwargs.pop('color', 'blue')

        # dpi for saving
        dpi = kwargs.pop('dpi', 100)

        # animation writer
        writer = kwargs.pop('writer', 'ffmpeg')

        if isinstance(t, (int, float)):
            t = jnp.linspace(0, t, 2)
        z = self.vflow(x, t, *args, uselast=False, **kwargs)
        iz= self.vintegrand(z.reshape(-1, self.ndim), *args, **kwargs).reshape(-1, t.size)

        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        ax2.set_ylim(-1.1, 1.1)
        line1, = ax1.plot([], [], marker='.', color=color, lw=2)
        line2, = ax2.plot([], [], ls='--', color=color)
        line3, = ax2.plot([], [], ls='-.', color=color)
        line4, = ax2.plot([], [], ls='-', color=color)

        def update(i):
            line1.set_data(z[:,i,dim].real, z[:,i,dim].imag)
            line2.set_data(x[:,dim], jnp.real(iz[:,i]))
            line3.set_data(x[:,dim], jnp.imag(iz[:,i]))
            line4.set_data(x[:,dim], jnp.abs(iz[:,i]))
            return line1, line2, line3, line4

        ani = FuncAnimation(fig, update, frames=t.size, interval=50, blit=False)
        ani.save(fname, writer=writer, dpi=dpi)

    def plot1d_action_map(self, n, *args, **kwargs):
        """
        Plot the action map in 1D

        Parameters:
            n (int): The number of points to plot
            args: Additional arguments to pass to the action function
            kwargs: Additional keyword arguments

        Returns:
            fig, ax: The figure and axes for the plot

        Notes:
            Acceptable keyword arguments for this function are:
            - fig: The figure for the plot
            - axes: The axes for the plot
            - cmap: The colormap for the plot (default is "PuOr")
        """
        # fig and axes
        if 'fig' in kwargs and 'axes' in kwargs:
            fig = kwargs.pop('fig')
            ax = kwargs.pop('axes')[0]
        else:
            fig, ax = plt.subplots()
        # cmap
        cmap = kwargs.pop('cmap', 'PuOr')
        # shading
        shading = kwargs.pop('shading', None)

        zx = jnp.linspace(-3,3,n)
        zy = jnp.linspace(-3,3,n)
        z = jnp.tile(zx, zy.size) + 1j*jnp.repeat(zy, zx.size)
        z = z.reshape(-1, self.ndim)
        s = self.vaction_s(z, *args, **kwargs).reshape(zx.size, zy.size)
        ax.pcolormesh(zx, zy, -s.real, cmap=cmap, alpha=1, shading=shading)
        return fig, (ax,)