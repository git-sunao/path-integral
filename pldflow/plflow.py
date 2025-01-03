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
# numpyro
import numpyro
import numpyro.distributions as npyro_dist
from numpyro.infer import MCMC, NUTS
# NF
import torch
import pyro
import pyro.distributions as pyro_dist
import pyro.distributions.transforms as T
# others
from inspect import signature
# plottting
from getdist import MCSamples
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Normalizing Flows
import flax.linen as nn
import optax
from sklearn import datasets, preprocessing
from .maf import MaskedAutoregressiveFlow
# odeint
from jax.experimental.ode import odeint
from .odeint import euler

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
        self.rescale_velocity = True
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
            i = self.integrand(z, *args, **kwargs)
            v = v * jnp.abs(i)
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
        """
        def step(z, dt):
            """One step of Euler integration."""
            z_new = z + dt * self.velocity(z, *args, **kwargs)
            return z_new, z_new

        def integrate(z0):
            """Perform Euler integration for a single initial condition."""
            z = scan(step, z0, jnp.diff(t))[1]
            z = jnp.concatenate([jnp.array([z0]), z])
            return z

        z = integrate(x.astype(self.complex_dtype))

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
        Approximated Jacobian of the Picard-Lefschetz flow at a given initial condition x.

        Parameters:
            x (array): The real-valued initial condition with shape (ndim,)
            t (array/scalar): The time array for integration
            args: Additional arguments to pass to the action function
            kwargs: Additional keyword arguments

        Returns:
            j (array): The approximated Jacobian of the flow evaluated at x with shape (ndim, ndim)
        """
        def step(z_and_j, dt):
            """One step of Euler integration."""
            z, j = z_and_j
            z_new = z + dt * self.velocity(z, *args, **kwargs)
            # j_new = j + dt * self.velocity_divergence(z, *args, **kwargs)
            j_new = j * jnp.exp(dt * self.velocity_divergence(z, *args, **kwargs))
            return (z_new, j_new), (z_new, j_new)
        
        def integrate(z0_and_j0):
            """Perform Euler integration for a single initial condition."""
            z, j = scan(step, z0_and_j0, jnp.diff(t))[1]
            z = jnp.concatenate([jnp.array([z0_and_j0[0]]), z])
            j = jnp.concatenate([jnp.array([z0_and_j0[1]]), j])
            return z, j
        
        z, j = integrate((x.astype(self.complex_dtype), 1.0))

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
        """
        error = kwargs.pop('error', False)
        def ij(x_sample):
            z = self.flow(x_sample, t, uselast=True, *args, **kwargs)
            j = self.flow_jacobian(x_sample, t, *args, uselast=True, **kwargs)
            i = self.integrand(z, *args, **kwargs)
            return i, j
        # We should cut samples with very small probability to avoid numerical issues
        min_lnp = kwargs.pop('min_lnp', -6)
        sel = lnp_samples > lnp_samples.max() + min_lnp
        # Integrate the integrand
        i, j = vmap(ij)(x_samples[sel])
        Z = jnp.mean(i*j*jnp.exp(-lnp_samples[sel]), axis=0)
        if error:
            dZ = jnp.std( (i*j*jnp.exp(-lnp_samples[sel])).real, axis=0)
            dZ+= jnp.std( (i*j*jnp.exp(-lnp_samples[sel])).imag, axis=0)*1j
            n_sample = jnp.sum(sel)
            dZ/= jnp.sqrt(n_sample)
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

    def vflow(self, z, t, *args, **kwargs):
        """Vectorized flow function"""
        f = vmap(lambda zin: self.flow(zin, t, *args, **kwargs), in_axes=0, out_axes=0)
        return f(z)

    def vflow_jacobian(self, z, t, *args, **kwargs):
        """Vectorized flow Jacobian function"""
        f = vmap(lambda zin: self.flow_jacobian(zin, t, *args, **kwargs), in_axes=0, out_axes=0)
        return f(z)
    
    def vflow_jacobian_approx(self, z, t, *args, **kwargs):
        """Vectorized approximated flow Jacobian function"""
        f = vmap(lambda zin: self.flow_jacobian_approx(zin, t, *args, **kwargs), in_axes=0, out_axes=0)
        return f(z)

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

    def plot1d(self, x, t, *args, **kwargs):
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

        if isinstance(t, (int, float)):
            t = jnp.linspace(0, t, 2)
        z = self.vflow(x, t, *args, uselast=False, **kwargs)
        iz= self.vintegrand(z.reshape(-1, self.ndim), *args, **kwargs).reshape(-1, t.size)

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
        """
        # dimension to plot
        dim = kwargs.pop('dim', 0)

        if isinstance(t, (int, float)):
            t = jnp.linspace(0, t, 2)
        z = self.vflow(x, t, *args, uselast=False, **kwargs)
        iz= self.vintegrand(z.reshape(-1, self.ndim), *args, **kwargs).reshape(-1, t.size)

        fig, (ax1, ax2) = self._plot1d_template()
        ax1.set_xlim(x.min(),x.max())
        ax1.set_ylim(x.min(),x.max())
        ax2.set_ylim(-1.1, 1.1)
        line1, = ax1.plot([], [], marker='.', color='blue', lw=2)
        line2, = ax2.plot([], [], ls='--', color='blue')
        line3, = ax2.plot([], [], ls='-.', color='blue')
        line4, = ax2.plot([], [], ls='-', color='blue')

        def update(i):
            line1.set_data(z[:,i,dim].real, z[:,i,dim].imag)
            line2.set_data(x[:,dim], jnp.real(iz[:,i]))
            line3.set_data(x[:,dim], jnp.imag(iz[:,i]))
            line4.set_data(x[:,dim], jnp.abs(iz[:,i]))
            return line1, line2, line3, line4
        
        ani = FuncAnimation(fig, update, frames=t.size, interval=50, blit=True)
        ani.save(fname, writer="pillow")


class HMCSampler(object):
    """
    Hamiltonian Monte Carlo sampler for the action

    Attributes:
        plmodel (PicardLefschetzModelBaseClass): The Picard-Lefschetz model
        priors (dict): The dictionary of priors for the parameters
        param_names (list): The list of parameter names
        group_names (list): The list of group names
        group_ndims (dict): The dictionary of group names and their dimensions
    """
    def __init__(self, plmodel, priors):
        """
        Initialize the HMC sampler for the action

        Parameters:
            plmodel (PicardLefschetzModelBaseClass): The Picard-Lefschetz model
            priors (dict): The dictionary of priors for the parameters
        """
        self.plmodel = plmodel
        self.set_priors(priors)

    def set_priors(self, priors):
        """
        Set the priors for the parameters of the action

        Parameters:
            priors (dict): The dictionary of priors for the parameters
        """
        # Set priors
        self.priors = dict()
        # 1. populate the priors from the grouped priors
        for k, v in priors.items():
            if len(v) == 2: continue
            vmin, vmax, ndim = v
            for i in range(ndim):
                self.priors[f"{k}{i+1}"] = (vmin, vmax)
        # 2. populate the priors for ungrouped priors
        for k, v in priors.items():
            if len(v) == 3: continue
            vmin, vmax = v
            self.priors[k] = (vmin, vmax)
        
        # Set parameter names
        self.param_names = []
        self.group_names = []
        # 1. populate the priors from the grouped priors
        for k, v in priors.items():
            if len(v) == 2: continue
            vmin, vmax, ndim = v
            for i in range(ndim):
                self.param_names.append(f"{k}{i+1}")
                self.group_names.append(k)
        # 2. populate the priors for ungrouped priors
        for k, v in priors.items():
            if len(v) == 3: continue
            if k in self.param_names: continue
            self.param_names.append(k)
            self.group_names.append(k)

        # Determine the dimenion of each group
        self.group_ndims = dict()
        for g, n in zip(self.group_names, self.param_names):
            dim = 0 if g == n else 1
            if g in self.group_ndims:
                self.group_ndims[g] += dim
            else:
                self.group_ndims[g] = dim

    def get_priors(self):
        """
        Get the priors for the parameters
        
        Returns:
            dict: The dictionary of priors for the parameters
        """
        return self.priors.copy()

    # Utility functions 
    def get_group_names(self, include=None, exclude=None):
        """
        Get the group names for the parameters

        Parameters:
            include (list): The list of group names to include
            exclude (list): The list of group names to exclude
        
        Returns:
            list: The list of group names
        """
        if include is None:
            include = self.group_names.copy()
        elif isinstance(include, str):
            include = [include]
        if exclude is None:
            exclude = []
        elif isinstance(exclude, str):
            exclude = [exclude]
        gnames = []
        for g in include:
            if g in exclude: continue
            if g in gnames: continue
            gnames.append(g)
        return gnames

    def get_param_names(self, include=None, exclude=None):
        """
        Get the parameter names

        Parameters:
            include (list): The list of parameter names to include
            exclude (list): The list of parameter names to exclude
        
        Returns:
            list: The list of parameter names
        """
        gnames = self.get_group_names(include=include, exclude=exclude)
        param_names = []
        for pname, gname in zip(self.param_names, self.group_names):
            if gname not in gnames: continue
            param_names.append(pname)
        return param_names

    def get_group_ndim(self, include=None, exclude=None):
        """
        Get the dimension of each group

        Parameters:
            include (list): The list of group names to include
            exclude (list): The list of group names to exclude
        
        Returns:
            dict: The dictionary of group names and their dimensions
        """
        gnames = self.get_group_names(include=include, exclude=exclude)
        group_ndims = dict()
        for gname, ndim in self.group_ndims.items():
            if gname not in gnames: continue
            group_ndims[gname] = ndim
        return group_ndims

    def get_ndim(self, include=None, exclude=None):
        """
        Get the total dimension of the parameters
        
        Parameters:
            include (list): The list of group names to include
            exclude (list): The list of group names to exclude

        Returns:
            int: The total dimension of the parameters
        """
        group_ndims = self.get_group_ndim(include=include, exclude=exclude)
        return sum(group_ndims.values())

    # Sampling related functions
    def likelihood_model(self, t=1.0):
        """
        The likelihood model for the parameters

        Parameters:
            t (array/float): The time for the flow

        Notes:
            The likelihood model is defined as a function for the parameters
            with the signature likelihood_model(t, **parameters)
        """ 
        # Genrate the parameters
        parameters = dict()
        for gname in self.get_group_names():
            ndim = self.get_group_ndim(gname)
            if ndim == 0:
                # generate a parameter as a scalar
                name = self.get_param_names(include=gname)[0]
                vmin, vmax = self.get_priors()[name]
                parameter = numpyro.sample(name, npyro_dist.Uniform(vmin, vmax))
            else:
                # generate a parameter as a vector
                parameter = []
                names = self.get_param_names(include=gname)                
                for name in names:
                    vmin, vmax = self.get_priors()[name]
                    p = numpyro.sample(name, npyro_dist.Uniform(vmin, vmax))
                    parameter.append(p)
                parameter = jnp.array(parameter)
            parameters[gname] = parameter
            
        # flow
        z = self.plmodel.flow(t=t, uselast=True, **parameters)
        # j = self.plmodel.flow_jacobian(t=t, uselast=True, **parameters)
        parameters.pop('x')

        # Integrand
        parameters['z'] = z
        lnp = -self.plmodel.action_s(**parameters).real
        # i   = self.plmodel.integrand(**parameters)
        # lnp = jnp.log(jnp.abs(i*j))

        # Set likelihood and derived values
        numpyro.factor('loglike', lnp)

    def sample(self, num_samples=5000, num_warmup=500, seed=0, **like_kwargs):
        """
        Sample the parameters using the likelihood model

        Parameters:
            num_samples (int): The number of samples to generate
            num_warmup (int): The number of warmup samples
            seed (int): The seed for random number generation
            like_kwargs (dict): The keyword arguments for the likelihood model
        """
        nuts_kernel = NUTS(self.likelihood_model)
        self.mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
        # cpu_device = jax.devices('cpu')[0]
        # with jax.default_device(cpu_device):
        self.mcmc.run(jax.random.PRNGKey(seed), **like_kwargs)

    # Getter functions
    def get_samples(self, format=jnp.array, names=None):
        """
        Get the samples from the MCMC run

        Parameters:
            format (type): The format of the samples (default is jnp.array)
            names (list): The list of parameter names to include

        Returns:
            array: The samples from the MCMC run
        """
        # Get the samples as a dictionary
        samples_dict = self.mcmc.get_samples()
        # List of names to be returned
        if names is None:
            names = list(samples_dict.keys())
        # Format the samples in the desired format
        if format == dict:
            samples = {name: samples_dict[name] for name in names}
        elif format == jnp.array:
            samples = jnp.array([samples_dict[k] for k in names]).T
        elif format == MCSamples:
            arrs    = jnp.array([samples_dict[k] for k in names], dtype=float).T
            ranges  = self.get_priors()
            samples = MCSamples(samples=arrs, names=names, ranges=ranges)
            samples.names = names # append names for the ease of use
        else:
            raise ValueError(f"Invalid format {format=}")
        return samples

class MAFModel(object):
    """
    Masked Autoregressive Flow model for the action

    Attributes:
        sampler (HMCSampler): The HMC sampler for the action
        model (MaskedAutoregressiveFlow): The MAF model
        trained_params (dict): The trained parameters of the MAF model
        loss_history_maf (list): The history of the training loss for the MAF model
    """
    def __init__(self, sampler, process=True):
        self.sampler = sampler
        self.ndim_x = self.sampler.get_ndim(include='x')
        self.ndim_c = self.sampler.get_ndim(exclude='x')
        self.process = process
        self.compute_processing_params()

    def build(self, **config_in):
        """
        Build the MAF model

        Parameters:
            config_in (dict): The configuration for the MAF model

        Notes:
            The configuration can include the following keys:
            - hidden_dims: The list of hidden dimensions for the MAF model (default is [128, 128])
            - activation: The activation function for the MAF model (default is "tanh")
            - n_transforms: The number of transformations for the MAF model (default is 4)
            - use_random_permutations: Whether to use random permutations (default is False)
        """
        # Default configuration
        config =   {'hidden_dims': [128, 128], \
                    'activation': "tanh", \
                    'n_transforms': 4, \
                    'use_random_permutations': False, 
                    'n_dim': self.ndim_x, \
                    'n_context': self.ndim_c}
        # Overwrite the default configuration
        config.update(config_in)
        # Initialize the MAF model
        self.model = MaskedAutoregressiveFlow(**config)

    def train(self, **config_in):
        """
        Train the MAF model using the samples from the sampler

        Parameters:
            config_in (dict): The configuration for the training

        Notes:
            The configuration can include the following keys:
            - seed: The seed for random number generation (default is 0)
            - learning_rate: The learning rate for the optimizer (default is 3e-4)
            - batch_size: The batch size for training (default is 64)
            - n_steps: The number of training steps (default is 1000)
        """
        # Default configuration
        config  =  {'seed': 0, \
                    'learning_rate': 3e-4, \
                    'batch_size': 64, \
                    'n_steps': 1000, 
                    'patience': 20}
        # Overwrite the default configuration
        config.update(config_in)

        # Initialize the model
        key    = jax.random.PRNGKey(config['seed'])
        x_test = jax.random.uniform(key=key, shape=(config['batch_size'], self.ndim_x))
        context= jax.random.uniform(key=key, shape=(config['batch_size'], self.ndim_c))
        params = self.model.init(key, x_test, context)

        # Optimizer
        optimizer = optax.adam(learning_rate=config['learning_rate'])
        opt_state = optimizer.init(params)

        # Loss function
        @jax.jit
        def loss_fn(params, x, context):
            return -jnp.mean(self.model.apply(params, x, context))
        
        # One step in the training loop
        @jax.jit
        def step(params, opt_state, batch):
            x, context = batch
            loss, grads = jax.value_and_grad(loss_fn)(params, x, context)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        # Samples for training
        samples = self.get_training_samples()

        # Training loop!
        best_loss = float("inf")
        steps_without_improvement = 0
        self.loss_history_maf = []
        key = jax.random.PRNGKey(config['seed'])
        for step_idx in tqdm(range(config['n_steps']), desc="Training MAF"):
            key, subkey = jax.random.split(key)
            indices = jax.random.choice(subkey, samples.shape[0], shape=(64,), replace=False)
            batch = (samples[indices, :self.ndim_x], samples[indices, self.ndim_x:])
            params, opt_state, loss = step(params, opt_state, batch)
            self.loss_history_maf.append(loss)

            # Early stopping logic
            if loss < best_loss:
                best_loss = loss
                steps_without_improvement = 0  # Reset the counter
            else:
                steps_without_improvement += 1  # Increment the counter

            # Stop training if no improvement for 'patience' steps
            if steps_without_improvement >= config['patience']:
                print(f"Early stopping at step {step_idx + 1} with best loss {best_loss}")
                break

        self.trained_params = params

    def plot_loss(self):
        """Plot the training loss"""
        plt.figure(figsize=(5,3))
        plt.plot(self.loss_history_maf)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

    @partial(jit, static_argnums=(0,2,3,4))
    def sample(self, context, num_samples=10000, seed=0, beta=1.0):
        """
        Sample from the trained MAF model

        Parameters:
            context (array): The context for the sampling with shape (n_context,)
            num_samples (int): The number of samples to generate
            seed (int): The seed for random number generation
        
        Returns:
            array: The samples from the MAF model with shape (num_samples, n_dim)
        """
        # Format the context shape if necessary
        if context.ndim == 0 or context.ndim == 1:
            context = jnp.tile(context, (num_samples, 1))

        # Tansform the context to the standardized form
        context = self.preprocess_context(context)

        # Sample from the trained network
        key = jax.random.PRNGKey(seed)
        x = self.model.apply(self.trained_params, num_samples=num_samples, rng=key, context=context, beta=beta, method='sample')

        # Postprocess the samples
        x = self.postprocess_x(x, context)

        return jnp.array(x)

    @partial(jit, static_argnums=(0,3))
    def log_prob(self, x, context, beta=1.0):
        """
        Caution: This is slow. Using sample_and_log_prob is recommended.

        Evaluate the log-probability of the samples

        Parameters:
            x (array): The samples with shape (num_samples, n_dim)
            context (array): The context for the sampling with shape (n_context,)

        Returns:
            array: The log-probability of the samples with shape (num_samples,)
        """
        # Format the context shape if necessary
        num_samples = x.shape[0]
        if context.ndim == 0 or context.ndim == 1:
            context = jnp.tile(context, (num_samples, 1))

        # Preprocess the samples
        x       = self.preprocess_x(x, context)
        context = self.preprocess_context(context)

        # Evaluate the log-probability of the samples
        log_prob = self.model.apply(self.trained_params, x=x, context=context, beta=beta, method='__call__')

        # Postprocess the log-probability
        log_prob = self.postprocess_logprob(log_prob)

        return log_prob

    @partial(jit, static_argnums=(0,2,3,4))
    def sample_and_log_prob(self, context, num_samples=10000, seed=0, beta=1.0):
        # Format the context shape if necessary
        if context.ndim == 0 or context.ndim == 1:
            context = jnp.tile(context, (num_samples, 1))

        # Tansform the context to the standardized form
        context = self.preprocess_context(context)

        # Sample from the trained network
        key = jax.random.PRNGKey(seed)
        x, log_prob = self.model.apply(self.trained_params, num_samples=num_samples, rng=key, context=context, beta=beta, method='sample_and_log_prob')
        
        # Postprocess the samples
        x = self.postprocess_x(x, context)
        log_prob = self.postprocess_logprob(log_prob)
        
        return jnp.array(x), jnp.array(log_prob)

    # Sample processing functions
    def compute_processing_params(self):
        """
        Compute the linear processing parameters for the MAF model.
        """
        # Get the samples of x and others separately
        names_x = self.sampler.get_param_names(include='x')
        names_c = self.sampler.get_param_names(exclude='x')
        samples_x = self.sampler.get_samples(format=jnp.array, names=names_x)
        samples_c = self.sampler.get_samples(format=jnp.array, names=names_c)

        if self.process:
            # Standardize the samples
            mean_x = jnp.mean(samples_x, axis=0)
            std_x  = jnp.std(samples_x, axis=0)
            samples_x_standardized = (samples_x - mean_x) / std_x
            mean_y = jnp.mean(samples_c, axis=0)
            std_y  = jnp.std(samples_c, axis=0)
            samples_c_standardized = (samples_c - mean_y) / std_y

            # Get the covariance matrix
            cov = jnp.cov(samples_x_standardized.T, samples_c_standardized.T)
            cov_cc = cov[self.ndim_x:, self.ndim_x:]
            cov_cx = cov[self.ndim_x:, :self.ndim_x]

            # Get the linear transformation coefficients
            dec_mat = jnp.linalg.solve(cov_cc, cov_cx).T

            # Save the linear transformation coefficients
            self.pp_params   = {'dec_mat': dec_mat, \
                                'mean_x': mean_x, \
                                'mean_c': mean_y, \
                                'std_x': std_x, \
                                'std_c': std_y}
        else:
            self.pp_params = {'dec_mat': jnp.zeros((self.ndim_x, self.ndim_c)), \
                              'mean_x': jnp.zeros(self.ndim_x), \
                              'mean_c': jnp.zeros(self.ndim_c), \
                              'std_x': jnp.ones(self.ndim_x), \
                              'std_c': jnp.ones(self.ndim_c)}

    def preprocess_context(self, samples_c):
        """
        Linear preprocess the context before inputting to the MAF model.
        This function applies the standardization of the context samples.

        Parameters:
            samples_c (array): The original samples of context with shape 
                               (num_samples, n_context)

        Returns:
            samples_c (array): The standardized samples of context with shape 
                               (num_samples, n_context)
        """
        # Get the linear transformation coefficients
        mean_c = self.pp_params['mean_c']
        std_c  = self.pp_params['std_c']

        # Standardize the samples
        samples_c = (samples_c - mean_c) / std_c

        return samples_c

    def postprocess_context(self, samples_c):
        """
        Linear postprocess the context after training the MAF model.
        This function applies the inverse transformation of the linear
        preprocess function.

        Parameters:
            samples_c (array): The standardized samples of context with shape
                               (num_samples, n_context)

        Returns:
            samples_c (array): The original samples of context with shape
                               (num_samples, n_context)
        """
        # Get the linear transformation coefficients
        mean_c = self.pp_params['mean_c']
        std_c  = self.pp_params['std_c']

        # Postprocess the samples
        samples_c = samples_c * std_c + mean_c

        return samples_c

    def preprocess_x(self, samples_x, samples_c):
        """
        Linear preprocess the samples before training the MAF model.
        If the x samples have linear correlation to the other model parameters,
        this can be removed analytically. This linear correlation is removed
        in this function.

        We first standardize the samples of x and context separately.
            X -> (X - mean(X)) / std(X)
            C -> (C - mean(C)) / std(C)
        Then we remove the linear correlation of x and context, assuming
            X = M C + S

        Parameters:
            samples_x (array): The samples of x with shape (num_samples, ndim)
            samples_c (array): The samples of context with shape (num_samples, n_context)
        
        Returns:
            samples_x (array): The linearly preprocessed samples with shape (num_samples, ndim)
        """
        # Get the linear transformation coefficients
        mean_c = self.pp_params['mean_c']
        std_c  = self.pp_params['std_c']
        mean_x = self.pp_params['mean_x']
        std_x  = self.pp_params['std_x']
        dec_mat= self.pp_params['dec_mat']

        # Standardize the samples
        samples_x_standardized = (samples_x - mean_x) / std_x
        samples_c_standardized = (samples_c - mean_c) / std_c

        # Decorrelate the x samples
        samples_x_dec = samples_x_standardized - samples_c_standardized @ dec_mat.T

        return samples_x_dec

    def postprocess_x(self, samples_x, samples_c):
        """
        Linear postprocess the samples after training the MAF model.
        This function applies the inverse transformation of the linear
        preprocess function.

        Parameters:
            samples_x (array): The standardized samples of x with shape 
                               (num_samples, ndim)
            samples_c (array): The standardized samples of context with shape 
                               (num_samples, n_context)

        Returns:
            samples_x (array): The postprocessed samples of x with shape 
                               (num_samples, ndim)
        """
        # Get the linear transformation coefficients
        mean_x = self.pp_params['mean_x']
        std_x  = self.pp_params['std_x']
        dec_mat= self.pp_params['dec_mat']
        
        # Add correlated part
        samples_x = samples_x + samples_c @ dec_mat.T

        # Postprocess the samples
        samples_x = samples_x * std_x + mean_x

        return samples_x

    def postprocess_logprob(self, logp):
        """
        Correct the log-probability of the samples after training the MAF model.
        Since we apply the linear preprocessing on the samples, the probabily we 
        train with the MAF model is not the original probability. This function
        corrects the log-probability to the original one. Since the preprocessing
        is linear, the correction is done analytically. Denoting the original
        sample `x` and the preprocessed sample `x`, the relation of the probability
        is derived by the probability conservation:
            dx P(x) = dx' P(x')
        So the correction is the jacobian of the transformation:
            log P(x) = log P(x') + log |det dx'/dx|
        In our case, the transformation is linear, so the correction is simply
            dx'/dx = 1 / std(x)
        Note that shift term has nothinig to do with the Jacobian.
        
        Parameters:
            logp (array): The log-probability of the preprocessed samples

        Returns:
            logp (array): The corrected log-probability of the samples
        """
        return logp - jnp.sum(jnp.log(self.pp_params['std_x']))

    def get_training_samples(self):
        """
        Get the preprocessed samples for training the MAF model.

        Returns:
            array: The preprocessed samples for training the MAF model
        """
        # Get the samples of x and others separately
        names_x = self.sampler.get_param_names(include='x')
        names_c = self.sampler.get_param_names(exclude='x')
        samples_x = self.sampler.get_samples(format=jnp.array, names=names_x)
        samples_c = self.sampler.get_samples(format=jnp.array, names=names_c)

        # Preprocess the samples
        samples_x = self.preprocess_x(samples_x, samples_c)
        samples_c = self.preprocess_context(samples_c)

        # Concatenate the samples
        samples = jnp.concatenate([samples_x, samples_c], axis=1)

        return samples