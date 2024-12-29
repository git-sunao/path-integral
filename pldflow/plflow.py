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
from jax.lax import scan
from tqdm import tqdm
# getdist
from getdist import MCSamples
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
    Z = \int dx e^{-S(x)}
    """
    ndim: int = 1
    def __init__(self):
        self.rescale_velocity = True
        self.complex_dtype = jnp.complex64

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

    # PL flow
    def flow_velocity(self, z, *args, **kwargs):
        """
        Velocity of the Picard-Leffschetz flow dz/dt = v(z)

        Parameters:
            z (array): The real or complex variable z with shape (ndim,)
            args: Additional arguments to pass to the action function
        
        Returns:
            v (array): The velocity of the flow evaluated at z with shape (ndim,)
        """
        # Gradient of the action
        dsdz = self.grad_s(z, *args, **kwargs)
        # Velocity for the flow
        v    = jnp.conj(dsdz)
        # Rescale the velocity
        if self.rescale_velocity:
            i = self.integrand(z, *args, **kwargs)
            v = v * jnp.abs(i)
        return v

    def flow(self, x, t, *args, **kwargs):
        """
        Solve the Picard-Lefschetz flow for a given initial condition x

        Parameters:
            x (array): The real-valued initial condition with shape (ndim,)
            t (array/scalar): The time array for integration
            args: Additional arguments to pass to the action function
            kwargs: Additional keyword arguments

        Returns:
            z (array): The solution of the flow at each time step with shape (ndim, t.size)

        Notes:
            The method of integration can be set with the keyword `method` (default is 'euler')
        """
        # pop the method
        method = kwargs.pop('method', 'euler')
        uselast= kwargs.pop('uselast', False)

        # Initial condition and flow velocity
        z0 = jnp.array(x, dtype=self.complex_dtype)
        flow_vel = lambda z, t: self.flow_velocity(z, *args, **kwargs)

        # Time array
        if isinstance(t, (int, float)):
            t = jnp.linspace(0, t, 2)
            uselast = True

        # Integration
        if method == 'euler':
            z = euler(flow_vel, z0, t)
        elif method == 'odeint':
            z = odeint(flow_vel, z0, t)
        else:
            raise ValueError(f"Invalid method {method=}")

        if uselast:
            return z[-1]
        else:
            return z
    
    def flow_jacobian(self, x, t, *args, **kwargs):
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
            z = self.flow(x, t, *args, **kwargs)
            return jnp.real(z), jnp.imag(z)
        j = jacrev(flow_split)(x)
        j = j[0] + 1j * j[1]
        return jnp.linalg.det(j)

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
        from time import time
        def zji(x_sample):
            t0 = time()
            z = self.flow(x_sample, t, uselast=True, *args, **kwargs)
            t1 = time()
            j = self.flow_jacobian(x_sample, t, uselast=True, *args, **kwargs)
            t2 = time()
            i = self.integrand(z, *args, **kwargs)
            t3 = time()
            print(f"Flow: {t1-t0:.2f} sec, Jacobian: {t2-t1:.2f} sec, Integrand: {t3-t2:.2f} sec ")
            return z, j, i 
        # We should cut samples with very small probability to avoid numerical issues
        z, j, i = vmap(zji)(x_samples)
        Z = jnp.mean(i*j*jnp.exp(-lnp_samples), axis=0)
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

    def vflow(self, z, t, *args, **kwargs):
        """Vectorized flow function"""
        f = vmap(lambda zin: self.flow(zin, t, *args, **kwargs), in_axes=0, out_axes=0)
        return f(z)

    def vflow_jacobian(self, z, t, *args, **kwargs):
        """Vectorized flow Jacobian function"""
        f = vmap(lambda zin: self.flow_jacobian(zin, t, *args, **kwargs), in_axes=0, out_axes=0)
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
        z = self.vflow(x, t, *args, **kwargs)
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
        z = self.vflow(x, t, *args, **kwargs)
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
            line2.set_data(x, jnp.real(iz[:,i]))
            line3.set_data(x, jnp.imag(iz[:,i]))
            line4.set_data(x, jnp.abs(iz[:,i]))
            return line1, line2, line3, line4
        
        ani = FuncAnimation(fig, update, frames=t.size, interval=50, blit=True)
        ani.save(fname, writer="pillow")


class HMCSampler(object):
    def __init__(self, plmodel, priors):
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
        return [g for g in include if g not in exclude]

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
        # Genrate the parameters
        parameters = dict()
        for gname in self.get_group_names():
            ndim = self.get_group_ndim(gname)
            if ndim == 0:
                name = self.get_param_names(include=gname)[0]
                vmin, vmax = self.get_priors()[name]
                parameter = numpyro.sample(gname, npyro_dist.Uniform(vmin, vmax))
            else:
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
        j = self.plmodel.flow_jacobian(t=t, uselast=True, **parameters)
        parameters.pop('x')

        # Integrand
        parameters['z'] = z
        logp = -self.plmodel.action_s(**parameters).real

        # Set likelihood and derived values
        numpyro.factor('loglike', logp)

    def sample(self, num_samples=5000, num_warmup=500, seed=0, **like_kwargs):
        nuts_kernel = NUTS(self.likelihood_model)
        self.mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
        cpu_device = jax.devices('cpu')[0]
        with jax.default_device(cpu_device):
            self.mcmc.run(jax.random.PRNGKey(seed), **like_kwargs)

    # Getter functions
    def get_samples(self, format=jnp.array, names=None):
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
    def __init__(self, sampler):
        self.sampler = sampler

    def build(self, **config_in):
        # Default configuration
        config =   {'hidden_dims': [128, 128], \
                    'activation': "tanh", \
                    'n_transforms': 4, \
                    'use_random_permutations': False}
        # Overwrite the default configuration
        config.update(config_in)
        # Dimension is always set from prior
        config['n_dim']     = self.sampler.get_ndim(include='x')
        config['n_context'] = self.sampler.get_ndim(exclude='x')
        # Initialize the MAF model
        self.model = MaskedAutoregressiveFlow(**config)

    def train(self, **config_in):
        # Default configuration
        config  =  {'seed': 0, \
                    'learning_rate': 3e-4, \
                    'batch_size': 64, \
                    'n_steps': 1000}
        # Overwrite the default configuration
        config.update(config_in)
        # Get n_dim and n_context from model
        config['n_dim']     = self.model.n_dim
        config['n_context'] = self.model.n_context

        # Initialize the model
        key    = jax.random.PRNGKey(config['seed'])
        x_test = jax.random.uniform(key=key, shape=(config['batch_size'], config['n_dim']))
        context= jax.random.uniform(key=key, shape=(config['batch_size'], config['n_context']))
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
        names   = self.sampler.get_param_names()
        samples = self.sampler.get_samples(format=jnp.array, names=names)

        # Training loop!
        self.loss_history_maf = []
        key = jax.random.PRNGKey(config['seed'])
        for _ in tqdm(range(config['n_steps']), desc="Training MAF"):
            key, subkey = jax.random.split(key)
            indices = jax.random.choice(subkey, samples.shape[0], shape=(64,), replace=False)
            batch = (samples[indices, :config['n_dim']], samples[indices, config['n_dim']:])
            params, opt_state, loss = step(params, opt_state, batch)
            self.loss_history_maf.append(loss)

        self.trained_params = params

    def plot_loss(self):
        plt.figure(figsize=(5,3))
        plt.plot(self.loss_history_maf)
        plt.yscale('log')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

    def sample(self, context, num_samples=10000, seed=0):
        key = jax.random.PRNGKey(seed)
        if context.ndim == 0 or context.ndim == 1:
            context = jnp.tile(context, (num_samples, 1))

        def sampling_fn(m):
            x = m.sample(num_samples=num_samples, rng=key, context=context, beta=1.0)
            return x

        x = nn.apply(sampling_fn, self.model)(self.trained_params)
        return jnp.array(x)

    def log_prob(self, x, context):
        # Get the number of samples from x input
        num_samples = x.shape[0]
        # Format the context shape if necessary
        if context.ndim == 0 or context.ndim == 1:
            context = jnp.tile(context, (num_samples, 1))

        def logprob_fn(m):
            lnp = m(x, context, beta=1.0)
            return lnp

        log_prob = nn.apply(logprob_fn, self.model)(self.trained_params)
        return log_prob