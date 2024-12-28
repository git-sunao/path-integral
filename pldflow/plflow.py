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
from jax.experimental.ode import odeint
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
# from nsf import NeuralSplineFlow

class PicardLefschetzModelBaseClass(object):
    """
    Z = \int dx e^{-S(x)}
    """
    def __init__(self):
        self.rescale_velocity = True
        self.complex_dtype = jnp.complex64

    # Functions
    def action_s(self, z, *args, **kwargs):
        raise NotImplementedError("action_s must be implemented in a subclass")

    def grad_s(self, z, *args, **kwargs):
        g = grad(self.action_s, argnums=0, holomorphic=True)
        return g(z, *args, **kwargs)

    def hessian_s(self, z, *args, **kwargs):
        h = hessian(self.action_s, argnums=0, holomorphic=True)
        return h(z, *args, **kwargs)

    def jacobian_s(self, z, *args, **kwargs):
        j = jacobian(self.action_s, argnums=0, holomorphic=True)
        return j(z, *args, **kwargs)

    def integrand(self, z, *args, **kwargs):
        s = self.action_s(z, *args, **kwargs)
        return jnp.exp(-s)

    # PL flow
    def flow_velocity(self, z, *args, **kwargs):
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
        z0 = jnp.array(x, dtype=self.complex_dtype)
        flow_vel = lambda z, t: self.flow_velocity(z, *args, **kwargs)
        if isinstance(t, (int, float)):
            t = jnp.linspace(0, t, 2)
            z = odeint(flow_vel, z0, t)
            return z[-1]
        else:
            z = odeint(flow_vel, z0, t)
            return z
    
    def flow_jacobian(self, x, t, *args, **kwargs):
        def flow_split(x):
            z = self.flow(x, t, *args, **kwargs)
            return jnp.real(z), jnp.imag(z)
        j = jacobian(flow_split, argnums=0)(x)
        j = j[0] + 1j * j[1]
        if j.ndim == 0:
            return j
        elif j.shape[0] == j.shape[1]:
            return jnp.linalg.det(j)
        else:
            raise ValueError(f"Unexpected shape for Jacobian: {j.shape}")

    # Integration with samples
    def integrate(self, x_samples, lnp_samples, t, *args, **kwargs):
        # We should cut samples with very small probability to avoid numerical issues
        from time import time
        t0 = time()
        z = self.vflow(x_samples, t, *args, **kwargs)
        t1 = time()
        j = self.vflow_jacobian(x_samples, t, *args, **kwargs)
        t2 = time()
        i = self.vintegrand(x_samples, *args, **kwargs)
        t3 = time()
        Z = jnp.mean(i*j*jnp.exp(-lnp_samples), axis=0)
        t4 = time()
        print(f"Flow: {t1-t0:.2f} s, Jacobian: {t2-t1:.2f} s, Integrand: {t3-t2:.2f} s, Mean: {t4-t3:.2f} s")
        return Z

    # Vectorized functions
    def vaction_s(self, z, *args, **kwargs):
        s = vmap(lambda zin: self.action_s(zin, *args, **kwargs), in_axes=0, out_axes=0)
        return s(z)

    def vgrad_s(self, z, *args, **kwargs):
        g = vmap(lambda zin: self.grad_s(zin, *args, **kwargs), in_axes=0, out_axes=0)
        return g(z)

    def vhessian_s(self, z, *args, **kwargs):
        h = vmap(lambda zin: self.hessian_s(zin, *args, **kwargs), in_axes=0, out_axes=0)
        return h(z)

    def vjacobian_s(self, z, *args, **kwargs):
        j = vmap(lambda zin: self.jacobian_s(zin, *args, **kwargs), in_axes=0, out_axes=0)
        return j(z)

    def vintegrand(self, z, *args, **kwargs):
        i = vmap(lambda zin: self.integrand(zin, *args, **kwargs), in_axes=0, out_axes=0)
        return i(z)

    def vflow(self, z, t, *args, **kwargs):
        f = vmap(lambda zin: self.flow(zin, t, *args, **kwargs), in_axes=0, out_axes=0)
        return f(z)

    def vflow_jacobian(self, z, t, *args, **kwargs):
        f = vmap(lambda zin: self.flow_jacobian(zin, t, *args, **kwargs), in_axes=0, out_axes=0)
        return f(z)

    # Plotter
    def _plot1d_template(self, fig=None, axes=None):
        # if fig is None and axes is None:
        #     fig = plt.figure(figsize=(5,6))
        #     ax1 = fig.add_axes((.1,.3,.8,.6))
        #     ax2 = fig.add_axes((.1,.05,.8,.15), sharex=ax1)
        # else:
        #     ax1, ax2 = axes
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
        if isinstance(t, (int, float)):
            t = jnp.linspace(0, t, 2)
        z = self.vflow(x, t, *args, **kwargs)
        iz= self.vintegrand(z, *args, **kwargs)
        ix= self.integrand(x, *args, **kwargs)

        color = []
        for i in range(t.size):
            color.append(plt.cm.bwr(i/(t.size-1)))

        fig, (ax1, ax2) = self._plot1d_template()
        # Plot the path
        for i in range(t.size):
            ax1.plot(z[:,i].real, z[:,i].imag, marker='.', color=color[i])
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
        if isinstance(t, (int, float)):
            t = jnp.linspace(0, t, 2)
        z = self.vflow(x, t, *args, **kwargs)
        iz= self.vintegrand(z, *args, **kwargs)
        ix= self.integrand(x, *args, **kwargs)

        fig, (ax1, ax2) = self._plot1d_template()
        ax1.set_xlim(x.min(),x.max())
        ax1.set_ylim(x.min(),x.max())
        ax2.set_ylim(-1.1, 1.1)
        line1, = ax1.plot([], [], marker='.', color='blue', lw=2)
        line2, = ax2.plot([], [], ls='--', color='blue')
        line3, = ax2.plot([], [], ls='-.', color='blue')
        line4, = ax2.plot([], [], ls='-', color='blue')

        def update(i):
            line1.set_data(z[:, i].real, z[:, i].imag)
            line2.set_data(x, jnp.real(iz[:, i]))
            line3.set_data(x, jnp.imag(iz[:, i]))
            line4.set_data(x, jnp.abs(iz[:, i]))
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
        
        Parameters
        ----------
        priors : dict
            Dictionary with the priors for the parameters of the action.
            The keys are the parameter names and the values are the priors.
            The priors can be either a tuple with the range (vmin, vmax) or
            a tuple with the range and the number of dimensions (vmin, vmax, ndim).
        """
        self.priors = dict()
        assert 'x' in priors, "Prior on `x` is required"
        for k, v in priors.items():
            # Get the range of the prior
            if len(v) == 2:
                vmin = v[0]
                vmax = v[1]
                ndim = 0
            elif len(v) == 3:
                vmin = v[0]
                vmax = v[1]
                ndim = v[2]
            else:
                raise ValueError(f"Invalid prior format {v=}")
            assert isinstance(vmin, (int, float)), f"Invalid vmin {vmin=}"
            assert isinstance(vmax, (int, float)), f"Invalid vmax {vmax=}"
            assert isinstance(ndim, int), f"Invalid ndim {ndim=}"
            self.priors[k] = (vmin, vmax, ndim)

    def get_param_names(self, separate=False):
        priors = self.priors.copy()
        # parameter name of x
        vmin, vmax, ndim = priors.pop('x')
        names_x = []
        if ndim == 0:
            names_x.append('x')
        else:
            for i in range(ndim):
                names_x.append(f'x{i+1}')
        # parameter name of other parameters
        names_o = []
        for k, v in priors.items():
            vmin, vmax, ndim = v
            if ndim == 0:
                names_o.append(k)
            else:
                for i in range(ndim):
                    names_o.append(f"{k}{i+1}")
        if separate:
            return names_x, names_o
        else:
            return names_x + names_o

    def get_n_dim(self):
        names_x, names_o = self.get_param_names(separate=True)
        return len(names_x)

    def get_n_context(self):
        names_x, names_o = self.get_param_names(separate=True)
        return len(names_o)

    def numpyro_model(self, t=1.0):
        # Genrate the parameters 
        parameters = dict()
        for k, v in self.priors.items():
            # Set the prior
            vmin, vmax, ndim = v
            if ndim == 0:
                p = numpyro.sample(k, npyro_dist.Uniform(vmin, vmax))
            else:
                p = []
                for i in range(ndim):
                    pi = numpyro.sample(f"{k}{i+1}", npyro_dist.Uniform(vmin, vmax))
                    p.append(pi)
                p = jnp.array(p)
            parameters[k] = p
            
        # flow
        z = self.plmodel.flow(t=t, **parameters)
        j = self.plmodel.flow_jacobian(t=t, **parameters)
        parameters.pop('x')

        # Integrand
        parameters['z'] = z
        i = self.plmodel.integrand(**parameters)

        # Target probability distribution
        amplt = jnp.abs(i)
        # phase = i*j/amplt

        # Set likelihood and derived values
        numpyro.factor('loglike', jnp.log(amplt))
        # numpyro.deterministic('phase', phase)
        # numpyro.deterministic('amplt', amplt)

    def sample(self, num_samples=10000, num_warmup=500, seed=0):
        nuts_kernel = NUTS(self.numpyro_model)
        self.mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
        cpu_device = jax.devices('cpu')[0]
        with jax.default_device(cpu_device):
            self.mcmc.run(jax.random.PRNGKey(seed))

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
            ranges  = dict()
            for k, v in self.priors.items():
                vmin, vmax, ndim = v
                if ndim == 0:
                    ranges[k] = [vmin, vmax]
                else:
                    for i in range(ndim):
                        ranges[f"{k}{i+1}"] = [vmin, vmax]
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
        config['n_dim']     = self.sampler.get_n_dim()
        config['n_context'] = self.sampler.get_n_context()
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
        names   = self.sampler.get_param_names(separate=False)
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