# jax
import jax
import jax.numpy as jnp
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
# Normalizing Flows
import flax.linen as nn
from flax.training.early_stopping import EarlyStopping
import optax
from optax import contrib
from sklearn import datasets, preprocessing
from .maf import MaskedAutoregressiveFlow
from .nsf import NeuralSplineFlow
# Sobol sampler
from scipy.stats import qmc
# Plot
import matplotlib.pyplot as plt
import numpy as np

class SobolSampler:
    def __init__(self, ranges, names=None):
        """
        ranges: dict
            パラメータ名をキー、(min, max)の範囲を値として指定
        names: list or None
            使用するパラメータ名のリスト。Noneの場合、rangesの全パラメータを使用
        """
        self.ranges = ranges
        self.names = names if names is not None else list(ranges.keys())
        self.ndim = len(self.names)
        self.sampler = qmc.Sobol(d=self.ndim, scramble=True)
        self.total_samples = 0  # 今までに生成したサンプル数

    def generate(self, n_samples):
        """
        Sobol列を生成して範囲をスケールします。
        n_samples: int
            生成するサンプル数
        """
        # Sobol列を生成
        samples = self.sampler.random(n=n_samples)

        # JAXでスケール変換
        def scale_sample(sample):
            return jnp.array([
                sample[i] * (self.ranges[name][1] - self.ranges[name][0]) + self.ranges[name][0]
                for i, name in enumerate(self.names)
            ])
        
        scaled_samples = jax.vmap(scale_sample)(samples)
        self.total_samples += n_samples
        return scaled_samples

class ConditionallikelihoodSampler:
    def __init__(self, likelihood, xranges, pranges=None):
        self.likelihood = likelihood
        self.set_xranges(xranges)
        self.set_pranges(pranges or {})
        self.initialize_sobol_sampler()
        self.initialize_samples_placeholder()

    def set_xranges(self, xranges):
        """
        This function defines the range of random variables.

        Parameters:
            xranges (dict): A dictionary of the form {name: (min, max)}.

        Notes:
            The format of xranges is as follows:
            xranges={
                # For n-dimensional random variables
                'x': (min, max, ndim),
                # If you want to overwrite the range of a dimension 
                # of an existing random variable,
                'x1': (min, max),
                'x2': (min, max),
                }
        """
        # Placeholder for the ranges
        self.xranges = dict()
        self.xgnames = dict()
        # 1. We popurate the ranges from the tuple with 3 entries
        for key, value in xranges.items():
            if not isinstance(value, tuple): continue
            if not len(value) == 3: continue
            self.xgnames[key] = {'is_vector':True, 'names':list()}
            for i in range(value[2]):
                self.xranges[f"{key}{i}"] = (value[0], value[1])
                self.xgnames[key]['names'].append(f"{key}{i}")
        # 2. We populate the ranges from the tuple with 2 entries
        for key, value in xranges.items():
            if not isinstance(value, tuple): continue
            if not len(value) == 2: continue
            self.xranges[key] = value
            if key not in self.xgnames:
                self.xgnames[key] = {'is_vector':False, 'names':[key]}            
        # 3. Define the random variable names
        self.xnames = list(self.xranges.keys())
        # 4. Define the number of random variables
        self.xndim  = len(self.xnames)
        
    def set_pranges(self, pranges):
        """
        This function defines the range of parameters.

        Parameters:
            pranges (dict): A dictionary of the form {name: (min, max)}.

        Notes:
            The format of pranges is as follows:
            pranges={
                # For n-dimensional random variables
                'p': (min, max, ndim),
                # If you want to overwrite the range of a dimension 
                # of an existing random variable,
                'p1': (min, max),
                'p2': (min, max),
                }
        """
        # 1. We popurate the ranges from the tuple with 3 entries
        self.pranges = dict()
        self.pgnames = dict()
        for key, value in pranges.items():
            if not isinstance(value, tuple): continue
            if not len(value) == 3: continue
            self.pgnames[key] = {'is_vector':True, 'names':list()}
            for i in range(value[2]):
                self.pranges[f"{key}{i}"] = (value[0], value[1])
                self.pgnames[key]['names'].append(f"{key}{i}")
        # 2. We populate the ranges from the tuple with 2 entries
        for key, value in pranges.items():
            if not isinstance(value, tuple): continue
            if not len(value) == 2: continue
            self.pranges[key] = value
            if key not in self.pgnames:
                self.pgnames[key] = {'is_vector':False, 'names':[key]}
        # 3. Define the parameter names
        self.pnames = list(self.pranges.keys())
        # 4. Define the number of parameters
        self.pndim  = len(self.pnames)

    def initialize_sobol_sampler(self):
        self.sobol = SobolSampler(self.pranges)

    def initialize_samples_placeholder(self):
        self.xsamples = jnp.zeros((0, self.xndim))
        self.psamples = jnp.zeros((0, self.pndim))

    def add_xpsamples(self, xsamples, psamples):
        self.xsamples = jnp.vstack([self.xsamples, xsamples])
        self.psamples = jnp.vstack([self.psamples, psamples])
        self.nsamples = self.xsamples.shape[0]

    def group_samples(self, gnames, names, samples):
        """
        This function groups samples into a dictionary.

        Parameters:
            gnames (dict): A dictionary of the grouped names.
            names (list): A list of names.
            samples (list): A list of samples.
        
        Returns:
            dict: A dictionary of the grouped samples.
        """
        gsamples = dict()
        for gname, info in gnames.items():
            is_vector = info['is_vector']
            if is_vector:
                _ = [samples[names.index(n)] for n in info['names']]
                gsamples[gname] = jnp.array(_)
            else:
                gsamples[gname] = samples[names.index(info['names'][0])]
        return gsamples

    def numpyro_likelihood_model(self, gpsample, **kwargs):
        # x variable
        gxsample = []
        for name in self.xnames:
            x = numpyro.sample(name, npyro_dist.Uniform(*self.xranges[name]))
            gxsample.append(x)
        gxsample = self.group_samples(self.xgnames, self.xnames, gxsample)

        # likelihood inputs
        likelihood_inputs = {}
        likelihood_inputs.update(gpsample)
        likelihood_inputs.update(gxsample)
        likelihood_inputs.update(kwargs)

        # likelihood
        lnlike = self.likelihood(**likelihood_inputs)

        numpyro.factor('loglike', lnlike)

    def sample_xsamples(self, num_warmup, num_samples, num_chains, 
            rng_key=None, seed=0, progress_bar=False, **like_kwargs):
        nuts_kernel = NUTS(self.numpyro_likelihood_model)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, 
            num_samples=num_samples, num_chains=num_chains, 
            progress_bar=progress_bar)
        mcmc.run(rng_key or jax.random.PRNGKey(seed), **like_kwargs)
        return mcmc

    def sample_xpsamples(self, num_psamples, num_warmup, num_xsamples, num_chains, 
            seed=0, xprogress_bar=False, pprogress_bar=True, thin=1, **like_kwargs):
        loop = range(num_psamples) if not pprogress_bar else tqdm(range(num_psamples))
        for _ in loop:
            psample = self.sobol.generate(1)[0]
            gpsample = self.group_samples(self.pgnames, self.pnames, psample)
            mcmc = self.sample_xsamples(num_warmup, num_xsamples, num_chains, 
                seed=seed, progress_bar=xprogress_bar, gpsample=gpsample, **like_kwargs)
            # Get new xpsamples
            xsamples = mcmc.get_samples()
            xsamples = jnp.transpose(jnp.array([xsamples[name] for name in self.xnames]))
            psamples = jnp.array([psample]*num_chains*num_xsamples)
            self.add_xpsamples(xsamples[::thin], psamples[::thin])


class NDESampler:
    def __init__(self, xsamples, csamples, normalize=True, decorrelate=False):
        self.xsamples   = xsamples
        self.csamples   = csamples
        self.nsamples   = xsamples.shape[0]
        self.xndim      = xsamples.shape[1]
        self.cndim      = csamples.shape[1]
        self.normalize  = normalize
        self.decorrelate= decorrelate
        self.compute_processing_params()

    def buildMAF(self, **config_in):
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
                    'n_dim': self.xndim, \
                    'n_context': self.cndim}
        # Overwrite the default configuration
        config.update(config_in)
        # Initialize the MAF model
        self.model = MaskedAutoregressiveFlow(**config)

    def buildNSF(self, **config_in):
        """
        Build the NSF model
        """
        # Default configuration
        config =   {'hidden_dims': [128, 128], \
                    'activation': "tanh", \
                    'n_transforms': 4, \
                    'n_dim': self.xndim, \
                    'n_context': self.cndim, \
                    'n_bins': 16, \
                    'range_min': self.xsamples.min(), \
                    'range_max': self.xsamples.max()}
        # Overwrite the default configuration
        config.update(config_in)
        # Initialize the MAF model
        self.model = NeuralSplineFlow(**config)

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
                    'batch_size': 64, \
                    'n_steps': 1000,  
                    'lr': 1e-3, 
                    # Early stopping
                    'patience': 20,
                    'min_delta': 1e-2}
        # Overwrite the default configuration
        config.update(config_in)

        # Initialize the model
        key    = jax.random.PRNGKey(config['seed'])
        x_test = jax.random.uniform(key=key, shape=(config['batch_size'], self.xndim))
        c_test = jax.random.uniform(key=key, shape=(config['batch_size'], self.cndim))
        params = self.model.init(key, x_test, c_test)

        # Optimizer
        optimizer = optax.adam(learning_rate=config['lr'])
        opt_state = optimizer.init(params)

        # Loss function
        @jax.jit
        def loss_fn(params, xsamples, csamples):
            return -jnp.mean(self.model.apply(params, xsamples, csamples))
        
        # One step in the training loop
        @jax.jit
        def step(params, opt_state, x_batch, c_batch):
            loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, c_batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        # Training loop!
        early_stop = EarlyStopping(min_delta=config['min_delta'], patience=config['patience'])
        self.loss_history = []
        key = jax.random.PRNGKey(config['seed'])
        for step_idx in tqdm(range(config['n_steps']), desc="Training MAF"):
            key, subkey = jax.random.split(key)
            indices = jax.random.choice(
                subkey, self.nsamples, \
                shape=(config['batch_size'],), replace=False
            )
            params, opt_state, loss = step(params, opt_state, \
                        self.xsamples[indices,:], self.csamples[indices,:])
            # Save the loss
            self.loss_history.append(loss)
            # Early stopping
            early_stop.update(loss)
            if early_stop.should_stop:
                print(f'Met early stopping criteria, breaking at epoch {epoch}')
                break

        self.trained_params = params
        self.opt_state = opt_state

    def plot_loss(self):
        """Plot the training loss"""
        plt.figure(figsize=(5,3))
        plt.plot(self.loss_history)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')

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
    def sample_and_log_prob(self, context, num_samples=1_000, seed=0, beta=1.0):
        """
        Sample from the trained MAF model and evaluate the log-probability

        Parameters:
            context (array): The context for the sampling with shape (n_context,)
            num_samples (int): The number of samples to generate
            seed (int): The seed for random number generation
            beta (float): The temperature for the sampling

        Returns:
            array: The samples from the MAF model with shape (num_samples, n_dim)
            array: The log-probability of the samples with shape (num_samples,)
        """
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
        self.processing_params = {}

        xsamples = self.xsamples
        csamples = self.csamples

        # Normalization
        if self.normalize:
            # Standardize the samples
            xmean = jnp.mean(self.xsamples, axis=0)
            xstd  = jnp.std(self.xsamples, axis=0)
            xsamples = (self.xsamples - xmean) / xstd
            cmean = jnp.mean(self.csamples, axis=0)
            cstd  = jnp.std(self.csamples, axis=0)
            csamples = (self.csamples - cmean) / cstd
            self.processing_params['xmean'] = xmean
            self.processing_params['xstd']  = xstd
            self.processing_params['cmean'] = cmean
            self.processing_params['cstd']  = cstd

        # Linear decorrelation
        if self.decorrelate:
            # Get the covariance matrix
            cov = jnp.cov(xsamples.T, csamples.T)
            cov_cc = cov[self.xndim:, self.xndim:]
            cov_cx = cov[self.xndim:, :self.xndim]
            # Get the linear transformation coefficients
            dec_mat = jnp.linalg.solve(cov_cc, cov_cx).T
            self.processing_params['dec_mat'] = dec_mat

    def preprocess_context(self, csamples):
        """
        Linear preprocess the context before inputting to the MAF model.
        This function applies the standardization of the context samples.

        Parameters:
            csamples (array): The original samples of context with shape 
                               (num_samples, n_context)

        Returns:
            csamples (array): The standardized samples of context with shape 
                               (num_samples, n_context)
        """
        csamples = jnp.array(csamples)

        if self.normalize:
            cmean = self.processing_params['cmean']
            cstd  = self.processing_params['cstd']
            csamples = (csamples - cmean) / cstd

        return csamples

    def postprocess_context(self, csamples):
        """
        Linear postprocess the context after training the MAF model.
        This function applies the inverse transformation of the linear
        preprocess function.

        Parameters:
            csamples (array): The standardized samples of context with shape
                               (num_samples, n_context)

        Returns:
            csamples (array): The original samples of context with shape
                               (num_samples, n_context)
        """
        csamples = jnp.array(csamples)

        if self.normalize:
            cmean = self.processing_params['cmean']
            cstd  = self.processing_params['cstd']
            csamples = csamples * cstd + cmean

        return csamples

    def preprocess_x(self, xsamples, csamples):
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
            xsamples (array): The samples of x with shape (num_samples, ndim)
            csamples (array): The samples of context with shape (num_samples, n_context)
        
        Returns:
            xsamples (array): The linearly preprocessed samples with shape (num_samples, ndim)
        """
        xsamples = jnp.array(xsamples)
        csamples = jnp.array(csamples)

        if self.normalize:
            cmean = self.processing_params['cmean']
            cstd  = self.processing_params['cstd']
            xmean = self.processing_params['xmean']
            xstd  = self.processing_params['xstd']
            xsamples = (xsamples - xmean) / xstd
            csamples = (csamples - cmean) / cstd

        if self.decorrelate:
            dec_mat = self.processing_params['dec_mat']
            xsamples = xsamples - csamples @ dec_mat.T
        
        return xsamples

    def postprocess_x(self, xsamples, csamples):
        """
        Linear postprocess the samples after training the MAF model.
        This function applies the inverse transformation of the linear
        preprocess function.

        Parameters:
            xsamples (array): The standardized samples of x with shape 
                               (num_samples, ndim)
            csamples (array): The standardized samples of context with shape 
                               (num_samples, n_context)

        Returns:
            xsamples (array): The postprocessed samples of x with shape 
                               (num_samples, ndim)
        """
        xsamples = jnp.array(xsamples)
        csamples = jnp.array(csamples)

        if self.decorrelate:
            dec_mat = self.processing_params['dec_mat']
            xsamples = xsamples + csamples @ dec_mat.T

        if self.normalize:
            xmean = self.processing_params['xmean']
            xstd  = self.processing_params['xstd']
            xsamples = xsamples * xstd + xmean

        return xsamples

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
        logp = jnp.array(logp)

        if self.normalize:
            xstd = self.processing_params['xstd']
            logp = logp - jnp.sum(jnp.log(xstd))

        return logp