import time
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '../'))
# from autoregressive import *  # Load from your provided files
# from bijectors import *  # Load from your provided files
from pldflow.maf import MaskedAutoregressiveFlow
from pldflow.nsf import NeuralSplineFlow
import optax

# Define target distribution
def target_distribution(x, p):
    return jnp.exp(-0.5 * jnp.dot(x - p, x - p))

# Function to train the flow model
def train_flow(flow_model, optimizer, num_steps=5000, lr=1e-3, batch_size=128, patience=200):
    params = flow_model.init(jax.random.PRNGKey(0), jnp.ones((batch_size, flow_model.n_dim)), jnp.ones((batch_size, flow_model.n_context)))
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, x_batch, p_batch):
        log_prob = flow_model.apply(params, x_batch, p_batch)
        return -jnp.mean(log_prob)

    @jax.jit
    def update(params, opt_state, x_batch, p_batch):
        # grads = jax.grad(loss_fn)(params, x_batch, p_batch)
        loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, p_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Generate training data
    rng = np.random.default_rng()
    p_train = rng.uniform(-4.0, 4.0, size=(10000, flow_model.n_context))
    x_train = rng.normal(loc=0.0, scale=1.0, size=(10000, flow_model.n_dim)) + p_train

    # Setting for training
    best_loss = float("inf")
    patience_counter = 0

    for step in tqdm(range(num_steps)):
        batch_indices = rng.choice(x_train.shape[0], batch_size, replace=False)
        x_batch = x_train[batch_indices]
        p_batch = p_train[batch_indices]
        params, opt_state, loss = update(params, opt_state, x_batch, p_batch)

        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter > patience:
            print(f"Early stopping at step {step}")
            break

    return params


# Experiment parameters
dimensions = [1, 5, 10, 20]
# dimensions = [1]
y_values = [-4.0, -2.0, 0.0, 2.0, 4.0]
samples_per_model = 5000

# Store results
results = {}

# configs
config_maf = {'n_dim':1, 'n_context':1, 'activation':'tanh', 'hidden_dims':[128, 128, 128]}
config_nsf = {'n_dim':1, 'n_context':1, 'activation':'gelu', 'range_min':-7.0, 'range_max':7.0, 'hidden_dims':[128, 128, 128]}

# Main experiment loop
for n in dimensions:
    config_maf['n_dim'] = n
    config_maf['n_context'] = n
    config_nsf['n_dim'] = n
    config_nsf['n_context'] = n

    for flow_name, FlowClass, config in zip(["MAF", "NSF"], [MaskedAutoregressiveFlow, NeuralSplineFlow], [config_maf, config_nsf]):
        print(f'Running {flow_name} (n={n})')
        # Initialize flow
        flow_model = FlowClass(**config)
        optimizer = optax.adam(learning_rate=1e-3)

        # Train the model
        start_time = time.time()
        trained_params = train_flow(flow_model, optimizer)
        training_time = time.time() - start_time

        @partial(jax.jit, static_argnums=(3,))
        def sample_flow(trained_params, key, condition, samples_per_model):
            return flow_model.apply(
                trained_params,
                rng=key,
                context=condition,
                num_samples=samples_per_model,
                method='sample'
            )

        # Generate samples
        samples = []
        sampling_times = []
        for y0 in y_values:
            condition = jnp.array([y0] + [0] * (n - 1))
            condition = jnp.tile(condition, samples_per_model).reshape(samples_per_model, n)
            key = jax.random.PRNGKey(0)
            start_time = time.time()
            # s = sample_flow(trained_params, key, condition, samples_per_model)
            s = flow_model.apply(trained_params, rng=key, context=condition, num_samples=samples_per_model, method='sample')
            sampling_times.append(time.time() - start_time)
            samples.append(s)

        results[(flow_name, n)] = {
            "training_time": training_time,
            "samples": samples,
            "sampling_times": sampling_times,
        }

# Plot results
for n in dimensions:
    for flow_name in ["MAF", "NSF"]:
        plt.figure(figsize=(12, 8))
        for i, y0 in enumerate(y_values):
            plt.hist(
                results[(flow_name, n)]["samples"][i][:, 0], bins=50, density=True, alpha=0.6, color=f"C{i}"
            )
            plt.axvline(y0, color=f"C{i}", linestyle="--")
            x = jnp.linspace(y0-3, y0+3, 1000)
            plt.plot(x, jnp.exp(-0.5*(x - y0)**2)/jnp.sqrt(2*jnp.pi), color=f"C{i}", linestyle="--")
        plt.savefig(os.path.join(here, f"compare_{flow_name}_n{n:02d}.png"))

# Print speed comparison
with open(os.path.join(here, "compare_speed.txt"), "w") as f:
    for n in dimensions:
        for flow_name in ["MAF", "NSF"]:
            training_time  = results[(flow_name, n)]["training_time"]
            sampling_time1 = results[(flow_name, n)]["sampling_times"][0]
            sampling_times = results[(flow_name, n)]["sampling_times"][1:]
            sampling_time  = f'{np.mean(sampling_times):.2f} +- {np.std(sampling_times):.2f} ({sampling_time1:.2f})'
            f.write(f"{flow_name} (n={n:02d}) Training Time: {training_time:.2f}s Sampling Time: {sampling_time}\n")
