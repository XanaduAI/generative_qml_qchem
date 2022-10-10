"""Classical neural network functionality for the Generative QQ project"""
import jax
import jax.numpy as jnp
import numpy as np
from .data import *

########### Feed-forward neural network ############

def prep_layer_params(m, n, key=None, zero=False):
    """
    Prepares layers for use in a feed-forward NN
    """
    if not zero:
        w_key, b_key = jax.random.split(key)
        w, b = jax.random.normal(w_key, (n, m)), jax.random.normal(b_key, (n,))
    else:
        w, b = jnp.zeros((n, m)), jnp.zeros((n,)) 

    return (w, b)

def network_params(sizes, key=None, zero=False):
    """
    Returns the parameters corresponding to a network
    """
    if key is not None:
        keys = jax.random.split(key, len(sizes))
    return [prep_layer_params(m, n, key=k, zero=zero) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def relu(x):
    """
    ReLU activation function
    """
    return jnp.maximum(0, x)


def shift_sig(x):
    """
    Shifted sigmoid
    """
    return jax.nn.sigmoid(x) - 0.5


def neural_network(NN, x, hidden_activation=jax.nn.sigmoid, out_activation=shift_sig):
    """
    Executes a feed-forward neural network for a given set of parameters NN and input x
    """
    vec = x # Initial vector passed into network
    # Runs through the neural network
    for W, b in NN[:-1]:
        vec = hidden_activation(jnp.dot(W, vec) + b) # Updates the vector
    
    W, b = NN[-1] # Last params
    final_vec = jnp.dot(W, vec) + b
    return 2 * jnp.pi * out_activation(final_vec) # Returns some set of angles


############## Quantum-classical model ###################

def exact_fidelity_loss_sample(model):
    """
    We assume a uniform distribution over training data
    Make sure that model() is differentiable with respect to JAX
    Assumes that the "model" function outputs the exact state vector, and that we have access to exact state vectors of training data
    """ 
    def loss_fn(theta, g, s):
        val = jnp.inner(jnp.conj(s), model(g, theta))
        return 1 - val * jnp.conj(val)
    return loss_fn

def batch_exact_fidelity_loss(model):
    """
    vmap of exact fidelity loss
    """
    return jax.vmap(exact_fidelity_loss_sample(model), in_axes=(None, 0, 0))

def exact_fidelity_loss(model, x, y):
    """
    Exact fidelity loss function
    """
    def fn(NN):
        samples = len(x)
        return jnp.real((1/samples) * jnp.sum(batch_exact_fidelity_loss(model)(NN, x, y)))
    return fn

def exact_log_fidelity_loss(model, x, y):
    """
    Exact log-fidelity loss function
    """
    def fn(NN):
        samples = len(x)
        return jnp.real(jnp.log((1 / samples) * jnp.sum(batch_exact_fidelity_loss(model)(NN, x, y))))
    return fn

################ Post-processing ###################

def model_energy(opt_model, H, weight):
    """
    Computes the energy of the states outputted by the model, for a particular parmaeter
    """
    def fn(x):
        H_R = H(x)

        model_val = opt_model(x)
        m =  np.real(np.dot(np.conj(model_val), H_R @ model_val))
    
        vec, val = allowed_vec_val_sparse(H_R, weight, 0)
        m2 = val
        v = vec

        return m, m2, model_val, v
    return fn