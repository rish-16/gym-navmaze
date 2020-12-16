import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt

class Actor(snt.Module):
    def __init__(self, dim_in, dim_out, ls_min=-10, ls_max=0):
        super().__init__()
        self.ls_max = ls_max
        self.ls_min = ls_min
        w_init = 1e-3
        
        self.l1 = snt.Linear(256)
        
        init = snt.initializers.RandomUniform(-w_init, w_init)
        self.mu_layer = snt.Linear(dim_out, initializer=init)
        self.log_std_layer = snt.Linear(dim_out, initializer=init)
        
    def __call__(self, state):
        x = tf.nn.relu(self.l1(state))
        
        mu = tf.nn.tanh(self.mu_layer(x))
        
        log_std = tf.nn.tanh(self.log_std_layer(x))
        log_std = 0.5 * self.ls_min + (self.ls_max - self.ls_min) * (log_std + 1)
                
        std = tf.exp(log_std)
        
        distrib = tfp.distributions.Normal(mu, std)
        action = dist.sample()
        
        return action, distrib    
                
class Critic(snt.Module):
    def __init__(self):
        super().__init__()
        self.l1 = snt.Linear(64)
        w_init = 1e-3
        init = snt.initializers.RandomUniform(-w_init, w_init)
        self.out = snt.Linear(1, initializer=init)
        
    def __call__(self, state):
        x = tf.nn.relu(self.l1(state))
        val = self.out(x)
        
        return val