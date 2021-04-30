import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

"""
Credits to https://github.com/4kasha/CartPole_PPO
"""


def layer_init(layer, scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class FCNet(nn.Module):
    """Fully Connected Model."""
    def __init__(self, state_size, seed, hidden_layers, use_reset, act_fnc=F.relu):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            seed (int): Random seed
            hidden_layers (list): Size of hidden_layers
            act_fnc : Activation function
            use_reset (bool): Weights initialization
        """
        super(FCNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        dims = [state_size, ] + hidden_layers
        if use_reset:
            self.layers = nn.ModuleList([layer_init(nn.Linear(in_put, out_put)) for in_put, out_put in zip(dims[:-1], dims[1:])])
        else:
            self.layers = nn.ModuleList([nn.Linear(in_put, out_put) for in_put, out_put in zip(dims[:-1], dims[1:])])
        self.act_fuc = act_fnc 
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.act_fuc(layer(x))
        return x

class Policy(nn.Module):
    """Actor Critic Model (shared weights)"""
    def __init__(self, state_size, action_size, seed, main_net):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            main_net (model): Common net for actor and critic 
        """
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.main_net = main_net
        
        self.fc_actor = layer_init(nn.Linear(main_net.feature_dim, action_size), 1e-3)
        self.fc_critic = layer_init(nn.Linear(main_net.feature_dim, 1), 1e-3)
    
    def forward(self, state):
        x = self.main_net(state)
        pi_a = self.fc_actor(x)
        prob = F.softmax(pi_a, dim=1)
        v = self.fc_critic(x)
        return prob, v        
        
    def act(self, state, action=None):
        prob, v = self.forward(state)
        dist = Categorical(prob)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'v': v.squeeze()}