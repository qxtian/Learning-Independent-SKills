# Networks of value / policy / decoder
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions import RelaxedOneHotCategorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete

class MLP(nn.Module):
    def __init__(self, layers, activation=torch.tanh, output_activation=None,
                 output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze
        
        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:

            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))
        return x.squeeze() if self.output_squeeze else x

class GaussianPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, output_activation, action_dim):
        super(GaussianPolicy, self).__init__()

        self.mu = MLP(layers=[input_dim] + list(hidden_dims) + [action_dim], activation=activation, output_activation=output_activation)
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim))

    def forward(self, x, a=None):
        policy = Normal(self.mu(x), self.log_std.exp())
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).mean(dim=1)
        if a is not None:
            logp = policy.log_prob(a).mean(dim=1)
        else:
            logp = None
        entropy = policy.entropy().mean(dim=1)

        return pi, logp, logp_pi, entropy

class CategoricalPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, output_activation, action_dim):
        super(CategoricalPolicy, self).__init__()

        self.logits = MLP(layers=[input_dim] + list(hidden_dims) + [action_dim], activation=activation)

    def forward(self, x, a=None):
        logits = self.logits(x)
        policy = Categorical(logits=logits)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()
        if a is not None:
            logp = policy.log_prob(a).squeeze()
        else:
            logp = None
        entropy = policy.entropy().squeeze()
        
        return pi, logp, logp_pi, entropy
    
class DiscPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, output_activation, con_dim):
        super(DiscPolicy, self).__init__()
        self.disc = MLP(layers=[input_dim] + list(hidden_dims) + [con_dim], activation=activation)

    def forward(self, seq, gt=None):
        logit_seq = self.disc(seq)
#        self.logits = torch.mean(logit_seq, dim=1)
        self.logits = logit_seq
        policy = Categorical(logits=self.logits)
        label = policy.sample()
        logp = policy.log_prob(label).squeeze()
        if gt is not None:
            loggt = policy.log_prob(gt).squeeze()
        else:
            loggt = None

        return label, loggt, logp
    
class DiscPolicyN(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, output_activation, con_dim):
        super(DiscPolicyN, self).__init__()
        self.disc = MLP(layers=[input_dim] + list(hidden_dims) + [con_dim], activation=activation)

    def forward(self, seq, gt=None):
        logit_seq = self.disc(seq)
        out = logit_seq.permute(0,2,1).squeeze()
##        self.logits = torch.mean(logit_seq, dim=1)
#        self.logits = logit_seq
#        policy = Categorical(logits=self.logits)
#        label = policy.sample()
#        logp = policy.log_prob(label).squeeze()
#        if gt is not None:
#            loggt = policy.log_prob(gt).squeeze()
#        else:
#            loggt = None
#        print(logit_seq.shape)
#        print(loggt.shape)
#        assert out.shape == loggt.shape
        return None, out, None
    

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_space, hidden_dims=(64, 64), activation=torch.tanh, output_activation=None, policy=None):
        super(ActorCritic, self).__init__()

        if policy is None:
            if isinstance(action_space, Box):
                self.policy = GaussianPolicy(input_dim, hidden_dims, activation, output_activation, action_space.shape[0])
                print('ActorCritc.policy = GaussianPolicy')
            elif isinstance(action_space, Discrete):
                self.policy = CategoricalPolicy(input_dim, hidden_dims, activation, output_activation, action_space.n)
                print('ActorCritc.policy = CategoricalPolicy')
        else:
            self.policy = policy(input_dim, hidden_dims, activation, output_activation, action_space)

        self.value_f = MLP(layers=[input_dim] + list(hidden_dims) + [1], activation=activation, output_squeeze=True)

    def forward(self, x, a=None):
        pi, logp, logp_pi, entropy = self.policy(x, a)
        v= self.value_f(x)

        return pi, logp, logp_pi, entropy, v

# Bidirectional LSTM for encoding trajectories
# Batch-first used
# input: (batch_size, seq_len, input_dim)
# inter_state: (batch_size, seq_len, 2*hidden_dims)
# linear_output: (batch_size, seq_len, context_dim)
# avg_logits: (batch_size, context_dim)
class Discriminator(nn.Module):
    def __init__(self, input_dim, context_dim, hidden_dims=(64,64), activation=torch.tanh, output_activation=None):
        super(Discriminator, self).__init__()

        self.policy = DiscPolicy(input_dim, hidden_dims, activation, output_activation, context_dim)

    def forward(self, seq, gt=None):
        pred, loggt, logp = self.policy(seq, gt)
        return pred, loggt, logp
    

class DiscriminatorN(nn.Module):
    def __init__(self, input_dim, context_dim, hidden_dims=(64,64), activation=torch.tanh, output_activation=None):
        super(DiscriminatorN, self).__init__()

        self.policy = DiscPolicyN(input_dim, hidden_dims, activation, output_activation, context_dim)

    def forward(self, seq, gt=None):
        pred, loggt, logp = self.policy(seq, gt)
        return pred, loggt, logp
    


def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
