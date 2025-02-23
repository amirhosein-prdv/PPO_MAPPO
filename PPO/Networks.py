import os
import numpy as np

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from utils import get_unique_log_dir


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNetwork(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_lr,
        fc_dims=[256, 256],
        chkpt_dir="./tmp/PPO-Agent",
    ):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, "actor_torch_ppo")

        layers = []
        in_features = state_dim
        for out_features in fc_dims:
            layers.append(layer_init(nn.Linear(in_features, out_features)))
            layers.append(nn.Tanh())
            in_features = out_features
        layers.append(layer_init(nn.Linear(in_features, action_dim), std=0.01))

        self.mean = nn.Sequential(*layers)
        self.logstd = nn.Parameter(T.zeros(1, action_dim))

        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr, eps=1e-5)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        action_mean = self.mean(state)
        action_logstd = self.logstd.expand_as(action_mean).exp()
        dist = Normal(action_mean, action_logstd)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(
        self,
        state_dim,
        critic_lr,
        fc_dims=[256, 256],
        chkpt_dir="./tmp/PPO-Agent",
    ):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, "critic_torch_ppo")

        layers = []
        in_features = state_dim
        for out_features in fc_dims:
            layers.append(layer_init(nn.Linear(in_features, out_features)))
            layers.append(nn.Tanh())
            in_features = out_features
        layers.append(layer_init(nn.Linear(in_features, 1), std=1.0))

        self.critic = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.parameters(), lr=critic_lr, eps=1e-5)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
