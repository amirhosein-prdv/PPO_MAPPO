import os
import numpy as np

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from typing import List, Tuple


def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_action: int,
        fc_dims: List[int],
    ) -> None:
        super(Actor, self).__init__()
        self.max_action = max_action

        layers = []
        in_features = input_dim
        for out_features in fc_dims:
            layers.append(layer_init(nn.Linear(in_features, out_features)))
            layers.append(nn.Tanh())
            in_features = out_features
        layers.append(layer_init(nn.Linear(in_features, output_dim), std=0.01))
        layers.append(nn.Tanh())

        self.mean = nn.Sequential(*layers)
        self.logstd = nn.Parameter(T.zeros(1, output_dim))

    def forward(self, state: T.Tensor) -> Normal:
        action_mean = self.mean(state) * self.max_action
        action_logstd = self.logstd.expand_as(action_mean).exp()
        dist = Normal(action_mean, action_logstd)
        return dist


class Critic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        fc_dims: List[int],
    ) -> None:
        super(Critic, self).__init__()

        layers = []
        in_features = input_dim
        for out_features in fc_dims:
            layers.append(layer_init(nn.Linear(in_features, out_features)))
            layers.append(nn.Tanh())
            in_features = out_features
        layers.append(layer_init(nn.Linear(in_features, 1), std=1.0))

        self.critic = nn.Sequential(*layers)

    def forward(self, state: T.Tensor) -> T.Tensor:
        value = self.critic(state)
        return value


class ActorCriticNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: int,
        feature_fc_dims: List[int],
        actor_fc_dims: List[int],
        critic_fc_dims: List[int],
        policy_lr=3e-4,
        chkpt_dir: str = None,
        model_name: str = None,
    ) -> None:
        super(ActorCriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, model_name)
        self.initial_lr = policy_lr

        # Shared feature extraction layers
        if len(feature_fc_dims) != 0:
            layers = []
            in_features = state_dim
            for out_features in feature_fc_dims:
                layers.append(layer_init(nn.Linear(in_features, out_features)))
                layers.append(nn.Tanh())
                in_features = out_features
            self.feature_extractor = nn.Sequential(*layers)
            input_dim = feature_fc_dims[-1]
        else:
            self.feature_extractor = nn.Flatten()
            input_dim = state_dim

        # Actor head
        self.actor = Actor(input_dim, action_dim, max_action, actor_fc_dims)

        # Critic head
        self.critic = Critic(input_dim, critic_fc_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=policy_lr, eps=1e-5)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state: T.Tensor) -> Tuple[Normal, T.Tensor]:
        features = self.feature_extractor(state)
        dist = self.actor(features)
        value = self.critic(features)
        return dist, value

    def save_checkpoint(self) -> None:
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self) -> None:
        self.load_state_dict(T.load(self.checkpoint_file))
