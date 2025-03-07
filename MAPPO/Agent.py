import numpy as np
import torch as T
from typing import Optional, Tuple, List

from .Networks import ActorCriticNetwork
from .utils import Logger, anneal_learning_rate


class Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: int = 1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        policy_kwargs: dict[str, List[int]] = {
            "feature": [32],
            "pi": [64, 64],
            "vf": [64, 64],
        },
        chkpt_dir: str = "./tmp/PPO-MultiAgent",
        model_name: str = "actor_critic_ppo",
    ) -> None:

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        feature_net = policy_kwargs["feature"]
        vf_net = policy_kwargs["vf"]
        pi_net = policy_kwargs["pi"]

        self.policy = ActorCriticNetwork(
            state_dim,
            action_dim,
            max_action,
            feature_fc_dims=feature_net,
            actor_fc_dims=pi_net,
            critic_fc_dims=vf_net,
            policy_lr=lr,
            chkpt_dir=chkpt_dir,
            model_name=model_name,
        )

        self.device = self.policy.device

    def anneal_lr(self, current_step: int, total_steps: int) -> None:
        """Anneal the learning rate of the policy network."""
        anneal_learning_rate(
            self.policy.optimizer, self.policy.initial_lr, current_step, total_steps
        )

    def get_value(self, observation: np.ndarray) -> float:
        state = T.tensor(np.array([observation]), dtype=T.float32).to(self.device)
        value = self.policy(state)[1]
        return value

    def get_action(
        self, observation: np.ndarray, action: Optional[T.Tensor] = None
    ) -> Tuple[np.ndarray, float, float]:
        state = T.tensor(np.array([observation]), dtype=T.float32).to(self.device)
        dist, value = self.policy(state)

        if action is None:
            action = dist.sample()
        logprobs = dist.log_prob(action).sum(-1)

        return action, logprobs, value

    def get_GAE_and_returns(self, reward_arr, values_arr, dones_arr, next_states_arr):
        advantages = np.zeros(len(reward_arr), dtype=np.float32)
        returns = np.zeros(len(reward_arr), dtype=np.float32)
        with T.no_grad():
            # Advantage for last step
            last_value = self.get_value(next_states_arr[-1]).item()
            delta_last = (
                reward_arr[-1]
                + self.gamma * last_value * (1 - int(dones_arr[-1]))
                - values_arr[-1]
            )
            advantages[-1] = delta_last

            # Calculate advantages for the values in memory (bootstrap)
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    delta = (
                        reward_arr[k]
                        + self.gamma * values_arr[k + 1] * (1 - int(dones_arr[k]))
                        - values_arr[k]
                    )
                    a_t += discount * delta
                    discount *= self.gamma * self.gae_lambda
                    if dones_arr[k]:
                        break
                advantages[t] = a_t + discount * advantages[k + 1]

            returns = advantages + values_arr
        return advantages, returns

    def save_models(self) -> None:
        print("... saving models ...")
        self.policy.save_checkpoint()

    def load_models(self) -> None:
        print("... loading models ...")
        self.policy.load_checkpoint()
