import numpy as np
import torch as T
from typing import Optional, Tuple, List

from .RolloutBuffer import RolloutBuffer
from .Networks import ActorNetwork, CriticNetwork, ActorCriticNetwork
from .utils import anneal_learning_rate
from .logger import Logger


class Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: int,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 3e-4,
        clip_coef: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        logger: Optional["Logger"] = None,
        policy_kwargs: dict[str, List[int]] = {
            "feature": [32],
            "pi": [64, 64],
            "vf": [64, 64],
        },
        chkpt_dir: str = "./tmp/PPO-Agent",
        model_name: str = "actor_critic_ppo",
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.n_epochs = n_epochs
        self.logger = logger

        # if self.logger is not None:
        #     logger.add_text(
        #         "hyperparameters",
        #         "|param|value|\n|-|-|\n%s"
        #         % (
        #             "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        #         ),
        #     )

        self.norm_adv = True
        self.clip_vloss = True

        feature_net = policy_kwargs["feature"]
        vf_net = policy_kwargs["vf"]
        pi_net = policy_kwargs["pi"]

        self.policy = ActorCriticNetwork(
            state_dim,
            action_dim,
            max_action=max_action,
            feature_fc_dims=feature_net,
            actor_fc_dims=pi_net,
            critic_fc_dims=vf_net,
            policy_lr=lr,
            chkpt_dir=chkpt_dir,
            model_name=model_name,
        )

        self.memory = RolloutBuffer(batch_size)
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

    def learn(self) -> None:
        # Generate Data
        (
            state_arr,
            action_arr,
            old_prob_arr,
            values_arr,
            next_states_arr,
            reward_arr,
            dones_arr,
        ) = self.memory.get_data()

        # Calculate Advantages
        advantages, returns = self.get_GAE_and_returns(
            reward_arr, values_arr, dones_arr, next_states_arr
        )

        returns = T.tensor(returns, dtype=T.float32).to(self.device)
        advantages = T.tensor(advantages, dtype=T.float32).to(self.device)
        values = T.tensor(values_arr, dtype=T.float32).to(self.device)

        clipfracs = []
        critic_buffer = []
        actor_buffer = []
        entropy_buffer = []
        approx_kl_buffer = []
        explained_var_buffer = []
        self.policy.train()
        for _ in range(self.n_epochs):
            # Generate batch data
            batches = self.memory.generate_batches()

            # Training
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.device)
                actions = T.tensor(action_arr[batch]).to(self.device)

                dist, new_value = self.policy(states)
                new_value = T.squeeze(new_value)

                dist_entropy = dist.entropy().sum(1, keepdim=True)
                new_probs = dist.log_prob(actions).sum(1)
                prob_ratio = (new_probs - old_probs).exp()
                prob_logratio = new_probs - old_probs

                with T.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    approx_kl = ((prob_ratio - 1) - prob_logratio).mean()
                    clipfracs += [
                        ((prob_ratio - 1.0).abs() > self.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                # nomalize advantage
                normalized_advantages = (
                    advantages[batch] - advantages[batch].mean()
                ) / (advantages[batch].std() + 1e-8)

                # calculate actor loss
                weighted_probs = normalized_advantages * prob_ratio
                weighted_clipped_probs = (
                    T.clamp(prob_ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    * normalized_advantages
                )
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # Calculate Critic loss
                # clip value loss
                clipped_value = values[batch] + T.clamp(
                    new_value - values[batch],
                    -self.clip_coef,
                    self.clip_coef,
                )
                critic_loss_clipped = (returns[batch] - clipped_value) ** 2
                critic_loss_unclipped = (returns[batch] - new_value) ** 2
                critic_loss_max = T.max(critic_loss_clipped, critic_loss_unclipped)
                critic_loss = 0.5 * critic_loss_max.mean()

                # Calculate entropy loss
                entropy_loss = dist_entropy.mean()

                total_loss = (
                    actor_loss
                    + self.vf_coef * critic_loss
                    - self.ent_coef * entropy_loss
                )

                self.policy.optimizer.zero_grad()
                total_loss.backward()
                T.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # Calculate explained variance
                y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )

                # Store training variable to log
                critic_buffer.append(critic_loss.item())
                actor_buffer.append(actor_loss.item())
                entropy_buffer.append(entropy_loss.item())
                approx_kl_buffer.append(approx_kl.item())
                explained_var_buffer.append(explained_var)

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        if self.logger is not None:
            # record for plotting purposes
            self.logger.add_scalar(
                "train/learning_rate", self.policy.optimizer.param_groups[0]["lr"]
            )
            self.logger.add_scalar("train/value_loss", np.mean(critic_buffer))
            self.logger.add_scalar("train/policy_loss", np.mean(actor_buffer))
            self.logger.add_scalar("train/entropy", np.mean(entropy_buffer))
            self.logger.add_scalar("train/approx_kl", np.mean(approx_kl_buffer))
            self.logger.add_scalar("train/clipfrac", np.mean(clipfracs))
            self.logger.add_scalar(
                "train/explained_variance", np.mean(explained_var_buffer)
            )

        self.memory.clear()

    def save_models(self) -> None:
        print("... saving models ...")
        self.policy.save_checkpoint()

    def load_models(self) -> None:
        print("... loading models ...")
        self.policy.load_checkpoint()
