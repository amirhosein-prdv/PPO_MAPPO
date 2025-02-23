import numpy as np
import torch as T
from typing import Optional, Tuple, List

from .RolloutBuffer import RolloutBuffer
from .Networks import ActorNetwork, CriticNetwork
from .utils import get_unique_log_dir, anneal_learning_rate


class Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
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
        logger: Optional[object] = None,
        chkpt_dir: str = "./tmp/PPO-Agent",
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

        chkpt_dir = get_unique_log_dir(chkpt_dir)

        self.actor = ActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            fc_dims=[64, 64],
            actor_lr=lr,
            chkpt_dir=chkpt_dir,
        )
        self.critic = CriticNetwork(
            state_dim=state_dim,
            fc_dims=[64, 64],
            critic_lr=lr,
            chkpt_dir=chkpt_dir,
        )
        self.memory = RolloutBuffer(batch_size)

    def anneal_actor_critic_lr(self, current_step: int, total_steps: int) -> None:
        """Anneal the learning rate of the actor and critic network."""
        anneal_learning_rate(
            self.actor.optimizer, self.actor.initial_lr, current_step, total_steps
        )
        anneal_learning_rate(
            self.critic.optimizer, self.critic.initial_lr, current_step, total_steps
        )

    def save_models(self) -> None:
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self) -> None:
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def get_value(self, observation: np.ndarray) -> float:
        state = T.tensor(np.array([observation]), dtype=T.float32).to(self.actor.device)
        return self.critic(state).item()

    def get_action(
        self, observation: np.ndarray, action: Optional[T.Tensor] = None
    ) -> Tuple[np.ndarray, float, float]:
        state = T.tensor(np.array([observation]), dtype=T.float32).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)

        if action is None:
            action = dist.sample()

        action_logprobs = dist.log_prob(action).sum(1).item()
        value = T.squeeze(value).item()

        return action.detach().cpu().numpy().squeeze(), action_logprobs, value

    def learn(self) -> None:
        clipfracs = []
        for _ in range(self.n_epochs):
            # Generate batch data
            (
                state_arr,
                action_arr,
                old_prob_arr,
                values_arr,
                next_states_arr,
                reward_arr,
                dones_arr,
                batches,
            ) = self.memory.generate_batches()

            # Calculate advantages for the values in memory (bootstrap)
            advantages = np.zeros(len(reward_arr), dtype=np.float32)
            returns = np.zeros(len(reward_arr), dtype=np.float32)
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
                advantages[t] = a_t

            # Advantage for last step
            last_value = self.get_value(next_states_arr[-1])
            delta_last = (
                reward_arr[-1]
                + self.gamma * last_value * (1 - int(dones_arr[-1]))
                - values_arr[-1]
            )
            advantages[-1] = delta_last
            returns = advantages + values_arr

            # Training
            returns = T.tensor(returns, dtype=T.float32).to(self.actor.device)
            advantages = T.tensor(advantages, dtype=T.float32).to(self.actor.device)
            values = T.tensor(values_arr, dtype=T.float32).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                new_value = self.critic(states)
                new_value = T.squeeze(new_value)

                dist_entropy = dist.entropy().sum(1, keepdim=True)
                new_probs = dist.log_prob(actions).sum(1)
                prob_ratio = new_probs.exp() / old_probs.exp()
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
                norm_advantages = (advantages[batch] - advantages[batch].mean()) / (
                    advantages[batch].std() + 1e-8
                )

                # calculate actor loss
                weighted_probs = norm_advantages * prob_ratio
                weighted_clipped_probs = (
                    T.clamp(prob_ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    * norm_advantages
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

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        if self.logger is not None:
            y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            global_step = self.logger.global_step
            add_scalar = self.logger.add_scalar()
            add_scalar(
                "charts/learning_rate",
                self.actor.optimizer.param_groups[0]["lr"],
                global_step,
            )
            add_scalar("losses/value_loss", critic_loss.item(), global_step)
            add_scalar("losses/policy_loss", actor_loss.item(), global_step)
            add_scalar("losses/entropy", entropy_loss.item(), global_step)
            add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            add_scalar("losses/explained_variance", explained_var, global_step)

        self.memory.clear()
