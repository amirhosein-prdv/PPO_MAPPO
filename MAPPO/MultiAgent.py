import torch as T
import numpy as np
from .utils import Logger
from typing import List, Optional, Tuple

from pettingzoo import ParallelEnv

from .Agent import Agent
from .RolloutBuffer import MultiAgentRolloutBuffer


class MultiAgent:
    def __init__(
        self,
        env: ParallelEnv,
        batch_size: int = 64,
        n_epochs: int = 10,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        logger: Optional["Logger"] = None,
        chkpt_dir: str = "./tmp/PPO-MultiAgent",
        policy_kwargs: dict[str, List[int]] = {
            "feature": [32],
            "pi": [64, 64],
            "vf": [64, 64],
        },
    ) -> None:

        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.n_epochs = n_epochs
        self.logger = logger

        self.norm_adv = True
        self.clip_vloss = True

        self.possible_agents = env.possible_agents
        self.num_agents = env.num_agents

        self.agents = {
            agent: Agent(
                state_dim=env.observation_space(agent),
                action_dim=env.action_space(agent),
                n_epochs=n_epochs,
                lr=lr,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_coef=clip_coef,
                vf_coef=vf_coef,
                ent_coef=ent_coef,
                max_grad_norm=max_grad_norm,
                target_kl=target_kl,
                logger=logger,
                policy_kwargs=policy_kwargs,
                chkpt_dir=chkpt_dir,
                model_name=f"actor-critic_{agent}",
            )
            for agent in self.possible_agents
        }

        self.memory = MultiAgentRolloutBuffer(batch_size, self.possible_agents)
        self.device = self.agents[0].device

    def learn(self) -> None:
        for agent in self.possible_agents:
            self._learn(agent)

    def _learn(self, agent_name: str) -> None:
        # Generate Data
        (
            state_arr,
            action_arr,
            old_prob_arr,
            values_arr,
            next_states_arr,
            reward_arr,
            dones_arr,
        ) = self.memory.get_data(agent_name)

        agent = self.agents[agent_name]

        # Calculate Advantages
        advantages, returns = agent.get_GAE_and_returns(
            self, reward_arr, values_arr, dones_arr, next_states_arr
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
        agent.policy.train()
        for _ in range(self.n_epochs):
            # Generate batch data
            batches = self.memory.generate_batches(agent_name)

            # Training
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.device)
                actions = T.tensor(action_arr[batch]).to(self.device)

                dist, new_value = self.policy(states)
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

                agent.policy.optimizer.zero_grad()
                total_loss.backward()
                T.nn.utils.clip_grad_norm_(
                    agent.policy.parameters(), self.max_grad_norm
                )
                agent.policy.optimizer.step()

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
                "losses/learning_rate", self.policy.optimizer.param_groups[0]["lr"]
            )
            self.logger.add_scalar(
                f"losses/agent_{agent_name}/value_loss", np.mean(critic_buffer)
            )
            self.logger.add_scalar(
                f"losses/agent_{agent_name}/policy_loss", np.mean(actor_buffer)
            )
            self.logger.add_scalar(
                f"losses/agent_{agent_name}/entropy", np.mean(entropy_buffer)
            )
            self.logger.add_scalar(
                f"losses/agent_{agent_name}/approx_kl", np.mean(approx_kl_buffer)
            )
            self.logger.add_scalar(
                f"losses/agent_{agent_name}/clipfrac", np.mean(clipfracs)
            )
            self.logger.add_scalar(
                f"losses/agent_{agent_name}/explained_variance",
                np.mean(explained_var_buffer),
            )

        self.memory.clear()

    def save_models(self) -> None:
        for agent in self.agents:
            agent.save_models()

    def load_models(self) -> None:
        for agent in self.agents:
            agent.load_models()
