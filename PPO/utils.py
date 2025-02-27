import os
import torch as T
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from typing import List


def plot_learning_curve(x: List[float], scores: List[float], figure_file: str) -> None:
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : (i + 1)])
    plt.plot(x, running_avg)
    plt.title("Running average of previous 100 scores")
    plt.savefig(figure_file)


def anneal_learning_rate(
    optimizer: T.optim.Optimizer,
    initial_lr: float,
    current_step: int,
    total_steps: int,
) -> None:
    """Anneal the learning rate linearly."""
    lr = initial_lr * (1 - (current_step / float(total_steps)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_unique_log_dir(base_dir: str) -> str:
    """Generates a unique log directory name by incrementing a suffix if needed."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return base_dir  # Return as is if it does not exist

    index = 1
    new_dir = f"{base_dir}_{index}"

    while os.path.exists(new_dir):
        index += 1
        new_dir = f"{base_dir}_{index}"

    os.makedirs(new_dir)
    return new_dir


class Logger:
    def __init__(self, log_dir: str) -> None:
        log_dir = get_unique_log_dir(log_dir)
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

    def update_global_step(self, step: int) -> None:
        self.global_step = step

    def add_scalar(
        self, tag: str, scalar_value: float, global_step: int = None
    ) -> None:
        """Adds a scalar value to the writer."""
        self.writer.add_scalar(tag, scalar_value, global_step or self.global_step)

    def add_scalars(
        self, main_tag: str, tag_scalar_dict: dict, global_step: int = None
    ) -> None:
        """Adds multiple scalar values to the writer."""
        self.writer.add_scalars(
            main_tag, tag_scalar_dict, global_step or self.global_step
        )

    def add_histogram(
        self,
        tag: str,
        values: T.Tensor,
        global_step: int = None,
        bins: str = "tensorflow",
    ) -> None:
        """Adds a histogram to the writer."""
        self.writer.add_histogram(tag, values, global_step or self.global_step, bins)

    def add_text(self, tag: str, text_string: str, global_step: int = None) -> None:
        """Adds text data to the writer."""
        self.writer.add_text(tag, text_string, global_step or self.global_step)

    def close(self) -> None:
        """Closes the writer."""
        self.writer.close()


class EvaluationLoggerCallback:
    def __init__(self, eval_env, agent, logger: Logger, eval_episodes=5, verbose=0):
        """
        Evaluate the model on multiple episodes after each policy update.

        :param eval_env: Gym-like environment for evaluation.
        :param eval_episodes: Number of episodes for evaluation.
        :param verbose: Verbosity level (0: no output, 1: info messages).
        """
        self.eval_env = eval_env
        self.agent = agent
        self.logger = logger
        self.eval_episodes = eval_episodes
        self.verbose = verbose

    def evaluate_policy(self):
        """Run evaluation episodes and step averaged metrics."""
        metrics_buffer = {"reward": []}

        self.agent.policy.eval()
        for _ in range(self.eval_episodes):
            state, _ = self.eval_env.reset()
            done = False

            while not done:
                action, logprob, value = self.agent.get_action(state)
                next_state, reward, terminated, truncated, info = self.eval_env.step(
                    action
                )
                done = terminated or truncated
                state = next_state

                metrics_buffer["reward"].append(reward)

                # Collect metrics from the 'info' dictionary

        # Compute averages and step them
        self.logger.add_scalar(
            "reward (eval)",
            sum(metrics_buffer["reward"]) / len(metrics_buffer["reward"]),
        )

        if self.verbose > 0:
            print("Evaluation metrics logged.")
