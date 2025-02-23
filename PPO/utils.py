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

    def add_scalar(self) -> callable:
        """Returns the writer's add_scalar function."""
        return self.writer.add_scalar

    def add_scalars(self) -> callable:
        """Returns the writer's add_scalars function."""
        return self.writer.add_scalars

    def add_histogram(self) -> callable:
        """Returns the writer's add_histogram function."""
        return self.writer.add_histogram

    def add_text(self) -> callable:
        """Returns the writer's add_text function."""
        return self.writer.add_text

    def close(self) -> callable:
        """Returns the writer's close function."""
        return self.writer.close
