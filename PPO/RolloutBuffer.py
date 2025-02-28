import numpy as np
from typing import List, Tuple


class RolloutBuffer:
    def __init__(self, batch_size: int):
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.logprobs: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.next_states: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

        self.batch_size: int = batch_size

    def get_data(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.logprobs),
            np.array(self.values),
            np.array(self.next_states),
            np.array(self.rewards),
            np.array(self.dones),
        )

    def generate_batches(
        self,
    ) -> List[np.ndarray]:
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        return batches

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        logprobs: np.ndarray,
        values: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprobs)
        self.values.append(values)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self) -> None:
        self.states = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.next_states = []
        self.rewards = []
        self.dones = []
