import numpy as np
from typing import List, Tuple, Dict


class MultiAgentRolloutBuffer:
    def __init__(self, batch_size: int, possible_agents: List[str]):
        self.states = {agent: [] for agent in possible_agents}
        self.actions = {agent: [] for agent in possible_agents}
        self.logprobs = {agent: [] for agent in possible_agents}
        self.values = {agent: [] for agent in possible_agents}
        self.next_states = {agent: [] for agent in possible_agents}
        self.rewards = {agent: [] for agent in possible_agents}
        self.dones = {agent: [] for agent in possible_agents}

        self.batch_size: int = batch_size
        self.possible_agents: List[str] = possible_agents

    def generate_batches(self, agent: str) -> List[np.ndarray]:
        n_states = len(self.states[agent])
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        return batches

    def get_data(self, agent: str) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
    ]:
        return (
            np.array(self.states[agent]),
            np.array(self.actions[agent]),
            np.array(self.logprobs[agent]),
            np.array(self.values[agent]),
            np.array(self.next_states[agent]),
            np.array(self.rewards[agent]),
            np.array(self.dones[agent]),
        )

    def store(
        self,
        state: Dict[str, np.ndarray],
        action: Dict[str, np.ndarray],
        logprobs: Dict[str, np.ndarray],
        values: Dict[str, np.ndarray],
        next_state: Dict[str, np.ndarray],
        reward: Dict[str, float],
        done: bool,
    ) -> None:
        for agent in self.possible_agents:
            self.states[agent].append(state[agent])
            self.actions[agent].append(action[agent])
            self.logprobs[agent].append(logprobs[agent])
            self.values[agent].append(values[agent])
            self.next_states[agent].append(next_state[agent])
            self.rewards[agent].append(reward[agent])
            self.dones[agent].append(done)

    def clear(self) -> None:
        self.states = {agent: [] for agent in self.possible_agents}
        self.actions = {agent: [] for agent in self.possible_agents}
        self.logprobs = {agent: [] for agent in self.possible_agents}
        self.values = {agent: [] for agent in self.possible_agents}
        self.next_states = {agent: [] for agent in self.possible_agents}
        self.rewards = {agent: [] for agent in self.possible_agents}
        self.dones = {agent: [] for agent in self.possible_agents}
