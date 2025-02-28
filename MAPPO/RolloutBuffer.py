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
        batches = []
        n_states = len(self.states[agent])
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        agent_batches = [indices[i : i + self.batch_size] for i in batch_start]
        batches.append(agent_batches)

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
        state: np.ndarray,
        action: np.ndarray,
        logprobs: np.ndarray,
        values: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        for agent in self.possible_agents:
            self.states[agent].append(state)
            self.actions[agent].append(action)
            self.logprobs[agent].append(logprobs)
            self.values[agent].append(values)
            self.next_states[agent].append(next_state)
            self.rewards[agent].append(reward)
            self.dones[agent].append(done)

    def clear(self) -> None:
        self.states = {agent: [] for agent in self.possible_agents}
        self.actions = {agent: [] for agent in self.possible_agents}
        self.logprobs = {agent: [] for agent in self.possible_agents}
        self.values = {agent: [] for agent in self.possible_agents}
        self.next_states = {agent: [] for agent in self.possible_agents}
        self.rewards = {agent: [] for agent in self.possible_agents}
        self.dones = {agent: [] for agent in self.possible_agents}
