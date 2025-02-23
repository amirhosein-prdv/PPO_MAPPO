import numpy as np


class RolloutBuffer:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.next_states = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.logprobs),
            np.array(self.values),
            np.array(self.next_states),
            np.array(self.rewards),
            np.array(self.dones),
            batches,
        )

    def store(self, state, action, logprobs, values, next_state, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprobs)
        self.values.append(values)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.next_states = []
        self.rewards = []
        self.dones = []
