from typing import Optional
import gym
import numpy as np
from PPO.Agent import Agent


def make_env(gym_id, seed: Optional[int] = None):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.RecordVideo(env, "./video")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        if seed is not None:
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        return env

    return thunk


if __name__ == "__main__":
    env = make_env("BipedalWalker-v3")
    env = env()

    n_epochs = 5
    batch_size = 50
    episode_num = 2
    timeLimit = env.spec.max_episode_steps

    agent = Agent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        batch_size=batch_size,
        n_epochs=n_epochs,
        chkpt_dir="./results/models/PPO_2",
    )

    agent.load_models()
    n_steps = 0
    for eps in range(episode_num):
        state, info = env.reset()
        done = False
        score = 0
        t_step = 0
        while not done:
            action, logprob, value = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            score += reward
            t_step += 1
        print(f"Episode {eps + 1} score: {score}")
    env.close()
