from typing import Optional
import gymnasium as gym
import numpy as np
from PPO.Agent import Agent
from PPO.utils import plot_learning_curve, Logger, get_unique_log_dir


def make_env(gym_id, seed: Optional[int] = None):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10), env.observation_space
        )
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

    n_epochs = 15
    batch_size = 32
    episode_num = 4000
    training_interval_step = 200
    timeLimit = env.spec.max_episode_steps

    chkpt_dir = "./results/models/PPO"
    chkpt_dir = get_unique_log_dir(chkpt_dir)

    figure_folder = "./results/plots/PPO"
    figure_file = get_unique_log_dir(figure_folder) + "/LearningCurve.png"

    logger = Logger(log_dir="./results/tb/PPO")

    policy_kwargs = {
        "feature": [64, 32],
        "pi": [256, 256],
        "vf": [256, 128],
    }

    agent = Agent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        policy_kwargs=policy_kwargs,
        batch_size=batch_size,
        n_epochs=n_epochs,
        logger=logger,
        chkpt_dir=chkpt_dir,
    )

    best_score = -np.inf
    score_history = []

    learn_steps = timeLimit * episode_num // training_interval_step
    learn_iters = 0
    avg_score = 0

    n_steps = 0
    for eps in range(episode_num):
        state, info = env.reset()
        done = False
        score = 0
        t_step = 0
        while not done:
            action, logprob, value = agent.get_action(state)
            action = action.detach().cpu().numpy().squeeze()
            logprob = logprob.item()
            value = value.item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            agent.memory.store(
                state,
                action,
                logprob,
                value,
                next_state,
                reward,
                done,
            )

            n_steps += 1
            t_step += 1
            logger.update_global_step(n_steps)
            logger.add_scalar("reward", reward)

            if n_steps % training_interval_step == 0:
                agent.anneal_lr(current_step=learn_iters, total_steps=learn_steps)
                agent.learn()
                learn_iters += 1
            state = next_state

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        logger.add_scalar("episodic mean reward", score / t_step, n_steps)
        logger.add_scalar("avg score", avg_score, n_steps)
        logger.add_scalar("charts/rollout_step", t_step, n_steps)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(
            "episode",
            eps,
            "score %.1f" % score,
            "avg score %.1f" % avg_score,
            "global_steps",
            n_steps,
            "episode_step",
            t_step,
        )
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
