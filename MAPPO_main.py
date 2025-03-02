from typing import Optional
import gymnasium as gym
import numpy as np
from MAPPO.MultiAgent import MultiAgent
from MAPPO.utils import plot_learning_curve, Logger, get_unique_log_dir


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

    multiAgents = MultiAgent(
        env=env,
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
            multiAgents.eval()
            action, logprob, value = multiAgents.get_action(state)
            action = {k: v.detach().cpu().numpy().squeeze() for k, v in action.items()}
            logprob = {k: v.item() for k, v in logprob.items()}
            value = {k: v.item() for k, v in value.items()}

            next_state, reward, terminated, truncated, info = env.step(action)
            done = any(terminated.values()) or any(truncated.values())
            score += sum([v for v in reward.values()])
            multiAgents.memory.store(
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
            logger.add_dict("rollout/step_reward", reward, n_steps)

            if n_steps % training_interval_step == 0:
                multiAgents.anneal_lr(current_step=learn_iters, total_steps=learn_steps)
                multiAgents.learn()
                learn_iters += 1
            state = next_state

        score_history.append(score)
        avg_score = np.mean(score_history[-10:])

        logger.add_scalar("rollout/avg score", avg_score, n_steps)
        logger.add_scalar("rollout/episode_reward", score / t_step, n_steps)
        logger.add_scalar("rollout/episode_len", t_step, n_steps)

        if avg_score > best_score:
            best_score = avg_score
            multiAgents.save_models()

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
