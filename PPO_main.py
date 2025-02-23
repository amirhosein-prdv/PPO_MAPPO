import gym
import numpy as np
from PPO.Agent import Agent
from PPO.utils import plot_learning_curve, Logger, get_unique_log_dir

if __name__ == "__main__":
    env = gym.make("BipedalWalker-v3")
    n_epochs = 5
    batch_size = 50
    episode_num = 400
    training_interval_step = 100

    logger = Logger(log_dir="./results/tb/PPO")

    agent = Agent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        batch_size=batch_size,
        n_epochs=n_epochs,
        logger=logger,
        chkpt_dir="./results/models/PPO",
    )

    figure_folder = "./results/plots/PPO"
    figure_file = get_unique_log_dir(figure_folder) + "/LearningCurve.png"

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0

    n_steps = 0

    for eps in range(episode_num):
        state, info = env.reset()
        done = False
        score = 0
        while not done:
            action, logprob, value = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            n_steps += 1
            logger.update_global_step(n_steps)
            score += reward
            agent.memory.store(state, action, logprob, value, next_state, reward, done)

            if n_steps % training_interval_step == 0:
                agent.learn()
                learn_iters += 1
            state = next_state
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(
            "episode",
            eps,
            "score %.1f" % score,
            "avg score %.1f" % avg_score,
            "time_steps",
            n_steps,
            "training_interval_steps",
            learn_iters,
        )
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
