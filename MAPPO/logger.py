import torch as T
from torch.utils.tensorboard import SummaryWriter

from .utils import get_unique_log_dir


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

    def add_dict(self, tag: str, dict_value: dict, global_step: int = None) -> None:
        for k, v in dict_value.items():
            self.writer.add_scalar(tag + f"/{k}", v, global_step or self.global_step)

    def close(self) -> None:
        """Closes the writer."""
        self.writer.close()


class EvaluationLogger:
    def __init__(
        self,
        eval_env,
        multiAgents,
        logger: Logger,
        eval_episodes=5,
        verbose=0,
    ):
        """
        Evaluate the model on multiple episodes after each policy update.

        :param eval_env: Gym-like environment for evaluation.
        :param eval_episodes: Number of episodes for evaluation.
        :param verbose: Verbosity level (0: no output, 1: info messages).
        """
        self.eval_env = eval_env
        self.multiAgents = multiAgents
        self.logger = logger
        self.eval_episodes = eval_episodes
        self.verbose = verbose

    def evaluate_and_log(self):
        """Run evaluation episodes and step averaged metrics."""
        metrics_buffer = {
            "reward": [],
            "rewards": {},
            "R_p_k": {},
            "R_c_k": {},
            "L_t": {},
            "var": {},
            "C_k": {},
            "downlinkPower_k": {},
            "objective": [],
            "agent_objective": {},
            "power_consuming": [],
            "constraints_error": {},
        }

        self.multiAgents.eval()
        for _ in range(self.eval_episodes):
            state, _ = self.eval_env.reset()
            done = False

            while not done:
                action, _, _ = self.multiAgents.get_action(state)
                action = {
                    k: v.detach().cpu().numpy().squeeze() for k, v in action.items()
                }

                next_state, rewards, terminated, truncated, infos = self.eval_env.step(
                    action
                )
                agent = list(infos.keys())[0]
                done = terminated[agent] or truncated[agent]
                state = next_state

                info = infos[agent]  # Assuming all agents share the same info structure

                metrics_buffer["reward"].append(info["reward"])

                for k, v in rewards.items():
                    metrics_buffer["rewards"].setdefault(k, []).append(v)

                # Collect metrics from the 'info' dictionary
                if "R_p_k" in info:
                    for key, value in info["R_p_k"].items():
                        for key_, value_ in value.items():
                            for user, Rate in enumerate(value_):
                                metrics_buffer["R_p_k"].setdefault(
                                    (key, key_, user), []
                                ).append(Rate)

                if "R_c_k" in info:
                    for key, value in info["R_c_k"].items():
                        for key_, value_ in value.items():
                            for user, Rate in enumerate(value_):
                                metrics_buffer["R_c_k"].setdefault(
                                    (key, key_, user), []
                                ).append(Rate)

                if "L_t" in info:
                    for key, value in enumerate(info["L_t"]):
                        metrics_buffer["L_t"].setdefault(key, []).append(value[0])

                if "var" in info:
                    for key, value in info["var"].items():
                        metrics_buffer["var"].setdefault(key, []).append(value[0])

                if "C_k" in info:
                    for key, value in enumerate(info["C_k"]):
                        metrics_buffer["C_k"].setdefault(key, []).append(value)

                if "downlinkPower_k" in info:
                    for key, value in enumerate(info["downlinkPower_k"]):
                        metrics_buffer["downlinkPower_k"].setdefault(key, []).append(
                            value
                        )

                if "objective" in info:
                    metrics_buffer["objective"].append(info["objective"])

                if "power_consuming" in info:
                    metrics_buffer["power_consuming"].append(info["power_consuming"][0])

                if "constraints_error" in info:
                    for key, value in info["constraints_error"].items():
                        metrics_buffer["constraints_error"].setdefault(key, []).append(
                            value
                        )

                if "agent_objective" in info:
                    for key, value in info["agent_objective"].items():
                        metrics_buffer["agent_objective"].setdefault(key, []).append(
                            value
                        )

        # Compute averages and log them
        rew = sum(metrics_buffer["reward"]) / len(metrics_buffer["reward"])
        self.logger.add_scalar("reward (eval)", rew)

        for k, v in metrics_buffer["rewards"].items():
            self.logger.add_scalar(f"reward {k} (eval)", sum(v) / len(v))

        if "R_p_k" in metrics_buffer:
            for (key, key_, user), values in metrics_buffer["R_p_k"].items():
                avg_value = sum(values) / len(values)
                self.logger.add_scalar(
                    f"R_p_k (eval)/{key}/RIS_{key_}/{user+1}", avg_value
                )

        if "R_c_k" in metrics_buffer:
            for (key, key_, user), values in metrics_buffer["R_c_k"].items():
                avg_value = sum(values) / len(values)
                self.logger.add_scalar(
                    f"R_c_k (eval)/{key}/RIS_{key_}/{user+1}", avg_value
                )

        if "L_t" in metrics_buffer:
            for key, value in metrics_buffer["L_t"].items():
                avg_L_t = sum(value) / len(value)
                self.logger.add_scalar(f"L_t (eval)/{key}", avg_L_t)

        if "var" in metrics_buffer:
            for key, values in metrics_buffer["var"].items():
                avg_value = sum(values) / len(values)
                self.logger.add_scalar(f"var (eval)/{key}", avg_value)

        if "C_k" in metrics_buffer:
            for key, value in metrics_buffer["C_k"].items():
                avg_C_k = sum(value) / len(value)
                self.logger.add_scalar(f"C_k (eval)/{key}", avg_C_k)

        if "objective" in metrics_buffer:
            avg_objective = sum(metrics_buffer["objective"]) / len(
                metrics_buffer["objective"]
            )
            self.logger.add_scalar("objective (eval)", avg_objective)

        if "power_consuming" in metrics_buffer:
            avg_power_consuming = sum(metrics_buffer["power_consuming"]) / len(
                metrics_buffer["power_consuming"]
            )
            self.logger.add_scalar("power_consuming (eval)", avg_power_consuming)

        if "constraints_error" in metrics_buffer:
            for key, values in metrics_buffer["constraints_error"].items():
                avg_value = sum(values) / len(values)
                self.logger.add_scalar(f"constraints_error (eval)/{key}", avg_value)

        if "agent_objective" in metrics_buffer:
            for key, values in metrics_buffer["agent_objective"].items():
                avg_value = sum(values) / len(values)
                self.logger.add_scalar(f"agent_objective (eval)/{key}", avg_value)

        if self.verbose > 0:
            print("Evaluation metrics logged.")


class StepLogger:
    def __init__(
        self, logger: Logger, step_interval=10, suffix_title="step", verbose=0
    ):
        self.suffix_title = suffix_title
        self.verbose = verbose
        self.logger = logger
        self.interval = step_interval
        self.step_counter = 0

        self.initialize_buffer()

    def initialize_buffer(self) -> None:
        self.metrics_buffer = {
            "rewards": {},
            "reward": [],
        }

    def record_log(self) -> None:
        if self.step_counter % self.interval == 0:
            self._log_buffered_metrics()
            self._reset_buffer()
            if self.verbose > 0:
                print("Step metrics logged.")

    def add_info(self, infos) -> None:
        self.step_counter += 1
        agent = list(infos.keys())[0]
        info = infos[agent]  # Assuming all agents share the same info structure
        self._buffer_metrics(info)

    def _buffer_metrics(self, info) -> None:

        if "rewards" in info:
            for k, v in info["rewards"].items():
                self.metrics_buffer["rewards"].setdefault(k, []).append(v)

        if "reward" in info:
            self.metrics_buffer["reward"].append(info["reward"])

    def _log_buffered_metrics(self) -> None:

        if "rewards" in self.metrics_buffer:
            for k, v in self.metrics_buffer["rewards"].items():
                self.logger.add_scalar(f"reward {k} (eval)", sum(v) / len(v))

        if "reward" in self.metrics_buffer:
            avg_value = sum(self.metrics_buffer["reward"]) / len(
                self.metrics_buffer["reward"]
            )
            self.logger.add_scalar(
                f"reward ({self.interval}-{self.suffix_title})", avg_value
            )

    def _reset_buffer(self) -> None:
        self.initialize_buffer()
