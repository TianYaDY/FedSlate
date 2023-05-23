import numpy as np
import time, datetime
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanLength':>15}"
                f"{'MeanReward_alpha':>15}{'MeanLoss_alpha':>15}{'MeanQValue_alpha':>15}"
                f"{'MeanReward_beta':>15}{'MeanLoss_beta':>15}{'MeanQValue_beta':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_lengths_plot = save_dir / "length_plot.jpg"

        self.ep_rewards_alpha_plot = save_dir / "reward_alpha_plot.jpg"
        self.ep_avg_losses_alpha_plot = save_dir / "loss_alpha_plot.jpg"
        self.ep_avg_qs_alpha_plot = save_dir / "q_alpha_plot.jpg"
        self.ep_rewards_beta_plot = save_dir / "reward_beta_plot.jpg"
        self.ep_avg_losses_beta_plot = save_dir / "loss_beta_plot.jpg"
        self.ep_avg_qs_beta_plot = save_dir / "q_beta_plot.jpg"

        # History metrics
        self.ep_lengths = []

        self.ep_rewards_alpha = []
        self.ep_avg_losses_alpha = []
        self.ep_avg_qs_alpha = []
        self.ep_rewards_beta = []
        self.ep_avg_losses_beta = []
        self.ep_avg_qs_beta = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_lengths = []

        self.moving_avg_ep_rewards_alpha = []
        self.moving_avg_ep_avg_losses_alpha = []
        self.moving_avg_ep_avg_qs_alpha = []
        self.moving_avg_ep_rewards_beta = []
        self.moving_avg_ep_avg_losses_beta = []
        self.moving_avg_ep_avg_qs_beta = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward_alpha, loss_alpha, q_alpha, reward_beta, loss_beta, q_beta):
        self.curr_ep_reward_alpha += reward_alpha
        self.curr_ep_reward_beta += reward_beta

        self.curr_ep_length += 1
        if loss_alpha:
            self.curr_ep_loss_alpha += loss_alpha
            self.curr_ep_q_alpha += q_alpha
            self.curr_ep_loss_length_alpha += 1
        if loss_beta:
            self.curr_ep_loss_beta += loss_beta
            self.curr_ep_q_beta += q_beta
            self.curr_ep_loss_length_beta += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards_alpha.append(self.curr_ep_reward_alpha)
        self.ep_rewards_beta.append(self.curr_ep_reward_beta)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length_alpha == 0:
            ep_avg_loss_alpha = 0
            ep_avg_q_alpha = 0
        else:
            ep_avg_loss_alpha = np.round(self.curr_ep_loss_alpha / self.curr_ep_loss_length_alpha, 5)
            ep_avg_q_alpha = np.round(self.curr_ep_q_alpha / self.curr_ep_loss_length_alpha, 5)
        if self.curr_ep_loss_length_beta == 0:
            ep_avg_loss_beta = 0
            ep_avg_q_beta = 0
        else:
            ep_avg_loss_beta = np.round(self.curr_ep_loss_beta / self.curr_ep_loss_length_beta, 5)
            ep_avg_q_beta = np.round(self.curr_ep_q_beta / self.curr_ep_loss_length_beta, 5)

        self.ep_avg_losses_alpha.append(ep_avg_loss_alpha)
        self.ep_avg_qs_alpha.append(ep_avg_q_alpha)
        self.ep_avg_losses_beta.append(ep_avg_loss_beta)
        self.ep_avg_qs_beta.append(ep_avg_q_beta)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_length = 0

        self.curr_ep_reward_alpha = 0.0
        self.curr_ep_loss_alpha = 0.0
        self.curr_ep_q_alpha = 0.0
        self.curr_ep_loss_length_alpha = 0
        self.curr_ep_reward_beta = 0.0
        self.curr_ep_loss_beta = 0.0
        self.curr_ep_q_beta = 0.0
        self.curr_ep_loss_length_beta = 0

    def record(self, episode, epsilon, step):
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)

        mean_ep_reward_alpha = np.round(np.mean(self.ep_rewards_alpha[-100:]), 3)
        mean_ep_loss_alpha = np.round(np.mean(self.ep_avg_losses_alpha[-100:]), 3)
        mean_ep_q_alpha = np.round(np.mean(self.ep_avg_qs_alpha[-100:]), 3)
        mean_ep_reward_beta = np.round(np.mean(self.ep_rewards_beta[-100:]), 3)
        mean_ep_loss_beta = np.round(np.mean(self.ep_avg_losses_beta[-100:]), 3)
        mean_ep_q_beta = np.round(np.mean(self.ep_avg_qs_beta[-100:]), 3)

        self.moving_avg_ep_lengths.append(mean_ep_length)

        self.moving_avg_ep_rewards_alpha.append(mean_ep_reward_alpha)
        self.moving_avg_ep_avg_losses_alpha.append(mean_ep_loss_alpha)
        self.moving_avg_ep_avg_qs_alpha.append(mean_ep_q_alpha)
        self.moving_avg_ep_rewards_beta.append(mean_ep_reward_beta)
        self.moving_avg_ep_avg_losses_beta.append(mean_ep_loss_beta)
        self.moving_avg_ep_avg_qs_beta.append(mean_ep_q_beta)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Reward {mean_ep_reward_alpha} - "
            f"Mean Loss {mean_ep_loss_alpha} - "
            f"Mean Q Value {mean_ep_q_alpha} - "
            f"Mean Reward {mean_ep_reward_beta} - "
            f"Mean Loss {mean_ep_loss_beta} - "
            f"Mean Q Value {mean_ep_q_beta} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_length:15.3f}"
                f"{mean_ep_reward_alpha:15.3f}{mean_ep_loss_alpha:15.3f}{mean_ep_q_alpha:15.3f}"
                f"{mean_ep_reward_beta:15.3f}{mean_ep_loss_beta:15.3f}{mean_ep_q_beta:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_lengths", "ep_rewards_alpha", "ep_avg_losses_alpha", "ep_avg_qs_alpha", "ep_rewards_beta",
                       "ep_avg_losses_beta", "ep_avg_qs_beta"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()
