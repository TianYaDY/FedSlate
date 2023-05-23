import torch
import random
import datetime
from pathlib import Path

import environments
import agents
import log


def generate_random_list(N, n):
    if n > N:
        print("Error: n should be less than or equal to N.")
        return

    nums = random.sample(range(0, N), n)
    return nums

user_features = 1
doc_features = 1
num_of_candidates = 500
slate_size = 10
batch_size = 32
num_contex = 5

save_dir = Path("checkpoints_rnd") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

logger = log.Logger(save_dir)


env = environments.RecsimEnv(num_of_candidates, slate_size, True, 42, 42)
step = 0
episodes = 5000
for e in range(episodes):
    env.env_ini(0)
    env.env_ini(1)
    while True:
        action_alpha = generate_random_list(num_of_candidates, slate_size)
        _, _, _, _, _, reward_alpha, done_alpha = env.env_step(action_alpha, 0)
        action_beta = generate_random_list(num_of_candidates, slate_size)
        _, _, _, _, _, reward_beta, done_beta = env.env_step(action_beta, 1)
        step += 1

        logger.log_step(reward_alpha, None, None, reward_beta, None, None)
        if done_beta:
            break
    logger.log_episode()
    if e % 20 == 0:
        logger.record(episode=e, epsilon=1, step=step)