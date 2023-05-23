import torch
import datetime
from pathlib import Path

import environments
import agents
import log

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

user_features = 1
doc_features = 1
num_of_candidates = 10
slate_size = 3
batch_size = 32
num_contex = 5

save_dir = Path("checkpoints_fed") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

logger = log.Logger(save_dir)

agent_alpha = agents.AgentAlpha(user_features, doc_features, num_of_candidates, slate_size, batch_size, num_contex)
agent_beta = agents.AgentBeta(user_features, doc_features, num_of_candidates, slate_size, batch_size)
agent_fed = agents.AgentFed(user_features, doc_features, num_of_candidates, slate_size, batch_size)

env = environments.RecsimEnv(num_of_candidates, slate_size, True, 42, 42)

episodes = 5000
for e in range(episodes):
    agent_fed.act_ini(agent_alpha, agent_beta, env)
    while True:
        done_alpha, reward_alpha, done_beta, reward_beta = agent_fed.act(agent_alpha, agent_beta, env)
        q_alpha, loss_alpha, q_beta, loss_beta = agent_fed.learn(agent_alpha, agent_beta)

        logger.log_step(reward_alpha, loss_alpha, q_alpha, reward_beta, loss_beta, q_beta)
        if done_beta:
            break
    logger.log_episode()
    if e % 20 == 0:
        logger.record(episode=e, epsilon=agent_fed.exploration_rate, step=agent_fed.curr_step)
