import torch
import numpy as np
from collections import deque
import random

import replay
import environments
import qnet
import slateq


class AgentAlpha(slateq.SlateQ):

    def __init__(self, user_features, doc_features, num_of_candidates, slate_size, batch_size, num_contex,
                 capacity=2000):
        self.user_features = user_features
        self.doc_features = doc_features
        self.num_of_candidates = num_of_candidates
        self.slate_size = slate_size
        self.batch_size = batch_size
        self.num_contex = num_contex

        self.state_dim = user_features + (
                doc_features * num_of_candidates + num_of_candidates) + num_contex * slate_size  # 状态中包含用户特征、文档特征(包括文档特征和文档ID)以及用户的上下文特征（也就是之前的反应）
        self.action_dim = slate_size

        self.response = deque(maxlen=num_contex)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = qnet.QNet(self.state_dim, self.num_of_candidates).to(self.device)

        self.replay = replay.ReplayMemory(capacity, (self.state_dim,), (self.action_dim,))

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.gamma = 0.9

    def compute_q_local_ini(self, env):
        for _ in range(self.num_contex):
            self.response.append(torch.zeros([self.slate_size], device=self.device))
        user, doc_id, doc_fea, click, engagement, reward, done = env.env_ini(0)
        state = torch.cat(
            [torch.tensor(user, device=self.device).view(-1), torch.tensor(doc_fea, device=self.device).view(-1),
             torch.tensor(doc_id, device=self.device).to(torch.float).view(-1)])
        for responses in self.response:
            state = torch.cat([state, responses.squeeze()])
        assert state.shape == torch.Size([self.state_dim])
        self.state = state
        return self.net(state, "online")

    def compute_q_local(self):
        return self.net.forward(self.state.view(1, -1), "online")


    def recommend(self, q_fed_alpha, env):
        user_obs = self.state[:self.user_features]
        doc_obs = self.state[self.user_features:(self.user_features + self.num_of_candidates * self.doc_features)]
        s, s_no_click = super().score_documents_torch(user_obs, doc_obs)
        slate = super().select_slate_greedy(s_no_click, s, q_fed_alpha)
        user, doc_id, doc_fea, click, engagement, reward, done = env.env_step(slate.cpu().numpy().tolist(), 0)
        self.response.append(torch.tensor(engagement, device=self.device))
        next_state = torch.cat(
            [torch.tensor(user, device=self.device).view(-1), torch.tensor(doc_fea, device=self.device).view(-1),
             torch.tensor(doc_id, device=self.device).to(torch.float).view(-1)])
        for responses in self.response:
            next_state = torch.cat([next_state, responses.view(-1)])

        self.replay.push(self.state.view(-1), slate.view(-1), torch.tensor(reward, device=self.device).view(-1),
                         torch.tensor(click, device=self.device),
                         next_state.squeeze(), torch.tensor(done, device=self.device).view(-1))
        self.state = next_state

        return done, reward

    def recommend_random(self, env):
        nums = list(range(self.num_of_candidates))
        random.shuffle(nums)
        slate = nums[:self.slate_size]
        user, doc_id, doc_fea, click, engagement, reward, done = env.env_step(slate, 0)
        self.response.append(torch.tensor(engagement, device=self.device))
        next_state = torch.cat(
            [torch.tensor(user, device=self.device).view(-1), torch.tensor(doc_fea, device=self.device).view(-1),
             torch.tensor(doc_id, device=self.device).to(torch.float).view(-1)])
        for responses in self.response:
            next_state = torch.cat([next_state, responses.squeeze()])

        self.replay.push(self.state.squeeze(), torch.tensor(slate, device=self.device), torch.tensor(reward, device=self.device).view(1),
                         torch.tensor(click, device=self.device),
                         next_state.squeeze(), torch.tensor(done, device=self.device).view(1))
        self.state = next_state
        return done, reward



    def compute_q_local_batch(self, ids):
        self.batch_states, self.batch_actions, self.batch_rewards, self.batch_clicks, self.batch_next_states, self.batch_terminals = self.replay.recall(
            ids)
        return self.net.forward(self.batch_states, "online"), self.net.forward(self.batch_next_states, "target")

    def update_q_net(self, q, q_next, agent_fed):

        assert q.shape == torch.Size([self.batch_size, self.num_of_candidates])
        assert q_next.shape == torch.Size([self.batch_size, self.num_of_candidates])

        doc_id = self.batch_states[:, (self.user_features + self.num_of_candidates * self.doc_features):(
                self.user_features + self.num_of_candidates * self.doc_features + self.num_of_candidates)]
        assert doc_id.shape == torch.Size([self.batch_size, self.num_of_candidates])
        selected_item = self.batch_actions * self.batch_clicks
        selected_item = selected_item.type(torch.int)
        assert selected_item.shape == torch.Size([self.batch_size, self.slate_size])
        selected_item = torch.sum(selected_item, dim=1, keepdim=True)
        q = torch.gather(q, 1, selected_item)
        q_next = super().compute_target_greedy_q(self.batch_rewards, self.gamma, q_next, self.batch_next_states,
                                                 self.batch_terminals)
        loss = self.loss_fn(q.view(self.batch_size,1), q_next.view(self.batch_size,1))
        agent_fed.optimizer.zero_grad()
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        agent_fed.optimizer.step()
        self.optimizer.step()
        return loss, q_next


class AgentBeta(slateq.SlateQ):

    def __init__(self, user_features, doc_features, num_of_candidates, slate_size, batch_size, capacity=2000):
        self.user_features = user_features
        self.doc_features = doc_features
        self.num_of_candidates = num_of_candidates
        self.slate_size = slate_size
        self.batch_size = batch_size

        self.state_dim = user_features + (
                doc_features * num_of_candidates + num_of_candidates)
        self.action_dim = slate_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = qnet.QNet(self.state_dim, self.num_of_candidates).to(self.device)

        self.replay = replay.ReplayMemory(capacity, (self.state_dim,), (self.action_dim,))

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.gamma = 0.9

    def compute_q_local_ini(self, env):
        user, doc_id, doc_fea, click, engagement, reward, done = env.env_ini(1)
        state = torch.cat(
            [torch.tensor(user, device=self.device).view(-1), torch.tensor(doc_fea, device=self.device).view(-1),
             torch.tensor(doc_id, device=self.device).to(torch.float).view(-1)])
        assert state.shape == torch.Size([self.state_dim])
        self.state = state
        return self.net(state, "online")

    def compute_q_local(self):
        return self.net.forward(self.state.view(1, -1), "online")

    def recommend(self, q_fed_beta, env):
        user_obs = self.state[:self.user_features]
        doc_obs = self.state[self.user_features:(self.user_features + self.num_of_candidates * self.doc_features)]
        s, s_no_click = super().score_documents_torch(user_obs, doc_obs)
        slate = super().select_slate_greedy(s_no_click, s, q_fed_beta)
        user, doc_id, doc_fea, click, engagement, reward, done = env.env_step(slate.cpu().numpy().tolist(), 1)
        next_state = torch.cat(
            [torch.tensor(user, device=self.device).view(-1), torch.tensor(doc_fea, device=self.device).view(-1),
             torch.tensor(doc_id, device=self.device).to(torch.float).view(-1)])
        self.replay.push(self.state.view(-1), slate.view(-1), torch.tensor(reward, device=self.device).view(-1),
                         torch.tensor(click, device=self.device),
                         next_state.squeeze(), torch.tensor(done, device=self.device).view(-1))

        self.state = next_state
        return done, reward

    def recommend_random(self, env):
        nums = list(range(self.num_of_candidates))
        random.shuffle(nums)
        slate = nums[:self.slate_size]

        user, doc_id, doc_fea, click, engagement, reward, done = env.env_step(slate, 1)
        next_state = torch.cat(
            [torch.tensor(user, device=self.device).view(-1), torch.tensor(doc_fea, device=self.device).view(-1),
             torch.tensor(doc_id, device=self.device).to(torch.float).view(-1)])
        self.replay.push(self.state.squeeze(), torch.tensor(slate, device=self.device), torch.tensor(reward, device=self.device).view(1),
                         torch.tensor(click, device=self.device),
                         next_state.squeeze(), torch.tensor(done, device=self.device).view(1))

        self.state = next_state
        return done, reward

    def compute_q_local_batch(self, ids):

        self.batch_states, self.batch_actions, _, _, _, _ = self.replay.recall(
            ids)
        return self.net.forward(self.batch_states, "online")

    def update_q_net(self, q_online, q_target, agent_fed):
        assert q_online.shape == torch.Size([self.batch_size, self.num_of_candidates])

        q_target = q_target.view(self.batch_size, 1)
        user_obs = self.batch_states[:, :self.user_features]
        doc_obs = self.batch_states[:,
                  self.user_features:(self.user_features + self.num_of_candidates * self.doc_features)]

        assert user_obs.shape == torch.Size([self.batch_size, self.user_features])
        assert doc_obs.shape == torch.Size([self.batch_size, self.num_of_candidates])

        greedy_q_list = []
        for i in range(self.batch_size):
            s, s_no_click = super().score_documents_torch(user_obs[i], doc_obs[i])
            q = q_online[i]
            slate = super().select_slate_greedy(s_no_click, s, q)
            p_selected = super().compute_probs_torch(slate, s, s_no_click)
            q_selected = torch.gather(q, 0, slate)
            greedy_q_list.append(
                torch.sum(input=p_selected * q_selected)
            )

        greedy_q_values = torch.stack(greedy_q_list)
        greedy_q_values = greedy_q_values.view(self.batch_size, 1)

        loss = self.loss_fn(greedy_q_values.view(self.batch_size,1), q_target.view(self.batch_size,1))
        agent_fed.optimizer.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        agent_fed.optimizer.step()
        self.optimizer.step()
        return loss


class AgentFed():

    def __init__(self, user_features, doc_features, num_of_candidates, slate_size, batch_size, capacity=2000):

        self.user_features = user_features
        self.doc_features = doc_features
        self.num_of_candidates = num_of_candidates
        self.slate_size = slate_size
        self.batch_size = batch_size
        self.capacity = capacity

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99995
        self.exploration_rate_min = 0.
        self.burnin = 5000  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 500  # no. of experiences between Q_target & Q_online sync
        self.curr_step = 0

        self.net = qnet.MLPNet(num_of_candidates * 2, num_of_candidates).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)

    def sync(self, agent_alpha, agent_beta):
        self.net.target.load_state_dict(self.net.online.state_dict())
        agent_alpha.net.target.load_state_dict(agent_alpha.net.online.state_dict())
        agent_beta.net.target.load_state_dict(agent_beta.net.online.state_dict())

    def act_ini(self, agent_alpha, agent_beta, env):
        q_alpha = agent_alpha.compute_q_local_ini(env).view(1, -1)
        q_beta = agent_beta.compute_q_local_ini(env).view(1, -1)
        q_alpha_fed = self.net.forward(torch.cat([q_alpha, q_beta], dim=1), "online")
        q_beta_fed = self.net.forward(torch.cat([q_beta, q_alpha], dim=1), "online")
        if np.random.rand() < self.exploration_rate:
            done_alpha, reward_alpha = agent_alpha.recommend_random(env)
        else:
            done_alpha, reward_alpha = agent_alpha.recommend(q_alpha_fed, env)

        if np.random.rand() < self.exploration_rate:
            done_beta, reward_beta = agent_beta.recommend_random(env)
        else:
            done_beta, reward_beta = agent_beta.recommend(q_beta_fed, env)

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1

        return done_alpha, reward_alpha, done_beta, reward_beta

    def act(self, agent_alpha, agent_beta, env):
        q_alpha = agent_alpha.compute_q_local().view(1, -1)
        q_beta = agent_beta.compute_q_local().view(1, -1)
        q_alpha_fed = self.net.forward(torch.cat([q_alpha, q_beta], dim=1), "online")
        q_beta_fed = self.net.forward(torch.cat([q_beta, q_alpha], dim=1), "online")
        if np.random.rand() < self.exploration_rate:
            done_alpha, reward_alpha = agent_alpha.recommend_random(env)
        else:
            done_alpha, reward_alpha = agent_alpha.recommend(q_alpha_fed, env)

        if np.random.rand() < self.exploration_rate:
            done_beta, reward_beta = agent_beta.recommend_random(env)
        else:
            done_beta, reward_beta = agent_beta.recommend(q_beta_fed, env)

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        if self.exploration_rate < 0.1:
            self.exploration_rate = 0

        # increment step
        self.curr_step += 1

        return done_alpha, reward_alpha, done_beta, reward_beta

    def learn(self, agent_alpha, agent_beta):

        if self.curr_step % self.sync_every == 0:
            self.sync(agent_alpha, agent_beta)

        if self.curr_step < self.burnin:
            return None, None, None, None

        if self.curr_step % self.learn_every != 0:
            return None, None, None, None

        nums = list(range(self.capacity))
        random.shuffle(nums)
        ids = nums[:self.batch_size]


        batch_q_alpha_online, batch_q_alpha_target = agent_alpha.compute_q_local_batch(ids)
        batch_q_beta_online = agent_beta.compute_q_local_batch(ids)

        q_alpha_fed_online = self.net(torch.cat([batch_q_alpha_online, batch_q_beta_online], dim=1), "online")
        q_alpha_fed_target = self.net(torch.cat([batch_q_alpha_target, batch_q_beta_online], dim=1), "target")

        loss_alpha, q_alpha_target = agent_alpha.update_q_net(q_alpha_fed_online, q_alpha_fed_target, self)
        batch_q_alpha_online_new, _ = agent_alpha.compute_q_local_batch(ids)
        q_beta_fed_online = self.net(torch.cat([batch_q_beta_online, batch_q_alpha_online_new], dim=1), "online")
        loss_beta = agent_beta.update_q_net(q_beta_fed_online, q_alpha_target, self)
        return batch_q_alpha_online.detach().cpu().mean().item(), loss_alpha.detach().cpu(), batch_q_beta_online.detach().cpu().mean().item(), loss_beta.detach().cpu()



