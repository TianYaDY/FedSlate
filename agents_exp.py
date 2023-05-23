import torch
from collections import deque
import random

import replay
import qnet
import slateq
import agents


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
        user, doc_id, doc_fea, click, engagement, reward, done = env.env_ini(1)
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
        user, doc_id, doc_fea, click, engagement, reward, done = env.env_step(slate.cpu().numpy().tolist(), 1)
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
        user, doc_id, doc_fea, click, engagement, reward, done = env.env_step(slate, 1)
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


class AgentFed(agents.AgentFed):

    def __init__(self, user_features, doc_features, num_of_candidates, slate_size, batch_size, capacity=2000):
        super().__init__(user_features, doc_features, num_of_candidates, slate_size, batch_size, capacity)

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
        batch_q_beta_online, batch_q_beta_target = agent_beta.compute_q_local_batch(ids)

        q_alpha_fed_online = self.net(torch.cat([batch_q_alpha_online, batch_q_beta_online], dim=1), "online")
        q_alpha_fed_target = self.net(torch.cat([batch_q_alpha_target, batch_q_beta_online], dim=1), "target")

        loss_alpha, q_alpha_target = agent_alpha.update_q_net(q_alpha_fed_online, q_alpha_fed_target, self)
        batch_q_alpha_online_new, batch_q_alpha_target_new = agent_alpha.compute_q_local_batch(ids)
        q_beta_fed_online = self.net(torch.cat([batch_q_beta_online, batch_q_alpha_target_new], dim=1), "online")
        q_beta_fed_target = self.net(torch.cat([batch_q_beta_target, batch_q_alpha_target_new], dim=1), "target")
        loss_beta, _ = agent_beta.update_q_net(q_beta_fed_online, q_beta_fed_target, self)
        return batch_q_alpha_online.detach().cpu().mean().item(), loss_alpha.detach().cpu(), batch_q_beta_online.detach().cpu().mean().item(), loss_beta.detach().cpu()
