import torch


class SlateQ():
    def __init__(self, user_features, doc_features, num_of_candidates, slate_size, batch_size):
        self.user_features = user_features
        self.num_of_candidates = num_of_candidates
        self.doc_features = doc_features
        self.slate_size = slate_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

    def score_documents_torch(self,
                              user_obs,
                              doc_obs,
                              no_click_mass=1.0,
                              is_mnl=True,
                              min_normalizer=-1.0):

        user_obs = user_obs.view(-1)
        doc_obs = doc_obs.view(-1)
        assert user_obs.shape == torch.Size([self.user_features])
        assert doc_obs.shape == torch.Size([self.num_of_candidates])

        scores = torch.sum(input=torch.mul(doc_obs.view(-1, 1),
                                           user_obs.view(1, -1)).view(self.num_of_candidates,
                                                                      self.user_features),
                           dim=1)
        all_scores = torch.cat([scores, torch.tensor([no_click_mass], device=self.device)], dim=0)
        if is_mnl:
            all_scores = torch.nn.functional.softmax(all_scores, dim=0)
        else:
            all_scores = all_scores - min_normalizer
        assert all_scores.shape == torch.Size([self.num_of_candidates + 1])
        return all_scores[:-1], all_scores[-1]

    def compute_probs_torch(self, slate, scores_torch, score_no_click_torch):

        slate = slate.squeeze()
        scores_torch = scores_torch.squeeze()

        assert slate.shape == torch.Size([self.slate_size])
        assert scores_torch.shape == torch.Size([self.num_of_candidates])
        all_scores = torch.cat([
            torch.gather(scores_torch, 0, slate).view(-1),
            score_no_click_torch.view(-1)
        ], dim=0)
        all_probs = all_scores / torch.sum(all_scores)
        assert all_probs.shape == torch.Size([self.slate_size + 1])
        return all_probs[:-1]

    def select_slate_greedy(self, s_no_click, s, q):

        s = s.view(-1)
        q = q.view(-1)
        assert s.shape == torch.Size([self.num_of_candidates])
        assert q.shape == torch.Size([self.num_of_candidates])

        def argmax(v, mask_inner):
            return torch.argmax((v - torch.min(v) + 1) * mask_inner, dim=0)

        numerator = torch.tensor(0., device=self.device)
        denominator = torch.tensor(0., device=self.device) + s_no_click
        mask_inner = torch.ones(q.size(0), device=self.device)

        def set_element(v, i, x):
            mask_inner = torch.nn.functional.one_hot(i, v.shape[0])
            v_new = torch.ones_like(v) * x
            return torch.where(mask_inner == 1, v_new, v)

        for _ in range(self.slate_size):
            k = argmax((numerator + s * q) / (denominator + s), mask_inner)
            mask_inner = set_element(mask_inner, k, 0)
            numerator = numerator + torch.gather(s * q, 0, k)
            denominator = denominator + torch.gather(s, 0, k)

        output_slate = torch.where(mask_inner == 0)[0].squeeze()
        assert output_slate.shape == torch.Size([self.slate_size])
        return output_slate

    def compute_target_greedy_q(self,
                                reward,
                                gamma,
                                next_q_values,
                                next_states,
                                terminals):

        assert reward.shape == torch.Size([self.batch_size])
        assert next_q_values.shape == torch.Size([self.batch_size, self.num_of_candidates])


        next_user_obs = next_states[:, :self.user_features]
        next_doc_obs = next_states[:, self.user_features:(self.user_features + self.num_of_candidates * self.doc_features)]

        assert next_user_obs.shape == torch.Size([self.batch_size, self.user_features])
        assert next_doc_obs.shape == torch.Size([self.batch_size, self.num_of_candidates])

        next_greedy_q_list = []
        for i in range(self.batch_size):
            s, s_no_click = self.score_documents_torch(next_user_obs[i], next_doc_obs[i])
            q = next_q_values[i]

            slate = self.select_slate_greedy(s_no_click, s, q)
            p_selected = self.compute_probs_torch(slate, s, s_no_click)
            q_selected = torch.gather(q, 0, slate)
            next_greedy_q_list.append(
                torch.sum(input=p_selected * q_selected)
            )

        next_greedy_q_values = torch.stack(next_greedy_q_list)

        target_q_values = reward + gamma * next_greedy_q_values * (
                1. - terminals.float()
        )
        assert target_q_values.shape == torch.Size([self.batch_size])
        return target_q_values
