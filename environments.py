import numpy as np
from gym import spaces
from scipy import stats

from recsim import document
from recsim import user
from recsim.choice_model import MultinomialLogitChoiceModel
from recsim.simulator import environment
from recsim.simulator import recsim_gym


class LTSDocument(document.AbstractDocument):
    def __init__(self, doc_id, kaleness):
        self.kaleness = kaleness
        # doc_id is an integer representing the unique ID of this document
        super(LTSDocument, self).__init__(doc_id)

    def create_observation(self):
        return np.array([self.kaleness])

    @staticmethod
    def observation_space():
        return spaces.Box(shape=(1,), dtype=np.float32, low=0.0, high=1.0)

    def __str__(self):
        return "Document {} with kaleness {}.".format(self._doc_id, self.kaleness)


class LTSDocumentSampler(document.AbstractDocumentSampler):
    def __init__(self, seed, doc_ctor=LTSDocument, **kwargs):
        super(LTSDocumentSampler, self).__init__(doc_ctor, **kwargs)
        self._doc_count = 0
        self.seed = seed
        self._rng = np.random.RandomState(self.seed)



    def sample_document(self):
        doc_features = {}
        doc_features['doc_id'] = self._doc_count
        doc_features['kaleness'] = self._rng.random_sample()
        self._doc_count += 1
        return self._doc_ctor(**doc_features)


class LTSUserState(user.AbstractUserState):
    def __init__(self, memory_discount, sensitivity, innovation_stddev,
                 choc_mean, choc_stddev, kale_mean, kale_stddev,
                 net_kaleness_exposure, time_budget, observation_noise_stddev=0.1
                 ):
        ## Transition model parameters
        ##############################
        self.memory_discount = memory_discount
        self.sensitivity = sensitivity
        self.innovation_stddev = innovation_stddev

        ## Engagement parameters
        self.choc_mean = choc_mean
        self.choc_stddev = choc_stddev
        self.kale_mean = kale_mean
        self.kale_stddev = kale_stddev

        ## State variables
        ##############################
        self.net_kaleness_exposure = net_kaleness_exposure
        self.satisfaction = 1 / (1 + np.exp(-sensitivity * net_kaleness_exposure))
        self.time_budget = time_budget

        # Noise
        self._observation_noise = observation_noise_stddev

    def create_observation(self):
        """User's state is not observable."""
        clip_low, clip_high = (-1.0 / (1.0 * self._observation_noise),
                               1.0 / (1.0 * self._observation_noise))
        noise = stats.truncnorm(
            clip_low, clip_high, loc=0.0, scale=self._observation_noise).rvs()
        noisy_sat = self.satisfaction + noise
        return np.array([noisy_sat, ])

    @staticmethod
    def observation_space():
        return spaces.Box(shape=(1,), dtype=np.float32, low=-2.0, high=2.0)

    # scoring function for use in the choice model -- the user is more likely to
    # click on more chocolatey content.
    def score_document(self, doc_obs):
        return 1 - doc_obs


class LTSStaticUserSampler(user.AbstractUserSampler):
    _state_parameters = None

    def __init__(self,
                 user_ctor=LTSUserState,
                 memory_discount=0.9,
                 sensitivity=0.01,
                 innovation_stddev=0.05,
                 choc_mean=5.0,
                 choc_stddev=1.0,
                 kale_mean=4.0,
                 kale_stddev=1.0,
                 time_budget=122,
                 **kwargs):
        self._state_parameters = {'memory_discount': memory_discount,
                                  'sensitivity': sensitivity,
                                  'innovation_stddev': innovation_stddev,
                                  'choc_mean': choc_mean,
                                  'choc_stddev': choc_stddev,
                                  'kale_mean': kale_mean,
                                  'kale_stddev': kale_stddev,
                                  'time_budget': time_budget
                                  }
        super(LTSStaticUserSampler, self).__init__(user_ctor, **kwargs)

    def sample_user(self):
        starting_nke = ((self._rng.random_sample() - .5) *
                        (1 / (1.0 - self._state_parameters['memory_discount'])))
        self._state_parameters['net_kaleness_exposure'] = starting_nke
        return self._user_ctor(**self._state_parameters)


class LTSResponse(user.AbstractResponse):
    # The maximum degree of engagement.
    MAX_ENGAGEMENT_MAGNITUDE = 100.0

    def __init__(self, clicked=False, engagement=0.0):
        self.clicked = clicked
        self.engagement = engagement

    def create_observation(self):
        return {'click': int(self.clicked), 'engagement': np.array(self.engagement)}

    @classmethod
    def response_space(cls):
        # `engagement` feature range is [0, MAX_ENGAGEMENT_MAGNITUDE]
        return spaces.Dict({
            'click':
                spaces.Discrete(2),
            'engagement':
                spaces.Box(
                    low=0.0,
                    high=cls.MAX_ENGAGEMENT_MAGNITUDE,
                    shape=tuple(),
                    dtype=np.float32)
        })


def user_init(self,
              slate_size,
              seed=0):
    super(LTSUserModel,
          self).__init__(LTSResponse,
                         LTSStaticUserSampler(LTSUserState,
                                              seed=seed), slate_size)
    self.choice_model = MultinomialLogitChoiceModel({})


def simulate_response(self, slate_documents):
    # List of empty responses
    responses = [self._response_model_ctor() for _ in slate_documents]
    # Get click from of choice model.
    self.choice_model.score_documents(
        self._user_state, [doc.create_observation() for doc in slate_documents])
    scores = self.choice_model.scores
    selected_index = self.choice_model.choose_item()
    # Populate clicked item.
    self._generate_response(slate_documents[selected_index],
                            responses[selected_index])
    return responses


def generate_response(self, doc, response):
    response.clicked = True
    # linear interpolation between choc and kale.
    engagement_loc = (doc.kaleness * self._user_state.choc_mean
                      + (1 - doc.kaleness) * self._user_state.kale_mean)
    engagement_loc *= self._user_state.satisfaction
    engagement_scale = (doc.kaleness * self._user_state.choc_stddev
                        + ((1 - doc.kaleness)
                           * self._user_state.kale_stddev))
    log_engagement = np.random.normal(loc=engagement_loc,
                                      scale=engagement_scale)
    response.engagement = np.exp(log_engagement)


def update_state(self, slate_documents, responses):
    for doc, response in zip(slate_documents, responses):
        if response.clicked:
            innovation = np.random.normal(scale=self._user_state.innovation_stddev)
            net_kaleness_exposure = (self._user_state.memory_discount
                                     * self._user_state.net_kaleness_exposure
                                     - 2.0 * (doc.kaleness - 0.5)
                                     + innovation
                                     )
            self._user_state.net_kaleness_exposure = net_kaleness_exposure
            satisfaction = 1 / (1.0 + np.exp(-self._user_state.sensitivity
                                             * net_kaleness_exposure)
                                )
            self._user_state.satisfaction = satisfaction
            self._user_state.time_budget -= 1
            return


def is_terminal(self):
    """Returns a boolean indicating if the session is over."""
    return self._user_state.time_budget <= 0


def clicked_engagement_reward(responses):
    reward = 0.0
    for response in responses:
        if response.clicked:
            reward += response.engagement
    return reward


LTSUserModel = type("LTSUserModel", (user.AbstractUserModel,),
                    {"__init__": user_init,
                     "is_terminal": is_terminal,
                     "update_state": update_state,
                     "simulate_response": simulate_response,
                     "_generate_response": generate_response})


#  模拟环境，需要在创建时输入slate大小，候选文档数量，是否重新采样
class RecsimEnv():
    def __init__(self, num_candidates, slate_size, resample_documents, env_seed_0, env_seed_1):
        assert num_candidates >= slate_size
        self.num_candidates = num_candidates
        self.slate_size = slate_size
        self.resample_documents = resample_documents
        #  创建两个文档采样器对象
        self.doc_model_1 = LTSDocumentSampler(env_seed_0)
        self.doc_model_2 = LTSDocumentSampler(env_seed_1)

        #  创建一个用户采样器对象，以便共享用户行为
        self.user_model = LTSUserModel(slate_size)

        #  创建两个不同的环境，但同属于一个用户
        self.env_0 = environment.Environment(
            self.user_model,
            self.doc_model_1,
            num_candidates,
            slate_size,
            resample_documents,

        )
        self.env_1 = environment.Environment(
            self.user_model,
            self.doc_model_2,
            num_candidates,
            slate_size,
            resample_documents)
        self.lts_gym_env_0 = recsim_gym.RecSimGymEnv(self.env_0, clicked_engagement_reward)
        self.lts_gym_env_1 = recsim_gym.RecSimGymEnv(self.env_1, clicked_engagement_reward)

    def env_ini(self, id):
        if id == 0:
            output = self.lts_gym_env_0.reset()
        elif id == 1:
            output = self.lts_gym_env_1.reset()
        else:
            raise ValueError(f"Invalid id {id}.")
        user, doc, response = output.values()
        doc_id = np.array(list(doc.keys())).astype(int)
        doc_fea = np.array(list(doc.values()))
        click = np.zeros([self.slate_size], dtype=int)
        engagement = np.zeros([self.slate_size])
        reward = np.array(0.)
        done = False
        return user.astype(np.float32), doc_id.astype(np.float32), doc_fea.astype(np.float32), click.astype(np.float32), engagement.astype(np.float32), reward.astype(np.float32), done

    def env_step(self, slate, id):
        if id == 0:
            output = self.lts_gym_env_0.step(slate)
        elif id == 1:
            output = self.lts_gym_env_1.step(slate)
        else:
            raise ValueError(f"Invalid id {id}.")
        user, doc, response = output[0].values()
        reward = np.array(output[1])
        doc_id = np.array(list(doc.keys())).astype(int)
        doc_fea = np.array(list(doc.values()))
        click = np.array(list(item['click'] for item in response))
        engagement = np.array(list(item['engagement'] for item in response))
        done = output[2]
        return user.astype(np.float32), doc_id.astype(np.float32), doc_fea.astype(np.float32), click.astype(np.float32), engagement.astype(np.float32), reward.astype(np.float32), done


if __name__ == "__main__":
    np.random.seed(42)
    env_test = RecsimEnv(10, 3, True)
    test_input = [0, 1, 2]
    print(env_test.env_ini(0))
    print(env_test.env_step(test_input, 0))
    print(env_test.env_ini(1))
    print(env_test.env_step(test_input, 1))

