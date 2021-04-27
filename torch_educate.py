import gym
from gym.core import ObservationWrapper
from gym.spaces import Box
import numpy as np
import torch
import torch.nn as nn
from replay_buffer import ReplayBuffer


class PreprocessiObs(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.angle_values = [3, 4, 5, 29, 30, 31]
        self.coordinates_values = [0, 1, 2, 26, 27, 28]
        self.overloads = [7, 8, 9, 10, 11]  # 10,11 for navigation Ny, Nz (last values?)
        self.speed = [6, 32]  # 15, 16, 17 target speed projections
        self.distance = [22]
        self.for_overload = [23, 24]
        self.angle_to_target = [25]
        self.target_speed = [15, 16, 17]

        self.min_coor = -5000
        self.max_coor = 20000
        self.state_size = (1, self.observation(self.env.get_obs).shape[0])
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.state_size)
        self.framebuffer = np.zeros(self.state_size[1], 'float32')

    # np.hstack([np.copy(self._coord),
    #            np.copy(self._euler),
    #            self._speed,
    #            np.copy(self._overload),
    #            np.copy(self._current_overloads),
    #            np.copy(self.dataForNzPN())])

    def observation(self, obs):
        new_obs = np.empty(0)
        for i, elem in enumerate(obs):
            if i in self.coordinates_values:
                elem = np.round(elem, 0)/1000
                # elem = np.round((elem - self.min_coor)/(self.max_coor - self.min_coor), 5)
            elif i in self.overloads:
                continue
                elem = np.round(elem, 2)
            elif i in self.angle_values:
                elem = np.hstack([elem, np.round(np.array(self.transform_to_trigonometry(elem)), 4)])
            elif i in self.speed:
                elem = np.round(elem)/1000
            elif i in self.distance:
                continue
                elem = np.round(elem, 0)/1000
            elif i in self.angle_to_target:
                elem = np.round(elem, 3)
            elif i in self.target_speed:
                continue
                elem = elem / 1000
            elif i in self.for_overload:
                continue
                if i == 24:
                    elem = np.round(elem, 3)
                else:
                    continue
            else:
                continue

            new_obs = np.hstack((new_obs, elem))  # Add embedings for manouver

        return new_obs.astype('float32')

    def transform_to_trigonometry(self, angle):
        return [np.sin(angle), np.cos(angle)]

    def reset(self, **info):
        return self.observation(self.env.reset(**info))
        # self.framebuffer = np.concatenate()

class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, layers_lst, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape
        # self.const_neurons = 128#int(np.ceil(2 / 3 * state_shape[1] + self.n_actions))

        self.net = nn.Sequential(
            nn.Linear(in_features=state_shape[1], out_features=layers_lst[1].in_features),
            *layers_lst,
            nn.Linear(in_features=layers_lst[-2].out_features, out_features=self.n_actions)
        )

        # self.dense1 = nn.Linear(in_features=state_shape[1], out_features=self.const_neurons)
        # self.relu1 = nn.LeakyReLU()
        # self.dense2 = nn.Linear(in_features=self.const_neurons, out_features=self.const_neurons)
        # self.relu2 = nn.LeakyReLU()
        # self.dense3 = nn.Linear(in_features=self.const_neurons, out_features=self.const_neurons)
        # self.relu3 = nn.LeakyReLU()
        # self.dense4 = nn.Linear(in_features=self.const_neurons, out_features=self.const_neurons)
        # self.relu4 = nn.LeakyReLU()
        # self.dense5 = nn.Linear(in_features=self.const_neurons, out_features=self.const_neurons)
        # self.relu5 = nn.LeakyReLU()
        # self.dense6 = nn.Linear(in_features=self.const_neurons, out_features=self.n_actions)

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """


        # qvalues = self.dense1(state_t)
        # qvalues = self.relu1(qvalues)
        # qvalues = self.dense2(qvalues)
        # qvalues = self.relu2(qvalues)
        # qvalues = self.dense3(qvalues)
        # qvalues = self.relu3(qvalues)
        # qvalues = self.dense4(qvalues)
        # qvalues = self.relu4(qvalues)
        # qvalues = self.dense5(qvalues)
        # qvalues = self.relu5(qvalues)
        # qvalues = self.dense6(qvalues)

        qvalues = self.net.forward(state_t)

        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"
        # assert len(
        #     qvalues.shape) == 2 and qvalues.shape[0] == state_t.shape[0] and qvalues.shape[1] == self.n_actions

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        # FIXME: max -> min
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)

def evaluate(env, agent, init_params, n_games=1, greedy=False, t_max=2000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    infos = []
    for _ in range(n_games):
        s = env.reset(**init_params())
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]  # FIXME: max -> min
            s, r, done, info = env.step(action)
            reward += r
            if done:
                print(info)
                infos.append(info["Distance"])
                break

        rewards.append(reward)
    print("Mean distance = ", np.mean(infos))
    return np.mean(rewards)


def play_and_record(initial_state, agent, env, exp_replay, n_steps=1, expert=False, prob_exp_random=0):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    # s = env.framebuffer add buffer
    # s = env.reset(**initial_state)
    s = env.observation(env.get_obs)
    reward = 0
    # temp_replay = ReplayBuffer(10**3)
    over = []
    r_l = []
    # rew = []
    # st = []
    # q = 0
    # Play the game for n_steps as per instructions above
    for i in range(n_steps):
        # q += 1
        if expert:
            env.wrap.rocket.grav_compensate()
            overload = env.wrap.rocket.proportionalCoefficients(k_z=2, k_y=2)
            over.append(overload)
            possible = env.wrap.findClosestFromLegal(overload)
            action = env.wrap.overloadsToNumber([possible])[0]
            action = action if np.random.uniform(0, 1) > prob_exp_random else np.random.choice(env.action_space.n)
        else:
            overload = env.wrap.rocket.proportionalCoefficients(k_z=2, k_y=2)
            qvalues = agent.get_qvalues([s])
            action = agent.sample_actions(qvalues=qvalues)[0]

        _s, r, done, info = env.step(action)

        reward += r
        r_l.append(r)

        exp_replay.add(s, action, r, _s, done)

        s = _s

        if done:
            s = env.reset(**initial_state())
            # print(len(r_l), np.sum(r_l), info)
            # print(reward)
            # r_l = []
            reward = 0
            # break

    return reward, s

            # z = {"r_euler": [0, np.random.uniform(np.deg2rad(-5), np.deg2rad(5)), 0],
            #      "t_euler": [0, np.random.uniform(np.deg2rad(-90), np.deg2rad(90)), 0]}
            # s = env.reset(**z)
            # print(q, info, reward)
            # rew.append(reward)
            # st.append(q)
            # q = 0
            # reward = 0




def compute_td_loss(states, actions, rewards, next_states, is_done,
                    agent, target_network,
                    gamma=0.99,
                    check_shapes=False,
                    device=torch.device('cpu')):
    """ Compute td loss using torch operations only. Use the formulae above. """
    states = torch.tensor(states, device=device, dtype=torch.float)  # shape: [batch_size, *state_shape]

    # for some torch reason should not make actions a tensor
    actions = torch.tensor(actions, device=device, dtype=torch.long)  # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float
    )  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)

    # compute q-values for all actions in next states
    predicted_next_qvalues = target_network(next_states)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(
        len(actions)), actions]

    # compute V*(next_states) using predicted next q-values
    # FIXME: maybe not working max -> min
    next_state_values, _ = torch.max(predicted_next_qvalues, dim=1)

    assert next_state_values.dim(
    ) == 1 and next_state_values.shape[0] == states.shape[0], "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    target_qvalues_for_actions = rewards + gamma * next_state_values * is_not_done

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions -
                       target_qvalues_for_actions.detach()) ** 2)

    if check_shapes:
        assert predicted_next_qvalues.data.dim(
        ) == 2, "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim(
        ) == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim(
        ) == 1, "there's something wrong with target q-values, they must be a vector"

    return loss
