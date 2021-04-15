import gym
from gym.core import ObservationWrapper
from gym.spaces import Box
import numpy as np
import torch
import torch.nn as nn


class PreprocessiObs(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.angle_values = [1, 6]
        self.coordinates_values = [0, 5]
        self.state_size = (1, self.observation(self.env.get_obs).shape[0])
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.state_size)
        self.framebuffer = np.zeros(self.state_size[1], 'float32')


    def observation(self, obs):
        new_obs = np.empty(0)
        for i, elem in enumerate(obs):
            if i in self.coordinates_values:
                elem = np.round(elem, -1)

            if i in self.angle_values:
                elem = np.array(list(map(self.transform_to_trigonometry, elem))).reshape(-1)

            new_obs = np.hstack((new_obs, elem))  # Add embedings for manouver

        return new_obs.astype('float32')

    def transform_to_trigonometry(self, angle):
        return [np.sin(angle), np.cos(angle)]

    def reset(self, **info):
        return self.observation(self.env.reset(**info))
        # self.framebuffer = np.concatenate()


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.const_neurons = 1248

        self.dense1 = nn.Linear(in_features=state_shape[1], out_features=self.const_neurons)
        self.relu1 = nn.LeakyReLU()
        self.dense2 = nn.Linear(in_features=self.const_neurons, out_features=self.const_neurons)
        self.relu2 = nn.LeakyReLU()
        self.dense3 = nn.Linear(in_features=self.const_neurons, out_features=self.const_neurons)
        self.relu3 = nn.LeakyReLU()
        self.dense4 = nn.Linear(in_features=self.const_neurons, out_features=self.const_neurons)
        self.relu4 = nn.LeakyReLU()
        self.dense5 = nn.Linear(in_features=self.const_neurons, out_features=self.const_neurons)
        self.relu5 = nn.LeakyReLU()
        self.dense6 = nn.Linear(in_features=self.const_neurons, out_features=self.n_actions)

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        qvalues = self.dense1(state_t)
        qvalues = self.relu1(qvalues)
        qvalues = self.dense2(qvalues)
        qvalues = self.relu2(qvalues)
        qvalues = self.dense3(qvalues)
        qvalues = self.relu3(qvalues)
        qvalues = self.dense4(qvalues)
        qvalues = self.relu4(qvalues)
        qvalues = self.dense5(qvalues)
        qvalues = self.relu5(qvalues)
        qvalues = self.dense6(qvalues)

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


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000, **init_params):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset(**init_params)
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmin(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)
    return np.mean(rewards)


def play_and_record(initial_state, agent, env, exp_replay, n_steps=1, expert=False):
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

    # Play the game for n_steps as per instructions above
    for _ in range(n_steps):

        if expert:
            env.wrap.rocket.grav_compensate()
            overload = env.wrap.rocket.proportionalCoefficients(k_z=2, k_y=2)
            possible = env.wrap.findClosestFromLegal(overload)
            action = env.wrap.overloadsToNumber([possible])[0]
        else:
            qvalues = agent.get_qvalues([s])
            action = agent.sample_actions(qvalues=qvalues)[0]

        _s, r, done, info = env.step(action)

        # if done and not info["Destroyed"]:
        #     r += 200

        reward += r
        exp_replay.add(s, action, r, _s, done)
        s = _s

        if done:
            s = env.reset(**initial_state)

    return reward, s

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
