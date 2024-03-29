import os
import numpy as np
import torch as torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir, main_stream, advantage_stream, value_stream):
        super(DuelingDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.n_neurons = 16

        self.fake_conv = nn.Sequential(
            nn.Linear(*input_dims, main_stream[0].in_features),
            nn.ReLU(),
            *main_stream
        )

        value_last_neurons = advantage_stream[-2].out_features if len(advantage_stream) else \
            main_stream[-2].out_features

        self.value_stream = nn.Sequential(
            *advantage_stream,
            nn.Linear(value_last_neurons, 1),
        )

        advantage_last_neurons = value_stream[-2].out_features if len(value_stream) else main_stream[-2].out_features
        self.advantage_stream = nn.Sequential(
            *value_stream,
            nn.Linear(advantage_last_neurons, n_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        res = self.fake_conv(state)

        V = self.value_stream(res)
        A = self.advantage_stream(res)

        return V, A

    def save_checkpoint(self, filename):
        print('... saving checkpoint ...')
        if not filename:
            torch.save(self.state_dict(), os.path.join(os.getcwd(), self.checkpoint_file))
        else:
            torch.save(self.state_dict(), os.path.join(os.getcwd(), filename))

    def load_checkpoint(self, filename):
        print('... loading checkpoint ...')
        if not filename:
            self.load_state_dict(torch.load(self.checkpoint_file))
        else:
            self.load_state_dict(torch.load(filename))


class Agent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, main_stream, advantage_stream, value_stream,
                 eps_min=0.01, eps_dec=5e-7,
                 replace=1000, chkpt_dir='tmp/dueling_ddqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims)

        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name='lunar_lander_dueling_ddqn_q_eval',
                                          chkpt_dir=self.chkpt_dir,
                                          main_stream=main_stream, advantage_stream=advantage_stream,
                                          value_stream=value_stream)

        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name='lunar_lander_dueling_ddqn_q_next',
                                          chkpt_dir=self.chkpt_dir,
                                          main_stream=main_stream, advantage_stream=advantage_stream,
                                          value_stream=value_stream)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation], dtype=torch.float).to(self.q_eval.device)
            value, advantage = self.q_eval.forward(state)
            action = torch.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def save_models(self, filename=""):
        self.q_eval.save_checkpoint(filename + "_q_eval")
        self.q_next.save_checkpoint(filename + "_q_next")

    def load_models(self, filename=""):
        self.q_eval.load_checkpoint(filename + "_q_eval")
        self.q_next.load_checkpoint(filename + "_q_next")

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return None

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = torch.tensor(state).to(self.q_eval.device)
        rewards = torch.tensor(reward).to(self.q_eval.device)
        dones = torch.tensor(done).to(self.q_eval.device)
        actions = torch.tensor(action).to(self.q_eval.device)
        states_ = torch.tensor(new_state).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = torch.add(V_s,
                           (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = torch.add(V_s_,
                           (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = torch.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = torch.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
        return loss.item()
