import gym
import numpy as np
import matplotlib.pyplot as plt
from torch_educate import PreprocessiObs, DQNAgent, evaluate, play_and_record, compute_td_loss
import torch
from replay_buffer import ReplayBuffer
import random
from IPython.display import clear_output
import utils
from tqdm import trange
import torch.nn as nn
from additional import graph
from draw_animation import drawAnimation
import sys
import os
import missile_env

seed = 43
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

rocket_info = [np.array([0, 0, 0]), 900, np.deg2rad([0, 0, 0])]
target_info = [np.array([18000, 0, 0]), 200, np.deg2rad([0, 0, 0])]

ENV_NAME = "missile_env:missile-env-v0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_env(seed, rocket_info, target_info):
    env = gym.make(ENV_NAME, rocket_info=rocket_info, target_info=target_info)  # create raw env
    if seed is not None:
        env.seed(seed)
    env = PreprocessiObs(env)
    return env


"""## Main loop
"""


def get_rand(min, max):
    return np.random.uniform(np.deg2rad(min), np.deg2rad(max))


def get_rand_ini():
    return {"r_euler": [0, get_rand(0, 0), 0],
            "t_euler": [0, get_rand(-90, 90), 0]}


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


@static_vars(ini_list=np.empty(0))
def reset_params(count=10000):
    # return {"r_euler": [0, np.deg2rad(-20), 0]}
    try:
        ini = reset_params.ini_list[0]
    except IndexError:
        reset_params.ini_list = np.hstack([reset_params.ini_list, [get_rand_ini() for _ in range(count)]])
        ini = reset_params.ini_list[0]

    reset_params.ini_list = np.delete(reset_params.ini_list, 0)
    return ini

    # return #{"la_coord": [np.random.uniform(-250, 250), np.random.uniform(-250, 250), np.random.uniform(-250, 250)]}


""" Expert play """

# exp_replay = ReplayBuffer(20*10 ** 4)
# for i in range(3000):
#     """ Ram consumable maybe check for available RAM """
#     play_and_record(initial_state=reset_params,
#                     agent=agent, env=env, exp_replay=exp_replay, n_steps=800, expert=False, prob_exp_random=0)
#     if len(exp_replay) == 2*10 ** 4:
#         break
#
# debug = True
# loss_log = []
# rew_log = []
#
# for i in range(100000000):
#     # train
#     play_and_record(initial_state=reset_params(),
#                     agent=agent, env=env, exp_replay=exp_replay, n_steps=50, expert=True, prob_exp_random=0)
#
#     s_, a_, r_, next_s_, done_ = exp_replay.sample(256)
#
#     # for i in range(s_.shape[0]):
#     #     agent.update(s_[i], a_[i], r_[i], next_s_[i])
#
#     loss = compute_td_loss(s_, a_, r_, next_s_, done_, agent, target_network)
#
#     loss.backward()
#     grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
#     opt.step()
#     opt.zero_grad()
#
#     stop = False
#
#     if stop:
#         break
#
#     if i % 15 == 0:
#         # Load agent weights into target_network
#         loss_log.append(loss)
#         target_network.load_state_dict(agent.state_dict())
#         if i % 4*15 == 0:
#             rew_log.append(evaluate(
#                 make_env(seed=step, rocket_info=rocket_info, target_info=target_info),
#                 agent, n_games=3, greedy=True, **reset_params()))
#             print(rew_log[-1])
#
#         print(loss_log[-1])#, rew_log[-1])


# if np.floor(i / 30) == 20:
#     break

""" Train loop """


class Tester:
    def __init__(self, nn_lst, buffer_size):
        self.env = make_env(seed=seed, rocket_info=rocket_info, target_info=target_info)

        self.state_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        self.state = self.env.reset(**reset_params())

        self.agent = DQNAgent(self.state_shape, self.n_actions, layers_lst=nn_lst, epsilon=1).to(device)

        self.target_network = DQNAgent(self.state_shape, self.n_actions, layers_lst=nn_lst).to(device)
        self.target_network.load_state_dict(self.agent.state_dict())

        self.timesteps_per_epoch = 100
        self.batch_size = 400
        self.total_steps = 3 * 10 ** 6
        self.decay_steps = 10 ** 5

        self.opt = torch.optim.Adam(self.agent.parameters(), lr=1e-4)

        self.init_epsilon = 0.8
        self.final_epsilon = 0.01

        self.loss_freq = 10
        self.refresh_target_network_freq = 50
        self.eval_freq = 50

        self.max_grad_norm = 50

        self.mean_rw_history = []
        self.td_loss_history = []
        self.grad_norm_history = []
        self.initial_state_v_history = []
        self.step = 0

        self.buffer_size = buffer_size
        self.exp_replay = ReplayBuffer(self.buffer_size)

    def fill_buffer(self):
        for i in range(1000):
            play_and_record(initial_state=reset_params,
                            agent=self.agent, env=self.env, exp_replay=self.exp_replay, n_steps=8 * 10 ** 2,
                            expert=False, prob_exp_random=0.5)

            if len(self.exp_replay) == self.buffer_size:
                break

    def train(self, total_steps, decay, save_folder):
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        state = self.env.reset(**reset_params())

        for step in trange(self.step, total_steps + 1):
            if not utils.is_enough_ram():
                try:
                    while True:
                        pass
                except KeyboardInterrupt:
                    pass

            self.agent.epsilon = utils.linear_decay(self.init_epsilon, self.final_epsilon, step, decay)

            # all_rew = exp_replay.sample(len(exp_replay))[2]
            # mean = all_rew[all_rew != 0].mean()
            _, state = play_and_record(reset_params, self.agent, self.env, self.exp_replay, self.timesteps_per_epoch)

            # train
            s_, a_, r_, next_s_, done_ = self.exp_replay.sample(self.batch_size)
            # for i in range(s_.shape[0]):
            #     agent.update(s_[i], a_[i], r_[i], next_s_[i])

            loss = compute_td_loss(s_, a_, r_, next_s_, done_, self.agent, self.target_network, gamma=0.95)

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.opt.step()
            self.opt.zero_grad()

            if step % self.loss_freq == 0:
                self.td_loss_history.append(loss.data.cpu().item())
                self.grad_norm_history.append(grad_norm)

            if step % self.refresh_target_network_freq == 0:
                # Load agent weights into target_network
                self.target_network.load_state_dict(self.agent.state_dict())

            # deb = True
            draw = True
            if step % self.eval_freq == 0:
                # if deb == True:
                self.mean_rw_history.append(evaluate(
                    make_env(seed=step, rocket_info=rocket_info, target_info=target_info),
                    self.agent, n_games=5, greedy=True, init_params=reset_params)
                )
                initial_state_q_values = self.agent.get_qvalues(
                    [make_env(seed=step, rocket_info=rocket_info, target_info=target_info).reset()]
                )
                self.initial_state_v_history.append(np.max(initial_state_q_values))

                clear_output(True)
                print("buffer size = %i, epsilon = %.5f" %
                      (len(self.exp_replay), self.agent.epsilon))

                print("Last reward = ", self.mean_rw_history[-1])

                if step % (self.eval_freq * 5) == 0 and draw:
                    plt.figure(figsize=[16, 9])

                    plt.subplot(2, 2, 1)
                    plt.title("Mean reward per life")
                    plt.plot(self.mean_rw_history)
                    plt.grid()

                    assert not np.isnan(self.td_loss_history[-1])
                    plt.subplot(2, 2, 2)
                    plt.title("TD loss history (smoothened)")
                    plt.plot(utils.smoothen(self.td_loss_history))
                    plt.grid()

                    plt.subplot(2, 2, 3)
                    plt.title("Initial state V")
                    plt.plot(self.initial_state_v_history)
                    plt.grid()

                    plt.subplot(2, 2, 4)
                    plt.title("Grad norm history (smoothened)")
                    plt.plot(utils.smoothen(self.grad_norm_history))
                    plt.grid()

                    plt.savefig(f"{save_folder}/{step}_log.png")
                    torch.save(self.agent, f"{save_folder}/{step}_agent.pt")
                    plt.close("all")

test_list = []

def layer(neurons):
    return nn.Linear(neurons, neurons), activation()

activation = nn.LeakyReLU

neurons = 32

test_list.append([activation(),
                  *layer(neurons)
                  ])

test_list.append([activation(),
                  *layer(neurons),
                  *layer(neurons)
                  ])

test_list.append([activation(),
                  *layer(neurons),
                  *layer(neurons),
                  *layer(neurons)
                  ])

neurons = 64

test_list.append([activation(),
                  *layer(neurons)
                  ])

test_list.append([activation(),
                  *layer(neurons),
                  *layer(neurons)
                  ])

test_list.append([activation(),
                  *layer(neurons),
                  *layer(neurons),
                  *layer(neurons)
                  ])

neurons = 128

test_list.append([activation(),
                  *layer(neurons)
                  ])

test_list.append([activation(),
                  *layer(neurons),
                  *layer(neurons)
                  ])

test_list.append([activation(),
                  *layer(neurons),
                  *layer(neurons),
                  *layer(neurons)
                  ])


for i, net in enumerate(test_list):
    print(net)
    t1 = Tester(nn_lst=net,
                buffer_size=10 ** 4)

    print("Filling buffer")
    t1.fill_buffer()
    t1.train(total_steps=2000, decay=1000,
             save_folder=f"{os.getcwd()}/log/9/{int((len(net)-1)/2)}_{net[1].in_features}")



# t = reset_params()
# # t["t_euler"][1] = -t["t_euler"][1]
#
# reward = 0
# env.reset(**t)
# log = np.array(env.get_obs)
# true_overload = np.empty(2)
#
# for _ in range(800):
#     env.wrap.rocket.grav_compensate()
#     overload = env.wrap.rocket.proportionalCoefficients(k_z=10, k_y=10)
#     if true_overload.shape[0] == 0:
#         true_overload = np.hstack((true_overload, overload))
#     else:
#         true_overload = np.vstack((true_overload, overload))
#
#     possible = env.wrap.findClosestFromLegal(overload)
#     action_num = env.wrap.overloadsToNumber([possible])[0]
#
#     # print("True = ", overload, "Possible = ", possible)
#     s = env.observation(env.get_obs)
#     s = torch.tensor(s, device=device, dtype=torch.float)
#     action_num = torch.argmax(agent(s))
#
#     ob, r, done, info = env.step(action_num)
#
#     reward += r
#
#     log = np.vstack((log, env.get_obs))
#     print("Distance to target = ", env.wrap.distance_to_target, np.rad2deg(env.wrap.rocket.angleToTarget))
#     if done:
#         break
#
# print("Reward = ", np.round(reward, 4))
#
# rocket_log = log[:, :26]
# la_log = log[:, 26:]
#
# graph(rocket_log=rocket_log, la_log=la_log, true_overload=true_overload)
#
# drawAnimation()

sys.exit()
